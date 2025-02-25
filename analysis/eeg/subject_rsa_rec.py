'''
This script performs subject-level reconstruction of gammatones from
the corresponding EEG data.

In brief, we collect audio and EEG data for morphs (during learning
task) and clear words (during listening task). We then build reconstruction
models that map EEG data back to gammatones (using temporal response
functions). Here, we train on morphs (and use the test set to build
out-of-sample predictions for all morphs, too).

NOTE: This script is relatively memory intensive. For me, one
run typically takes about 90 minutes and requires about 32GB of RAM.
Of course, this will also depend on your architecture (as well as `n_workers`).

OPTIONS:
    id                  -   Participant identifier (pid/sid).
    fs                  -   Sampling frequency of gammatones (default = 200; must match `subject_rsa_enc.py`).
    n_k1                -   Length of kernel to apply to EEG data (in samples) (default = 1).
    n_k2                -   Length of kernel to apply to gammatones (in samples) (default = 20).
    off_L               -   Length of offset to apply to gammatones (in samples) (default = 100) (i.e., removes baseline).
    max_L               -   Maximum length of gammatones (in samples) (default = 300).
    b_clear             -   Should we use clear words as training data? (default = False)
    n_folds             -   Number of folds for cross-validation (default = 5).
    n_permutations      -   Number of permutations to run (default = 100).
    n_workers           -   Number of workers for parallelisation (default = 4).
    n_alphas            -   Number of alphas to test (default = 20).
    t_min               -   Minimum time to consider for reconstruction model (in seconds) (default = 0.0).
    t_max               -   Maximum time to consider for reconstruction model (in seconds) (default = 0.25).
    backend             -   Which backend to use (numpy/torch, default = torch)?
    device              -   Which device to use, if applicable (cpu/cuda/mps, default = cuda)?
    maxtask             -   After `n` tasks, we should reset forked processes. (default = 1; this is relevant for CUDA, can be higher on CPU)

EXAMPLE USAGE:
    python subject_rsa_rec.py id=0002 n_permutations=10
'''

try:
    '''
    NOTE: On intel machines, sklearn-intelex should be
    installed to speed up sklearn quite substantially.
    
    Please see:
    
        https://github.com/intel/scikit-learn-intelex
    '''
    
    from sklearnex import patch_sklearn
    patch_sklearn(verbose = False)
except:
    pass

import time
import numpy as np
import torch
import pandas as pd
import mne, os, sys, copy
import auxiliary as aux
import data, rsa
import gzip, pickle
import scipy, sklearn, warnings
from multiprocessing import freeze_support, set_start_method

sys.path.append('../spaces/')
import embeddings as emb

from typing import Union, Dict, Any

def load_eeg(f_in: str, fs: int = 200) -> np.ndarray:
    '''
    Loads the eeg data and passes the relevant segment on
    to be downsampled to the desired output frequency. Note
    that downsampling is performed through binning here.
    
    INPUTS:
        f_in        -   File to load
        fs          -   Desired sampling frequency (default = 200Hz)
    
    OUTPUTS:
        binned      -   Binned eeg segments
    '''
    
    # read data and info
    eeg = mne.read_epochs(f_in, verbose = False)
    info = eeg.info
    current_fs = info['sfreq']
    eeg = eeg.get_data(copy = False)
    
    return rsa.signal.binned(eeg, fs = fs, current_fs = current_fs), info

def worker_reconstruct(task: Dict, settings: tuple[Any]) -> Dict:
    '''
    Performs one permutation of the reconstruction.
    
    INPUTS:
        task        -   Task to perform. Dictionary includes:
                            p_i:    Permutation index
                            i:      Iteration index
                            indc:   Indices of the current permutation
        settings    -   Settings for the reconstruction task. Tuple includes:
                            ts:         Time stamp
                            N:          Number of permutations in total
                            n_folds:    Number of folds
                            alphas:     Alphas to test
                            t_min:      Minimum time for reconstruction
                            t_max:      Maximum time for reconstruction
                            fs:         Sampling frequency
                            X_pl1:      Predictors for the reconstruction from clear words (`trials` x `channels` x `time`)
                            X_mt1:      Predictors for the reconstruction from morphs (`trials` x `channels` x `time`)
                            y:          Targets for morphs (`trials` x `features` x `time`)
                            backend:    Which backend to use?
                            device:     Which device to use?

    OUTPUTS:
        result  -   Dictionary containing:
                        p_i:    Permutation index
                        r:      Out of sample reconstruction performance
                        p:      Patterns recovered from reconstruction model
                        mt1:    Gammatones reconstructed (out of sample) from morphs
                        pl1:    Gammatones reconstructed (out of sample) from clear words
    '''
    
    # grab task and settings
    p_i, i, indc = task['p_i'], task['i'], task['indc']
    ts, N, n_folds, alphas, t_min, t_max, fs, X_pl1, X_mt1, y, backend, device = settings
    
    # log progress
    aux.progressbar(i, N, ts, msg = f'[REC]')
    
    # move data, if required
    if backend == 'torch':
        # check if cuda
        if device == 'cuda':
            # distribute across devices
            device = f'cuda:{i % torch.cuda.device_count()}'
        
        # move
        X_pl1, X_mt1, y = torch.from_numpy(X_pl1).to(torch.float64).to(device), torch.from_numpy(X_mt1).to(torch.float64).to(device), torch.from_numpy(y).to(torch.float64).to(device)
        alphas = torch.from_numpy(alphas).to(torch.float64).to(device)
    
    # setup cross-validation
    kf = sklearn.model_selection.KFold(n_splits = n_folds)
    
    # setup models
    if backend == 'torch': rf = rsa.analysis.estimators.torch.TimeDelayed(t_min, t_max, fs, alphas = alphas, patterns = True)
    else: rf = rsa.analysis.estimators.TimeDelayed(t_min, t_max, fs, alphas = alphas, patterns = True)
    
    # setup data containers
    if backend == 'torch':
        p_h = torch.zeros((n_folds, y.shape[1], X_mt1.shape[1], rf.window.shape[0]), dtype = X_pl1.dtype, device = X_pl1.device)
        y_h_mt1 = torch.zeros_like(y, dtype = X_pl1.dtype, device = X_pl1.device) * torch.nan
        y_h_pl1 = torch.zeros((n_folds, X_pl1.shape[0], y.shape[1], y.shape[2]), dtype = X_pl1.dtype, device = X_pl1.device) * torch.nan
        oos_r = torch.zeros((n_folds, y.shape[1]), dtype = X_pl1.dtype, device = X_pl1.device)
    else:
        p_h = np.zeros((n_folds, y.shape[1], X_mt1.shape[1], rf.window.shape[0]))
        y_h_mt1 = np.zeros_like(y) * np.nan
        y_h_pl1 = np.zeros((n_folds, X_pl1.shape[0], y.shape[1], y.shape[2])) * np.nan
        oos_r = np.zeros((n_folds, y.shape[1]))
    
    # loop over folds
    for f_i, (kf_train, kf_test) in enumerate(kf.split(X_mt1, y)):
        # get shuffled indices
        train, test = indc[kf_train], indc[kf_test]
        
        if backend == 'torch':
            train, test = torch.from_numpy(train).to(torch.int64).to(device), torch.from_numpy(test).to(torch.int64).to(device)
        
        # get mean and std
        if backend == 'torch':
            X_mu, X_sigma = torch.mean(X_mt1[train,:,:], dim = 0, keepdim = True), torch.std(X_mt1[train,:,:], dim = 0, keepdim = True)
            y_mu, y_sigma = torch.mean(y[train,:,:], dim = 0, keepdim = True), torch.std(y[train,:,:], dim = 0, keepdim = True)
        else:
            X_mu, X_sigma = X_mt1[train,:,:].mean(axis = 0, keepdims = True), X_mt1[train,:,:].std(axis = 0, keepdims = True)
            y_mu, y_sigma = y[train,:,:].mean(axis = 0, keepdims = True), y[train,:,:].std(axis = 0, keepdims = True)
        
        # normalise data (in audio, NaN may occur)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            X0_i = (X_mt1 - X_mu) / X_sigma
            X1_i = (X_pl1 - X_mu) / X_sigma
            y_i = (y - y_mu) / y_sigma
            
            if backend == 'torch':
                X0_i[torch.isnan(X0_i)] = 0.0
                X1_i[torch.isnan(X1_i)] = 0.0
                y_i[torch.isnan(y_i)] = 0.0
            else:
                X0_i[np.isnan(X0_i)] = 0.0
                X1_i[np.isnan(X1_i)] = 0.0
                y_i[np.isnan(y_i)] = 0.0
        
        # train model
        rf.fit(X0_i[train,:,:], y_i[train,:,:])
        
        # grab patterns
        if backend == 'torch': p_h[f_i,:,:,:] = rf.pattern_.to(p_h.dtype)
        else: p_h[f_i,:,:,:] = rf.pattern_.copy()
        
        if backend == 'torch':
            # evaluate model
            y_h_mt1[test,:,:] = rf.predict(X0_i[test,:,:], invert_y = True).to(y_h_mt1.dtype)
            oos_r[f_i,:] = torch.nanmean(rsa.math.torch.pearsonr(y_h_mt1[test,:,:].swapaxes(0, 2),
                                                                 y_i[test,:,:].swapaxes(0, 2)), dim = 0)
            
            # apply to clear data
            y_h_pl1[f_i,:,:,:] = rf.predict(X1_i, invert_y = True).to(y_h_pl1.dtype)
        else:
            # evaluate model
            y_h_mt1[test,:,:] = rf.predict(X0_i[test,:,:], invert_y = True)
            oos_r[f_i,:] = np.nanmean(rsa.math.pearsonr(y_h_mt1[test,:,:].swapaxes(0, 2),
                                                        y_i[test,:,:].swapaxes(0, 2)), axis = 0)

            # apply to clear data
            y_h_pl1[f_i,:,:,:] = rf.predict(X1_i, invert_y = True)
        
        # finally, undo transforms
        y_h_mt1[test,:,:] = (y_h_mt1[test,:,:] * y_sigma) + y_mu
        y_h_pl1[f_i,:,:,:] = (y_h_pl1[f_i,:,:,:] * y_sigma) + y_mu
    
    # move to cpu now, if required
    if backend == 'torch':
        p_h = p_h.cpu().numpy()
        y_h_mt1 = y_h_mt1.cpu().numpy()
        y_h_pl1 = y_h_pl1.cpu().numpy()
        oos_r = oos_r.cpu().numpy()
    
    # average over folds
    y_h_pl1 = np.nanmean(y_h_pl1, axis = 0)
    oos_r = np.nanmean(oos_r, axis = 0)
    p_h = np.nanmean(p_h, axis = 0)
    
    return dict(p_i = p_i, r = oos_r, p = p_h, mt1 = y_h_mt1, pl1 = y_h_pl1)

if __name__ == '__main__':
    '''
    Start reconstructing gammatones.
    '''
    
    # call freeze support for MP later
    freeze_support()
    
    # make sure we are not going to have trouble with locks from fork()
    set_start_method('spawn')
    
    # grab participant
    id = aux.get_opt('id', cast = str)      # identifier (either sid or pid)
    subject = data.Subjects[id]             # load subject
    sid, pid = subject.sid, subject.pid     # get session & participant id
    
    # processing options
    fs = aux.get_opt('fs', default = 200, cast = int)                           # frequency to downsample to (and reconstruct gammatones at)
    n_k1 = aux.get_opt('n_k1', default = 1, cast = int)                         # length of smoothing kernel for neural data (in samples)
    n_k2 = aux.get_opt('n_k2', default = 20, cast = int)                        # length of smoothing kernel for gammatones (in samples)
    off_L = aux.get_opt('off_L', default = 100, cast = int)                     # offset where data begins (i.e., cuts off prestim)
    max_L = aux.get_opt('max_L', default = 300, cast = int)                     # maximum length of episodes to consider (including prestim)
    b_clear = bool(aux.get_opt('b_clear', default = 0, cast = int))             # should we train on clear data?
    n_folds = aux.get_opt('n_folds', default = 5, cast = int)                   # number of folds for cross-validation
    n_permutations = aux.get_opt('n_permutations', default = 100, cast = int)   # number of permutations to run
    n_workers = aux.get_opt('n_workers', default = 4, cast = int)               # number of workers for parallelisation
    n_alphas = aux.get_opt('n_alphas', default = 20, cast = int)                # number of alphas to test
    t_min = aux.get_opt('t_min', default = 0.0, cast = float)                   # minimum time to consider for reconstruction model (in seconds)
    t_max = aux.get_opt('t_max', default = 0.25, cast = float)                  # maximum time to consider for reconstruction model (in seconds)
    backend = aux.get_opt('backend', default = 'torch', cast = str)             # which backend to use (numpy/torch)?
    device = aux.get_opt('device', default = 'cpu', cast = str)                 # which device to use, if applicable (cpu/cuda/mps)?
    maxtask = aux.get_opt('maxtask', default = 1, cast = int)                   # for multiprocessing, when do we kill a child? this is for efficient garbage collection (and keeping memory required to a minimum)

    print(f'-------- RSA: REC --------')
    print(f'[OPTS]\tsid={sid}\tpid={pid}')
    print(f'[OPTS]\tfs={fs}\tb_clear={b_clear}')
    print(f'[OPTS]\tn_k1={n_k1}\tn_k2={n_k2}')
    print(f'[OPTS]\toff_L={off_L}\tmax_L={max_L}')
    print(f'[OPTS]\tfolds={n_folds}\tpermutations={n_permutations}')
    print(f'[OPTS]\tworkers={n_workers}\talphas={n_alphas}')
    print(f'[OPTS]\tt_min={t_min}\tt_max={t_max}')
    print(f'[OPTS]\tbackend={backend}\tdevice={device}')
    print(f'--------------------------')
    
    # load & downsample (through binning) eeg data from both tasks
    print(f'[PRE] Loading and binning eeg data...')
    X_pl1, _ = load_eeg(f'./data/preprocessed/eeg/sub{sid}/rsa-PL1-epo.fif', fs = fs)
    X_mt1, info = load_eeg(f'./data/preprocessed/eeg/sub{sid}/rsa-MT1-epo.fif', fs = fs)
    
    # load behaviour from both tasks
    print(f'[PRE] Loading raw behaviour...')
    df = pd.read_csv(f'./data/raw/beh/sub{sid}/{pid}.csv', sep = ',')
    beh_pl1 = df.loc[(df.type == data.defs.TRIAL_PL1)].reset_index(drop = True)
    beh_mt1 = df.loc[(df.type == data.defs.TRIAL_MT_PRACTICE) | (df.type == data.defs.TRIAL_MT_CONTROL) | (df.type == data.defs.TRIAL_MT_MAIN)].reset_index(drop = True)
    
    # load embedding
    print(f'[PRE] Loading embedding...')
    G = emb.glove.load_embedding(f_in = './data/preprocessed/misc/glove/w2v_50D.txt') # mini version
    
    # start loading audio
    print(f'[PRE] Loading audio data...')
    y_p = np.zeros((len(beh_pl1), 28, max_L))
    
    # loop over clera trials quickly
    for i in range(X_pl1.shape[0]):
        # grab trial
        trial = beh_pl1.loc[i]
        
        # grab envelope
        f = trial.stimulus.split('/')[-1].split('.')[0]
        audio = np.load(f'./data/preprocessed/audio/gt-{fs}Hz/clear/{f}.npy')
        y_p[i,:,100:100+audio.shape[1]] = audio[:,0:y_p.shape[2]-100]
    
    # setup dummies for morphs
    labels = []
    targets = []
    clear_t, clear_a = [], []
    
    y = np.zeros((len(beh_mt1), 28, max_L))
    y_t = y.copy()
    y_a = y.copy()
    
    min_L = np.inf
    
    # loop over trials
    for i in range(X_mt1.shape[0]):
        # grab trial
        trial = beh_mt1.loc[i]
        
        # make sure it isnt control or practice
        if trial.type in [data.defs.TRIAL_MT_CONTROL, data.defs.TRIAL_MT_PRACTICE]: 
            labels.append('-')
            targets.append('-')
            clear_t.append(0)
            clear_a.append(0)
            continue

        # grab envelope 
        f = '.'.join(trial.stimulus.split('/')[-1].split('.')[0:-1])
        audio = np.load(f'./data/preprocessed/audio/gt-{fs}Hz/morphed/{f}.npy')
        y[i,:,100:100+audio.shape[1]] = audio[:,0:y.shape[2]-100]
        
        # grab identifying information
        a, b = f.split('_')[0].split('-')
        v0, v1 = f.split('_')[1].split('-')
        r = f.split('_')[4][1]
        
        # find target
        target = trial.options_0
        target = 'wert' if target == 'Wert' else target
        target = 'warten' if target == 'waten' else target
        
        # find mapping
        T = 'T' if target == a else 'D'
        Tv = v0 if target == a else v1
        A = 'D' if target == a else 'T'
        Av = v1 if target == a else v0
        
        # load clear files
        f_t = np.load(f'./data/preprocessed/audio/gt-{fs}Hz/clear/{a}-{b}_{T}{Tv}.npy')
        y_t[i,:,100:100+f_t.shape[1]] = f_t[:,0:y_t.shape[2]-100]
        
        f_a = np.load(f'./data/preprocessed/audio/gt-{fs}Hz/clear/{a}-{b}_{A}{Av}.npy')
        y_a[i,:,100:100+f_a.shape[1]] = f_a[:,0:y_a.shape[2]-100]

        # find indices of clear audio
        indx_t = np.where(beh_pl1.stimulus == f'./audio_clear/{a}-{b}_{T}{Tv}.wav')[0][0]
        indx_a = np.where(beh_pl1.stimulus == f'./audio_clear/{a}-{b}_{A}{Av}.wav')[0][0]
        clear_t.append(indx_t)
        clear_a.append(indx_a)
        
        # keep track of L
        min_L = np.array([f_t.shape[1], f_a.shape[1], audio.shape[1], min_L]).min()
        
        # add label
        labels.append(f)
        targets.append(trial.options_0)
    
    # cast as numpy
    clear_t, clear_a = np.array(clear_t), np.array(clear_a)
    labels, targets = np.array(labels), np.array(targets)
    
    # remove invalid trials
    valid = np.where(labels != '-')[0]
    labels, targets = labels[valid], targets[valid]
    clear_t, clear_a = clear_t[valid], clear_a[valid]
    X_mt1 = X_mt1[valid,:,:]
    y, y_t, y_a = y[valid,:,:], y_t[valid,:,:], y_a[valid,:,:]
    
    # apply smoothing
    kernel_n = rsa.signal.boxcar(n_k1)
    kernel_a = rsa.signal.boxcar(n_k2)
    
    X_pl1, X_mt1 = rsa.signal.smoothen(X_pl1, kernel_n, axis = 2), rsa.signal.smoothen(X_mt1, kernel_n, axis = 2)
    y, y_t, y_a, y_p = rsa.signal.smoothen(y, kernel_a, axis = 2), rsa.signal.smoothen(y_t, kernel_a, axis = 2), rsa.signal.smoothen(y_a, kernel_a, axis = 2), rsa.signal.smoothen(y_p, kernel_a, axis = 2)
    
    # cut to relevant segments
    X_pl1, X_mt1 = X_pl1[:,:,off_L:max_L], X_mt1[:,:,off_L:max_L]
    y, y_t, y_a, y_p = y[:,:,off_L:max_L], y_t[:,:,off_L:max_L], y_a[:,:,off_L:max_L], y_p[:,:,off_L:max_L]
    
    # rescale SPLs real quick (morphs were SPL scaled, clear words were not)
    max_y = y.max(axis = (1, 2), keepdims = True)
    max_t = y_t.max(axis = (1, 2), keepdims = True)
    max_a = y_a.max(axis = (1, 2), keepdims = True)
    max_p = y_p.max(axis = (1, 2), keepdims = True)
    
    y_t = y_t * (max_y / max_t)
    y_a = y_a * (max_y / max_a)
    y_p = y_p * (max_y.max() / max_p)
    
    # finally, make a decision: What data do we train on?
    if b_clear:
        # here, we simply swap the variables - it's dirty but does the trick
        X_pl1, X_mt1, y = X_mt1, X_pl1, y_p
    
    # setup jobs
    print(f'[REC] Setting up jobs...')
    jobs = []
    indcs = np.array([np.random.choice(np.arange(0, X_mt1.shape[0], 1), size = (X_mt1.shape[0],), replace = False) for _ in np.arange(0, n_permutations, 1)])
    alphas = np.logspace(-5, 10, n_alphas)
    
    for p_i in range(n_permutations):
        jobs.append(dict(p_i = p_i, i = len(jobs), indc = indcs[p_i,:]))
    
    # run jobs
    print(f'[REC] Reconstructing gammatones...')
    processor = emb.mp.Processor(workers = n_workers, maxtasksperchild = maxtask)
    ts = time.time()
    outputs = processor.run(jobs, external = worker_reconstruct, settings = (ts, len(jobs), n_folds, alphas, t_min, t_max, fs, X_pl1, X_mt1, y, backend, device))
    
    # setup data structure
    print(f'')
    print(f'[REC] Collecting results...')
    window = (np.arange(t_min, t_max + 1 / fs, 1 / fs) * fs).astype(int)
    
    r = np.zeros((n_permutations, y.shape[1]))
    P = np.zeros((n_permutations, y.shape[1], X_mt1.shape[1], window.shape[0]))
    
    y_h_mt1 = np.zeros((n_permutations, *y.shape))
    y_h_pl1 = np.zeros((n_permutations, X_pl1.shape[0], y.shape[1], X_pl1.shape[2]))
    
    # loop over outputs
    for output in outputs:
        # get tags
        p_i = output['p_i']
        
        # grab data
        r[p_i,:] = output['r']
        P[p_i,:,:,:] = output['p']
        
        y_h_mt1[p_i,:,:,:] = output['mt1']
        y_h_pl1[p_i,:,:,:] = output['pl1']
    
    # make sure directory exists
    dir_out = f'./data/processed/eeg/sub{sid}/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    # save results
    print(f'[REC] Saving results...')
    
    if b_clear == False:
        with gzip.open(f'{dir_out}rec-data.pkl.gz', 'wb') as f:
            pickle.dump((np.nanmean(r, axis = 0), np.nanmean(P, axis = 0), labels, targets, clear_t, clear_a, y, y_t, y_a, np.nanmean(y_h_mt1, axis = 0), np.nanmean(y_h_pl1, axis = 0)), f)
    else:
        with gzip.open(f'{dir_out}rec-data-clear.pkl.gz', 'wb') as f:
            pickle.dump((np.nanmean(r, axis = 0), np.nanmean(P, axis = 0), labels, targets, clear_t, clear_a, y, y_t, y_a, np.nanmean(y_h_pl1, axis = 0), np.nanmean(y_h_mt1, axis = 0)), f)