'''
This script performs a subject-level encoding analysis of the
full between-item representational similarity of decoded percepts
during our learning task.

In brief, unlike our main approach where we model within-item
similarity (because it allows us to make relatively few hard
mathematical commitments while being highly sensitive to the
directionality of the effect), here we construct full RDMs of
size 240x240x200 (instead of 120x2x2x200). We repeat this for
all predictors of interest, which--wherever possible--we split
into sharpening and prediction error components:

    sharpening = observation x expectation
    prediction error = observation - expectation

before running linear encoders to find the (combination of)
predictors that best explains the observed sensory RSMs (using
repeated cross-validation).

NOTE: This complementary approach was included to address some
methodological concerns of reviewer #4, but results con-
verge with our main findings.

NOTE: Because reviewer #4 was also skeptical of the top-k spect-
rogram approach, here we set k=1 such that there is simply no
averaging. We also use this script for controlling for the in-
fluence of length, by setting the temporal window of our analysis
to 0.48s only (which is the shortest possible audio length).

NOTE: This script now relies on MVPy which is a package I am
writing to make analysis on GPUs easier. If you are at all
familiar with sklearn and MNE, the syntax here should be very
easy to follow.

If you want to run this, please make sure you have MVPy installed:

    pip install git+https://github.com/FabulousFabs/MVPy.git

NOTE: This requires that subject_rsa_rec.py has been run to re-
construct gammatones from the morph data.

OPTIONS:
    id                  -   Participant identifier (pid/sid).
    fs                  -   Sampling frequency of gammatones (default = 200; must match `subject_rsa_rec.py`).
    n_topk              -   Number of top-k items to compute (default = 5).
    n_k1                -   Length of kernel to apply to similarity matrices (in samples) (default = 10).
    pad_L               -   Length of padding to apply to similarity matrices (in samples) before smoothing (should be >= n_k1) (default = n_k1).
    n_mod               -   Should a specific model be fit _only_? (-1 = False, else 0, 1, 2, 3)
    b_brain             -   Should we use targets/alternatives extracted from EEG data? (default = False)
    b_clear             -   Should we use clear words as training data? (default = False)
    b_acoustic          -   Should we use acoustic features?
    b_semantic          -   Should we use semantic features?
    n_folds             -   Number of folds for cross-validation (default = 5).
    n_permutations      -   Number of permutations to run (default = 50).
    n_workers           -   Number of workers for parallelisation (default = 5).
    n_alphas            -   Number of alphas to test (default = 20).
    backend             -   Which backend to use (numpy/torch, default = torch)?
    device              -   Which device to use, if applicable (cpu/cuda/mps, default = cuda)?
    maxtask             -   After `n` tasks, we should reset forked processes. (default = 1; this is relevant for CUDA, can be higher on CPU)

EXAMPLE USAGE:
    python subject_rsa_enc_full.py id=0002 n_permutations=10
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
from datetime import timedelta

sys.path.append('../spaces/')
import embeddings as emb

import mvpy as mv
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold

from typing import Union, Dict, Any, Callable

def get_topk(n_topk: int, beh: pd.core.frame.DataFrame, targets: np.ndarray, priors: np.ndarray, G: Any, va: int, context: Union[str, None] = None, contexts: np.ndarray = np.array(['essen', 'fashion', 'outdoor', 'technik', 'politik', 'unterhaltung'])) -> np.ndarray:
    '''
    Obtain the expected audio given a top-k prediction.
    
    INPUTS:
        n_topk   -  Number of top-k predictions to consider
        beh      -  Behavioral data frame
        targets  -  Array of all the target words
        priors   -  Semantic priors
        G        -  Semantic space
        va       -  Variant_a of the morph
        context  -  Context of the morph
        contexts -  All contexts (in the correct order, relative to priors) (default = ['essen', 'fashion', 'outdoor', 'technik', 'politik', 'unterhaltung'])
    
    OUTPUTS:
        exp      -  Expected audio (`features` x `time`)
    '''
    
    # find context id and candidates
    if context is None:
        c_id = 0
        candidates = np.unique(beh.options_0.tolist())
    else: 
        c_id = np.where(contexts == context)[0]
        candidates = np.unique(beh.loc[(beh.context == context)].options_0.tolist())
    
    # find semantics and goodness of fit
    semantics = np.zeros((len(candidates), 50))
    fits = np.zeros((len(candidates)))
    
    for j, word in enumerate(candidates):
        semantics[j,:] = G[word.lower()]
        fits[j] = rsa.math.cosine(priors[c_id,:].squeeze(), semantics[j,:])
    
    # determine top-k
    order = fits.argsort()[::-1]
    topk = order[0:n_topk]
    
    # setup expectations
    expectation = np.zeros((n_topk, max_L, 28))
    
    # ignore NaNs produced by zeros
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        for j, k in enumerate(topk):
            # find morph where this item would be target
            indx = np.where(targets == candidates[k])[0][0]

            # find associated a & b
            morph_k = labels[indx]
            k_a, k_b = morph_k.split('_')[0].split('-')

            # find associated audio file
            T = 'T' if k_a.lower() == candidates[k].lower() else 'D'

            # load gammatone
            y_k = np.load(f'./data/preprocessed/audio/gt-{fs}Hz/clear/{k_a}-{k_b}_{T}{va}.npy')

            # track expectation
            L_k = min(200, y_k.shape[1])
            expectation[j,0:L_k] = y_k[:,0:L_k].T
    
    # create acoustic prior
    prior_k = (fits[topk] + 1 ) / 2 # normalise to [0, 1]
    prior_k = prior_k / prior_k.sum() # take softmax
    exp_audio = (prior_k[:,None,None] * expectation).sum(axis = 0).T
    
    return exp_audio

def worker_encode(task: Dict, settings: tuple[Any]) -> Dict:
    '''
    Compute all permutations of the RSA encoding model.
    
    task = dict(p_i = 0, i = 1, indc = np.arange(rdms.shape[0])),
    settings = (ts, n_permutations, n_folds, alphas, rdms, n_mod, n_workers, backend, device)
    
    INPUTS:
        task        - Dictionary of task parameters:
                        p_i: Permutation index (unused here)
                        i: Current iteration (unused here)
                        indc: Indices to use for starting permutation
        settings    - Tuple of settings
                        ts: Start time
                        N: number of permutations
                        n_folds: Number of folds
                        alphas: Alphas to test
                        rdms: All pre-computed RDMs (y, morph, rnd-acc (sh), rnd-acc (pe), rnd-sem (sh), rnd-sem (pe), pred-acc-sp (sh), pred-acc-inv (sh), pred-acc-sp (pe), pred-acc-inv (pe), pred-sem-sp (sh), pred-sem-inv (sh), pred-sem-sp (pe), pred-sem-inv (pe))
                        n_mod: Specific model to compute (or -1 to compute all)
                        n_workers: Workers to use in mvpy
                        backend: Backend to use for mvpy
                        device: Device to use for mvpy
    OUTPUTS:
        result      -   Dictionary of results:
                            p_i: Permutation index
                            ß: Raw coefficients from models
                            ß_r: Normalised coefficients (as R2).
                            r: Out of sample performance.
    '''
    
    # grab task and settings
    p_i, i, indc = task['p_i'], task['i'], task['indc']
    ts, N, n_folds, alphas, rdms, n_mod, n_workers, backend, device = settings

    # permute data
    X = rdms[indc][:,1:,:]
    y = rdms[indc][:,0,None,:]
    
    # setup tests
    test_models = np.array([
        # morph, rnd-acc (sh), rnd-acc (pe), rnd-sem (sh), rnd-sem (pe), pred-acc-sp (sh), pred-acc-inv (sh), pred-acc-sp (pe), pred-acc-inv (pe), pred-sem-sp (sh), pred-sem-inv (sh), pred-sem-sp (pe), pred-sem-inv (pe)
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], #  baseline
        [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], # + pred-acc-inv   (sh)
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], # + pred-acc-sp    (sh)
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], # + pred-acc-both  (sh)
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0], # + pred-acc-inv   (pe)
        [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], # + pred-acc-sp    (pe)
        [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0], # + pred-acc-both  (pe)
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], # + pred-acc-both  (sh+pe)
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0], # + pred-sem-inv   (sh)
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0], # + pred-sem-sp    (sh)
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0], # + pred-sem-both  (sh)
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], # + pred-sem-inv   (pe)
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0], # + pred-sem-sp    (pe)
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1], # + pred-sem-both  (pe)
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], # + pred-sem-both  (sh+pe)
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0], # + acc-sem        (sh)
        [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], # + acc-sem        (pe)
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # + full           (sh+pe)
    ]).astype(bool)
    
    # transfer data
    if backend == 'torch':
        alphas = torch.from_numpy(alphas).float().to(device)
        X = torch.from_numpy(X).float().to(device)
        y = torch.from_numpy(y).float().to(device)
        test_models = torch.from_numpy(test_models).to(torch.bool).to(device)
    
    # setup models and kfold
    model = make_pipeline(
        mv.estimators.Scaler().to_numpy() if backend == 'numpy' else mv.estimators.Scaler().to_torch(),
        mv.estimators.Sliding(
            mv.estimators.Encoder(
                alphas
            ),
            dims = np.array([-1]) if backend == 'numpy' else torch.tensor([-1]),
            n_jobs = n_workers,
            verbose = False
        )
    )

    kf = RepeatedKFold(
        n_splits = n_folds,
        n_repeats = N
    )
    
    # setup dummies
    oos_r = np.zeros((test_models.shape[0], n_folds * N, X.shape[-1]))
    ß = np.zeros((test_models.shape[0], n_folds * N, test_models.shape[1], X.shape[-1]))
    
    if backend == 'torch':
        oos_r = torch.from_numpy(oos_r).float().to(device)
        ß = torch.from_numpy(ß).float().to(device)

    # run model
    eta = '-'
    for i, (train, test) in enumerate(kf.split(X[:,:,0])):
        time_taken = time.time() - ts
        after = str(timedelta(seconds = int(time_taken)))
        print(f'[ENC] Encoding RSA... {((i + 1) / (n_folds * N) * 100):06.2f}% after {after} (ETA: {eta})', end = '\r')
        
        for j, mask in enumerate(test_models):
            # check if we should skip
            if n_mod > -1:
                if j != n_mod:
                    continue
            
            # grab data
            X_j = X[:,mask,:]
            if backend == 'numpy':
                y_j = y.copy()
            else:
                y_j = y.clone()
            
            if len(X_j.shape) == 2:
                X_j = X_j[:,None,:]
            
            try:
                # fit model
                model.fit(X_j[train], y_j[train])

                # test model
                y_h = model.predict(X_j[test])
                oos_r[j,i,:] = mv.math.pearsonr(y_h.T, y_j[test].T).mean(-1)

                # save coefs
                ß[j,i,mask,:] = model[-1].collect('coef_').squeeze().T
            except:
                print(f'SVD precision error f_i={i}, m_j={j}')
                
                oos_r[j,i,:] = torch.nan if backend == 'torch' else np.nan
                ß[j,i,mask,:] = torch.nan if backend == 'torch' else np.nan
        
        time_taken = time.time() - ts
        completed = (i + 1) / (n_folds * N)
        time_eta = time_taken / completed - time_taken
        eta = str(timedelta(seconds = int(time_eta)))
    
    '''
    NOTE: To avoid issues of scaling (i.e., different models at different alphas), we will
    compute coefficients as R2 here in a manner very similar to:

        Gwilliams, L., King, J.R., Marantz, A., & Poeppel, D. (2022). Neural dynamics of phoneme sequences reveal position-invariant code for content and order. Nature Communications, 13, 6606. 10.1038/s41467-022-34326-1
    
    except that we use absolute values of ß for the normalisation, because both +- are 
    meaningful to us. We do, however, also save original coefficients for posterity.
    '''
    
    # move back to numpy device
    if backend == 'torch':
        ß = ß.cpu().numpy()
        oos_r = oos_r.cpu().numpy()
    
    ß_r = (ß / np.abs(ß).sum(axis = 2, keepdims = True).max(axis = 3, keepdims = True))
    
    # average over folds
    ß = np.nanmean(ß, axis = 1)
    ß_r = np.nanmean(ß_r, axis = 1)
    oos_r = np.nanmean(oos_r, axis = 1)
    
    return dict(p_i = p_i, ß = ß, ß_r = ß_r, r = oos_r)

if __name__ == '__main__':
    '''
    Start encoding of reconstructed gammatones.
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
    fs = aux.get_opt('fs', default = 200, cast = int)                           # frequency of reconstructed gammatones (must match subject_rsa_rec.py)
    n_topk = aux.get_opt('n_topk', default = 5, cast = int)                     # number of top-k predictions to consider
    n_k1 = aux.get_opt('n_k1', default = 10, cast = int)                        # length of smoothing kernel (in samples)
    pad_L = aux.get_opt('pad_L', default = n_k1, cast = int)                    # padding to apply before smoothing (should be >= n_k1)
    b_brain = bool(aux.get_opt('b_brain', default = 0, cast = int))             # should we use brain data for targets/alternatives?
    b_clear = bool(aux.get_opt('b_clear', default = 0, cast = int))             # should we load clears from clear training?
    b_morph = bool(aux.get_opt('b_morph', default = 0, cast = int))             # should we load morphs from clear training?
    n_mod = aux.get_opt('n_mod', default = -1, cast = int)                      # should we fit _only a specific_ model? (give index)
    n_folds = aux.get_opt('n_folds', default = 5, cast = int)                   # number of folds for cross-validation
    n_permutations = aux.get_opt('n_permutations', default = 50, cast = int)    # number of permutations to run
    n_workers = aux.get_opt('n_workers', default = 8, cast = int)               # number of parallel workers to use
    n_alphas = aux.get_opt('n_alphas', default = 20, cast = int)                # number of alphas to consider
    b_acoustic = bool(aux.get_opt('b_acoustic', default = 1, cast = int))       # should we include acoustic predictors in models?
    b_semantic = bool(aux.get_opt('b_semantic', default = 1, cast = int))       # should we include semantic predictors in models?
    backend = aux.get_opt('backend', default = 'torch', cast = str)             # which backend to use? (numpy/torch)
    device = aux.get_opt('device', default = 'cpu', cast = str)                 # which device to use, if applicable? (cpu/cuda/mps)
    maxtask = aux.get_opt('maxtask', default = 1, cast = int)                   # for multiprocessing, when do we kill a child? this is for efficient garbage collection (and keeping memory required to a minimum)
    
    print(f'-------- RSA: ENC --------')
    print(f'[OPTS]\tsid={sid}\tpid={pid}')
    print(f'[OPTS]\tfs={fs}\tn_k1={n_k1}')
    print(f'[OPTS]\tpad_L={pad_L}\tn_topk={n_topk}')
    print(f'[OPTS]\tb_brain={b_brain}')
    print(f'[OPTS]\tb_clear={b_clear}\tb_morph={b_morph}')
    print(f'[OPTS]\tn_mod={n_mod}')
    print(f'[OPTS]\tb_acoustic={b_acoustic}\tb_semantic={b_semantic}')
    print(f'[OPTS]\tn_folds={n_folds}\tn_permutations={n_permutations}')
    print(f'[OPTS]\tn_workers={n_workers}\tn_alphas={n_alphas}')
    print(f'[OPTS]\tbackend={backend}\tdevice={device}')
    print(f'--------------------------')
    
    # setup output directory
    dir_out = f'./data/processed/eeg/sub{sid}/'

    # load data from reconstruction
    print(f'[PRE] Loading reconstruction data...')
    with gzip.open(f'{dir_out}rec-data.pkl.gz', 'rb') as f:
        (r, P, labels, targets, clear_t, clear_a, y, y_t, y_a, y_h_mt1, y_h_pl1) = pickle.load(f)
    
    with gzip.open(f'{dir_out}rec-data-clear.pkl.gz', 'rb') as f:
        (_, _, _, _, _, _, _, _, _, y_h_mt1_c, y_h_pl1_c) = pickle.load(f)
    
    # decide what data to use (trained on clear? morph?)
    if b_clear:
        y_h_pl1 = y_h_pl1_c # set clears from clear
    
    if b_morph:
        y_h_mt1 = y_h_mt1_c # set morphs from clear
    
    # set maximum segment length
    max_L = y.shape[2]
    
    # load embedding
    print(f'[PRE] Loading embedding...')
    G = emb.glove.load_embedding(f_in = './data/preprocessed/misc/glove/w2v_50D.txt') # mini version
    
    # load behaviour
    print(f'[PRE] Loading behaviour...')
    df = pd.read_csv(f'./data/raw/beh/sub{sid}/{pid}.csv')
    beh = df.loc[(df.type == data.defs.TRIAL_MT_MAIN)].reset_index(drop = True)

    # setup contexts & priors
    contexts = np.array(['essen', 'fashion', 'outdoor', 'technik', 'politik', 'unterhaltung'])
    priors = np.random.normal(loc = 0, scale = 0.1, size = (6, 50))
    priors_uns = np.random.normal(loc = 0, scale = 0.1, size = (6, 50))

    # setup variables
    exp_audio = np.zeros_like(y)
    uns_audio = np.zeros_like(y)
    ran_audio = np.zeros_like(y)

    exp_sem = np.zeros((y.shape[0], 50, y.shape[2]))
    uns_sem = np.zeros_like(exp_sem)
    ran_sem = np.zeros_like(exp_sem)
    
    t_sem = np.zeros_like(exp_sem)
    a_sem = np.zeros_like(exp_sem)

    # loop over trials
    print(f'[PRE] Loading trial data...')
    for i in range(len(labels)):
        # grab trial data
        label, trial = labels[i], beh.loc[i]
        context = trial.context
        
        # grab descriptors from label
        a, b = label.split('_')[0].split('-')
        va, vb = label.split('_')[1].split('-')
        r = int(label.split('_')[4][1])
        
        # get context id
        c_id = np.where(contexts == context)[0][0]
        
        # get expected audios
        exp_audio[i,:,:] = get_topk(n_topk, beh, targets, priors, G, va, context = context)
        uns_audio[i,:,:] = get_topk(n_topk, beh, targets, priors_uns, G, va, context = None)
        ran_audio[i,:,:] = get_topk(n_topk, beh, targets, np.random.normal(loc = 0, scale = 0.1, size = (6, 50)), G, va, context = None)
        
        # track semantic prior
        exp_sem[i,:,:] = priors[c_id,:].squeeze()[:,None]
        uns_sem[i,:,:] = priors_uns[0,:].squeeze()[:,None]
        ran_sem[i,:,:] = np.random.normal(loc = 0, scale = 0.1, size = (6, 50))[0,:,None]
        t_sem[i,:,:] = G[trial.options_0.lower()].squeeze()[:,None]
        a_sem[i,:,:] = G[trial.options_1.lower()].squeeze()[:,None]
        
        # update priors
        with gzip.open(f'./data/raw/rtfe/sub{sid}/{pid}_t{trial.no}_{context}.pkl.gz', 'rb') as f:
            _, _, _, posterior, _, _ = pickle.load(f)
        priors = posterior.T
        
        # update unspecific priors
        with gzip.open(f'./data/raw/rtfe-unspecific/sub{sid}/{pid}_t{trial.no}_essen.pkl.gz', 'rb') as f:
            _, _, _, posterior, _, _ = pickle.load(f)
        priors_uns = posterior.T
    
    # compute hypotheses
    ran_sh = ran_audio * y
    exp_sh = exp_audio * y
    uns_sh = uns_audio * y
    ran_pe = (y - ran_audio)
    exp_pe = (y - exp_audio)
    uns_pe = (y - uns_audio)
    sem_ran_sh = ran_sem * t_sem
    sem_exp_sh = exp_sem * t_sem
    sem_uns_sh = uns_sem * t_sem
    sem_ran_pe = (t_sem - ran_sem)
    sem_exp_pe = (t_sem - exp_sem)
    sem_uns_pe = (t_sem - uns_sem)
    
    # compute similarities
    print(f'[RSA] Computing similarities...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        rsa_model = make_pipeline(
            mv.estimators.RSA(
                estimator = mv.math.cosine,
                verbose = True,
                n_jobs = n_workers
            ).to_numpy()
        )
        
        tri_x, tri_y = np.triu_indices(y_h_mt1.shape[0], k = 1)
        
        rdms = []
        
        for i, data in enumerate([
            y_h_mt1, y, ran_sh, ran_pe, sem_ran_sh, sem_ran_pe, exp_sh, uns_sh, exp_pe, uns_pe, sem_exp_sh, sem_uns_sh, sem_exp_pe, sem_uns_pe
        ]):
            rdm = rsa_model.fit(data)[-1].full_rdm()
            rdm[np.isnan(rdm)] = 0.0
            
            rdm = rsa.signal.smoothen(
                rdm,
                rsa.signal.boxcar(n_k1),
                axis = -1
            )
            
            rdms.append(rdm[tri_x, tri_y])

        rdms = np.array(rdms).swapaxes(0, 1)
    
    # start task
    print(f'[ENC] Starting RSA...')
    alphas = np.logspace(-5, 10, n_alphas)
    ts = time.time()
    
    outputs = worker_encode(
        task = dict(p_i = 0, i = 1, indc = np.arange(rdms.shape[0])),
        settings = (ts, n_permutations, n_folds, alphas, rdms, n_mod, n_workers, backend, device)
    )
    
    ß = outputs['ß']
    ß_r = outputs['ß_r']
    oos_r = outputs['r']
    
    # make sure directory exists
    dir_out = f'./data/processed/eeg/sub{sid}/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    # save results
    print(f'')
    print(f'[ENC] Saving results...')
    with gzip.open(f'{dir_out}rec-rdmF-enc-b{int(b_brain)}-m{int(b_morph)}-c{int(b_clear)}-k{int(n_topk)}-m{n_mod}.pkl.gz', 'wb') as f:
        pickle.dump((ß, ß_r, oos_r), f)