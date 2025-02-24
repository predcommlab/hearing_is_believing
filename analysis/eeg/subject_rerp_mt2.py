'''
This script performs subject-level rERP analysis in the final task. 

Briefly, we collect predictors for word onset, phoneme onset, 
acoustic envelopes, acoustic edges, phonotactic surprisal, 
lexical surprisal, invariant acoustic surprisal, invariant 
semantic surprisal, speaker acoustic surprisal, and speaker 
semantic surprisal before fitting temporal response functions 
over the EEG data.

We fit four types of models, including either baseline,
+acoustic-semantic, -acoustic+semantic, or +acoustic+semantic
effects of interest. These can be probed either for contributions
of speaker-specific or -invariant effects.

OPTIONS
    id                  -   Subject id
    fs                  -   Sampling frequency (default = 200)
    b_con               -   Should we include congruent trials? (default = True)
    b_inc               -   Should we include incongruent trials? (default = False)
    b_zsc               -   Should we normalise all data and predictors? (default = False)
    b_spa               -   Should we enforce sparsity in design matrices? (default = False)
    b_bsl               -   Should we apply a baseline correction before model fitting? (default = False)
    s_mod               -   Which types of predictors should we evaluate (spc/inv for specific/invariant, default = spc)?
    n_workers           -   Number of workers for parallelisation (default = 5)
    n_alphas            -   Number of alphas to test (default = 20)
    n_folds             -   Number of folds for cross-validation (default = 5)
    n_permutations      -   Number of permutations to run (default = 50)
    n_topk              -   Number of top-k items to compute (default = 5)
    n_edge_l            -   Length to consider for acoustic edges (default = 10)
    n_maxl              -   Maximum length of epoch (default = 2000)
    n_tmin              -   Minimum timepoint for temporal response functions (default = -0.8)
    n_tmax              -   Maximum timepoint for temporal response functions (default = 0.1)
    backend             -   What backend to use for model fitting? (numpy/torch, default = torch)
    device              -   What device to use for model fitting, if applicable? (cpu/cuda/mps, default = cuda)
    maxtask             -   After `n` tasks, we should reset forked processes. (default = 1; this is relevant for CUDA, can be higher on CPU)

EXAMPLE USAGE:
    python subject_rerp_mt2.py id=0002 n_permutations=10 backend=torch device=cuda
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

'''
NOTE: On intel machines, I would strongly recommend
installing an appropriate numpy version. For regular
versions, issues with SVDs may occur and model fitting
may generally be slower.

Please see:
    https://pypi.org/project/intel-numpy/

NOTE: If you are not on an intel machine, please make
sure to run numpy using CBLAS, if possible. On macs,
this should be the default. You can check by calling:

    import numpy as np
    np.show_config()
'''

import time, re
import numpy as np
import pandas as pd
import mne, os, sys, copy
import aux, data, rsa, models
import gzip, pickle
import scipy, sklearn, torch, warnings
from multiprocessing import freeze_support, set_start_method

sys.path.append('../spaces/')
import embeddings as emb

from typing import Union, Dict, Any, Callable

def load_eeg(f_in: str, fs: int = 200, bsl: bool = False) -> tuple[np.ndarray, Dict]:
    '''
    Loads the eeg data and passes the relevant segment on
    to be downsampled to the desired output frequency. Note
    that downsampling is performed through decimation.
    
    INPUTS:
        f_in        -   File to load
        fs          -   Desired sampling frequency (default = 200Hz)
        bsl         -   Should we baseline correct the data? (default = False)
    
    OUTPUTS:
        eeg         -   Preprocessed EEG data
        info        -   Info struct of data
    '''
    
    # read data and info
    eeg = mne.read_epochs(f_in, verbose = False)
    info = eeg.info
    current_fs = eeg.info['sfreq']
    
    # filter the data; this is principally useless
    # because we have already applied it to these
    # data, but i am keeping it here for posterity,
    # i guess
    eeg = eeg.filter(l_freq = None, h_freq = 15.0, method = 'iir', iir_params = dict(order = 1, ftype = 'butter'), verbose = False)
    
    # downsample to fs
    # NOTE: i'm not low pass filtering here because data are
    # already low-passed @ 15Hz, which is enough for 200Hz fs
    decimation = np.round(current_fs / fs).astype(int)
    eeg = eeg.decimate(decimation).get_data(copy = False)
    
    # apply baseline, if required
    if bsl:
        eeg = (eeg - eeg[:,:,0:100].mean(axis = 2, keepdims = True))
    
    return (eeg, info)

def load_annotation(f_in: str, fs: int = 200) -> pd.core.frame.DataFrame:
    '''
    Load an annotation file.
    
    INPUTS:
        f_in        -   File to load
        fs          -   Sampling frequency (default = 200)
    
    OUTPUTS:
        ann         -   Annotation data frame
    '''
    
    # load original data frame
    ann = pd.read_csv(f_in, sep = ';')
    
    # remove invalid tokens
    ann = ann.loc[(ann.MAU != '<p:>') & (ann.MAU != '?')].reset_index(drop = True)
    phoneme = re.compile(r'[^a-z@0-9]+', re.IGNORECASE)
    phonemes = [phoneme.sub('', tk) for tk in ann.MAU.tolist()]
    
    # grab info
    onsets = [int(on * (fs / 32000)) for on in ann.BEGIN.tolist()]
    lengths = [int(dur * (fs / 32000)) for dur in ann.DURATION.tolist()]
    offsets = [onset + length for onset, length in zip(onsets, lengths)]
    
    # add to data frame
    ann['phoneme'] = phonemes
    ann['onset'] = onsets
    ann['offset'] = offsets
    ann['length'] = lengths
    
    return ann

def get_topk_surprisal(observed: np.ndarray, n_topk: int, beh: pd.core.frame.DataFrame, targets: np.ndarray, priors: np.ndarray, G: Any, va: int, context: Union[str, None] = None, contexts: np.ndarray = np.array(['essen', 'fashion', 'outdoor', 'technik', 'politik', 'unterhaltung'])) -> np.ndarray:
    '''
    Obtain the acoustic surprisal given a top-k prediction.
    
    INPUTS:
        observed -  Observed gammatone (`features` x `time`)
        n_topk   -  Number of top-k predictions to consider
        beh      -  Behavioral data frame (learning task!)
        targets  -  Array of all the target words (learning task!)
        priors   -  Semantic priors
        G        -  Semantic space
        va       -  Variant_a of the word
        context  -  Context of the word
        contexts -  All contexts (in the correct order, relative to priors) (default = ['essen', 'fashion', 'outdoor', 'technik', 'politik', 'unterhaltung'])
    
    OUTPUTS:
        surprisal   -  Surprisal at observed gammatone (`time`)
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
    
    # setup expectation
    expectation = np.zeros((n_topk, observed.shape[1], 28))
    
    # ignore NaNs produced by zeros
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        for j, k in enumerate(topk):
            # find morph where this item would be target
            indx = np.where(targets == candidates[k])[0][0]

            # find associated a & b
            clear_k = labels[indx]
            k_a, k_b = clear_k.split('_')[0].split('-')

            # find associated audio file
            T = 'T' if k_a.lower() == candidates[k].lower() else 'D'
            # load gammatone
            y_k = np.load(f'./data/preprocessed/audio/gt-{fs}Hz/clear/{k_a}-{k_b}_{T}{va}.npy')
            
            # track expectation
            L_k = min(observed.shape[1], y_k.shape[1])
            expectation[j,0:L_k,:] = y_k[:,0:L_k].T
    
    # create acoustic prior
    prior_k = (fits[topk] + 1 ) / 2 # normalise to [0, 1]
    prior_k = prior_k / prior_k.sum() # take softmax
    
    # compute surprisal
    surprisal_k = -np.log2(rsa.math.cosine(observed.T[None,:,:] * np.ones_like(expectation), expectation) + 1)
    surprisal = np.nansum((prior_k[:,None] * surprisal_k), axis = 0)
    
    return surprisal

def permutation_worker(task: Dict, settings: Union[tuple, None] = None) -> Union[int, Dict]:
    '''
    Computes one permutation of the rERP model fitting procedure.
    
    INPUTS:
        task        -   Task data. Dictionary includes:
                            p_i:    Permutation index
                            indc:   Indicies to permute the data set
                            i:      Current iteration
        settings    -   Settings. Tuple includes:
                            ts:         Time stamp
                            N:          Total number of iterations
                            n_tmin:     Minimum time point to consider for TRF model
                            n_tmax:     Maximum time point to consider for TRF model
                            fs:         Sampling frequency
                            n_folds:    Number of folds for cross-validation
                            alphas:     Regularisation parameters to test
                            b_spa:      Should we enforce sparsity?
                            X:          Design matrix (`trials` x `features` x `time`)
                            X_mask:     Masking matrix for coefficients (`models` x `features`)
                            fit_models: Array including all models to be fit (indices, `N`).
                            y:          EEG data (`trials` x `channels` x `time`)
                            backend:    What backend should be used (numpy/torch)?
                            device:     If torch, what device (cpu/cuda/mps)?
    
    OUTPUTS:
        result      -   Results dictionary for permutation. Dictionary includes:
                            p_i:    Permutation index
                            ß:      Beta estimates (`models` x `channels` x `features` x `time`)
                            ß_r2:   Beta estimates, but R2 transformed (`models` x `channels` x `features` x `time`)
                            r:      Out of sample accuracy (pearson correlation) (`models` x `channels`)
    '''
    
    # unpack task data and settings
    p_i, indc, i = task['p_i'], task['indc'], task['i']
    ts, N, n_tmin, n_tmax, fs, n_folds, alphas, b_spa, X, X_mask, fit_models, y, backend, device = settings
    
    # log progress
    aux.progressbar(i, N, ts, msg = f'[ERP]')
    
    # setup scaler to enforce sparsity (or not)
    scaler = dict(with_mean = False, with_std = False) if b_spa else dict()
    
    # setup cross-validation procedure
    kf = sklearn.model_selection.KFold(n_splits = n_folds)

    # move to device, if required
    if backend == 'torch':
        if device == 'cuda': 
            # set cuda device
            device = f'cuda:{i % torch.cuda.device_count()}'

        X, X_mask, y = torch.from_numpy(X).to(torch.float64).to(device), torch.from_numpy(X_mask).to(torch.bool).to(device), torch.from_numpy(y).to(torch.float64).to(device)
        alphas, indc = torch.from_numpy(alphas).to(torch.float64).to(device), torch.from_numpy(indc).to(torch.int64).to(device)
    
    # setup model, depending on backend
    if backend == 'torch': trf = rsa.analysis.estimators.torch.TimeDelayed(n_tmin, n_tmax, fs, alphas = alphas, scaler = scaler)
    else: trf = rsa.analysis.estimators.TimeDelayed(n_tmin, n_tmax, fs, alphas = alphas, scaler = scaler)

    # setup data containers
    if backend == 'torch':
        ß = torch.zeros((len(X_mask), n_folds, y.shape[1], X.shape[1], trf.window.shape[0]), dtype = X.dtype, device = X.device)
        r = torch.zeros((len(X_mask), n_folds, y.shape[1]), dtype = X.dtype, device = X.device)
    else:
        ß = np.zeros((len(X_mask), n_folds, y.shape[1], X.shape[1], trf.window.shape[0]))
        r = np.zeros((len(X_mask), n_folds, y.shape[1]))
    
    # permute data
    X, y = X[indc,:,:], y[indc,:,:]

    # loop over models
    for m_i, mask in enumerate(X_mask):
        # check if model is required
        if m_i not in fit_models: continue

        # masked model
        X_i = X[:,mask,:]
        
        # loop over folds
        for f_i, (train, test) in enumerate(kf.split(X, y)):
            # convert to torch, if required
            if backend == 'torch': 
                train, test = torch.from_numpy(train).to(torch.int64).to(device), torch.from_numpy(test).to(torch.int64).to(device)
                trf = rsa.analysis.estimators.torch.TimeDelayed(n_tmin, n_tmax, fs, alphas = alphas, scaler = scaler)

            # fit model
            trf.fit(X_i[train,:,:], y[train,:,:])
            
            # grab beta estimates and evaluate OOS
            if backend == 'torch': 
                ß[m_i,f_i,:,mask,:] = trf.coef_.to(ß.dtype)
                r[m_i,f_i,:] = trf.score(X_i[test,:,:], y[test,:,:]).to(r.dtype)
            else:
                ß[m_i,f_i,:,mask,:] = trf.coef_.swapaxes(0, 1)
                r[m_i,f_i,:] = trf.score(X_i[test,:,:], y[test,:,:])
            
            if backend == 'torch': del trf
        
        if backend == 'torch': del X_i
    
    # if torch, move to cpu now
    if backend == 'torch':
        ß = ß.cpu().numpy()
        r = r.cpu().numpy()
    
    '''
    NOTE: To avoid issues with scaling (caused by different penalisation parameters), we
    compute R2 transformed beta estimates similar to the appraoch in:

        Gwilliams, L., King, J.R., Marantz, A., & Poeppel, D. (2022). Neural dynamics of phoneme sequences reveal position-invariant code for content and order. Nature Communications, 13, 6606. 10.1038/s41467-022-34326-1
    
    except that we use absolute values of ß for the normalisation, because both +- are 
    meaningful to us. We do, however, also save original coefficients for posterity.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # we may get slice warnings if not all models are fit

        ß_r2 = ß / np.abs(ß).sum(axis = 3, keepdims = True).max(axis = 4, keepdims = True)
        
        # average over folds
        ß = np.nanmean(ß, axis = 1)
        ß_r2 = np.nanmean(ß_r2, axis = 1)
        r = np.nanmean(r, axis = 1)
    
    return dict(p_i = p_i, ß = ß, ß_r2 = ß_r2, r = r)

if __name__ == '__main__':
    '''
    Start rERP approach.
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
    fs = aux.get_opt('fs', default = 200, cast = int)                           # frequency to downsample signal to (through binnning)
    b_con = bool(aux.get_opt('b_con', default = 1, cast = int))                 # should we include congruent trials?
    b_inc = bool(aux.get_opt('b_inc', default = 0, cast = int))                 # should we include incongruent trials?
    b_zsc = bool(aux.get_opt('b_zsc', default = 0, cast = int))                 # should we normalise all data and predictors?
    b_spa = bool(aux.get_opt('b_spa', default = 0, cast = int))                 # should we enforce sparsity? (i.e., are we interested in TRF-like events or more temporally distributed computations?)
    b_bsl = bool(aux.get_opt('b_bsl', default = 0, cast = int))                 # should we baseline correct epochs?
    s_mod = aux.get_opt('s_mod', default = 'spc', cast = str)                   # which hypotheses do we test here? (spc = specific, inv = invariant)
    n_workers = aux.get_opt('n_workers', default = 4, cast = int)               # how many jobs to use
    n_alphas = aux.get_opt('n_alphas', default = 20, cast = int)                # how many alphas to test
    n_folds = aux.get_opt('n_folds', default = 5, cast = int)                   # how many folds to use for CV
    n_permutations = aux.get_opt('n_permutations', default = 50, cast = int)    # how many permutations to perform
    n_topk = aux.get_opt('n_topk', default = 5, cast = int)                     # how many top words to use for acoustic surprisal
    n_edge_L = aux.get_opt('n_edge_l', default = 10, cast = int)                # how many samples to consider in acoustic edge computation
    n_maxL = aux.get_opt('n_maxl', default = 2000, cast = int)                  # total duration after onset (in epochs)
    n_tmin = aux.get_opt('n_tmin', default = -0.8, cast = float)                # minimum to consider for time delaying ridge
    n_tmax = aux.get_opt('n_tmax', default = 0.1, cast = float)                 # maximum to consider for time delaying ridge
    backend = aux.get_opt('backend', default = 'torch', cast = str)             # what backend to use for model fitting? (torch/numpy)
    device = aux.get_opt('device', default = 'cuda', cast = str)                # what device to use for model fitting? (cpu/gpu, gpu only available in torch)
    maxtask = aux.get_opt('maxtask', default = 1, cast = int)                   # for multiprocessing, when do we kill a child? this is for efficient garbage collection (and keeping memory required to a minimum)

    # dump opts
    print(f'---------- rERP ----------')
    print(f'[OPTS]\tsid={sid}\tpid={pid}')
    print(f'[OPTS]\tfs={fs}\t\tb_zsc={b_zsc}')
    print(f'[OPTS]\tb_spa={b_spa}\tb_bsl={b_bsl}')
    print(f'[OPTS]\tb_con={b_con}\tb_inc={b_inc}')
    print(f'[OPTS]\tn_workers={n_workers}\tn_alphas={n_alphas}')
    print(f'[OPTS]\tn_folds={n_folds}\tn_permutations={n_permutations}')
    print(f'[OPTS]\tn_topk={n_topk}\tn_edge_l={n_edge_L}')
    print(f'[OPTS]\tn_maxl={n_maxL}\tn_tmin={n_tmin}\tn_tmax={n_tmax}')
    print(f'[OPTS]\ts_mod={s_mod}')
    print(f'[OPTS]\tbackend={backend}\tdevice={device}')
    print(f'--------------------------')
    
    # load & prepare eeg data
    print(f'[PRE] Loading and preparing eeg data...')
    eeg_mt2, info = load_eeg(f'./data/preprocessed/eeg/sub{sid}/rerp-MT2-epo.fif', fs = fs, bsl = b_bsl)
    
    # load behaviour from both tasks
    print(f'[PRE] Loading raw behaviour...')
    df = pd.read_csv(f'./data/raw/beh/sub{sid}/{pid}.csv', sep = ',')
    beh_mt1 = df.loc[(df.type == data.defs.TRIAL_MT_MAIN) | (df.type == data.defs.TRIAL_MT_CONTROL)].reset_index(drop = True)
    beh_mt2 = df.loc[(df.type == data.defs.TRIAL_ET_BAD) | (df.type == data.defs.TRIAL_ET_GOOD)].reset_index(drop = True)
    
    print(f'[PRE] Loading SUBTLEX-de...')
    subtlex = pd.read_excel('../spaces/dumps/subtlex-de.xlsx')
    sub_dict = np.array(subtlex.Word.tolist())
    
    # load flagged trials
    print(f'[PRE] Loading bad trials...')
    with open(f'./data/preprocessed/eeg/sub{sid}/rerp_badtrials.npy', 'rb') as f:
        bad_trials = np.load(f)
    
    # load embedding
    print(f'[PRE] Loading embedding...')
    G = emb.glove.load_embedding(f_in = '../spaces/glove-german/w2v_50D.txt')
    
    # load collapsed phoneme df
    print(f'[PRE] Loading phoneme data...')
    collapsed = pd.read_csv('./data/preprocessed/misc/phonetics/collapsed.csv')
    
    # load last RTFE positions
    print(f'[PRE] Loading RTFE priors...')
    contexts = np.array(['essen', 'fashion', 'outdoor', 'technik', 'politik', 'unterhaltung'])
    last_con = beh_mt1.context.tolist()[-1]

    with gzip.open(f'./data/raw/rtfe/sub{sid}/{pid}_t263_{last_con}.pkl.gz', 'rb') as f:
        _, _, _, posterior, _, _ = pickle.load(f)
    context_pos = posterior.T
    
    # load unspecific priors (invariant)
    with gzip.open(f'./data/raw/rtfe-unspecific/sub{sid}/{pid}_t263_essen.pkl.gz', 'rb') as f:
        _, _, _, posterior, _, _ = pickle.load(f)
    context_pos_uns = posterior.T
    
    # load phonotactic model
    print(f'[PRE] Loading PTX model...')
    p_mod_bl = models.PTX(collapsed)
    
    # load targets and labels from mt1
    targets = np.array(beh_mt1.options_0.tolist())
    labels = []
    for i in range(len(targets)):
        stimulus = beh_mt1.loc[i].stimulus
        a, b = stimulus.split('/')[-1].split('_')[0].split('-')
        labels.append(f'{a}-{b}')
    labels = np.array(labels)
    
    # setup baseline offset
    print(f'[PRE] Preparing predictors...')
    bl_off = int(500 * (fs / 1000))
    
    # setup data and mask for MT1
    context = np.array(beh_mt1.context.tolist())
    stimuli = np.array(beh_mt1.stimulus.tolist())
    opt0 = np.array(beh_mt1.options_0.tolist())
    types = np.array(beh_mt1.type.tolist())
    mask = types == data.defs.TRIAL_MT_MAIN
    
    # obtain data
    L = int(500 * (fs / 1000) + n_maxL * (fs / 1000))
    
    tag_pho = np.zeros((120, L))              # tags for phoneme onsets
    tag_wrd = np.zeros((120, L))              # tags for word onsets
    acc_env = np.zeros((120, L))              # acoustic envelope
    acc_edge = np.zeros((120, L))             # acoustic envelope edges
    surprisal_ptx = np.zeros((120, L))        # phonotactic surprisal
    surprisal_lex = np.zeros((120, L))        # lexical surprisal
    surprisal_acc_inv = np.zeros((120, L))    # top-k acoustic surprisal from invariant priors
    surprisal_sem_inv = np.zeros((120, L))    # semantic surprisal from invariant priors
    surprisal_acc_spc = np.zeros((120, L))    # top-k acoustic surprisal from speaker priors
    surprisal_sem_spc = np.zeros((120, L))    # semantic surprisal from speaker priors
    tag_bad_pho = np.zeros((120, L))          # tag for a phoneme in a bad word
    tag_bad_wrd = np.zeros((120, L))          # tag for a bad word
    y = np.zeros((120, 60, L))
    
    # loop over trials
    for i in np.arange(0, 120, 1):
        # load trial and position
        trial = beh_mt2.loc[i]
        pos = np.where(contexts == trial.context)[0][0]
        is_bad = int(trial.type == data.defs.TRIAL_ET_BAD)
        
        # get descriptives
        a, b = trial.stimulus.split('/')[-1].split('_')[0].split('-')
        T, v = trial.stimulus.split('/')[-1].split('_')[1]
        r = int(trial.stimulus.split('/')[-1].split('_')[3][1])
        
        # load audio data
        f_aud = f'{a}-{b}_{T}{v}_12ch_r{r}_cs'
        env = np.load(f'./data/preprocessed/audio/gt-{fs}Hz/vocoded/{f_aud}.npy')
        
        # compute broadband envelope and edges
        b_env = (np.abs(env) ** 0.6).sum(axis = 0)
        b_edge = np.nanstd(np.lib.stride_tricks.sliding_window_view(np.concatenate(([np.nan] * (n_edge_L - 1), b_env)), (n_edge_L,), axis = 0), axis = 1)
        
        # compute semantic surprisal (invariant)
        exp = context_pos_uns[0,:]
        obs = G[trial.options_0.lower()]
        fit = np.dot(exp, obs) / (np.linalg.norm(exp) * np.linalg.norm(obs))
        surprisal_sem_inv[i,bl_off] = -np.log2(1 + fit)
        
        # compute semantic surprisal (speaker)
        exp = context_pos[pos,:]
        obs = G[trial.options_0.lower()]
        fit = np.dot(exp, obs) / (np.linalg.norm(exp) * np.linalg.norm(obs))
        surprisal_sem_spc[i,bl_off] = -np.log2(1 + fit)
        
        # obtain lexical surprisal
        t = trial.options_0
        ts = t if t != 'Wild' else 'wild' # fix for subtlex
        indx = np.where(sub_dict == ts)[0]
        if len(indx) > 0: surprisal_lex[i,bl_off] = -np.log2(subtlex.loc[indx].SUBTLEX.tolist()[0])
        
        # read phonemes
        f_aud = f'{a}-{b}_{T}{v}'
        ann = load_annotation(f'./data/preprocessed/audio/annotated/clear/{f_aud}.csv', fs = fs)
        
        # read audio
        f_aud = f'{a}-{b}_{T}{v}_12ch_r{r}_cs'
        obs = np.load(f'./data/preprocessed/audio/gt-200Hz/vocoded/{f_aud}.npy')
        
        # compute top-k surprisal (and ignore 0 warnings)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            i_s = get_topk_surprisal(obs, n_topk, beh_mt1, targets, context_pos_uns, G, v, context = None) # invariant
            s_s = get_topk_surprisal(obs, n_topk, beh_mt1, targets, context_pos, G, v, context = trial.context) # specific
        
        # compute surprisal
        b_s, b_e, _ = p_mod_bl.trajectory(ann.phoneme.tolist())
        
        # set impulses per phoneme
        for j, pho in enumerate(ann.phoneme.tolist()):
            # find timing
            on, off = ann.loc[j].onset, ann.loc[j].offset
            
            # make sure we don't go too far
            if (bl_off + on) > surprisal_ptx.shape[1]: continue
            
            # set impulses
            acc_env[i,bl_off+on] = b_env[on:off].mean()         # envelopes
            acc_edge[i,bl_off+on] = b_edge[on:off].mean()       # edges
            surprisal_ptx[i,bl_off+on] = b_s[j]                 # phonotactic
            surprisal_acc_inv[i,bl_off+on] = i_s[on:off].mean() # top-k acoustic (invariant)
            surprisal_acc_spc[i,bl_off+on] = s_s[on:off].mean() # top-k acoustic (speaker)
            
            # set phoneme tag
            tag_pho[i,bl_off+on] = 1.0
            tag_bad_pho[i,bl_off+on] = is_bad
        
        # set word tag
        tag_wrd[i,bl_off] = 1.0
        tag_bad_wrd[i,bl_off] = is_bad
        
        # set neural response
        y[i,:,:] = eeg_mt2[i,:,0:y.shape[2]]
    
    # setup models
    X = np.array([tag_pho, tag_wrd, tag_bad_pho, tag_bad_wrd, acc_env, acc_edge, surprisal_ptx, surprisal_lex, surprisal_acc_inv, surprisal_sem_inv, surprisal_acc_spc, surprisal_sem_spc, tag_bad_pho * surprisal_acc_spc, tag_bad_wrd * surprisal_sem_spc]).swapaxes(0, 1)
    
    if s_mod == 'spc':
        # setup models testing specific effects
        X_mask = np.array([[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], # control (baseline)
                           [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], # +acoustic -semantic -bad
                           [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0], # -acoustic +semantic -bad
                           [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], # +acoustic +semantic -bad
                           [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0], # +acoustic -semantic +bad
                           [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1], # -acoustic +semantic +bad
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # +acoustic +semantic +bad
                          ]).astype(bool)
    
        if b_con == False: X_mask[:,[2, 3, 10, 11]] = False
        if b_inc == False: X_mask[:,[2, 3, 12, 13]] = False
        
        fit_models = np.arange(7)
        if (b_con) & (b_inc == False): fit_models = np.array([0, 1, 2, 3])
        if (b_con == False) & (b_inc): fit_models = np.array([0, 4, 5, 6])
    else:
        # setup models testing invariant effects
        X_mask = np.array([[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], # control (baseline)
                           [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], # +acoustic -semantic
                           [1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0], # -acoustic +semantic
                           [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], # +acoustic +semantic
                          ]).astype(bool)

        fit_models = np.arange(4)
    
    # make trial selection
    trials = []
    
    if b_con: trials = np.concatenate((trials, np.where(beh_mt2.type == data.defs.TRIAL_ET_GOOD)[0]))
    if b_inc: trials = np.concatenate((trials, np.where(beh_mt2.type == data.defs.TRIAL_ET_BAD)[0]))
    
    trials = np.array(trials)
    trials = np.setdiff1d(trials, bad_trials).astype(int) # remove flagged trials

    # data selection
    X, y = X[trials,:,:], y[trials,:,:]
    X, y = X[:,:,50:400], y[:,:,50:400]
    
    # normalise while preserving sparsity, if required
    if b_zsc:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            for x_i in range(4, X.shape[1]):
                X_i = X[:,x_i,:]
                
                indc = np.nonzero(X_i)
                X_i[indc] = (X_i[indc] - X_i[indc].mean()) / X_i[indc].std()
                X[:,x_i,:] = X_i
            
            y = (y - y.mean(axis = (0, 2), keepdims = True)) / y.std(axis = (0, 2), keepdims = True)

    # start worker pool
    print(f'[ERP] Starting worker pool...')
    processor = emb.mp.Processor(workers = n_workers, maxtasksperchild = maxtask)
    
    # create jobs
    print(f'[ERP] Preparing jobs...')
    jobs = []
    
    alphas = np.logspace(-5, 10, n_alphas)
    permutation_indc = np.array([np.random.choice(np.arange(0, trials.shape[0], 1), size = (trials.shape[0],), replace = False) for _ in range(n_permutations)])

    for p_i in np.arange(0, n_permutations, 1):
        job = dict(p_i = p_i, indc = permutation_indc[p_i,:], i = len(jobs))
        jobs.append(job)
    
    # run jobs
    print(f'[ERP] Computing rERP...')
    ts = time.time()
    outputs = processor.run(jobs, external = permutation_worker, settings = (ts, len(jobs), n_tmin, n_tmax, fs, n_folds, alphas, b_spa, X, X_mask, fit_models, y, backend, device))
    
    # allocate memory
    n_delays = int((n_tmax - n_tmin) * fs) + 1
    
    ß = np.zeros((n_permutations, X_mask.shape[0], y.shape[1], X.shape[1], n_delays))
    ß_r2 = np.zeros_like(ß)
    r = np.zeros((n_permutations, X_mask.shape[0], y.shape[1]))
    
    # grab results
    print(f'')
    print(f'[ERP] Collecting results...')
    for output in outputs:
        # grab indices
        p_i = output['p_i']

        # grab data
        ß[p_i] = output['ß']
        ß_r2[p_i] = output['ß_r2']
        r[p_i] = output['r']
    
    # save results
    print(f'[ERP] Saving results...')
    dir_out = f'./data/processed/eeg/sub{sid}/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        with gzip.open(dir_out + f'rerp-mt2-c{int(b_con)}-i{int(b_inc)}-k{n_topk}-z{int(b_zsc)}-s{int(b_spa)}-b{int(b_bsl)}-{s_mod}.pkl.gz', 'wb') as f:
            pickle.dump((np.nanmean(ß, axis = 0), np.nanmean(ß_r2, axis = 0), np.nanmean(r, axis = 0)), f)