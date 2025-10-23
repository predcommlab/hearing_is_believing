'''
This script performs a subject-level encoding analyis of the 
representational similarity of the percepts during our learning
task.

In brief, we compute observed similarities of the percepts and
targets/alternatives as well as hypothesis similarities (based 
on raw audio, top-k expected audio, and semantics, utilising
random, speaker-specific or speaker-invariant priors). We then
run encoding models to reconstruct the observed similarities 
from our hypotheses and compare different models based on out-
of-sample prediction accuracy.

NOTE: This is a script that performs this analysis using foil
priors that have consistent semantic categories, but are not
the real priors of any participant at any point in time. This
is done to see whether top-k acoustic predictions truly do
capture meaningful spectral expectations.

NOTE: This requires that subject_rsa_rec.py has been run to re-
construct gammatones from the morph data.

NOTE: This script is relatively memory intensive. For me, one 
run may take 90-120 minutes and requires ~32GB of RAM. Of course,
this will also depend on your architecture (as well as `n_workers`).

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
    python subject_rsa_enc.py id=0002 n_permutations=10
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

def compute_sim(x: np.ndarray, y: np.ndarray, f: Callable = rsa.math.cosine, pad_L: int = 0, kernel_L: int = 1) -> np.ndarray:
    '''
    Compute similarity between tensors `x` and `y`.
    
    INPUTS:
        x           -   Tensor of shape (`trials` x `features` x `time`)
        y           -   Tensor of shape (`trials` x `features` x `time`)
        f           -   Similarity function (default = rsa.math.cosine)
        pad_L       -   Number of time points to pad at the beginning and end of the signal (for smoothing)
        kernel_L    -   Length of smoothing kernel (default = 1)
    
    OUTPUTS:
        sim         -   Similarity matrix of shape (`trials` x `time`)
    '''
    
    # compute similarity
    sim = f(x.swapaxes(1, 2), y.swapaxes(1, 2))
    
    # apply padding and smoothen
    sim = np.concatenate((np.zeros((sim.shape[0], pad_L)), sim, np.zeros((sim.shape[0], pad_L))), axis = 1)
    sim = rsa.signal.smoothen(sim, rsa.signal.boxcar(kernel_L), axis = 1)
    
    # undo padding
    sim = sim[:,pad_L:-pad_L]
    
    return sim

def worker_encode(task: Dict, settings: tuple[Any]) -> Dict:
    '''
    Computes the RSA encoding model for one permutation of the data.
    
    INPUTS:
        task        -   Dictionary of task parameters:
                            p_i: Permutation index
                            i: Current iteration
                            indc: Indices to use for permuting.
        settings    -   Tuple of settings:
                            ts: Start time
                            N: Number of iterations
                            n_folds: Number of folds
                            alphas: Regularisation parameters
                            b_acoustic: Should acoustic predictors be used?
                            b_semantic: Should semantic predictors be used?
                            n_mod: Should a specific model be used _only_? (-1 = False, else 0, 1, 2, 3)
                            sim_n_mt: Similarity matrix for morph-target (neural)
                            sim_n_ma: Similarity matrix for morph-alternative (neural)
                            sim_a_mt: Similarity matrix for morph-target (audio)
                            sim_a_ma: Similarity matrix for morph-alternative (audio)
                            sim_a_et: Similarity matrix for expectation-target (audio)
                            sim_a_ea: Similarity matrix for expectation-alternative (audio)
                            sim_a_ut: Similarity matrix for expectation-target (audio, unspecific)
                            sim_a_ua: Similarity matrix for expectation-alternative (audio, unspecific)
                            sim_a_rt: Similarity matrix for expectation-target (audio, random)
                            sim_a_ra: Similarity matrix for expectation-alternative (audio, random)
                            sim_s_et: Similarity matrix for expectation-target (semantic)
                            sim_s_ea: Similarity matrix for expectation-alternative (semantic)
                            sim_s_ut: Similarity matrix for expectation-target (semantic, unspecific)
                            sim_s_ua: Similarity matrix for expectation-alternative (semantic, unspecific)
                            sim_s_rt: Similarity matrix for expectation-target (semantic, random)
                            sim_s_ra: Similarity matrix for expectation-alternative (semantic, random)
                            backend: Which backend to use? (numpy/torch)
                            device: Which device to use, if applicable? (cpu/cuda/mps)
    
    OUTPUTS:
        result      -   Dictionary of results:
                            p_i: Permutation index
                            ß: Raw coefficients from models
                            ß_r: Normalised coefficients (as R2).
                            r: Out of sample performance.
    '''
    
    # grab task and settings
    p_i, i, indc = task['p_i'], task['i'], task['indc']
    ts, N, n_folds, alphas, b_acoustic, b_semantic, n_mod, sim_n_mt, sim_n_ma, sim_a_mt, sim_a_ma, sim_a_et, sim_a_ea, sim_a_ut, sim_a_ua, sim_a_rt, sim_a_ra, sim_s_et, sim_s_ea, sim_s_ut, sim_s_ua, sim_s_rt, sim_s_ra, backend, device = settings
    
    # log progress
    aux.progressbar(i, N, ts, msg = f'[ENC]')
    
    # permute data
    sim_n_mt, sim_n_ma = sim_n_mt[indc,:], sim_n_ma[indc,:]
    sim_a_mt, sim_a_ma = sim_a_mt[indc,:], sim_a_ma[indc,:]
    sim_a_et, sim_a_ea = sim_a_et[indc,:], sim_a_ea[indc,:]
    sim_a_ut, sim_a_ua = sim_a_ut[indc,:], sim_a_ua[indc,:]
    sim_a_rt, sim_a_ra = sim_a_rt[indc,:], sim_a_ra[indc,:]
    sim_s_et, sim_s_ea = sim_s_et[indc,:], sim_s_ea[indc,:]
    sim_s_ut, sim_s_ua = sim_s_ut[indc,:], sim_s_ua[indc,:]
    sim_s_rt, sim_s_ra = sim_s_rt[indc,:], sim_s_ra[indc,:]
    
    # setup masks for models
    acc, sem = int(b_acoustic), int(b_semantic)
    models = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, acc * 1, sem * 1, 0, 0],
                       [1, 1, 1, 1, 0, 0, acc * 1, sem * 1],
                       [1, 1, 1, 1, acc * 1, sem * 1, acc * 1, sem * 1]]).astype(bool)
    
    # setup data
    X = np.array([np.concatenate((np.ones_like(sim_a_mt), np.ones_like(sim_a_ma)), axis = 0),
                  np.concatenate((sim_a_mt, sim_a_ma), axis = 0), 
                  np.concatenate((sim_a_rt, sim_a_ra), axis = 0), 
                  np.concatenate((sim_s_rt, sim_s_ra), axis = 0), 
                  np.concatenate((sim_a_et, sim_a_ea), axis = 0),
                  np.concatenate((sim_s_et, sim_s_ea), axis = 0),
                  np.concatenate((sim_a_ut, sim_a_ua), axis = 0),
                  np.concatenate((sim_s_ut, sim_s_ua), axis = 0)]).swapaxes(0, 1)
    y = np.concatenate((sim_n_mt, sim_n_ma), axis = 0)[:,None,:]
    
    # select model, if desired
    if n_mod > -1: models = models[n_mod][None,:]
    
    # move to device, where relevant
    if backend == 'torch':
        # check if cuda
        if device == 'cuda':
            # distribute across devices
            device = f'cuda:{i % torch.cuda.device_count()}'
        
        # move
        X, y = torch.from_numpy(X).to(torch.float64).to(device), torch.from_numpy(y).to(torch.float64).to(device)
        alphas, models = torch.from_numpy(alphas).to(torch.float64).to(device), torch.from_numpy(models).to(torch.bool).to(device)
    
    # setup cross-validation
    kf = sklearn.model_selection.KFold(n_splits = n_folds)
    
    # setup encoder
    if backend == 'torch': encoder = rsa.analysis.estimators.torch.SparseEncoder(alphas = alphas)
    else: encoder = rsa.analysis.estimators.SparseEncoder(alphas = alphas)

    # setup data containers
    if backend == 'torch':
        ß = torch.zeros((n_folds, models.shape[0], X.shape[1], X.shape[2]), dtype = X.dtype, device = X.device)
        oos_r = torch.zeros((n_folds, models.shape[0]), dtype = X.dtype, device = X.device)
    else:
        ß = np.zeros((n_folds, models.shape[0], X.shape[1], X.shape[2]))
        oos_r = np.zeros((n_folds, models.shape[0]))
    
    # loop over folds
    for f_i, (kf_train, kf_test) in enumerate(kf.split(sim_n_mt, sim_n_ma)):
        # obtain real indices
        # NOTE: This guarantees that we will always have target/alternative in
        # the same fold, essentially. We do this to avoid (potential) biases
        # from having more of one or the other in training data.
        train, test = np.concatenate((kf_train, kf_train + sim_n_mt.shape[0])), np.concatenate((kf_test, kf_test + sim_n_mt.shape[0]))
        
        # move, if required
        if backend == 'torch':
            train, test = torch.from_numpy(train).to(torch.int64).to(X.device), torch.from_numpy(test).to(torch.int64).to(X.device)
        
        # loop over models
        for m_i in range(models.shape[0]):
            # grab data
            if backend == 'torch': 
                X_m = X[:,models[m_i],:].clone()
                y_m = y.clone()
            else: 
                X_m = X[:,models[m_i],:].copy()
                y_m = y.copy()
            
            # get mean and std
            if backend == 'torch':
                def nanstd(o, dim = 0, keepdim = False):
                    '''
                    Compute the standard deviation of a tensor, ignoring NaN values.
                    '''
                    
                    result = torch.sqrt(
                                torch.nanmean(
                                    torch.pow( torch.abs(o - torch.nanmean(o, dim = dim).unsqueeze(dim)), 2),
                                    dim=dim
                                )
                            )

                    if keepdim:
                        result = result.unsqueeze(dim)
                    
                    return result
                
                X_mu, X_sigma = torch.nanmean(X_m[train,:,:], dim = 0, keepdim = True), nanstd(X_m[train,:,:], dim = 0, keepdim = True)
                y_mu, y_sigma = torch.nanmean(y[train,:,:], dim = 0, keepdim = True), nanstd(y[train,:,:], dim = 0, keepdim = True)
            else:
                X_mu, X_sigma = np.nanmean(X_m[train,:,:], axis = 0, keepdims = True), np.nanstd(X_m[train,:,:], axis = 0, keepdims = True)
                y_mu, y_sigma = np.nanmean(y_m[train,:,:], axis = 0, keepdims = True), np.nanstd(y_m[train,:,:], axis = 0, keepdims = True)
            
            # normalise data (in audio, NaN may occur)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                # apply normalisation (without intercept term)
                X_m[:,1:,:] = (X_m[:,1:,:] - X_mu[:,1:,:]) / X_sigma[:,1:,:]
                y_m = (y_m - y_mu) / y_sigma

                # reset infinities/zeros
                if backend == 'torch':
                    X_m[torch.isfinite(X_m) == False] = 0.0
                    y_m[torch.isfinite(y_m) == False] = 0.0
                else:
                    X_m[np.isfinite(X_m) == False] = 0.0
                    y_m[np.isfinite(y_m) == False] = 0.0
            
            '''
            NOTE: For this model fitting procedure, there are edge cases that
            may fail. This occurs due to numerical precision errors that will
            prevent the SVD from converging.
            
            One way to fix this would be to use a different solver. However,
            this is problematic for us given that SVD allows us to solve all
            alphas in parallel, i.e. would come at a very significant loss in
            efficiency.
            
            Given that the issue is exceedingly rare, we will opt for performance
            here and let these cases fail. Instead, then, we NaN them out. We run
            enough permutations to warrant this either way, if it even fails to
            begin with.
            
            TODO: Monitor the situation to see if this even occurs at all during
            our model fitting procedure.
            
            NOTE: As of 23/10/2024, I have now run the full batch and can confirm
            that this is exceedingly rare (about 1:5000). As such, it will happen
            occasionally, but is far from being a problem for our analysis.
            '''

            try:
                # fit model
                encoder.fit(X_m[train,:,:], y_m[train,:,:])
                
                if backend == 'torch':
                    # save coefficients
                    ß[f_i,m_i,models[m_i],:] = encoder.coef_.squeeze().to(ß.dtype)
                    
                    # evaluate model
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        
                        y_h = encoder.predict(X_m[test,:,:], invert_y = True)
                        oos_r[f_i,m_i] = torch.nanmean(rsa.math.torch.pearsonr(y_h.T, y_m[test,:,:].T))
                else:
                    # save coefficients
                    ß[f_i,m_i,models[m_i],:] = encoder.coef_.squeeze().copy()
                    
                    # evaluate model (again, NaNs may occur due to 0s)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        
                        y_h = encoder.predict(X_m[test,:,:], invert_y = True)
                        oos_r[f_i,m_i] = np.nanmean(rsa.math.pearsonr(y_h.T, y_m[test,:,:].T))
            except:
                print(f'[ENC] SVD precision error (p={p_i}, m={m_i}, f={f_i}).')
                
                if backend == 'torch':
                    ß[f_i,m_i,models[m_i],:] = torch.nan
                    oos_r[f_i,m_i] = torch.nan
                else:
                    ß[f_i,m_i,models[m_i],:] = np.nan
                    oos_r[f_i,m_i] = np.nan
                    
    # if torch, move back to cpu
    if backend == 'torch':
        ß = ß.cpu().numpy()
        oos_r = oos_r.cpu().numpy()
    
    '''
    NOTE: To avoid issues of scaling (i.e., different models at different alphas), we will
    compute coefficients as R2 here in a manner very similar to:

        Gwilliams, L., King, J.R., Marantz, A., & Poeppel, D. (2022). Neural dynamics of phoneme sequences reveal position-invariant code for content and order. Nature Communications, 13, 6606. 10.1038/s41467-022-34326-1
    
    except that we use absolute values of ß for the normalisation, because both +- are 
    meaningful to us. We do, however, also save original coefficients for posterity.
    '''
    
    ß_r = (ß / np.abs(ß).sum(axis = 2, keepdims = True).max(axis = 3, keepdims = True))
    
    # average over folds
    ß = np.nanmean(ß, axis = 0)
    ß_r = np.nanmean(ß_r, axis = 0)
    oos_r = np.nanmean(oos_r, axis = 0)
    
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
    n_workers = aux.get_opt('n_workers', default = 5, cast = int)               # number of parallel workers to use
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
        with gzip.open(f'./data/raw/rtfe-random/sub{sid}/{pid}_t{trial.no}_{context}.pkl.gz', 'rb') as f:
            _, _, _, posterior, _, _ = pickle.load(f)
        priors = posterior.T
        
        # update unspecific priors
        with gzip.open(f'./data/raw/rtfe-unspecific-random/sub{sid}/{pid}_t{trial.no}_essen.pkl.gz', 'rb') as f:
            _, _, _, posterior, _, _ = pickle.load(f)
        priors_uns = posterior.T
    
    # compute similarities
    print(f'[RSA] Computing similarities...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # do we use clears from brain data?
        if b_brain:
            sim_n_mt = compute_sim(y_h_mt1, y_h_pl1[clear_t,:,:], pad_L = pad_L, kernel_L = n_k1)
            sim_n_ma = compute_sim(y_h_mt1, y_h_pl1[clear_a,:,:], pad_L = pad_L, kernel_L = n_k1)
        else:
            sim_n_mt = compute_sim(y_h_mt1, y_t, pad_L = pad_L, kernel_L = n_k1)
            sim_n_ma = compute_sim(y_h_mt1, y_a, pad_L = pad_L, kernel_L = n_k1)

        sim_a_mt = compute_sim(y, y_t, pad_L = pad_L, kernel_L = n_k1)
        sim_a_ma = compute_sim(y, y_a, pad_L = pad_L, kernel_L = n_k1)

        sim_a_et = compute_sim(exp_audio, y_t, pad_L = pad_L, kernel_L = n_k1)
        sim_a_ea = compute_sim(exp_audio, y_a, pad_L = pad_L, kernel_L = n_k1)
        
        sim_a_ut = compute_sim(uns_audio, y_t, pad_L = pad_L, kernel_L = n_k1)
        sim_a_ua = compute_sim(uns_audio, y_a, pad_L = pad_L, kernel_L = n_k1)
        
        sim_a_rt = compute_sim(ran_audio, y_t, pad_L = pad_L, kernel_L = n_k1)
        sim_a_ra = compute_sim(ran_audio, y_a, pad_L = pad_L, kernel_L = n_k1)
        
        sim_s_et = compute_sim(exp_sem, t_sem, pad_L = pad_L, kernel_L = n_k1)
        sim_s_ea = compute_sim(exp_sem, a_sem, pad_L = pad_L, kernel_L = n_k1)
        
        sim_s_ut = compute_sim(uns_sem, t_sem, pad_L = pad_L, kernel_L = n_k1)
        sim_s_ua = compute_sim(uns_sem, a_sem, pad_L = pad_L, kernel_L = n_k1)
        
        sim_s_rt = compute_sim(ran_sem, t_sem, pad_L = pad_L, kernel_L = n_k1)
        sim_s_ra = compute_sim(ran_sem, a_sem, pad_L = pad_L, kernel_L = n_k1)
    
    # start worker pool
    print(f'[ENC] Starting worker pool...')
    processor = emb.mp.Processor(workers = n_workers, maxtasksperchild = maxtask)
    
    # create jobs
    print(f'[ENC] Preparing jobs...')
    jobs = []
    
    indcs = np.array([np.random.choice(np.arange(0, sim_n_mt.shape[0], 1), size = (sim_n_mt.shape[0],), replace = False) for _ in np.arange(0, n_permutations, 1)])
    alphas = np.logspace(-5, 10, n_alphas)
    
    for p_i in range(n_permutations):
        jobs.append(dict(p_i = p_i, i = len(jobs), indc = indcs[p_i,:]))
    
    # run jobs
    print(f'[ENC] Encoding RSA...')
    processor = emb.mp.Processor(workers = n_workers)
    ts = time.time()
    outputs = processor.run(jobs, external = worker_encode, settings = (ts, len(jobs), n_folds, alphas, b_acoustic, b_semantic, n_mod, sim_n_mt, sim_n_ma, sim_a_mt, sim_a_ma, sim_a_et, sim_a_ea, sim_a_ut, sim_a_ua, sim_a_rt, sim_a_ra, sim_s_et, sim_s_ea, sim_s_ut, sim_s_ua, sim_s_rt, sim_s_ra, backend, device))
    
    # collect data
    print(f'')
    print(f'[ENC] Collecting results...')
    ß = np.zeros((n_permutations, *outputs[0]['ß'].shape))
    ß_r = np.zeros_like(ß)
    r = np.zeros((n_permutations, outputs[0]['r'].shape[0]))
    
    # loop over outputs
    for output in outputs:
        # get tags
        p_i = output['p_i']
        
        # grab data
        ß[p_i] = output['ß']
        ß_r[p_i] = output['ß_r']
        r[p_i] = output['r']
    
    # make sure directory exists
    dir_out = f'./data/processed/eeg/sub{sid}/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    # save results
    print(f'[ENC] Saving results...')
    with gzip.open(f'{dir_out}rec-encran-b{int(b_brain)}-m{int(b_morph)}-c{int(b_clear)}-a{int(b_acoustic)}-s{int(b_semantic)}-k{int(n_topk)}-m{n_mod}.pkl.gz', 'wb') as f:
        pickle.dump((np.nanmean(ß, axis = 0), np.nanmean(ß_r, axis = 0), np.nanmean(r, axis = 0)), f)