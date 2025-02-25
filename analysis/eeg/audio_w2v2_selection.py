'''
Because we have morphs in the learning task but would still
like to estimate semantic effects that must technically be
in reference to word_a or word_b within the morph (e.g., for
morph /_i:/ between /si:/ and /ti:/, we must compute surprisal
either for /si:/ or /ti:/), we would like to use an LLM as a
surrogate model. This is because we can feed it the morphed
audio data rather than having to rely on word-level data.

To do this, we want to know with some certainty that the LLM
does show some sensitivity to the control predictors of interest
(i.e., phonotactic, lexical and semantic surprisal). We also
want to know what layer to use as a statistical surrogate.

Consequently, here we use a brief narrative segment (Alice
in Wonderland, read by same speaker), compute the predictors
of interest and then train B2B models to decode predictors
from layer activations. This also allows us to disentangle
the unique variance explained within each predictor. We can
then select a layer based on the highest decoding accuracy.

OPTIONS
    fs                  -   Sampling frequency (default = 200)
    vocoded             -   Should we use vocoded data? (default = False)
    n_features          -   Number of features to use (default = 10)
    n_workers           -   Number of workers for parallelisation (default = 5)
    n_alphas            -   Number of alphas to test (default = 20)
    n_permutations      -   Number of permutations to run (default = 50)
    backend             -   What backend to use for model fitting? (numpy/torch, default = numpy)
    device              -   What device to use for model fitting, if applicable? (cpu/cuda/mps, default = cpu)
    maxtask             -   After `n` tasks, we should reset forked processes. (default = 1; this is relevant for CUDA, can be higher on CPU)

EXAMPLE USAGE:
    python audio_w2v2_selection.py n_permutations=10
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

import time, re
import numpy as np
import pandas as pd
import mne, os, sys, copy
import auxiliary as aux
import data, rsa, models
import gzip, pickle
import scipy, sklearn, torch, warnings

from multiprocessing import freeze_support, set_start_method

sys.path.append('../spaces/')
import embeddings as emb

from typing import Union, Dict, Any, Callable

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

def permutation_worker(task: Dict, settings: Union[tuple, None] = None) -> Union[int, Dict]:
    '''
    Computes one permutation of the B2B modelling proecedure.
    
    INPUTS:
        task        -   Task dictionary. Dictionary includes:
                            p_i:    Permutation index
                            indc:   Indices for current permutation
                            i:      Current iteration
        settings    -   Settings. Tuple includes:
                            ts:         Start time
                            N:          Number of permutations
                            alphas:     Alphas to test
                            X:          Activation matrix (`layers` x `phonemes` x `features` x `time`)
                            y:          Surprisal data (`phonemes` x `features` x `time`)
                            backend:    Backend to use.
                            device:     Device to use
    
    OUTPUTS:
        results     -   Results dictionary. Dictionary includes:
                            p_i:    Permutation index
                            S:      Feature contributions (`layers` x `features`)
                            r:      Decoding performance (`layers` x `features`)
    '''
    
    # unpack task data and settings
    p_i, indc, i = task['p_i'], task['indc'], task['i']
    ts, N, alphas, X, y, backend, device = settings
    
    # progressbar
    aux.progressbar(i, N, ts, msg = f'[B2B]')
    
    # setup place holders
    if backend == 'torch':
        S = torch.zeros((X.shape[0], y.shape[1]))
        r = torch.zeros((X.shape[0], y.shape[1]))
        
        # move to device
        if device == 'cuda': 
            # set cuda device
            device = f'cuda:{i % torch.cuda.device_count()}'
        
        X, y = torch.from_numpy(X).to(torch.float64).to(device), torch.from_numpy(y).to(torch.float64).to(device)
        alphas, indc = torch.from_numpy(alphas).to(torch.float64).to(device), torch.from_numpy(indc).to(torch.int64).to(device)
    else:
        S = np.zeros((X.shape[0], y.shape[1]))
        r = np.zeros((X.shape[0], y.shape[1]))
    
    # setup estimators
    estimator = rsa.analysis.estimators.torch.B2B if backend == 'torch' else rsa.analysis.estimators.B2B
    model = rsa.analysis.estimators.torch.Temporal(estimator = estimator, alphas = alphas) if backend == 'torch' else rsa.analysis.estimators.Temporal(estimator = estimator, alphas = alphas)
    
    # loop over layers
    for i, X_i in enumerate(X):
        # quickly remove 0-cols
        if backend == 'torch': 
            valid = np.where(X_i.cpu().numpy().any(axis = (0, 2)))[0]
            valid = torch.from_numpy(valid).to(torch.int64).to(device)
        else:
            valid = np.where(X_i.any(axis = (0, 2)))[0]
        X_i = X_i[:,valid,:]
        
        # setup indices
        train = indc
        test = indc[indc.shape[0] // 2:] # corresponds to split_b in b2b
        
        # fit model
        model.fit(X_i[train,:,:], y[train,:,:])
        
        # evaluate
        if backend == 'torch':
            S[i] = model.collect('S').mean(1).to(S.dtype)
            y_h = torch.cat([model.estimator_[i].scaler_out.inverse_transform(model.estimator_[i].decoder.predict(model.estimator_[i].scaler_ins.transform(X_i[test,:,i])))[None,:] for i in range(model.estimator_.shape[0])]).swapaxes(0, 1).swapaxes(1, 2).to(y.dtype).to(y.device)
            r[i] = torch.nanmean(rsa.math.torch.pearsonr(y[test,:,:].transpose(0, 2), y_h.transpose(0, 2)), dim = 0).to(r.dtype)
        else:
            S[i] = model.collect('S').mean(axis = 1)
            y_h = np.array([model.estimator_[i].scaler_out.inverse_transform(model.estimator_[i].decoder.predict(model.estimator_[i].scaler_ins.transform(X_i[test,:,i]))) for i in range(model.estimator_.shape[0])]).swapaxes(0, 1).swapaxes(1, 2)
            r[i] = np.nanmean(rsa.math.pearsonr(y[test,:,:].T, y_h.T), axis = 0)
    
    # move to cpu, if required
    if backend == 'torch':
        S, r = S.cpu().numpy(), r.cpu().numpy()
    
    return dict(p_i = p_i, S = S, r = r)

if __name__ == '__main__':
    '''
    Start B2B approach.
    '''
    
    # call freeze support for MP later
    freeze_support()
    
    # make sure we are not going to have trouble with locks from fork()
    set_start_method('spawn')
    
    # get options
    fs = aux.get_opt('fs', default = 200, cast = int)                           # what sampling frequency to use? (default = 200)
    vocoded = aux.get_opt('vocoded', default = 0, cast = bool)                  # should we use vocoded data? (default = False)
    n_features = aux.get_opt('n_features', default = 10, cast = int)            # which PCA projection should we use? (default = 10)
    n_workers = aux.get_opt('n_workers', default = 1, cast = int)               # how many workers to use? (default = 1)
    n_alphas = aux.get_opt('n_alphas', default = 20, cast = int)                # how many alpha values to test? (default = 20)
    n_permutations = aux.get_opt('n_permutations', default = 50, cast = int)    # how many permutations to use? (default = 100)
    backend = aux.get_opt('backend', default = 'numpy', cast = str)             # which backend to use? (default = numpy)
    device = aux.get_opt('device', default = 'cpu', cast = str)                 # which device to use, if applicable? (cpu/mps/cuda, default = cpu)
    maxtask = aux.get_opt('maxtask', default = 1, cast = int)                   # when do we reset workers (useful on CUDA, otherwise irrelevant; default = 1)
    
    # dump opts
    print(f'------- B2B: w2v2 --------')
    print(f'[OPTS]\tfs={fs}\t\tvocoded={vocoded}')
    print(f'[OPTS]\tn_features={n_features}\tn_workers={n_workers}')
    print(f'[OPTS]\tn_alphas={n_alphas}\tn_permutations={n_permutations}')
    print(f'[OPTS]\tbackend={backend}\tdevice={device}')
    print(f'--------------------------')
    
    # load phoneme data
    print(f'[PRE] Loading phoneme data...')
    collapsed = pd.read_csv('./data/preprocessed/misc/phonetics/collapsed.csv')
    
    # load embedding
    print(f'[PRE] Loading embedding...')
    G = emb.glove.load_embedding(f_in = '../spaces/glove-german/w2v_50D.txt')
    
    # obtain centre position
    words = G.key_to_index.keys()
    locs = np.array([G[word] for word in words])
    centre = locs.mean(axis = 0)
    
    # load SUBTLEX-de
    print(f'[PRE] Loading SUBTLEX-de...')
    subtlex = pd.read_excel('../spaces/dumps/subtlex-de.xlsx')
    sub_dict = np.array(subtlex.Word.tolist())
    
    # load annotation & gammatone data
    print(f'[PRE] Loading narrative data...')
    ann_pl2 = load_annotation(f'./data/preprocessed/audio/annotated/narrative/narrative.csv', fs = fs)
    if vocoded: gtg_pl2 = np.load(f'./data/preprocessed/audio/gt-200Hz/narrative/narrative_12ch.npy').T
    else: gtg_pl2 = np.load(f'./data/preprocessed/audio/gt-200Hz/narrative/narrative.npy').T
    
    # load phonotactic model
    print(f'[PRE] Loading PTX model...')
    ptx = models.PTX(collapsed)
    
    # retrieve phoneme tags
    phonemes = np.array(ann_pl2.phoneme.tolist())
    phoneme_onsets = np.array(ann_pl2.onset.tolist())
    phoneme_offsets = np.array(ann_pl2.offset.tolist())
    unq_phonemes, unq_N = np.unique(phonemes, return_counts = True)
    
    # retrieve word tags
    tokens = np.unique(ann_pl2.TOKEN.tolist())
    tokens_indx_s = np.array([np.argmax(np.array(ann_pl2.TOKEN.tolist()) == token) for token in tokens])
    tokens_indx_e = np.array([len(ann_pl2.TOKEN.tolist()) - np.argmax(np.array(np.array(ann_pl2.TOKEN.tolist()) == token)[::-1]) - 1 for token in tokens])
    token_onsets = np.array(ann_pl2.loc[tokens_indx_s].onset.tolist())
    token_offsets = np.array(ann_pl2.loc[tokens_indx_e].offset.tolist())
    
    # prepare model structure
    print(f'[PRE] Selecting models...')
    
    if vocoded:
        dnn = dict(gammatone = dict(fs = 200, path = './data/preprocessed/audio/gt-200Hz/narrative/narrative_12ch.npy', label = 'gammatones'))
        for i in np.arange(0, 7, 1): dnn[f'conv_L{i}'] = dict(fs = int(3200 / (2**i)), path = f'./data/preprocessed/audio/w2v2/reduced{n_features}_narrative_12ch/narrative_12ch_conv_L{i}.npy', label = fr'conv $L_{{{i}}}$')
        for i in np.arange(0, 24, 1): dnn[f'transformer_L{i}'] = dict(fs = 50, path = f'./data/preprocessed/audio/w2v2/reduced{n_features}_narrative_12ch/narrative_12ch_transformer_L{i}.npy', label = fr'trans $L_{{{i}}}$')
        dnn[f'decoder'] = dict(fs = 50, path = f'./data/preprocessed/audio/w2v2/reduced{n_features}_narrative_12ch/narrative_12ch_decoder.npy', label = r'decoder')
    else:
        dnn = dict(gammatone = dict(fs = 200, path = './data/preprocessed/audio/gt-200Hz/narrative/narrative.npy', label = 'gammatones'))
        for i in np.arange(0, 7, 1): dnn[f'conv_L{i}'] = dict(fs = int(3200 / (2**i)), path = f'./data/preprocessed/audio/w2v2/reduced{n_features}_narrative/narrative_conv_L{i}.npy', label = fr'conv $L_{{{i}}}$')
        for i in np.arange(0, 24, 1): dnn[f'transformer_L{i}'] = dict(fs = 50, path = f'./data/preprocessed/audio/w2v2/reduced{n_features}_narrative/narrative_transformer_L{i}.npy', label = fr'trans $L_{{{i}}}$')
        dnn[f'decoder'] = dict(fs = 50, path = f'./data/preprocessed/audio/w2v2/reduced{n_features}_narrative/narrative_decoder.npy', label = r'decoder')
    
    # extract surprisal data from narrative
    print(f'[PRE] Extracting surprisal data...')
    y = np.zeros((phonemes.shape[0], 3, 100)) # phonemes x surprisal x time

    p_i = 0
    p_N = len(tokens)
    ts = time.time()

    # loop over all tokens
    for t_i, token in enumerate(tokens):
        aux.progressbar(t_i, p_N, ts, msg = f'[PRE]')
        
        # get phonemes and onsets/offsets
        phonemes_i = np.array(ann_pl2.loc[ann_pl2.TOKEN == token].phoneme.tolist())
        s_i, e_i = np.array(ann_pl2.loc[ann_pl2.TOKEN == token].onset.tolist()), np.array(ann_pl2.loc[ann_pl2.TOKEN == token].offset.tolist())
        
        # compute trajectory from phonotactic model
        y_i, _, _ = ptx.trajectory(phonemes_i)
        
        # grab lexical and semantic surprisal, if available
        w_i = ann_pl2.loc[ann_pl2.TOKEN == token].ORT.tolist()[0]
        indx = sub_dict == w_i
        if indx.sum() > 0: L_i = -np.log2(subtlex.loc[indx].SUBTLEX.tolist()[0])
        else: L_i = 0
        if w_i.lower() in G: S_i = -np.log2(rsa.math.cosine(G[w_i.lower()], centre) + 1)
        else: S_i = 0
        
        # loop over phonemes in token
        for pho_i, (pho, on, off) in enumerate(zip(phonemes_i, s_i, e_i)):
            # save surprisal values
            y[p_i,0] = y_i[pho_i]
            y[p_i,1] = L_i
            y[p_i,2] = S_i
            
            # phoneme tally
            p_i += 1
    print(f'')
    
    # extract model activations
    print(f'[PRE] Extracting model activations...')
    
    X = np.zeros((len(dnn), phonemes.shape[0], np.max([28, n_features]), 100))
    F = np.zeros((len(dnn),))
    
    l_N = len(dnn)
    ts = time.time()
    
    # loop over layers
    for l_i, layer_key in enumerate(dnn):
        aux.progressbar(l_i, l_N, ts, msg = f'[PRE]')
        
        # grab layer
        layer = dnn[layer_key]

        # grab activations
        x_l = np.load(layer['path'])
        if layer_key == 'gammatone': x_l = x_l.T
        
        # interpolate samples linearly
        z_l = np.floor(np.linspace(0, x_l.shape[0] - 1, gtg_pl2.shape[0])).astype(int)
        x_l = x_l[z_l,:]
        
        # loop over phonemes and set activations
        for i, (on, off) in enumerate(zip(phoneme_onsets, phoneme_offsets)):
            X[l_i,i,0:x_l.shape[1],:] = x_l[on:on+100].T
    print(f'')
    
    # start worker pool
    print(f'[B2B] Starting worker pool...')
    processor = emb.mp.Processor(workers = n_workers, maxtasksperchild = maxtask)
    
    # create jobs
    print(f'[B2B] Preparing jobs...')
    jobs = []
    
    alphas = np.logspace(-5, 10, n_alphas)
    permutation_indc = np.array([np.random.choice(np.arange(0, y.shape[0], 1), size = (y.shape[0],), replace = False) for _ in range(n_permutations)])
    
    for p_i in np.arange(0, n_permutations, 1):
        job = dict(p_i = p_i, indc = permutation_indc[p_i,:], i = len(jobs))
        jobs.append(job)
    
    # run jobs
    print(f'[B2B] Decoding...')
    ts = time.time()
    outputs = processor.run(jobs, external = permutation_worker, settings = (ts, len(jobs), alphas, X, y, backend, device))
    
    # allocate memory
    S = np.zeros((n_permutations, X.shape[0], y.shape[1]))
    r = np.zeros((n_permutations, X.shape[0], y.shape[1]))
    
    # grab results
    print(f'')
    print(f'[B2B] Collecting results...')
    for output in outputs:
        # grab indices
        p_i = output['p_i']

        # grab data
        S[p_i] = output['S']
        r[p_i] = output['r']
    
    # do evaluation
    print(f'[B2B] Evaluating layers...')
    
    # convert to R2
    R = S / (S.sum(axis = 2, keepdims = True).max(axis = 1, keepdims = True))
    
    # compute geometric mean of ranks and take median
    ranks = R.shape[1] + 1 - scipy.stats.rankdata(R, axis = 1)
    median = np.median(ranks.prod(axis = 2).argmin(axis = 1)).astype(int)
    best = list(dnn.keys())[median]
    print(f'[B2B] Best layer identified: {best}.')
    
    # save results
    print(f'[B2B] Saving results...')
    dir_out = f'./data/processed/w2v2/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    with gzip.open(dir_out + f'b2b_f{n_features}.pkl.gz', 'wb') as f:
        pickle.dump((best, median, R, S, r), f)