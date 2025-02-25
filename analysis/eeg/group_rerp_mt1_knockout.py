'''
Quick script to run encoding analysis over
all subjects in learning task.

All options available to `subject_rerp_mt1.py` can
be passed to this script.

NOTE: Running these analyses with `s_bsl=llm` (default)
requires that w2v2 activations have been extracted, pro-
jected and evaluated. Please see:

    `audio_w2v2.py`, `audio_w2v2_pca.py`, `audio_w2v2_selection.py`

NOTE: Running these analyses requires you to specify which model to
evaluate. Consequently, this should be done _after_ evaluating results
from rERP analysis.
'''

import auxiliary as aux
import os, data

if __name__ == '__main__':
    # meta options
    start_from = aux.get_opt('from', cast = int, default = 0)       # start at sid no. N

    # processing options
    fs = aux.get_opt('fs', default = 200, cast = int)                           # frequency to downsample signal to (through binnning)
    b_zsc = bool(aux.get_opt('b_zsc', default = 0, cast = int))                 # should we normalise all data and predictors?
    b_spa = bool(aux.get_opt('b_spa', default = 0, cast = int))                 # should we enforce sparsity? (i.e., are we interested in TRF-like events or more temporally distributed computations?)
    b_bsl = bool(aux.get_opt('b_bsl', default = 0, cast = int))                 # should we baseline correct epochs?
    s_mod = aux.get_opt('s_mod', default = 'spc', cast = str)                   # which hypotheses do we test here? (spc = specific, inv = invariant)
    s_bsl = aux.get_opt('s_bsl', default = 'llm', cast = str)                   # which baseline models to use? (tar/alt/llm for target/alternative/large language model).
    n_features = aux.get_opt('n_features', default = 10, cast = int)            # how many features to use from LLM?
    n_mod = aux.get_opt('n_mod', default = 2, cast = int)                       # which model to test (default = 2)?
    a_coefs = aux.get_opt('a_coefs', default = 'all', cast = str)               # which coefficients to test (default = -1)?
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
    
    # loop over subjects
    for subject in data.Subjects.trim():
        # skip if desired
        if int(subject.sid) < start_from: 
            print(f'[ERP] Skipping subject{subject.sid}')
            continue
        
        # run subject
        os.system(f"python subject_rerp_mt1_knockout.py id={subject.sid} fs={fs} b_zsc={int(b_zsc)} b_spa={int(b_spa)} s_mod={s_mod} s_bsl={s_bsl} n_features={n_features} n_mod={n_mod} a_coefs={a_coefs} b_bsl={int(b_bsl)} n_workers={n_workers} n_alphas={n_alphas} n_folds={n_folds} n_permutations={n_permutations} n_topk={n_topk} n_edge_l={n_edge_L} n_maxl={n_maxL} n_tmin={n_tmin} n_tmax={n_tmax} backend={backend} device={device} maxtask={maxtask}")