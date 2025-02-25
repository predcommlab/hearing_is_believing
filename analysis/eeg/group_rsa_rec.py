'''
Quick script to run stimulus reconstruction models 
over all subjects.

All options supported by `subject_rsa_rec.py` are
available here.
'''

import os, data
import auxiliary as aux

if __name__ == '__main__':
    # meta options
    start_from = aux.get_opt('from', cast = int, default = 0)       # start at sid no. N

    # processing options
    fs = aux.get_opt('fs', default = 200, cast = int)                           # frequency to downsample to (and reconstruct gammatones at)
    n_k1 = aux.get_opt('n_k1', default = 1, cast = int)                         # length of smoothing kernel for neural data (in samples)
    n_k2 = aux.get_opt('n_k2', default = 20, cast = int)                        # length of smoothing kernel for gammatones (in samples)
    off_L = aux.get_opt('off_L', default = 100, cast = int)                     # offset where data begins (i.e., cuts off prestim)
    max_L = aux.get_opt('max_L', default = 300, cast = int)                     # maximum length of episodes to consider (including prestim)
    b_clear = bool(aux.get_opt('b_clear', default = 0, cast = int))             # should we train on clear data?
    n_folds = aux.get_opt('n_folds', default = 5, cast = int)                   # number of folds for cross-validation
    n_permutations = aux.get_opt('n_permutations', default = 100, cast = int)   # number of permutations to run
    n_workers = aux.get_opt('n_workers', default = 2, cast = int)               # number of workers for parallelisation
    n_alphas = aux.get_opt('n_alphas', default = 20, cast = int)                # number of alphas to test
    t_min = aux.get_opt('t_min', default = 0.0, cast = float)                   # minimum time to consider for reconstruction model (in seconds)
    t_max = aux.get_opt('t_max', default = 0.25, cast = float)                  # maximum time to consider for reconstruction model (in seconds)
    backend = aux.get_opt('backend', default = 'torch', cast = str)             # which backend to use (numpy/torch)?
    device = aux.get_opt('device', default = 'cuda', cast = str)                # which device to use, if applicable (cpu/cuda/mps)?
    
    # loop over subjects
    for subject in data.Subjects.trim():
        # skip if desired
        if int(subject.sid) < start_from: 
            print(f'[REC] Skipping subject{subject.sid}')
            continue
        
        # run subject
        os.system(f"python subject_rsa_rec.py id={subject.sid} fs={fs} n_k1={n_k1} n_k2={n_k2} off_L={off_L} max_L={max_L} b_clear={int(b_clear)} n_folds={n_folds} n_permutations={n_permutations} n_workers={n_workers} n_alphas={n_alphas} t_min={t_min} t_max={t_max} backend={backend} device={device}")