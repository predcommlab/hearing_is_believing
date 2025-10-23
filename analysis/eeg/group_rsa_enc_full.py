'''
Quick script to run between-item similarity encoders over all subjects.

NOTE: This requires an installation of MVPy. See subject script.

All options from `subject_rsa_enc_full.py` are available here.
'''

import auxiliary as aux
import os, data

if __name__ == '__main__':
    # meta options
    start_from = aux.get_opt('from', cast = int, default = 0)       # start at sid no. N

    # processing options
    fs = aux.get_opt('fs', default = 200, cast = int)                           # frequency of reconstructed gammatones (must match subject_rsa_rec.py)
    n_topk = aux.get_opt('n_topk', default = 1, cast = int)                     # number of top-k predictions to consider
    n_k1 = aux.get_opt('n_k1', default = 10, cast = int)                        # length of smoothing kernel (in samples)
    pad_L = aux.get_opt('pad_L', default = n_k1, cast = int)                    # padding to apply before smoothing (should be >= n_k1)
    b_brain = bool(aux.get_opt('b_brain', default = 0, cast = int))             # should we use brain data for targets/alternatives?
    b_clear = bool(aux.get_opt('b_clear', default = 0, cast = int))             # should we load clears from clear training?
    b_morph = bool(aux.get_opt('b_morph', default = 0, cast = int))             # should we load morphs from clear training?
    n_mod = aux.get_opt('n_mod', default = -1, cast = int)                      # should we fit _only a specific_ model? (give index)
    n_folds = aux.get_opt('n_folds', default = 5, cast = int)                   # number of folds for cross-validation
    n_permutations = aux.get_opt('n_permutations', default = 50, cast = int)    # number of permutations to run
    n_workers = aux.get_opt('n_workers', default = 1, cast = int)               # number of parallel workers to use
    n_alphas = aux.get_opt('n_alphas', default = 20, cast = int)                # number of alphas to consider
    backend = aux.get_opt('backend', default = 'torch', cast = str)             # which backend should we use?
    device = aux.get_opt('device', default = 'cuda', cast = str)                # which device should we use, if applicable?

    # loop over subjects
    for subject in data.Subjects.trim():
        # skip if desired
        if int(subject.sid) < start_from: 
            print(f'[REC] Skipping subject{subject.sid}')
            continue
        
        # run subject
        os.system(f"python subject_rsa_enc_full.py id={subject.sid} fs={fs} n_k1={n_k1} n_topk={n_topk} pad_L={pad_L} b_brain={int(b_brain)} b_clear={int(b_clear)} b_morph={int(b_morph)} n_mod={n_mod} n_folds={n_folds} n_permutations={n_permutations} n_workers={n_workers} n_alphas={n_alphas} backend={backend} device={device}")