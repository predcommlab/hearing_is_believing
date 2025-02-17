'''
Quick script to traverse all participant data in the
experiment folder and estimate the free energy model 
at any point in time for each participant. Note that 
this script handles _all_ spaces, but not necessarily
at once. E.g., a 'normal' run of this will run spaces
for speaker-specific and -invariant priors with
generalised and individual posteriors with word em-
beddings taken from GloVe. If you would like to run,
for example, the control posteriors (simulated and
random choices), please supply `-control`. Similarly,
GloVe may be switched out for other embeddings by
supplying, e.g., `-gpt`.
'''

import sys
sys.path.append('../spaces/')

import embeddings as emb
import glob
from typing import Any
from multiprocessing import freeze_support

def worker(task: tuple[Any, str, str, int, int], **kwargs: Any) -> bool:
    '''
    External worker function (complementing embeddings::mp::Processor) that
    handles model fitting and evaluation.
    '''

    # parameters
    model_type, path, ppn, priors, posteriors = task
    gamma_func = lambda x: 1e-2

    # status
    print(f'...Running ({model_type}, {priors}, {posteriors}, {ppn})...')

    # create model
    model = model_type(f_in = ppn, type_priors = priors, type_posteriors = posteriors, path = path, **kwargs)

    # fit
    if (model_type  == emb.model.FreeEnergy): model.fit(gamma = gamma_func)
    else: model.fit()
    
    # evaluate
    model.evaluate()

    return True

# setup paths
f_T = '../spaces/dumps/stimuli_pairs_from_old.xlsx'
f_C = '../spaces/dumps/stimuli_pairs_control.xlsx'
f_P = '../spaces/dumps/stimuli_pairs_practice.xlsx'

'''
setup models to simulate here.

Note that, for the PyMC models, it isn't necessarily advisable to run them
using multiprocessing given that, firstly, PyMC already capitalises on MP,
making it relatively moot, and, secondly, they may occasionally fail (which
we currently don't deal with in this script). Instead, these models should
probably be fit using the accompanying file for fitting single participants.
This is, of course, only relevant if we even use the IO PyMC models, given
that FE and IO should be almost identical to begin with (they are funda-
mentally approximating the same thing anyway).

Also note that type_posteriors should be chosen here (unless you want
to compute both).
'''

# system arguments
control = '-control' in sys.argv[1:]
use_gpt = '-gpt' in sys.argv[1:]
use_bert = '-bert' in sys.argv[1:] if not use_gpt else False
use_llama = '-llama' in sys.argv[1:] if not use_gpt else False

# setup desired args
path_space = './data/models_space/' if not use_gpt and not use_bert and not use_llama and not control else \
             './data/models_space_gpt/' if use_gpt and not control else \
             './data/models_space_bert/' if use_bert and not control else \
             './data/models_space_llama/' if use_llama and not control else \
             './data/models_simulated/' if not use_gpt and not use_bert and control else \
             './data/models_simulated_gpt/' if use_gpt and control else \
             './data/models_simulated_bert/' if use_bert and control else \
             './data/models_simulated_llama/'

if control:
    types_models = [(emb.model.FreeEnergy, path_space)]
    types_priors = [emb.model.PRIOR_LOCAL, emb.model.PRIOR_GLOBAL]
    types_posteriors = [emb.model.POSTERIOR_SIMULATED, emb.model.POSTERIOR_RANDOM]
else:
    types_models = [(emb.model.FreeEnergy, path_space)]
    types_priors = [emb.model.PRIOR_LOCAL, emb.model.PRIOR_GLOBAL]
    types_posteriors = [emb.model.POSTERIOR_CORRECT, emb.model.POSTERIOR_CHOICE]

# enter main
if __name__ == "__main__":
    freeze_support()

    # get mp wrapper
    print('Setting up multiprocessing wrapper...')
    processor = emb.mp.Processor(workers = 4)

    # load glove
    print('Loading GloVe...')
    f_E = '../spaces/text-embedding-ada-002/w2v-50D.txt' if use_gpt else \
          '../spaces/bert-base-german-cased/w2v-50D.txt' if use_bert else \
          '../spaces/llama-7b/w2v-50D.txt' if use_llama else \
          '../spaces/glove-german/w2v_50D.txt'
    G = emb.glove.load_embedding(f_in = f_E)
    
    # load tasks
    print('Setting up tasks...')
    tasks = [f for f in glob.glob('./data/raw/*.csv')]
    T = []
    for ppn in tasks:
        for (model, path) in types_models:
            for prior in types_priors:
                for posterior in types_posteriors:
                    T.append((model, path, ppn, prior, posterior))
    
    # send tasks to workers
    print('Spawning workers...')
    processor.run(T, external = worker, timeout = None, f_T = f_T, f_C = f_C, f_P = f_P, G = G, verbose = False)