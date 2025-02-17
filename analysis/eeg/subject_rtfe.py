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
sys.path.append('../experiments/eeg/')

import embeddings as emb
import data, rsa

from multiprocessing import freeze_support
import numpy as np
import pandas as pd
import pickle
import gzip
import os

from typing import Any, Union

class RTFE:
    '''
    This is an implementation of the free-energy model detailed in our previous
    paper (to be referenced once submitted). Here, the key changes are that we
    are now doing this in real-time and that we are only computing the local
    idiosyncratic model.

    For reference, we are maximising:

        F = ln f(phi; p_mu_s, p_sigma_s) + ln f(M; p_mu_m, p_sigma_m)
    
    We do this by assuming input nodes for speaker identity and the sensory
    signal (i.e., w2v embedding) as well as error nodes for priors and the
    sensory signal and a prediction node (phi). These are solved through
    a couple of differential equations. For more detail, please see the
    original paper or code below.
    '''

    def __init__(self, sid: str, pid: str, dir: str):
        '''
        '''

        # setup pid
        self.pid = pid
        self.sid = sid
        self.dir = dir

        # load reduced glove
        self.G = emb.glove.load_embedding(f_in = '../../exp4_eeg/resources/model/w2v_50D.txt')
        
        # load priors
        with open('../../exp4_eeg/resources/model/priors.npy', 'rb') as f:
            self.priors = np.load(f)
        
        # setup speakers
        self.speakers = np.array(['essen', 'fashion', 'outdoor', 'technik', 'politik', 'unterhaltung'])

        # setup model
        self.I = np.eye(len(self.speakers))
        self.p_mu_s = np.tile(self.priors, (self.I.shape[0], 1)).T
        self.p_sigma_s = np.random.uniform(low = 1, high = 5, size = (len(self.G['kayak']),))
        self.p_sigma_m = np.random.uniform(low = 1, high = 5, size = (len(self.G['kayak']),))
        self.gamma = 1e-2
        self.T = 100

    
    def step(self, id: int, choice: str, speaker: str, learning: bool = True):
        '''
        '''

        # current sensations and initialise phi
        self.I_t = self.I[self.speakers == speaker,:].squeeze().reshape((len(self.speakers),))
        self.M = self.G[choice.lower()]
        self.phi = np.dot(self.p_mu_s, self.I_t)

        # setup error representations
        self.epsilon_s = self.phi*0
        self.epsilon_m = self.M*0

        # setup history bins
        epsilon_s = np.zeros((self.T, self.epsilon_s.shape[0]))
        epsilon_m = np.zeros((self.T, self.epsilon_m.shape[0]))
        phi = np.zeros((self.T, self.phi.shape[0]))

        # compute T model steps
        for i in np.arange(0, self.T, 1):
            # get derivatives
            self.d_epsilon_s = self.phi - np.dot(self.p_mu_s, self.I_t) - self.p_sigma_s * self.epsilon_s
            self.d_epsilon_m = self.M - self.phi - self.p_sigma_m * self.epsilon_m

            self.d_phi = self.epsilon_m - self.epsilon_s
            self.d_p_mu_s = self.epsilon_s
            self.d_p_sigma_s = .5 * (self.epsilon_s**2 - 1 / self.p_sigma_s)
            self.d_p_sigma_m = .5 * (self.epsilon_m**2 - 1 / self.p_sigma_m)

            # update model
            if learning:
                self.epsilon_s = self.epsilon_s + self.gamma * self.d_epsilon_s
                self.epsilon_m = self.epsilon_m + self.gamma * self.d_epsilon_m
                self.phi = self.phi + self.gamma * self.d_phi
                self.p_mu_s = self.p_mu_s + self.gamma * (self.I_t[:,np.newaxis] @ self.d_p_mu_s[np.newaxis,:]).T
                self.p_sigma_s = np.maximum(self.p_sigma_s + self.gamma * self.d_p_sigma_s, np.ones_like(self.p_sigma_s))
                self.p_sigma_m = np.maximum(self.p_sigma_m + self.gamma * self.d_p_sigma_m, np.ones_like(self.p_sigma_m))
            
            # write history
            epsilon_s[i,:] = self.epsilon_s
            epsilon_m[i,:] = self.epsilon_m
            phi[i,:] = self.phi
        
        # save data
        with gzip.open(os.path.join(self.dir, f'{self.pid}_t{id}_{speaker}.pkl.gz'), 'wb') as f:
            pickle.dump((epsilon_s, epsilon_m, phi, self.p_mu_s, self.p_sigma_s, self.p_sigma_m), f)

def worker(task: tuple[Any, str, str, int, int], **kwargs: Any) -> bool:
    '''
    External worker function (complementing embeddings::mp::Processor) that
    handles model fitting and evaluation.
    '''

    # parameters
    sid, pid, dir, generalised, unspecific, congruent, incongruent, f_in = task
    
    # status
    print(f'...{sid}')

    # create model
    model = RTFE(sid, pid, dir)
    
    # load data
    df = pd.read_csv(f_in)
    df = df.loc[(df.type == data.defs.TRIAL_MT_PRACTICE) | (df.type == data.defs.TRIAL_MT_CONTROL) | (df.type == data.defs.TRIAL_MT_MAIN) | (df.type == data.defs.TRIAL_ET_GOOD) | (df.type == data.defs.TRIAL_ET_BAD)].reset_index(drop = True)
    
    # loop over data
    for i in range(len(df)):
        # get trial
        trial = df.loc[i]
        
        # determine word
        if generalised:
            # correct option
            word = trial.options_0
        else:
            # choice
            word = trial.options_0 if (trial.correct == True) else trial.options_1 if ((trial.correct == True) & (np.isnan(trial.rt) == False)) else trial.options_0
        
        # determine context
        if unspecific:
            context = model.speakers[0]
        else:
            context = trial.context
        
        # determine if there should be an update step
        learning = True
        if (congruent) & (trial.type == data.defs.TRIAL_ET_BAD): learning = False
        if (incongruent) & (trial.type == data.defs.TRIAL_ET_GOOD): learning = False
        
        # step
        model.step(i - 6, word, context, learning = learning)
        
    return True

# setup paths
f_T = '../spaces/dumps/stimuli_pairs_from_old.xlsx'
f_C = '../spaces/dumps/stimuli_pairs_control.xlsx'
f_P = '../spaces/dumps/stimuli_pairs_practice.xlsx'

# system arguments
use_gpt = '-gpt' in sys.argv[1:]
use_bert = '-bert' in sys.argv[1:] if not use_gpt else False
use_llama = '-llama' in sys.argv[1:] if not use_gpt else False

# setup mode
generalised = '-generalised' in sys.argv[1:]
unspecific = '-unspecific' in sys.argv[1:]
congruent = '-congruent' in sys.argv[1:]
incongruent = '-incongruent' in sys.argv[1:]

# setup extension
folder = 'rtfe'

if use_gpt:
    folder += '-gpt'
elif use_bert:
    folder += '-bert'
elif use_llama:
    folder += '-llama'

if generalised:
    folder += '-generalised'
if unspecific:
    folder += '-unspecific'
if congruent:
    folder += '-congruent'
if incongruent:
    folder += '-incongruent'

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
    
    # make sure super directory exists
    dir = f'./data/raw/{folder}/'
    if os.path.isdir(dir) == False: os.mkdir(dir)

    # load tasks
    print('Setting up tasks...')    
    T = []
    for sub in data.Subjects.select(lambda sub: int(sub.sid) >= 2):
        dir = f'./data/raw/{folder}/sub{sub.sid}/'
        if os.path.isdir(dir) == False: os.mkdir(dir)
        T.append((sub.sid, sub.pid, dir, generalised, unspecific, congruent, incongruent, f'./data/raw/beh/sub{sub.sid}/{sub.pid}.csv'))
    
    # send tasks to workers
    print('Spawning workers...')
    processor.run(T, external = worker, timeout = None, f_T = f_T, f_C = f_C, f_P = f_P, G = G, verbose = False)