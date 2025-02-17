import sys
sys.path.append('../semantic_spaces/')

import numpy as np
import pickle
import gzip
import os

import embeddings as emb

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

    def __init__(self, pid: str):
        '''
        '''

        # setup pid
        self.pid = pid

        # load reduced glove
        self.G = emb.glove.load_embedding(f_in = './resources/model/w2v_50D.txt')
        
        # load priors
        with open('./resources/model/priors.npy', 'rb') as f:
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

    
    def step(self, id: int, choice: str, speaker: str):
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
        with gzip.open(f'./models/{self.pid}_t{id}_{speaker}.pkl.gz', 'wb') as f:
            pickle.dump((epsilon_s, epsilon_m, phi, self.p_mu_s, self.p_sigma_s, self.p_sigma_m), f)