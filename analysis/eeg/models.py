'''
Models used across multiple scripts.
'''

import sys, time
import numpy as np

import rsa

from typing import Any, Union, Callable

class InformationTheoretic:
    '''
    Placeholder class to obtain IT measures
    through inheritance.
    '''
    
    def __init__(self) -> None:
        '''
        Never used.
        '''
        
        pass
    
    def norm(self, x: np.ndarray) -> np.ndarray:
        '''
        Computes a simple normalisation over values in x.
        
        INPUTS:
            x   -   Values to be normalised
        
        OUTPUTS:
            y   -   Normalised values
        '''
        
        return x / x.sum()
    
    def softmax(self, x: np.ndarray, T: int = 1) -> np.ndarray:
        '''
        Computes a softmax over the values in x.
        
        INPUTS:
            x   -   Values to be normalised
            T   -   Temperature of the softmax function (default = 1.0)
        
        OUTPUTS:
            y   -   Probability distribution
        '''
        
        return np.exp(x / T) / np.exp(x / T).sum()
    
    def surprisal(self, x: np.ndarray, eps: float = 0.0) -> np.ndarray:
        '''
        Computes the surprisal associated with a probability.
        
        INPUTS:
            x   -   Probability of event
            eps -   Epsilon to add to phoneme to avoid zero
        
        OUTPUTS:
            y   -   Surprisal at event
        '''
        
        return -np.log2(x + eps)
    
    def entropy(self, x: np.ndarray, eps: float = 0.0) -> np.ndarray:
        '''
        Computes entropy of a distribution.
        
        INPUTS:
            x   -   Probability distribution
            eps -   Epsilon to add to probabilities to avoid zero
        
        OUTPUTS:
            y   -   Entropy of distribution
        '''
        
        return -((x + eps) * np.log2(x + eps)).sum()

class PTX(InformationTheoretic):
    '''
    Phonotactic model. This allows obtaining the unfolding
    probabilities of individual phonemes in a word, weighted
    by the lexical expectancy (frequency) and phonotactic
    constraints.
    
    For a model that is also sensitive to a semantic prior,
    please see PTXS.
    '''
    
    def __init__(self, df, f_pho: str = f'./data/preprocessed/misc/phonetics/phonemes.npy') -> None:
        '''
        Setup the model.
        
        INPUTS:
            df      -   DataFrame containing the collapsed word/phoneme lists
            f_pho   -   File for phoneme names (default = './data/preprocessed/misc/phonetics/phonemes.npy')
        '''
        
        # load unique phonemes
        self.vocab = np.load(f_pho)
        
        # setup required strucutures
        self.df = df # collapsed dataframe
        
        # compute baseline
        self.update_prior()
    
    def update_prior(self) -> None:
        '''
        Updates baseline phoneme probabilities.
        '''
        
        # compute baseline
        candidates = self.df.loc[self.df.indx == 0]
        self.baseline, self.baseline_c = self.compute_distribution(candidates)
    
    def compute_distribution(self, candidates, eps = 1e-6):
        '''
        Computes the word probabilities conditioned on
        lexical frequency given the candidate words:
        
        p(word_k) = n \sum_{c = 0}{C} p(lex_c)

            where
                
                n = 1 / [\sum_{k = 0}{K} p(phoneme_k)]
                p(lex_c) = softmax(log_{10}(f), 1)
        
        INPUTS:
            candidates  -   Candidate words that can be considered (from collapsed dataframe)
            eps         -   Epsilon to add to phoneme probabilities (to avoid zeros)
        
        OUTPUTS:
            p   -   Normalised probabilities for all phonemes
        '''
        
        # find next phonemes
        indc = np.array(candidates.next_i.tolist())
        
        # compute lexical probabilities
        p_l = np.log10(candidates.freq.tolist())
        p_l = self.softmax(p_l, T = 1)
        
        # compute phoneme probabilities
        p = np.ones((len(self.vocab),)) * eps
        for p_l_i, indx in zip(p_l, indc): p[indx] += p_l_i
        p = self.norm(p)
        
        return p, p_l
    
    def trajectory(self, phonemes):
        '''
        Computes surprisal and entropy for each phoneme.
        
        INPUTS:
            phonemes    -   List of phonemes of the target word
        
        OUTPUTS:
            s   -   Surprisal at each phoneme
            e   -   Entropy at each phoneme
            c   -   Cohort entropy at each phoneme
        '''
        
        # setup data containers
        s = np.zeros((len(phonemes),))
        e = np.zeros((len(phonemes),))
        c = np.zeros((len(phonemes),))
        
        # add values from baseline
        indx = np.where(self.vocab == phonemes[0])[0][0]
        s[0] = self.surprisal(self.baseline[indx])
        e[0] = self.entropy(self.baseline)
        c[0] = self.entropy(self.baseline_c)
        
        # loop over following phonemes
        for i in range(len(phonemes) - 1):
            # find candidates
            context = ''.join(phonemes[0:i+1])
            candidates = self.df.loc[self.df.phonemes == context]
            
            # compute distribution
            p, p_c = self.compute_distribution(candidates)
            
            # compute metrics
            indx = np.where(self.vocab == phonemes[i+1])[0][0]
            s[i+1] = self.surprisal(p[indx])
            e[i+1] = self.entropy(p)
            c[i+1] = self.entropy(p_c)
        
        return (s, e, c)

class PTXS(InformationTheoretic):
    '''
    Phontactic model. In essence, this will allow obtaining
    the probability of individual phonemes in a word, weighted
    by the lexical expectancy (frequency), phoneme constraints
    (i.e., transitions), and precision-weighted semantic exp-
    ectancy.
    
    Note: This includes a semantic weighting! For a baseline
    class, please see PTX.
    '''
    
    def __init__(self, df, G, r, p, f_pho = f'./data/preprocessed/misc/phonetics/phonemes.npy') -> None:
        '''
        Setup the model and initial priors.
        
        INPUTS:
            df      -   DataFrame containing the collapsed word/phoneme lists.
            G       -   GloVe object
            r       -   Prior (see also: update_priors())
            p       -   Prior strength (see also: update_priors())
            f_pho   -   File for phoneme names (default = './data/preprocessed/misc/phonetics/phonemes.npy')
        '''
        
        # load unique phonemes
        self.vocab = np.load(f_pho)
        
        # setup required data structures
        self.df = df # collapsed dataframe
        self.G = G # glove
        
        # compute baseline
        self.update_prior(r, p)
    
    def update_prior(self, r, p):
        '''
        Updates the prior and prior strength, then
        recomputes the baseline phoneme distribution.
        
        INPUTS:
            r   -   Prior (GloVe embedding).
            p   -   Prior strength (0-1).
        '''
        
        # update values
        self.r = r
        self.p = np.clip(p, 0, 1 - 5e-2)
        
        # recompute baseline
        candidates = self.df.loc[self.df.indx == 0]
        self.baseline, self.baseline_c = self.compute_distribution(candidates)
    
    def compute_distribution(self, candidates, eps = 1e-6):
        '''
        Computes the phoneme probabilities conditioned on
        lexical frequency and semantic expectancy given
        the candidate words:
        
        p(phoneme_k) = n \sum_{c = 0}{C} p(lex_c) * p(sem_c)

            where
                
                n = 1 / [\sum_{k = 0}{K} p(phoneme_k)]
                p(lex_c) = softmax(log_{10}(f), 1)
                p(sem_c) = softmax(cs(r, G(c)), 1 - p)
        
        INPUTS:
            candidates  -   Candidate words that can be considered (from collapsed dataframe)
            eps         -   Epsilon to add to phoneme probabilities (to avoid zeros)
        
        OUTPUTS:
            p   -   Normalised probabilities for all phonemes
        '''
        
        # find next phonemes
        indc = np.array(candidates.next_i.tolist())
        
        # compute lexical probabilities
        p_l = np.log10(candidates.freq.tolist())
        p_l = self.softmax(p_l, T = 1)
        
        # compute semantic probabilities
        p_s = np.array([rsa.math.cosine(self.r, self.G[c.lower()]) for c in candidates.word.tolist()])
        p_s = self.softmax(p_s, T = 1 - self.p)
        
        # compute phoneme probabilities
        p = np.ones((len(self.vocab),)) * eps
        for p_l_i, p_s_i, indx in zip(p_l, p_s, indc): p[indx] += p_l_i * p_s_i
        p = self.norm(p)
        
        # compute cohort probabilities
        p_c = self.softmax(p_l * p_s, T = 1)
        
        return p, p_c
    
    def trajectory(self, phonemes):
        '''
        Computes surprisal and entropy for each phoneme.
        
        INPUTS:
            phonemes    -   List of phonemes of the target word
        
        OUTPUTS:
            s   -   Surprisal at each phoneme
            e   -   Entropy at each phoneme
            c   -   Cohort entropy at each phoneme.
        '''
        
        # setup data containers
        s = np.zeros((len(phonemes),))
        e = np.zeros((len(phonemes),))
        c = np.zeros((len(phonemes),))
        
        # add values from baseline
        indx = np.where(self.vocab == phonemes[0])[0][0]
        s[0] = self.surprisal(self.baseline[indx])
        e[0] = self.entropy(self.baseline)
        c[0] = self.entropy(self.baseline_c)
        
        # loop over following phonemes
        for i in range(len(phonemes) - 1):
            # find candidates
            context = ''.join(phonemes[0:i+1])
            candidates = self.df.loc[self.df.phonemes == context]
            
            # compute distribution
            p, p_c = self.compute_distribution(candidates)
            
            # compute metrics
            indx = np.where(self.vocab == phonemes[i+1])[0][0]
            s[i+1] = self.surprisal(p[indx])
            e[i+1] = self.entropy(p)
            c[i+1] = self.entropy(p_c)
        
        return (s, e, c)