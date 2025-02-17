'''
Some auxiliary functions for statistics
'''

import numpy as np

from typing import Union, Any

def bootstrap(x: np.ndarray, N: int = 10000) -> np.ndarray:
    '''
    Obtain a bootstrapped view of `x`.
    
    INPUTS:
        x       -   Array to bootstrap
        N       -   Number of bootstraps (default = 10000)

    OUTPUTS:
        x       -   Bootstrapped view
    '''
    
    # return a boostrapped view
    return x[np.random.choice(x.shape[0], replace = True, size = (N, x.shape[0]))]

def bootstrap_se(x: np.ndarray, N: int = 10000) -> np.ndarray:
    '''
    Compute the standard error of `x` via bootstrapping.
    
    INPUTS:
        x       -   Array to bootstrap
        N       -   Number of bootstraps (default = 10000)
    
    OUTPUTS:
        se      -   Standard error
    '''
    
    # compute bootstrapped standard error
    x = bootstrap(x, N = N)
    se = x.mean(axis = 1).std(axis = 0)
    
    return se

def cohens_d(x: np.ndarray, y: Union[None, np.ndarray] = None, popmean: Union[None, float, np.ndarray] = 0.0, paired: bool = False) -> float:
    '''
    Compute Cohen's d for one-sample, paired samples or student's t-tests.
    
    INPUTS:
        x       -   First sample (`subjects` x ...)
        y       -   Second sample (`subjects` x ...)
        popmean -   Population mean (default = 0.0)
        paired  -   Are samples paired?
    
    OUTPUTS:
        d       -   Cohen's d (...)
    '''
    
    if y is None:
        # one-sample t-test
        numerator = x.mean(axis = 0, keepdims = True) - popmean
        denominator = x.std(axis = 0, keepdims = True)
    else:
        if paired:
            # paired samples t-test
            z = (x - y)
            numerator = z.mean(axis = 0, keepdims = True)
            denominator = z.std(axis = 0, keepdims = True)
        else:
            # student's t-test
            numerator = x.mean(axis = 0, keepdims = True) - y.mean(axis = 0, keepdims = True)
            denominator = np.sqrt((np.sum((x - x.mean(axis = 0, keepdims = True)) ** 2) + np.sum((y - y.mean(axis = 0, keepdims = True)) ** 2)) / (x.shape[0] + y.shape[0] - 2))
    
    # compute d
    d = numerator / denominator
    
    return d

def bonferroni(p: Union[list[float], np.ndarray]) -> np.ndarray:
    '''
    Perform a Bonferroni correction.
    
    INPUTS:
        p       -   p-values

    OUTPUTS:
        p       -   Bonferroni corrected p-values
    '''
    
    return np.clip(np.array(p) * len(p), 0, 1)

def bonferroni_holm(p: Union[list[float], np.ndarray]) -> np.ndarray:
    '''
    Perform a Bonferroni-Holm correction.
    
    INPUTS:
        p   -   p-values
    
    OUTPUTS:
        p   -   Bonferroni-Holm corrected p-values
    '''
    
    return np.clip(np.array(p) * (len(p) - np.argsort(p).argsort()), 0, 1)

def fdr(p: Union[list[float], np.ndarray], method: str = 'bh') -> np.ndarray:
    '''
    Perform a False Discovery Rate (FDR) correction.
    
    INPUTS:
        p       -   p-values
        method  -   Method to use (bh/by, default = 'bh')

    OUTPUTS:
        p       -   FDR corrected p-values
    '''
    
    if method not in ['bh', 'by']: method = 'bh'
    
    # sort
    o = np.argsort(p)
    q = np.take(p, o)
    
    # compute q values
    F = np.arange(1, len(p)+1, 1) / float(len(p))
    if method == 'by': F /= np.sum(1. / np.arange(1, len(p)+1, 1))
    q = q / F
    q = np.clip(np.minimum.accumulate(q[::-1])[::-1], 0, 1)
    
    # unsort
    q_o = np.zeros_like(q)
    q_o[o] = q
    
    return q_o