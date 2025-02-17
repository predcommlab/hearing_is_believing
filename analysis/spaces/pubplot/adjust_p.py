import sys
from .internals import *
import numpy as np
from typing import Union

# grab module
__pub = sys.modules['pubplot']

def bonferroni(p: Union[list[float], np.ndarray]) -> np.ndarray:
    '''
    '''
    
    return np.clip(np.array(p) * len(p), 0, 1)

def bonferroni_holm(p: Union[list[float], np.ndarray]) -> np.ndarray:
    '''
    '''
    
    return np.clip(np.array(p) * (len(p) - np.argsort(p).argsort()), 0, 1)

def fdr(p: Union[list[float], np.ndarray], method: str = 'bh') -> np.ndarray:
    '''
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