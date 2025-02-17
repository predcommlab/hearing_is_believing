'''
embeddings::power.py

Common functions for power simulations. Available as embeddings.power.*
'''

from .internal import *
import numpy as np
from scipy.stats import chisquare as chsq
from scipy.stats import ttest_ind as tt_ind

def chisquare(N: int, R: int, p: np.ndarray, k: int = 1e4, alpha: float = 5e-2) -> tuple[float, np.ndarray]:
    '''
    Computes a power analysis for a chisquare test with probability vector `p` for implicit conditions over
    `N` participants being measured in `R` repeated measures. Optionally, number of samples `k` as well as
    `alpha`-level may be adjusted.

    Returns tuple of power `P` and length-`k` vector of p-values `X` of type np.ndarray.
    '''
    
    if type(p) == list: p = np.array(p)
    if type(k) == float: k = int(k)
    
    assert((type(N) == int) and (N > 0)) or critical(ref = 'embeddings::power::chisquare()', msg = f'`N` must be of type int and greater than zero.')
    assert((type(R) == int) and (R > 0)) or critical(ref = 'embeddings::power::chisquare()', msg = f'`R` must be of type int and greater than zero.')
    assert(type(p) == np.ndarray) or critical(ref = 'embeddings::power::chisquare()', msg = f'`p` must be a vector of type np.ndarray.')
    assert(p.sum() == 1) or critical(ref = 'embeddings::power::chisquare()', msg = f'`p` must sum to 1.')
    assert((type(k) == int) and (k > 0)) or critical(ref = 'embeddings::power::chisquare()', msg = f'`k` must be of type int and greater than zero.')
    assert((type(alpha) == float) and (alpha < 1)) or critical(ref = 'embeddings::power::chisquare()', msg = f'`alpha` must be of type float and smaller than one.')

    # sample `k` times and compute chisquare
    X = []
    for i in np.arange(0, k, 1):
        x = np.random.choice(np.arange(0, p.shape[0], 1), size = (N, R), p = p)
        _, p_x = chsq(np.array([np.sum(x.flatten() == j) for j in np.arange(0, p.shape[0], 1)]))
        X.append(p_x)

    # compute power from true positives
    P = np.round(np.sum(np.array(X) <= alpha) / k * 100, 2)

    return (P, np.array(X))

def ttest_ind(N: int, R: int, d: float, k: int = 1e4, alpha: float = 5e-2, alt: str = 'two-sided') -> tuple[float, np.ndarray]:
    '''
    Computes power analysis for independent samples t-test with Cohen's `d` between two implicit conditions 
    over `N` participants being measured in `R` repeated measures (here, this should almost always be 1).
    Optionally, number of samples `k` as well as `alpha`-level may be adjusted. Further, alternative
    hypothesis `alt` may be supplied (two-sided, less, greater).

    Returns tuple of power `P` and length-`k` vector of p-values `X` of type np.ndarray.
    '''

    if type(k) == float: k = int(k)

    assert((type(N) == int) and (N > 0)) or critical(ref = 'embeddings::power::ttest_ind()', msg = f'`N` must be of type int and greater than zero.')
    assert((type(R) == int) and (R > 0)) or critical(ref = 'embeddings::power::ttest_ind()', msg = f'`R` must be of type int and greater than zero.')
    assert(type(d) == float) or critical(ref = 'embeddings::power::ttest_ind()', msg = f'`p` must be a vector of type np.ndarray.')
    assert((type(k) == int) and (k > 0)) or critical(ref = 'embeddings::power::ttest_ind()', msg = f'`k` must be of type int and greater than zero.')
    assert((type(alpha) == float) and (alpha < 1)) or critical(ref = 'embeddings::power::ttest_ind()', msg = f'`alpha` must be of type float and smaller than one.')
    assert((type(alt) == str) and (alt in ['two-sided', 'less', 'greater'])) or critical(ref = 'embeddings::power::ttest_ind()', msg = f'`alt` must be of type string and in [two-sided, less, greater].')

    # sample `k` times and compute t-test
    X = []
    for i in np.arange(0, k, 1):
        x0 = np.random.normal(loc = 0.0, scale = 1.0, size = (N, R))
        x1 = np.random.normal(loc = 0.0 + 1.0 * d, scale = 1.0, size = (N, R))
        _, p_x = tt_ind(x0.flatten(), x1.flatten(), alternative = alt)
        X.append(p_x)
    
    # compute power from true positives
    P = np.round(np.sum(np.array(X) <= alpha) / k * 100, 2)

    return (P, np.array(X))
