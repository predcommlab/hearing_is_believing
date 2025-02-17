'''
Auxiliary functions for handling (pseudo-)trials.
'''

import numpy as np
from sklearn.covariance import LedoitWolf

from typing import Any, Union, Callable

def group_by(X: np.ndarray, Y: np.ndarray, conds: Union[np.ndarray, list[str]]) -> np.ndarray:
    '''
    Group the EEG data by conditions.
    
    INPUTS:
        X       -   EEG data
        Y       -   EEG labels
        conds   -   Unique conditions
    '''
    
    # setup outputs
    Z = np.zeros((len(conds), int(X.shape[0] / len(conds)), X.shape[1], X.shape[2]))
    
    # loop over conditions and group
    for i, cond in enumerate(conds):
        indcs = np.where(Y == cond)[0]
        Z[i,:,:,:] = X[indcs,:,:]
    
    return Z

def pseudo(X: np.ndarray, conds: Union[np.ndarray, list[str]], N: int, use_indices: Union[None, np.ndarray] = None, mu: Callable = np.mean) -> tuple[np.ndarray, np.ndarray]:
    '''
    Generates `N` pseudotrials from the data `X` for
    the unique conditions `conds`.
    
    INPUTS:
        X           -   Grouped EEG data
        conds       -   Unique conditions
        N           -   Number of trials to generate
        use_indices -   A set of permutation indices to use from prior run (or None).
    
    OUTPUTS:
        Y       -   Pseudotrials
        indices -   Permutation indices
    '''
    
    # prepare outputs, get N per pseudotrial
    Z = np.zeros((X.shape[0], N, X.shape[2], X.shape[3]))
    N_t = int(X.shape[1] / N)
    
    # load precomputed indices, if desired
    if use_indices is None: saved_indices = np.zeros((len(conds), X.shape[1]))
    else: saved_indices = use_indices
    
    # loop over conditions
    for i, cond in enumerate(conds):
        # either permute and save or use precomputed
        if use_indices is None:
            indcs = np.arange(0, X.shape[1], 1)
            np.random.shuffle(indcs)
            saved_indices[i,:] = indcs
        else:
            indcs = saved_indices[i,:]
        
        # generate pseudotrial
        for j in np.arange(0, N, 1):
            Z[i,j,:,:] = mu(X[i,indcs[j*N_t:(j+1)*N_t],:,:], axis = 0)
    
    return (Z, saved_indices.astype(int))

def whiten(X: np.ndarray, trials: Union[np.ndarray, None] = None, ie: Union[np.ndarray, None] = None) -> tuple[np.ndarray, np.ndarray]:
    '''
    Whiten the grouped pseudotrials. For more information, please refer to
    the following papers:
        
        Ledoit, O., & Wolf, M. (2003). Honey, I shrunk the sample covariance matrix. UPF Economics and Business Working Paper, 691. 10.2139/ssrn.433840
        Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88, 365-411. 10.1016/S0047-259X(03)00096-4
        Guggenmos, M., Sterzer, P., & Cichy, R.M. (2018). Multivariate pattern analysis for MEG: A comparison of dissimilarity measures. NeuroImage, 173, 434-447. 10.1016/j.neuroimage.2018.02.044
        
    INPUTS:
        X       -   Pseudotrial data.
        trials  -   Indices of trials to include in covariance estimation.
        ie      -   Precision matrix (if precomputed).
    
    OUTPUTS:
        W       -   Whitened data
        ie      -   Precision matrix
    '''
    
    # if not supplied, compute whitening estimator
    if ie is None:
        # compute epsilon
        eps = np.zeros((X.shape[3], X.shape[0], X.shape[2], X.shape[2])) * np.nan
        trials = np.arange(0, X.shape[1], 1).astype(int) if trials is None else trials
        
        for t in np.arange(0, X.shape[3], 1):
            for c in np.arange(0, X.shape[0], 1):
                X_tc = X[c,trials,:,t]
                eps[t,c,:,:] = LedoitWolf().fit(X_tc).covariance_
        
        # average epsilon and take inverse
        ee = eps.mean(axis = (0, 1))
        ie = np.linalg.inv(ee)
    
    # apply whitening
    W = np.zeros(X.shape)
    
    for t in np.arange(0, X.shape[3], 1):
        for c in np.arange(0, X.shape[0], 1):
            for k in np.arange(0, X.shape[1], 1):
                W[c,k,:,t] = X[c,k,:,t] @ ie
    
    return (W, ie)

def normalise(X: np.ndarray, axis: Union[int, tuple[int]] = (0, 3), mu: Union[np.ndarray, None] = None, sigma: Union[np.ndarray, None] = None, NaN: bool = False) -> np.ndarray:
    '''
    Normalises pseudotrials by z-scoring
    along `axis` (default = 0, 3).
    
    INPUTS:
        X       -   Pseudotrial data.
        axis    -   Axes to z-score over.
        mu      -   Precomputed means (or None).
        sigma   -   Precomputed standard deviation (or None).
        NaN     -   Should nan functions be used or not?
    
    OUTPUTS:
        Z       -   Normalised pseudotrials
    '''
    
    if NaN == False: 
        mu = X.mean(axis = axis, keepdims = True) if mu is None else mu
        sigma = X.std(axis = axis, keepdims = True) if sigma is None else sigma
    else:
        mu = np.nanmean(X, axis = axis, keepdims = True) if mu is None else mu
        sigma = np.nanstd(X, axis = axis, keepdims = True) if sigma is None else sigma
    
    return (X - mu) / sigma