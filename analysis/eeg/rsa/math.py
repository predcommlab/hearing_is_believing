'''
Auxiliary functions for common math operations
for RSAs. Typically, these are reimplementations
of things found in scipy or numpy, but for higher
dimensional arrays.
'''

import numpy as np
from scipy.stats import rankdata
from typing import Any, Union, Callable

from . import math_torch as torch

def euclidean_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Computes 1D euclidean distances between x and y.
    
    INPUTS:
        x   -   Vector
        y   -   Vector
    
    OUTPUTS:
        d   -   Euclidean distance
    '''
    
    return np.sqrt(np.sum((x - y)**2))

def euclidean_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Computes 2D euclidean distances between vectors in x and y.
    
    INPUTS:
        x   -   Matrix (samples x features)
        y   -   Matrix (samples x features)
    
    OUTPUTS:
        d   -   Euclidean distances
    '''
    
    return np.sqrt(np.sum((x - y)**2, axis = 1))

def euclidean_3d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Computes 3D euclidean distances between vectors in x and y.
    
    INPUTS:
        x   -   Matrix (samples x samples x features)
        y   -   Matrix (samples x samples x features)
    
    OUTPUTS:
        d   -   Euclidean distances
    '''
    
    return np.sqrt(np.sum((x - y)**2, axis = 2))

def euclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute 1D, 2D or 3D euclidean distances. Please always
    supply features as the last dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        d   -   Euclidean distance(s)
    '''
    
    if (1 == np.array([len(x.shape), len(y.shape)])).all(): return euclidean_1d(x, y)
    elif (2 == np.array([len(x.shape), len(y.shape)])).all(): return euclidean_2d(x, y)
    elif (3 == np.array([len(x.shape), len(y.shape)])).all(): return euclidean_3d(x, y)
    
    raise NotImplementedError

def mahalanobis_1d(x: np.ndarray, y: np.ndarray, Σ: np.ndarray) -> np.ndarray:
    '''
    Computes 1D mahalanobis distance.
    
    INPUTS:
        x   -   Vector
        y   -   Vector
        Σ   -   Feature covariance
    
    OUTPUTS:
        d   -   Mahalanobis distance
    '''
    
    return np.sqrt(np.dot(x - y, Σ).dot((x - y).T))

def mahalanobis_2d(x: np.ndarray, y: np.ndarray, Σ: np.ndarray) -> np.ndarray:
    '''
    Computes 2D mahalanobis distances between vectors in x and y.
    
    INPUTS:
        x   -   Matrix (samples x features)
        y   -   Matrix (samples x features)
        Σ   -   Feature covariance
    
    OUTPUTS:
        d   -   Mahalanobis distances
    '''
    
    d = x - y
    return np.sqrt(np.dot(d, Σ).dot(d.T)).diagonal()

def mahalanobis_3d(x: np.ndarray, y: np.ndarray, Σ: np.ndarray) -> np.ndarray:
    '''
    Computes 3D mahalanobis distances between vectors in x and y.
    
    INPUTS:
        x   -   Tensor (samples x samples x features)
        y   -   Tensor (samples x samples x features)
        Σ   -   Feature covariance
    
    OUTPUTS:
        d   - Mahalanobis distances
    '''
    
    d = x - y
    return np.sqrt(np.dot(d, Σ).dot(d.swapaxes(1, 2)).swapaxes(1, 2).diagonal().diagonal())

def mahalanobis(x: np.ndarray, y: np.ndarray, Σ: np.ndarray) -> np.ndarray:
    '''
    Computes 1D, 2D or 3D mahalanobis distances. Please always supply
    features as the last dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
        Σ   -   Feature covariance
    
    OUTPUTS:
        d   -   Mahalanobis distance(s)
    '''
    
    if (1 == np.array([len(x.shape), len(y.shape)])).all(): return mahalanobis_1d(x, y, Σ)
    elif (2 == np.array([len(x.shape), len(y.shape)])).all(): return mahalanobis_2d(x, y, Σ)
    elif (3 == np.array([len(x.shape), len(y.shape)])).all(): return mahalanobis_3d(x, y, Σ)

    raise NotImplementedError

def cosine_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Computes 1D cosine similarity.
    
    INPUTS:
        x   -   Vector
        y   -   Vector
    
    OUTPUTS:
        s   -   Similarity
    '''
    
    return (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))

def cosine_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Computes 2D cosine similarity between vectors in x and y.
    
    INPUTS:
        x   -   Matrix (samples x features)
        y   -   Matrix (samples x features)
    
    OUTPUTS:
        s   -   Similarities
    '''
    
    return np.sum(x * y, axis = 1) / (np.linalg.norm(x, axis = 1) * np.linalg.norm(y, axis = 1))

def cosine_3d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Computes 3D cosine similarity between vectors in x and y.
    
    INPUTS:
        x   -   Tensor (samples x samples x features)
        y   -   Tensor (samples x samples x features)
    
    OUTPUTS:
        s   -   Similarities
    '''
    
    return np.sum(x * y, axis = 2) / (np.linalg.norm(x, axis = 2) * np.linalg.norm(y, axis = 2))

def cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Computes 1D, 2D or 3D cosine similarities. Please always
    supply features as the final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        s   -   Similarities
    '''
    
    if (1 == np.array([len(x.shape), len(y.shape)])).all(): return cosine_1d(x, y)
    elif (2 == np.array([len(x.shape), len(y.shape)])).all(): return cosine_2d(x, y)
    elif (3 == np.array([len(x.shape), len(y.shape)])).all(): return cosine_3d(x, y)

    raise NotImplementedError

def cosine_d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Computes 1D, 2D or 3D cosine distances. Please
    always supply features as the final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        d   -   Distances
    '''
    
    return 1 - cosine(x, y)

def pearsonr_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute pearson correlation.
    
    INPUTS:
        x   -   Vector
        y   -   Vector
    
    OUTPUTS:
        r   -   Correlation
    '''
    
    μ_x, μ_y = x.mean(), y.mean()
    return np.sum((x - μ_x) * (y - μ_y)) / np.sqrt(np.sum((x - μ_x)**2) * np.sum((y - μ_y)**2))

def pearsonr_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute pearson correlations between vectors in x and y.
    
    INPUTS:
        x   -   Matrix (samples x features)
        y   -   Matrix (samples x features)
    
    OUTPUTS:
        r   -   Correlations
    '''
    
    μ_x, μ_y = x.mean(axis = 1, keepdims = True), y.mean(axis = 1, keepdims = True)
    return np.sum((x - μ_x) * (y - μ_y), axis = 1) / np.sqrt(np.sum((x - μ_x)**2, axis = 1) * np.sum((y - μ_y)**2, axis = 1))

def pearsonr_3d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute pearson correlations between vectors in x and y.
    
    INPUTS:
        x   -   Tensor (samples x samples x features)
        y   -   Tensor (samples x samples x features)
    
    OUTPUTS:
        r   -   Correlations
    '''
    
    μ_x, μ_y = x.mean(axis = 2, keepdims = True), y.mean(axis = 2, keepdims = True)
    return np.sum((x - μ_x) * (y - μ_y), axis = 2) / np.sqrt(np.sum((x - μ_x)**2, axis = 2) * np.sum((y - μ_y)**2, axis = 2))

def pearsonr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute 1D, 2D or 3D pearson correlations. Please always
    supply features as the final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        r   -   Correlations
    '''
    
    if (1 == np.array([len(x.shape), len(y.shape)])).all(): return pearsonr_1d(x, y)
    elif (2 == np.array([len(x.shape), len(y.shape)])).all(): return pearsonr_2d(x, y)
    elif (3 == np.array([len(x.shape), len(y.shape)])).all(): return pearsonr_3d(x, y)

    raise NotImplementedError

def pearsonr_d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute 1D, 2D or 3D pearson distances. Please always
    supply features as the final dimension.
    
    INPUTS:
        x
        y
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        d   -   Distances
    '''
    
    return 1 - pearsonr(x, y)

def spearmanr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute 1D, 2D or 3D spearman correlations. Please always
    supply features as the final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        ρ   -   Correlations
    '''
    
    if (1 == np.array([len(x.shape), len(y.shape)])).all(): return pearsonr_1d(rankdata(x), rankdata(y))
    elif (2 == np.array([len(x.shape), len(y.shape)])).all(): return pearsonr_2d(rankdata(x, axis = 1), rankdata(y, axis = 1))
    elif (3 == np.array([len(x.shape), len(y.shape)])).all(): return pearsonr_3d(rankdata(x, axis = 2), rankdata(y, axis = 2))

    raise NotImplementedError

def spearmanr_d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Compute 1D, 2D or 3D spearman distances. Please always
    supply the features as the final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        d   -   Distances
    '''
    
    return 1 - spearmanr(x, y)

'''
Unit tests
'''

def unit_tests_2d(N: int = 100, f: int = 25) -> bool:
    '''
    Runs unit tests over all 2D functions.
    '''
    
    import warnings, scipy
    
    tests = []
    
    # generate some 2D data
    x, y = np.random.normal(size = (N, f)), np.random.normal(size = (N, f)) 
    Σ = np.cov(np.concatenate((x, y)).T)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        # euclidean distance
        ed_spy = np.array([scipy.spatial.distance.euclidean(x[i,:], y[i,:]) for i in np.arange(0, N, 1)])
        ed_rsa = euclidean(x, y)
        tests.append((ed_spy, ed_rsa))
        
        # mahalanobis distance
        md_spy = np.array([scipy.spatial.distance.mahalanobis(x[i,:], y[i,:], Σ) for i in np.arange(0, N, 1)])
        md_rsa = mahalanobis(x, y, Σ)
        tests.append((md_spy, md_rsa))
        
        # cosine distance
        cd_spy = np.array([scipy.spatial.distance.cosine(x[i,:], y[i,:]) for i in np.arange(0, N, 1)])
        cd_rsa = cosine_d(x, y)
        tests.append((cd_spy, cd_rsa))
        
        # pearson correlation
        pc_spy = np.array([scipy.stats.pearsonr(x[i,:], y[i,:]).statistic for i in np.arange(0, N, 1)])
        pc_rsa = pearsonr(x, y)
        tests.append((pc_spy, pc_rsa))
        
        # spearmanr correlation
        sc_spy = np.array([scipy.stats.spearmanr(x[i,:], y[i,:]).statistic for i in np.arange(0, N, 1)])
        sc_rsa = spearmanr(x, y)
        tests.append((sc_spy, sc_rsa))
    
    # loop over tests
    for test in tests:
        # unpack results
        spy, rsa = test
        
        # assert equality
        assert(np.isclose(spy, rsa).all())
    
    return True

def unit_tests_3d(N: int = 100, f: int = 25) -> bool:
    '''
    Runs unit tests over all 3D functions.
    '''
    
    import warnings, scipy
    
    tests = []
    
    # generate some 3D data
    x, y = np.random.normal(size = (N, N, f)), np.random.normal(size = (N, N, f))
    Σ = np.array([np.cov(np.concatenate((x[i,:,:], y[i,:,:])).T) for i in np.arange(0, N, 1)]).mean(axis = 0)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        # euclidean distance
        ed_spy = np.array([[scipy.spatial.distance.euclidean(x[i,j,:], y[i,j,:]) for j in np.arange(0, N, 1)] for i in np.arange(0, N, 1)])
        ed_rsa = euclidean(x, y)
        tests.append((ed_spy, ed_rsa))
        
        # mahalanobis distance
        md_spy = np.array([[scipy.spatial.distance.mahalanobis(x[i,j,:], y[i,j,:], Σ) for j in np.arange(0, N, 1)] for i in np.arange(0, N, 1)])
        md_rsa = mahalanobis(x, y, Σ)
        tests.append((md_spy, md_rsa))
        
        # cosine distance
        cd_spy = np.array([[scipy.spatial.distance.cosine(x[i,j,:], y[i,j,:]) for j in np.arange(0, N, 1)] for i in np.arange(0, N, 1)])
        cd_rsa = cosine_d(x, y)
        tests.append((cd_spy, cd_rsa))
        
        # pearson correlation
        pc_spy = np.array([[scipy.stats.pearsonr(x[i,j,:], y[i,j,:]).statistic for j in np.arange(0, N, 1)] for i in np.arange(0, N, 1)])
        pc_rsa = pearsonr(x, y)
        tests.append((pc_spy, pc_rsa))
        
        # spearmanr correlation
        sc_spy = np.array([[scipy.stats.spearmanr(x[i,j,:], y[i,j,:]).statistic for j in np.arange(0, N, 1)] for i in np.arange(0, N, 1)])
        sc_rsa = spearmanr(x, y)
        tests.append((sc_spy, sc_rsa))
    
    # loop over tests
    for test in tests:
        # unpack results
        spy, rsa = test
        
        # assert equality
        assert(np.isclose(spy, rsa).all())
    
    return True