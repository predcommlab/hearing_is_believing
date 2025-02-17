'''
Reimplementations of rsa.math.* functions using torch.
'''

import torch
import numpy as np
from typing import Any, Union, Callable

def euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Compute N-d euclidean distances. Please always
    supply features as the last dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        d   -   Euclidean distance(s)
    '''
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    return torch.sqrt(torch.sum((x - y)**2, -1))

def mahalanobis_1d(x: torch.Tensor, y: torch.Tensor, Σ: torch.Tensor) -> torch.Tensor:
    '''
    Computes 1D mahalanobis distance.
    
    INPUTS:
        x   -   Vector
        y   -   Vector
        Σ   -   Feature covariance
    
    OUTPUTS:
        d   -   Mahalanobis distance
    '''
    
    d = x - y
    return torch.sqrt(d.dot(Σ).dot(d.T))

def mahalanobis_2d(x: torch.Tensor, y: torch.Tensor, Σ: torch.Tensor) -> torch.Tensor:
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
    return torch.sqrt(d.mm(Σ).mm(d.T).diagonal())

def mahalanobis_3d(x: torch.Tensor, y: torch.Tensor, Σ: torch.Tensor) -> torch.Tensor:
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
    return torch.sqrt((d @ Σ @ d.swapaxes(1, 2)).swapaxes(0, 2).diagonal())

def mahalanobis(x: torch.Tensor, y: torch.Tensor, Σ: torch.Tensor) -> torch.Tensor:
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
    
    if (1 == torch.Tensor([len(x.shape), len(y.shape)])).all(): return mahalanobis_1d(x, y, Σ)
    elif (2 == torch.Tensor([len(x.shape), len(y.shape)])).all(): return mahalanobis_2d(x, y, Σ)
    elif (3 == torch.Tensor([len(x.shape), len(y.shape)])).all(): return mahalanobis_3d(x, y, Σ)
    
    raise NotImplementedError

def cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Computes N-d cosine similarities. Please always
    supply features as the final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        s   -   Similarities
    '''
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    return (x * y).sum(-1) / (torch.linalg.norm(x, dim = -1) * torch.linalg.norm(y, dim = -1))

def cosine_d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Computes N-d cosine distances. Please
    always supply features as the final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        d   -   Distances
    '''
    
    return 1 - cosine(x, y)

def pearsonr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Compute N-d pearson correlations. Please always
    supply features as the final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        r   -   Correlations
    '''
    
    if x.shape != y.shape:
        raise ValueError('`x` and `y` must have the same shape.')
    
    μ_x, μ_y = x.mean(-1, keepdim = True), y.mean(-1, keepdim = True)
    return torch.sum((x - μ_x) * (y - μ_y), -1) / torch.sqrt(torch.sum((x - μ_x)**2, -1) * torch.sum((y - μ_y)**2, -1))

def pearsonr_d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Compute N-d pearson distances. Please always
    supply features as the final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        r   -   Distances
    '''
    
    return 1 - pearsonr(x, y)

def rankdata(x: torch.Tensor) -> torch.Tensor:
    '''
    Rank data with ties as averages. Note
    that features must be the final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor

    OUTPUTS:
        r   -   Ranked data
    '''
    
    # sort tensor
    v, i = x.sort(dim = -1)
    
    # setup rank tensor
    r = torch.zeros_like(v, dtype = x.dtype, device = x.device)

    # loop over features to find ranks (with ties)
    for f_i in range(r.shape[-1]):
        r[...,f_i,None] = (v < v[...,f_i,None]).sum(-1, keepdim = True) + 1

    # resolve ties through average
    for f_i in range(r.shape[-1]):
        delta = (v == v[...,f_i,None]).sum(-1, keepdim = True) - 1
        r[...,f_i,None] += (delta / 2).to(dtype = x.dtype)

    # unsort ranked data
    r = r.gather(-1, i.argsort(-1))
    
    return r

def spearmanr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Compute N-d spearman correlations. Please always
    supply features as the final dimension.
    
    INPUTS:
        x   -   Vector/Matrix/Tensor
        y   -   Vector/Matrix/Tensor
    
    OUTPUTS:
        ρ   -   Correlations
    '''
    
    return pearsonr(rankdata(x), rankdata(y))

def spearmanr_d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Compute N-d spearman distances. Please always
    supply features as the final dimension.
    
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

def unit_tests_2d(N: int = 100, f: int = 25, device: str = 'cpu') -> bool:
    '''
    Runs unit tests over all 2D functions.
    '''
    
    import warnings, scipy
    
    tests = []
    
    # generate some 2D data
    x, y = np.random.normal(size = (N, f)).astype(np.float32), np.random.normal(size = (N, f)).astype(np.float32)
    Σ = np.cov(np.concatenate((x, y)).T).astype(np.float32)
    
    # move to torch
    x_t, y_t, Σ_t = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), torch.from_numpy(Σ).to(device)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        # euclidean distance
        ed_spy = np.array([scipy.spatial.distance.euclidean(x[i,:], y[i,:]) for i in np.arange(0, N, 1)])
        ed_rsa = euclidean(x_t, y_t).cpu().numpy()
        tests.append((ed_spy, ed_rsa))
        
        # mahalanobis distance
        md_spy = np.array([scipy.spatial.distance.mahalanobis(x[i,:], y[i,:], Σ) for i in np.arange(0, N, 1)])
        md_rsa = mahalanobis(x_t, y_t, Σ_t).cpu().numpy()
        tests.append((md_spy, md_rsa))
        
        # cosine distance
        cd_spy = np.array([scipy.spatial.distance.cosine(x[i,:], y[i,:]) for i in np.arange(0, N, 1)])
        cd_rsa = cosine_d(x_t, y_t).cpu().numpy()
        tests.append((cd_spy, cd_rsa))
        
        # pearson correlation
        pc_spy = np.array([scipy.stats.pearsonr(x[i,:], y[i,:]).statistic for i in np.arange(0, N, 1)])
        pc_rsa = pearsonr(x_t, y_t).cpu().numpy()
        tests.append((pc_spy, pc_rsa))
        
        # spearmanr correlation
        sc_spy = np.array([scipy.stats.spearmanr(x[i,:], y[i,:]).statistic for i in np.arange(0, N, 1)])
        sc_rsa = spearmanr(x_t, y_t).cpu().numpy()
        tests.append((sc_spy, sc_rsa))
    
    # loop over tests
    for test in tests:
        # unpack results
        spy, rsa = test
        
        # assert equality
        assert(np.isclose(spy, rsa, atol = 1e-3).all())
    
    return True

def unit_tests_3d(N: int = 100, f: int = 25, device: str = 'cpu') -> bool:
    '''
    Runs unit tests over all 3D functions.
    '''
    
    import warnings, scipy
    
    tests = []
    
    # generate some 3D data
    x, y = np.random.normal(size = (N, N, f)).astype(np.float32), np.random.normal(size = (N, N, f)).astype(np.float32)
    Σ = np.array([np.cov(np.concatenate((x[i,:,:], y[i,:,:])).T) for i in np.arange(0, N, 1)]).mean(axis = 0).astype(np.float32)
    
    # move to torch
    x_t, y_t, Σ_t = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), torch.from_numpy(Σ).to(device)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        # euclidean distance
        ed_spy = np.array([[scipy.spatial.distance.euclidean(x[i,j,:], y[i,j,:]) for j in np.arange(0, N, 1)] for i in np.arange(0, N, 1)])
        ed_rsa = euclidean(x_t, y_t).cpu().numpy()
        tests.append((ed_spy, ed_rsa))
        
        # mahalanobis distance
        md_spy = np.array([[scipy.spatial.distance.mahalanobis(x[i,j,:], y[i,j,:], Σ) for j in np.arange(0, N, 1)] for i in np.arange(0, N, 1)])
        md_rsa = mahalanobis(x_t, y_t, Σ_t).cpu().numpy()
        tests.append((md_spy, md_rsa))
        
        # cosine distance
        cd_spy = np.array([[scipy.spatial.distance.cosine(x[i,j,:], y[i,j,:]) for j in np.arange(0, N, 1)] for i in np.arange(0, N, 1)])
        cd_rsa = cosine_d(x_t, y_t).cpu().numpy()
        tests.append((cd_spy, cd_rsa))
        
        # pearson correlation
        pc_spy = np.array([[scipy.stats.pearsonr(x[i,j,:], y[i,j,:]).statistic for j in np.arange(0, N, 1)] for i in np.arange(0, N, 1)])
        pc_rsa = pearsonr(x_t, y_t).cpu().numpy()
        tests.append((pc_spy, pc_rsa))
        
        # spearmanr correlation
        sc_spy = np.array([[scipy.stats.spearmanr(x[i,j,:], y[i,j,:]).statistic for j in np.arange(0, N, 1)] for i in np.arange(0, N, 1)])
        sc_rsa = spearmanr(x_t, y_t).cpu().numpy()
        tests.append((sc_spy, sc_rsa))
    
    # loop over tests
    for test in tests:
        # unpack results
        spy, rsa = test
        
        # assert equality
        assert(np.isclose(spy, rsa, atol = 1e-3).all())
    
    return True