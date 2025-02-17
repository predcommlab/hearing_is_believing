'''
Auxiliary functions to make multivariate
analyses a bit easier.

@TODO: I think principally these should be estimators.
'''

from .math import *
from . import math_torch
from . import estimators

import torch, mne
import numpy as np
import warnings

from typing import Any, Union, Callable, Dict

def compute_rdms(X: list[Any], f: Callable = spearmanr_d, backend: str = 'numpy', **kwargs: Any) -> np.ndarray:
    '''
    '''
    
    # validate backend
    assert(backend in ['numpy', 'torch'])
    
    # check data type
    if backend == 'numpy' and type(X[0]) == torch.Tensor:
        warnings.warn('Backend requested is `numpy`, but data are in `torch`. Switching backend to `torch`.')
        backend = 'torch'
    
    # check backend functions
    if backend == 'torch':
        f = math_torch.euclidean if f == euclidean else f
        f = math_torch.mahalanobis if f == mahalanobis else f
        f = math_torch.cosine if f == cosine else f
        f = math_torch.cosine_d if f == cosine_d else f
        f = math_torch.pearsonr if f == pearsonr else f
        f = math_torch.pearsonr_d if f == pearsonr_d else f
        f = math_torch.spearmanr if f == spearmanr else f
        f = math_torch.spearmanr_d if f == spearmanr_d else f
    
    # set backend
    bk = torch if backend == 'torch' else np
    
    # setup rdms
    rdm = bk.zeros((X[0].shape[1], len(X), len(X)))
    
    # start loop
    for i in bk.arange(0, len(X), 1):
        for j in bk.arange(0, len(X), 1):
            # skip mirrored entries
            if j < i: continue
            
            # pass data to metric function
            if f in [mahalanobis, math_torch.mahalanobis]: rdm[:,i,j] = rdm[j,i] = f(X[i].T, X[j].T, kwargs['iv'])
            else: rdm[:,i,j] = rdm[:,j,i] = f(X[i].T, X[j].T, *kwargs)
    
    return rdm

'''
def get_channel_loclab(info: mne._fiff.meas_info.Info, ch_type: Union[str, list[str], tuple[str], mne.utils._bunch.NamedInt] = 'all') -> tuple[np.ndarray]:
    if isinstance(ch_type, str) or isinstance(ch_type, mne.utils._bunch.NamedInt):
        ch_type = [ch_type,]
    
    if isinstance(ch_type, tuple):
        ch_type = list(ch_type)
    
    ch_type_mne = []
    
    for i in range(len(ch_type)):
        if ch_type[i] in ['eeg', 'all']: ch_type_mne.append(mne._fiff.constants.FIFF.FIFFV_EEG_CH)
        elif ch_type[i] in ['meg', 'all']: ch_type_mne.append(mne._fiff.constants.FIFF.FIFFV_MEG_CH)
        elif ch_type[i] in ['mag', 'all']: ch_type_mne.append(mne._fiff.constants.FIFF.FIFFV_MEG_CH)
        elif ch_type[i] in ['grad', 'all']: ch_type_mne.append(mne._fiff.constants.FIFF.FIFFV_MEG_CH)
        elif ch_type[i] in ['eog', 'all']: ch_type_mne.append(mne._fiff.constants.FIFF.FIFFV_EOG_CH)
        elif ch_type[i] in ['ecg', 'all']: ch_type_mne.append(mne._fiff.constants.FIFF.FIFFV_ECG_CH)
        elif ch_type[i] in ['emg', 'all']: ch_type_mne.append(mne._fiff.constants.FIFF.FIFFV_EMG_CH)
        elif isinstance(ch_type[i], mne.utils._bunch.NamedInt): ch_type_mne.append(ch_type[i])
        else: raise ValueError(f'Unknown `ch_type` provided (`ch_type[{i}]` = {ch_type[i]}).')
    
    types = np.array([info['chs'][i]['kind'] for i in range(len(info['chs']))])
    indc = np.where([types[i] in ch_type_mne for i in range(types.shape[0])])[0]
    
    if len(indc) < 1: raise ValueError(f'No channels of type `{ch_type}` found.')
    
    loc, label = np.array([info['chs'][i]['loc'][0:3] for i in indc]), np.array([info['chs'][i]['ch_name'] for i in indc])
    
    return loc, label
'''

def searchlight(X: list[Any], info: dict, r: float = 0.04, k: int = 5, method: str = 'sphere', f: Callable = compute_rdms, f_args: Dict = dict(f = spearmanr_d)) -> np.ndarray:
    '''
    '''
    
    # check method
    assert(method in ['sphere', 'topk'])
    
    # obtain channel positions
    ch_pos = np.zeros((len(info['chs']), 3))

    for i, ch in enumerate(info['chs']):
        ch_pos[i,:] = ch['loc'][0:3]
    
    # compute distances
    d = compute_rdms(np.array([ch_pos, ch_pos]).swapaxes(0, 1).swapaxes(1, 2), f = euclidean)[0,:,:]
    
    # find channels within spheres or closest k
    if method == 'sphere': s = [np.where(d[i,:] < r) for i in np.arange(0, d.shape[0], 1)]
    else: s = [d[i,d[i,:].argsort()[0:k]] for i in np.arange(0, d.shape[0], 1)]
    
    # prepare data arrays
    X = np.array(X)
    Y = np.zeros((len(s), X[0].shape[1], len(X), len(X)))
    
    # compute statistic within searchlights
    for i, s_i in enumerate(s):
        Y[i,:,:,:] = f([X[c_i,s_i,:].squeeze() for c_i in np.arange(0, X.shape[0], 1)], **f_args)

    return Y

'''
Unit tests
'''

def unit_tests_compute_rdms() -> bool:
    '''
    Computes RDMs over a set of orthogonal trials
    that must produce an identity matrix in cosine
    comparisons.
    
    Fails iff rdm != identitiy matrix.
    '''
    
    # generate some orthogonal 'trials'
    x = np.array([[[1, 0, 0]], 
                  [[0, 1, 0]], 
                  [[0, 0, 1]]]).swapaxes(1, 2)

    # compute rdms
    rdm = compute_rdms([*x], f = cosine).squeeze()
    
    # make sure it's an identity matrix
    assert(np.all(rdm == np.eye(3)))
    
    return True