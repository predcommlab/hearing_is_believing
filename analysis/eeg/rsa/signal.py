'''
Auxiliary functions to help with signal
processing.
'''

from typing import Any, Union, Callable, Dict

import numpy as np
import sklearn, scipy

from .math import *

def align_y_with_x(x: np.ndarray, y: np.ndarray, f: Callable = cosine, epsilon: float = 1e-27) -> int:
    '''
    '''
    
    # get parameters
    _, T = x.shape
    offsets = np.arange(-T, T, 1).astype(int)
    L = max(x.shape[1], y.shape[1])

    # populate shifted matrix
    y_h = np.zeros((offsets.shape[0], x.shape[0], int(3 * L)))
    for offset in offsets: y_h[(offset+T),:,(offset+T):(offset+T+y.shape[1])] = y

    # populate reference matrix
    x_h = np.zeros((offsets.shape[0], x.shape[0], int(3 * L)))
    for offset in offsets: x_h[(offset+T),:,(0+T):(0+T+x.shape[1])] = x

    # compute similarities
    y_e = f(x_h + epsilon, y_h + epsilon)

    # find best offset for shifting
    best = np.argmax(np.nanmean(y_e, axis = 1))

    return offsets[best]

def align_by_mismatch(p0: Any, p1: Any, fs: int = 200, fs_ph: int = 32000) -> tuple[int, int, int, int]:
    '''
    '''

    # find length, initiate point of uniqueness
    L = min(len(p0[2]), len(p1[2]))
    m = 0

    # loop over phonemes
    for i in np.arange(0, L, 1):
        # compare phonemes (and break at mismatch)
        if p0[2][i] != p1[2][i]:
            m = i
            break
    
    # compute mismatch timings (as an offset) within sequences (and durations)
    return (np.floor(p0[0][m] * (fs / fs_ph)).astype(int), np.floor(p1[0][m] * (fs / fs_ph)).astype(int), 
            np.floor(p0[1][m] * (fs / fs_ph)).astype(int), np.floor(p1[1][m] * (fs / fs_ph)).astype(int))

def aligned(x: np.ndarray, mor: int, pou: int, pad: Union[bool, tuple[int, int]] = False, epsilon: float = 0) -> np.ndarray:
    '''
    '''
    
    # if desired, create left and right paddings
    if pad:
        pad_L, pad_R = pad
        x0 = np.ones((x.shape[0], pad_L + x.shape[1] + pad_R)) * epsilon
        x0[:,pad_L:pad_L+x.shape[1]] = x
        x = x0
    
    # apply mor alignment
    y = np.ones(x.shape) * epsilon
    if mor < 0: y[:,0:x.shape[1]+mor] = x[:,np.abs(mor):]
    elif mor > 0: y[:,mor:x.shape[1]+mor] = x[:,0:-mor]
    else: y[:,0:x.shape[1]] = x

    # apply pou alignment
    z = np.ones(y.shape) * epsilon
    z[:,0:y.shape[1]-pou] = y[:,pou:]

    return z

def regress_out(x: np.ndarray, covariates: np.ndarray, demean: bool = True) -> np.ndarray:
    '''
    Remove `covariates` from `x`.
    
    INPUTS:
        x           -   Signal to clean
        covariates  -   Covariates to remove from signal
        demean      -   Should output signal be demeaned?
    
    OUTPUTS:
        y           -   Output signal
    '''
    
    # fit x as a function of covariates
    lm = sklearn.linear_model.LinearRegression()
    lm.fit(covariates.T, x)
    
    # return residuals
    return (x - (np.dot(lm.coef_, covariates) + (lm.intercept_ * int(demean))))

def maxZ(x: np.ndarray, axis_z: Union[int, tuple[int]] = (0, 2), axis_max: int = 2, axis_mu: Union[int, tuple[int]] = (0, 1)) -> np.ndarray:
    '''
    Find the absolute maximum Z values in `x`.
    
    INPUTS:
        x           -   Input signal
        axis_z      -   Axis to z-score along (default = (0, 2))
        axis_max    -   Axis to take the maximum along (default = 2)
    
    OUTPUTS:
        max(Z)      -   Absolute z-values
    '''
    
    # obtain Z
    Z = (x - x.mean(axis = axis_z, keepdims = True)) / x.std(axis = axis_z, keepdims = True)
    
    # obtain maximum absolute z-scores along axis
    Z_max = np.max(np.abs(Z), axis = axis_max)
    
    return Z_max

def binned(x: np.ndarray, fs: int = 200, current_fs: int = 1000, f: Callable = np.mean, f_args: Union[Dict, None] = None) -> np.ndarray:
    '''
    Downsample some signal `x` from `current_fs` to `fs` through some form of binning. Note
    that `x` should be 3D or 4D.
    
    INPUTS:
        x           -   Signal to downsample (`time` should be last axis)
        fs          -   Desired sampling frequency
        current_fs  -   Current sampling frequency
        f           -   Function to apply for binning (default = np.mean)
        f_args      -   Additional arguments to apply to `f`.
    
    OUTPUTS:
        y           -   Binned signal
    '''
    
    if len(x.shape) == 4:
        # setup arguments
        if f_args is None: f_args = dict(axis = 3)
        elif 'axis' not in f_args: f_args['axis'] = 3
        
        # preallocate memory
        bins = np.zeros((*x.shape[0:3], int(x.shape[3] * (fs / current_fs))))
        
        # complete binning process
        for b in np.arange(0, bins.shape[3], 1):
            bins[:,:,:,b] = f(x[:,:,:,b * np.floor(current_fs / fs).astype(int):(b+1) * np.ceil(current_fs / fs).astype(int)], **f_args)
    elif len(x.shape) == 3:
        # setup arguments
        if f_args is None: f_args = dict(axis = 2)
        elif 'axis' not in f_args: f_args['axis'] = 2
        
        # preallocate memory
        bins = np.zeros((x.shape[0], x.shape[1], int(x.shape[2] * (fs / current_fs))))
        
        # complete binning process
        for b in np.arange(0, bins.shape[2], 1):
            bins[:,:,b] = f(x[:,:,b * np.ceil(current_fs / fs).astype(int):(b+1) * np.ceil(current_fs / fs).astype(int)], **f_args)
    else:
        raise ValueError(f'`x` must be 3D or 4D, but not shape {x.shape}.')
    
    return bins

def boxcar(L: int) -> np.ndarray:
    '''
    Create a boxcar kernel.
    
    INPUTS:
        L   -   Length of kernel
    
    OUTPUTS:
        k   -   Kernel
    '''
    
    # return kernel
    return np.ones((L,))

def gaussian(L: int, sigma: float = 1.0) -> np.ndarray:
    '''
    Create a gaussian kernel.
    
    INPUTS:
        L       -   Length of kernel
        sigma   -   Sigma of kernel (default = 1)
    
    OUTPUTS:
        k       -   Kernel
    '''
    
    # compute radius
    r = np.linspace(-int(L / 2) + 0.5, int(L / 2) - 0.5, L)
    
    # compute kernel
    return np.array([1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-float(x)**2 / (2 * sigma**2)) for x in r])

def half_gaussian(L: int, sigma: float = 1.0, tail: str = 'left') -> np.ndarray:
    '''
    Create a half-gaussian kernel.
    
    INPUTS:
        L       -   Length of kernel
        sigma   -   Sigma of gaussian (default = 1)
        tail    -   Which tail to use? (default = 'left')
    
    OUTPUTS:
        k       -   Kernel
    '''
    
    # adjust length to be even
    if L % 2 == 1: L -= 1
    
    # get full gaussian
    full = gaussian(2 * L + 1, sigma = sigma)
    
    # slice desired tail
    half = np.zeros(full.shape)
    if tail == 'left': half[0:L+1] = full[0:L+1]
    elif tail == 'right': half[L+1:] = full[L+1]
    
    return half

def smoothen(x: np.ndarray, k: np.ndarray, axis: int = 1, mode: str = 'same', nan_mode: str = 'ignore') -> np.ndarray:
    '''
    Apply some smoothing kernel `k` over signal `x`.
    
    INPUTS:
        x           -   Arbitrary signal to smoothen
        k           -   Smoothing kernel (see boxcar, gaussian, half_gaussian)
        axis        -   Axis along which to apply the smoothing (default = 1)
        mode        -   Convolution mode (default = 'same')
        nan_mode    -   How to treat NaNs? (default = 'ignore', opt. 'propagate')
    
    OUTPUTS:
        y           -   Smoothed signal
    '''
    
    # NaN-free smoothing
    y = np.copy(x)
    y[np.isnan(x)] = 0
    y = np.apply_along_axis(scipy.signal.convolve, axis, y, k, mode = mode) / k.sum()
    
    # NaN-correction smoothing
    z = np.ones_like(x)
    z[np.isnan(x)] = 0
    z = np.apply_along_axis(scipy.signal.convolve, axis, z, k, mode = mode) / k.sum()
    
    # compute NaN-corrected signal and propagate NaNs
    z = y / z
    if nan_mode == 'propagate': z[np.isnan(x)] = np.nan
    
    return z

def dtw(x: np.ndarray, y: np.ndarray, cost_f: Callable = cosine_d) -> np.ndarray:
    '''
    Implements dynamic time warping between two gammatone
    spectrograms. In brief, this implements the DTW as in a 
    version for MATLAB provided by Dan Ellis:
        
        https://www.ee.columbia.edu/~dpwe/LabROSA/matlab/dtw/
    
    The main difference here is that this expects a batch of
    signals in x/y that should be aligned in parallel (leading
    to a substantial improvement in speed).
    
    INPUTS:
        x       -   Input signals `x` (`trials` x `channels` x `time`)
        y       -   Input signals `y` (`trials` x `channels` x `time`)
        cost_f  -   Cost function to use (default = rsa.math.cosine_d)
    
    OUTPUTS:
        z       -   Output signals `y` warped onto `x` (`trials` x `channels` x `time`)
    '''
    
    # make sure shapes match
    assert(np.all(x.shape == y.shape))
    
    # setup dims
    T, I, J = x.shape[0], x.shape[2], y.shape[2]
    
    # setup dtw tensor
    D = np.ones((T, I + 1, J + 1)) * np.inf
    D[:,0,0] = 0.0
    
    # setup cost tensor
    C = np.zeros((T, I, J))
    for t_i in range(I):
        C[:,t_i,:] = cost_f((x[:,:,t_i,np.newaxis] * np.ones_like(y)).swapaxes(1, 2),
                             y.swapaxes(1, 2))
    C[np.isnan(C)] = np.nanmax(C)
    
    # walk through matrices
    R = np.zeros_like(D)
    
    for i in range(1, I):
        for j in range(1, J):
            # setup costs and moves
            c_ij = C[:,i,j]
            opts = np.array([C[:,i-1,j-1], # match
                             C[:,i-1,j],   # insertion
                             C[:,i,j-1]])  # deletion
            
            # accumulate and determine moves
            D[:,i-1,j-1] = c_ij + opts.min(axis = 0)
            R[:,i,j] = opts.argmin(axis = 0)
    
    # cast R as int
    R = R.astype(int)
    
    # setup traversal
    max_steps = max(I, J) * 2
    
    step = 1
    i, j = np.ones((T,)) * (I - 1), np.ones((T,)) * (J - 1)
    p, q = np.zeros((T, max_steps)), np.zeros((T, max_steps))
    p[:,0] = i
    q[:,0] = j
    
    while np.any((i > 0) | (j > 0)):
        # break if we're going too far
        if (step >= max_steps): break
        
        # find all current moves
        c_ij = R[np.arange(T),i.astype(int),j.astype(int)].astype(int)
        
        # update matches
        i[c_ij == 0] -= 1
        j[c_ij == 0] -= 1
        
        # update insertions
        i[c_ij == 1] -= 1
        
        # update deletions
        j[c_ij == 2] -= 1
        
        # clip moves, just in case
        i, j = np.clip(i, 0, I - 1), np.clip(j, 0, J - 1)
        
        # register as p, q
        p[:,step] = i
        q[:,step] = j
        
        # tally
        step += 1
    
    # cast as int
    p = p.astype(int)
    q = q.astype(int)
    
    # invert trajectories
    p = np.flip(p, axis = 1)
    q = np.flip(q, axis = 1)
    
    # setup Q segments and warping
    Q = np.zeros((T, J))
    z = np.zeros_like(y)
    t = np.arange(I)
    
    # move through time
    for t_i in t:
        # find all occurrences where p >= time
        t_trl, t_tps = np.where(p >= t_i)
        
        # find first occurence per trial
        indc = np.nonzero(np.r_[1, np.diff(t_trl)[:-1]])[0]
        trls = indc // max_steps
        Q[trls,t_i] = q[trls,t_tps[indc]]
        
        # warp items at time
        z[:,:,t_i] = y[np.arange(T),:,Q[:,t_i].astype(int)]
    
    return z