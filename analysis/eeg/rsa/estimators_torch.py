'''
Estimators that are GPU-ready to make analyses quick and easy.

NOTE: Support for torch is still experimental. I have weeded
out most bugs and, generally, it seems to work quite well. However,
there are some things to keep in mind. One of the key things 
that I have found testing simulated data and real neural data
so far is that data should always be 64-bit floats (`torch.float64`).
Otherwise, we run into precision issues quite regularly that may
very negatively impact model fit. Beyond that, for simulated
data, please make sure to not fit a perfect model, because the
fitting of coefficients requires some degree of error to be
correctly handled. Obviously, this is not a concern in any
sort of real world setting, but more of a reminder to myself.

@TODO: Write type checks and appropriate warnings?

@TODO: I have looked into the FFT method and it's absolutely
possible to use multiple alphas at once. Implement this as a
faster alternative for TimeDelayed?
'''

from .math_torch import *

import torch
import numpy as np
import sklearn, scipy

from typing import Any, Union, Callable, Dict

class StandardScaler(sklearn.base.BaseEstimator):
    '''
    This ports `mne.preprocessing.StandardScaler` to torch, except
    it's rather crude and doesn't do their consecutive fitting approach.
    '''
    
    def __init__(self, copy: bool = False, with_mean: bool = True, with_std: bool = True) -> None:
        '''
        Create new scaler.
        
        INPUTS:
            copy        -   Whether to copy data.
            with_mean   -   Whether to center data.
            with_std    -   Whether to scale data.
        '''
        
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        
    def _reset(self) -> None:
        '''
        Resets scaler.
        '''

        del self.mean_
        del self.var_
        del self.scale_
    
    def fit(self, X: torch.Tensor, y: Union[torch.Tensor, None] = None, sample_weight: Union[torch.Tensor, None] = None):
        '''
        Fit the scaler.
        
        INPUTS:
            X               -   X data (`trials` x `features`)
            y               -   y data (`trials` x `features`)
            sample_weight   -   Sample weights (`trials`)
        
        OUTPUTS:
            self
        '''
        
        # check X data
        assert(type(X) == torch.Tensor)
        
        # check y data
        if y is not None:
            assert(type(y) == torch.Tensor)
            assert(y.shape[0] == X.shape[0])
        
        # check weights
        if sample_weight is not None:
            assert(type(sample_weight) == torch.Tensor)
            assert(sample_weight.shape[0] == X.shape[0])
            
            if len(sample_weight.shape) == 1:
                sample_weight = sample_weight[:, None]
                
        # weighted approach
        if sample_weight is not None:
            n = (sample_weight != 0.).sum()
            self.mean_ = ((X * sample_weight).sum(0) / sample_weight.sum())[None,:]
            self.var_ = (((X - self.mean_)**2 * sample_weight).sum(0) / ((n - 1) / n * sample_weight.sum()))[None,:]
            self.scale_ = torch.sqrt(self.var_)
        else:
            self.mean_ = X.mean(0, keepdim = True)
            self.var_ = X.var(0, keepdim = True)
            self.scale_ = torch.sqrt(self.var_)
        
        return self
    
    def transform(self, X: torch.Tensor, copy: Union[bool, None] = None) -> torch.Tensor:
        '''
        Transforms data.
        
        INPUTS:
            X       -   X data (`trials` x `features`)
            copy    -   Whether to copy data.

        OUTPUTS:
            X       -   Transformed data (`trials` x `features`)
        '''
        
        # check data
        assert(type(X) == torch.Tensor)
        
        # check attributes
        assert(hasattr(self, 'mean_'))
        assert(hasattr(self, 'scale_'))
        
        # demean
        if self.with_mean:
            X = X - self.mean_
        
        # scale
        if self.with_std:
            X = X / self.scale_
        
        return X
    
    def inverse_transform(self, X: torch.tensor, copy: Union[bool, None] = None) -> torch.Tensor:
        '''
        Inverse transform data.
        
        INPUTS:
            X       -   X data (`trials` x `features`)
            copy    -   Whether to copy data.
        
        OUTPUTS:
            X       -   Transformed data (`trials` x `features`)
        '''

        # check data
        assert(type(X) == torch.Tensor)
        
        # check attributes
        assert(hasattr(self, 'mean_'))
        assert(hasattr(self, 'scale_'))

        # scale
        if self.with_std:
            X = X * self.scale_

        # add mean
        if self.with_mean:
            X = X + self.mean_

        return X

class RidgeCV(sklearn.base.BaseEstimator):
    '''
    This ports `sklearn.linear_model.RidgeCV` to torch. This is a pretty
    crude modification of J.R. King's torch_ridge (and, of course, the
    original version).
    
    Adapted from:
        github.com/kingjr/torch_ridge/blob/master/torch_ridge/ridge.py
        github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/_ridge.py
    
    Original authors:
        Jean-Remi King <jeanremi.king@gmail.com>
    
    Original authors:
        Mathieu Blondel <mathieu@mblondel.org>
        Reuben Fletcher-Costin <reuben.fletchercostin@gmail.com>
        Fabian Pedregosa <fabian@fseoane.net>
        Michael Eickenberg <michael.eickenberg@nsup.org>
    '''
    
    def __init__(self, alphas: Union[torch.Tensor, int, float, list, np.ndarray] = torch.Tensor([1]), fit_intercept: bool = True, normalise: bool = False, alpha_per_target: bool = False) -> None:
        '''
        Create a new RidgeCV model.
        
        INPUTS:
            alphas              -   Regularisation parameters.
            fit_intercept       -   Should we fit an intercept?
            normalise           -   Should we normalise data (for fitting)?
            alpha_per_target    -   Should regularisation parameters be estimated for each feature?
        '''
        
        # check alphas
        if type(alphas) in [int, float]:
            alphas = torch.Tensor([alphas])
        
        if type(alphas) == np.ndarray:
            alphas = torch.from_numpy(alphas)
        
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalise = normalise
        self.alpha_per_target = alpha_per_target
    
    def _preprocess(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor]:
        '''
        Preprocess data for fitting.
        
        INPUTS:
            X       -   X data (`trials` x `features`)
            y       -   y data (`trials` x `features`)

        OUTPUTS:
            X           -   Preprocessed X data (`trials` x `features`)
            y           -   Preprocessed y data (`trials` x `features`)
            X_offset    -   X offsets
            X_scale     -   X scales
            y_offset    -   y offsets
            y_scale     -   y scales
        '''
        
        if self.fit_intercept:
            # find and demean
            X_offset = X.mean(0)
            X -= X_offset
            
            # same for scale, if required
            if self.normalise:
                X_scale = torch.sqrt((X**2).sum(0))
                X /= X_scale
            else:
                X_scale = torch.ones(X.shape[1], dtype = X.dtype, device = X.device)
            
            # same for y
            y_offset = y.mean(0)
            y -= y_offset
        else:
            X_offset = torch.zeros(X.shape[1], dtype = X.dtype, device = X.device)
            X_scale = torch.ones(X.shape[1], dtype = X.dtype, device = X.device)
            y_offset = torch.zeros(y.shape[1], dtype = X.dtype, device = X.device)
        
        return (X, y, X_offset, X_scale, y_offset)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        '''
        Fit the RidgeCV model.
        
        INPUTS:
            X       -   X data (`trials` x `features`)
            y       -   y data (`trials` x `features`)
        
        OUTPUTS:
            self
        '''
        
        # check data types
        assert(type(X) == torch.Tensor)
        assert(type(y) == torch.Tensor)
        
        # check shapes
        assert(X.shape[0] == y.shape[0])
        
        # ensure y shape
        if len(y.shape) == 1:
            y = y[:, None]
        
        # get dims
        n, n_x = X.shape
        _, n_y = y.shape
        
        # preprocess
        X, y, X_offset, X_scale, y_offset = self._preprocess(X, y)
        
        # decomposition
        U, S, _ = torch.linalg.svd(X, full_matrices = False)
        v = S ** 2
        Uy = U.transpose(0, 1) @ y
        
        # perform LOO per alpha
        cv_duals = torch.zeros((self.alphas.shape[0], n, n_y), dtype = X.dtype, device = X.device)
        cv_errors = torch.zeros_like(cv_duals, dtype = X.dtype, device = X.device)
        
        for a_i, alpha in enumerate(self.alphas):
            # solve for duals
            w = ((v + alpha) ** - 1) - alpha ** -1
            cv_duals[a_i] = U @ torch.diag(w) @ Uy + alpha ** -1 * y
            
            # compute errors
            G = ((w * U**2).sum(-1) + alpha ** -1)[:,None]
            cv_errors[a_i] = cv_duals[a_i] / G
        
        # if required, find alpha per feature
        if self.alpha_per_target:
            best = (cv_errors**2).mean(1).argmin(0)
            
            duals = torch.zeros((n, n_y), dtype = X.dtype, device = X.device)
            errors = torch.zeros_like(duals, dtype = X.dtype, device = X.device)
            
            for a_i, best_a in enumerate(best):
                duals[:,a_i] = cv_duals[best_a,:,a_i]
                errors[:,a_i] = cv_errors[best_a,:,a_i]
        else:
            best = (cv_errors.reshape(len(self.alphas), -1)**2).mean(1).argmin(0)
            duals = cv_duals[best]
            errors = cv_errors[best]
        
        # set alpha and coef
        self.alpha_ = self.alphas[best]
        self.coef_ = duals.transpose(0, 1) @ X
        
        # find intercept
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale[None,:]
            Xoff_coef = X_offset[None,:] @ self.coef_.transpose(1, 0)
            self.intercept_ = y_offset - Xoff_coef
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Get predictions from fitted model.
        
        INPUTS:
            X       -   X data (`trials` x `features`)

        OUTPUTS:
            y       -   y data (`trials` x `features`)
        '''
        
        # make sure model was fit
        assert(hasattr(self, 'coef_'))
        assert(hasattr(self, 'intercept_'))
        
        return X @ self.coef_.transpose(1, 0) + self.intercept_

class TimeDelayed(sklearn.base.BaseEstimator):
    '''
    This is a port of rsa.analysis.estimators.TimeDelayed, but using torch instead of numpy/sklearn.
    '''
    
    def __init__(self, t_min: float = 0.0, t_max: float = 0.5, fs: float = 200, estimator: sklearn.base.BaseEstimator = RidgeCV, alphas: Union[torch.Tensor, list, int] = torch.Tensor([1]), scaler: Dict = dict(), patterns: bool = False, **kwargs) -> None:
        '''
        Creates a TimeDelayed estimator.
        
        INPUTS:
            t_min       -   Minimum time delay to consider (in seconds). Positive values indicate X is delayed relative to y.
            t_max       -   Maximum time delay to consider (in seconds). Positive values indicate X is delayed relative to y.
            fs          -   Sampling frequency (Hz).
            estimator   -   Estimator to use (default = rsa.analysis.estimators.torch.RidgeCV)
            alphas      -   Alphas to test (default = [1])
            scaler      -   Options to use for scaler (default = dict()), see `rsa.analysis.estimators.torch.StandardScaler` for options)
            patterns    -   Should inverse patterns be returned? (default = False, only applicable for reconstruction models).
            kwargs      -   **kwargs to pass to estimator
        '''
        
        # setup options
        self.alphas = alphas
        self.patterns = patterns
        self.kwargs = kwargs
        self.opts_scaler = scaler
        
        # setup window options
        self.t_min, self.t_max = t_min, t_max
        self.fs = fs
        
        t_neg = np.arange(0, np.abs(np.ceil(t_min * fs)) + 1, 1)
        t_pos = np.arange(0, np.abs(np.ceil(t_max * fs)) + 1, 1)
        self.window = torch.from_numpy(np.unique(np.concatenate((-t_neg[::-1], t_pos))).astype(int))
        
        # setup estimators
        self.scaler_X = StandardScaler(**self.opts_scaler)
        self.scaler_y = StandardScaler(**self.opts_scaler)
        self.estimator = estimator(alphas = self.alphas, **kwargs)
    
    def make_design(self, X: torch.Tensor, y: Union[torch.Tensor, None] = None) -> Union[torch.Tensor, tuple[torch.Tensor]]:
        '''
        Create time delayed design matrix.
        
        INPUTS:
            X   -   X data (`trials` x `channels` x `time`)
            y   -   y data (`trials` x `features` x `time`)
        
        OTPUTS:
            X_t -   Time delayed design matrix (`trials * time` x `channels * window`)
            y_t -   Time delayed outcomes (`trials * time` x `features`)
        '''
        
        # setup design matrix & outcomes
        X_t = torch.zeros((X.shape[0] * X.shape[2], X.shape[1] * self.window.shape[0])).to(X.device)
        if y is not None: y_t = torch.zeros((X.shape[0] * X.shape[2], y.shape[1])).to(y.device)
        
        # loop over trials
        for i in range(X.shape[0]):
            # loop over time points
            for j in range(X.shape[2]):
                # grab index in widened matrix
                ij = i * X.shape[2] + j
                
                # setup window indices
                win_j = torch.clip(j + self.window, min = 0, max = X.shape[2] - 1).to(X.device)
                cor_j = ((j + self.window) < 0) | ((j + self.window) > (X.shape[2] - 1))
                cor_j = (cor_j[None,:] * torch.ones((X.shape[1], cor_j.shape[0]), dtype = bool)).to(X.device)
                
                # write data
                X_ij = X[i,:,win_j]
                X_ij[cor_j] = 0.0 # correct edges
                X_t[ij,:] = X_ij.T.flatten()
                
                if y is not None: y_t[ij,:] = y[i,:,j]
        
        if y is not None: return X_t, y_t
        return X_t
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Fit time delayed model
        
        INPUTS:
            X   -   X data (`trials` x `channels` x `time`)
            y   -   y data (`trials` x `features` x `time`)
        '''
        
        # check data
        if len(y.shape) == 2:
            y = y[:,None,:]
        
        # keep track of dimensionality
        f, c, w = y.shape[1], X.shape[1], self.window.shape[0]
        self.f, self.c, self.w = f, c, w
        
        # create lagged matrices
        X, y = self.make_design(X, y)
        
        # setup scaling
        self.scaler_X.fit(X)
        self.scaler_y.fit(y)
        
        # fit model
        X, y = self.scaler_X.transform(X), self.scaler_y.transform(y)
        self.estimator.fit(X, y)
        
        # setup coefficients
        self.coef_ = self.estimator.coef_.reshape((f, w, c)).swapaxes(1, 2)
        self.coef_ = self.coef_.flip(2)
        
        # if required, estimate patterns
        if self.patterns:
            X = (X - X.mean(0))
            S_X = torch.cov(X.T)
            
            # obtain covariance of y
            if y.shape[1] > 1:
                y = (y - y.mean(0, keepdim = True))
                P_y = torch.linalg.pinv(torch.cov(y.T))
            else:
                P_y = 1.0 / float(y.shape[0] - 1)
            
            # obtain inverse pattern
            if y.shape[1] > 1: self.pattern_ = S_X.mm(self.estimator.coef_.T).mm(P_y)
            else: self.pattern_ = S_X.mm(self.estimator.coef_.T) * P_y
            self.pattern_ = self.pattern_.T.reshape((f, w, c)).swapaxes(1, 2)
    
    def predict(self, X: torch.Tensor, reshape: bool = True, invert_y: bool = False) -> torch.Tensor:
        '''
        Obtain model predictions.
        
        INPUTS:
            X           -   X data (`trials` x `channels` x `time`)
            reshape     -   Should predictions be reshaped to original dimensions? (default = True)
            invert_y    -   Should predictions be inverted to original scale? (default = False)

        OUTPUTS:
            y           -   Predictions (`trials` x `features` x `time`)
        '''
        
        # keep track of dimensions
        n, c, t = X.shape
        
        # transform X
        X = self.make_design(X)
        X = self.scaler_X.transform(X)
        
        # make and reshape predictions
        y = self.estimator.predict(X)
        if invert_y: y = self.scaler_y.inverse_transform(y)
        if reshape: y = y.reshape((n, t, self.f)).swapaxes(1, 2)
        
        return y

    def score(self, X: torch.Tensor, y: torch.Tensor, method: Union[str, Callable] = 'corrcoef') -> np.ndarray:
        '''
        Score the model.
        
        INPUTS:
            X           -   X data (`trials` x `channels` x `time`)
            y           -   y data (`trials` x `features` x `time`)
            method      -   Method for scoring (default = 'corrcoef')

        OUTPUTS:
            score       -   Score (`features` x `time`)
        '''
        
        # grab design matrix for y
        _, y_i = self.make_design(X, y)
        y_i = self.scaler_y.transform(y_i)
        
        # make predictions
        y_h = self.predict(X, reshape = False)
        
        # compute pearson correlation
        return pearsonr(y_h.permute(*torch.arange(y_h.ndim - 1, -1, -1)), y_i.permute(*torch.arange(y_i.ndim - 1, -1, -1)))

class Decoder(sklearn.base.BaseEstimator):
    '''
    This is a port of rsa.analysis.estimators.Decoder but using torch instead of numpy/sklearn.
    '''
    
    def __init__(self, estimator: sklearn.base.BaseEstimator = RidgeCV, alphas: Union[np.ndarray, list, int, torch.Tensor] = torch.Tensor([1]), scaler: Dict = dict(), fit_intercept: bool = True, **kwargs) -> None:
        '''
        Create new decoder.
        
        INPUTS:
            estimator       -   Estimator to use (default = `RidgeCV`)
            alphas          -   Alphas to use (default = [1])
            scaler          -   Options to pass to scaler (default = dict()). See `rsa.analysis.estimators.torch.StandardScaler` for options.
            fit_intercept   -   Fit intercept? (default = True)
            kwargs          -   kwargs to pass to estimator
        '''
        
        # setup options
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.kwargs = kwargs
        self.opts_scaler = scaler
        
        # setup estimators
        self.scaler = StandardScaler(**self.opts_scaler)
        self.estimator = estimator(alphas = self.alphas, fit_intercept = self.fit_intercept, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Fit decoder. Note that this will also estimate
        inverse patterns that are available as `decoder.pattern_`.
        
        For more information on patterns, please see:

            Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.D., Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 96-110. 10.1016/j.neuroimage.2013.10.067
        
        
        INPUTS:
            X   -   X data (`trials` x `channels`)
            y   -   y data (`trials` x `features`)
        '''
        
        # check data
        if y.ndim == 1:
            y = y[:, None]
        
        # fit input scaler
        self.scaler.fit(X)
        
        # fit decoder
        self.estimator.fit(self.scaler.transform(X), y)
        
        # obtain covariance of X
        X = self.scaler.transform(X)
        X = (X - X.mean(0, keepdim = True))
        S_X = torch.cov(X.T)
        
        # obtain covariance of y
        if y.shape[1] > 1:
            y = (y - y.mean(0, keepdim = True))
            P_y = torch.linalg.pinv(torch.cov(y.T))
        else:
            P_y = 1.0 / float(y.shape[0] - 1)
        
        # obtain inverse pattern
        if y.shape[1] > 1: self.pattern_ = S_X.mm(self.estimator.coef_.T).mm(P_y)
        else: self.pattern_ = S_X.mm(self.estimator.coef_.T) * P_y
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Obtain decoder predictions.
        
        INPUTS:
            X   -   X data (`trials` x `channels`)

        OUTPUTS:
            y   -   Predictions (`trials` x `features`)
        '''
        
        return self.estimator.predict(self.scaler.transform(X))

class Encoder(sklearn.base.BaseEstimator):
    '''
    This is a port of rsa.analysis.estimators.Encoder but using torch instead of numpy/sklearn.
    '''
    
    def __init__(self, estimator: sklearn.base.BaseEstimator = RidgeCV, alphas: Union[np.ndarray, list, int, torch.Tensor] = torch.Tensor([1]), scaler: Dict = dict(), fit_intercept: bool = True, **kwargs) -> None:
        '''
        Create encoder class.
        
        INPUTS:
            estimator       -   Estimator to use (default = `RidgeCV`)
            alphas          -   Alphas to use (default = [1])
            scaler          -   Options to pass to scalers (default = dict()). See `rsa.analysis.estimators.torch.StandardScaler` for options.
            fit_intercept   -   Fit intercept? (default = True)
            kwargs          -   kwargs to pass to estimator.
        '''
        
        # setup options
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.kwargs = kwargs
        self.opts_scaler = scaler
        
        # setup estimators
        self.scaler_X = StandardScaler(**self.opts_scaler)
        self.scaler_y = StandardScaler(**self.opts_scaler)
        self.estimator = estimator(alphas = self.alphas, fit_intercept = self.fit_intercept, **kwargs)
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        '''
        Fit encoder.
        
        INPUTS:
            X   -   X data (`trials` x `features`)
            y   -   y data (`trials` x `channels`)
        '''
        
        # check data
        if y.ndim == 1:
            y = y[:, None]
        
        # fit input scaler
        self.scaler_X.fit(X)
        self.scaler_y.fit(y)
        
        # fit decoder
        self.estimator.fit(self.scaler_X.transform(X), self.scaler_y.transform(y))
        
        # write data
        self.alpha_ = self.estimator.alpha_.clone()
        self.coef_ = self.estimator.coef_.clone()
    
    def predict(self, X: torch.Tensor, invert_y: bool = False) -> torch.Tensor:
        '''
        Make predictions.
        
        INPUTS:
            X               -   X data (`trials` x `features`)
            invert_y        -   Invert y to original scale? (default = False)

        OUTPUTS:
            y               -   Predictions (`trials` x `channels`)
        '''
        
        # make prediction
        y = self.estimator.predict(self.scaler_X.transform(X))
        
        # invert if required
        if invert_y: y = self.scaler_y.inverse_transform(y)
        
        return y

    def score(self, X: torch.Tensor, y: torch.Tensor, method: str = 'corrcoef') -> torch.Tensor:
        '''
        '''
        
        # obtain predictions
        y_h = self.predict(X, invert_y = True)
        
        # obtain score
        f = euclidean if method == 'mse' else pearsonr
        s = torch.nanmean(f(y.permute(*torch.arange(y.ndim - 1, -1, -1)), y_h.permute(*torch.arange(y_h.ndim - 1, -1, -1))))
        
        return s

class Temporal(sklearn.base.BaseEstimator):
    '''
    This is a port of rsa.analysis.estimators.Temporal but using torch instead of numpy/sklearn.
    '''
    
    def __init__(self, *args, estimator: sklearn.base.BaseEstimator = Decoder, opts: Dict = dict(), **kwargs) -> None:
        '''
        Creates a new temporal estimator.
        
        INPUTS:
            estimator       -   Estimator to use (default = `Decoder`)
            args            -   args to pass to estimator
            opts            -   Options to pass to estimator
            kwargs          -   kwargs to pass to estimator
        '''
        
        # save options
        self.estimator = estimator
        self.args = args
        self.kwargs = kwargs
        self.opts = opts

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        '''
        Fits the estimators.
        
        INPUTS:
            X   -   X data (`trials` x `channels` x `time`)
            y   -   y data (`trials` x `features` x `time`)
        '''
        
        # create estimator structure
        self.estimator_ = np.empty((X.shape[-1],), dtype = object)
        
        # loop over time
        for t_i in range(X.shape[-1]):
            # fit estimator
            self.estimator_[t_i] = self.estimator(*self.args, **self.opts, **self.kwargs)
            self.estimator_[t_i].fit(X[:,:,t_i], y[:,:,t_i])
    
    def collect(self, x: Union[str, tuple[str], list[str]]) -> Union[torch.Tensor, tuple[torch.Tensor]]:
        '''
        Collects an attribute from all estimators.
        
        INPUTS:
            x   -   Attribute(s) to collect
        
        OUTPUTS:
            y   -   Collected attribute (`*` x `time` or tuple of `*` x `time`s)
        '''
        
        # check attribute
        if type(x) == str:
            x = (x,)
        
        if type(x) == list:
            x = tuple(x)
        
        # check attributes exist
        for x_i in x: assert(hasattr(self.estimator_[0], x_i))
        
        # set output structure
        y0 = [getattr(self.estimator_[0], x_i) for x_i in x]
        y = [torch.zeros((self.estimator_.shape[0], *y0_i.shape)) for y0_i in y0]
        
        # fill in data
        for i, y0_i in enumerate(y0): y[i][0] = y0_i
        for t_i in range(1, self.estimator_.shape[0]):
            for i, x_i in enumerate(x): y[i][t_i] = getattr(self.estimator_[t_i], x_i)
        
        # reshape data
        for i, _ in enumerate(x): y[i] = y[i].moveaxis(0, -1)
        
        # finally, check output structure
        if len(x) == 1: return y[0]
        
        return y
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Make predictions in all estimators.
        
        INPUTS:
            X   -   X data (`trials` x `channels` x `time`)

        OUTPUTS:
            y   -   Predictions (`trials` x `features` x `time`)
        '''
        
        # check dims
        assert(X.shape[-1] == self.estimator_.shape[0])
        
        # generate output structure
        y_0 = self.estimator_[0].predict(X[:,:,0])
        y_h = torch.zeros((X.shape[-1], *y_0.shape))
        
        # fill in predictions
        y_h[0] = y_0
        for t_i in range(1, X.shape[-1]):
            y_h[t_i] = self.estimator_[t_i].predict(X[:,:,t_i])
        
        # make time axis last axis
        y_h = y_h.moveaxis(0, -1)
        
        return y_h
    
    def score(self, X: np.ndarray, y: np.ndarray, method: str = 'corrcoef') -> torch.Tensor:
        '''
        Score the estimators.
        
        INPUTS:
            X           -   X data (`trials` x `channels` x `time`)
            y           -   y data (`trials` x `features` x `time`)
            method      -   Method to use for scoring (default = `corrcoef`)

        OUTPUTS:
            score       -   Score (`features` x `time`)
        '''
        
        # obtain predictions
        y_h = self.predict(X)
        
        # obtain score
        f = euclidean if method == 'mse' else pearsonr
        s = torch.nanmean(f(y.swapaxes(0, 2), y_h.swapaxes(0, 2)))
        
        return s

class Generalisation(Temporal):
    '''
    This is a port of rsa.analysis.estimators.Generalisation but using torch instead of numpy/sklearn.
    '''
    
    def __init__(self, *args, **kwargs):
        '''
        Create a new temporal generalization estimator.
        
        For more information, see `Temporal`.
        '''
        
        super(Generalisation, self).__init__(*args, **kwargs)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Make predictions in all estimators.
        
        INPUTS:
            X   -   X data (`trials` x `channels` x `time`)
        
        OUTPUTS:
            y   -   Predictions (`trials` x `features` x `time (train)` x `time (test)`)
        '''
        
        # check dims
        assert(X.shape[-1] == self.estimator_.shape[0])
        
        # generate output structure
        y_0 = self.estimator_[0].predict(X[:,:,0])
        y_h = torch.zeros((X.shape[-1], X.shape[-1], *y_0.shape))
        
        # fill in predictions
        for t_i in range(0, X.shape[-1]):
            for t_j in range(0, X.shape[-1]):
                y_h[t_i,t_j] = self.estimator_[t_i].predict(X[:,:,t_j])
        
        # make time axis last axis
        y_h = y_h.moveaxis(0, -1)
        y_h = y_h.moveaxis(0, -1)
        
        return y_h
    
    def score(self, X: torch.Tensor, y: torch.Tensor, method: str = 'corrcoef') -> np.ndarray:
        '''
        Score the estimators.
        
        INPUTS:
            X           -   X data (`trials` x `channels` x `time`)
            y           -   y data (`trials` x `features` x `time`)
            method      -   Method to use for scoring (default = `corrcoef`)
        
        OUTPUTS:
            score       -   Score (`features` x `time (train)` x `time (test)`)
        '''
        
        # obtain predictions
        y_h = self.predict(X)
        
        # setup output data
        r = np.zeros(y_h.shape[1:])
        
        # loop over features
        for f_i in range(y_h.shape[1]):
            # score predictions
            r[f_i] = pearsonr(torch.moveaxis(y[:,f_i,None,:] * torch.ones((y_h.shape[0], *y_h.shape[2:])), 0, 2),
                              torch.moveaxis(y_h[:,f_i,:,:], 0, 2))
        
        return r

class SparseEncoder(Encoder):
    '''
    This is a port of rsa.analysis.estimators.SparseEncoder but using torch instead of numpy/sklearn.
    '''
    
    def __init__(self, *args, **kwargs):
        '''
        Creates new sparse encoder. See `Encoder` for options.
        '''
        
        super(SparseEncoder, self).__init__(*args, **kwargs)
    
    def make_design(self, X: torch.Tensor, y: Union[torch.Tensor, None] = None) -> Union[torch.Tensor, tuple[torch.Tensor]]:
        '''
        Creates a sparse design matrix that includes all
        predictors at all time points.
        
        INPUTS:
            X       -   Input data (`trials` x `features` x `timepoints`)
            y       -   Output data (`trials` x `channels` x `timepoints`)
        
        OUTPUTS:
            X_t     -   Design matrix (`trials * timepoints` x `features * timepoints`)
            y_t     -   Outcomes (`trials * timepoints` x `channels`)
        '''
        
        # setup design matrix & outcomes
        X_t = torch.zeros((X.shape[0] * X.shape[2], X.shape[1] * X.shape[2]), dtype = X.dtype, device = X.device)
        if y is not None: y_t = torch.zeros((y.shape[0] * y.shape[2], y.shape[1]), dtype = y.dtype, device = y.device)

        # loop over trials
        for i in range(X.shape[0]):
            # loop over time points
            for j in range(X.shape[2]):
                # grab index in matrix
                ij = i * X.shape[2] + j
                
                # write data
                X_t[ij,j*X.shape[1]:(j+1)*X.shape[1]] = X[i,:,j]
                if y is not None: y_t[ij,:] = y[i,:,j]
        
        if y is not None: return (X_t, y_t)
        return X_t
        
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        '''
        Fit the sparse encoder. ß-coefficients
        are available in  `self.coef_` (`channels` x 
        `features` x `timepoints`).
        
        INPUTS:
            X       -   Input data (`trials` x `features` x `timepoints`)
            y       -   Output data (`trials` x `channels` x `timepoints`)
        '''
        
        # check dimensionality
        if y.ndim == 2:
            y = y[:,None,:]
        
        # grab dimensionality
        self.n, self.f, self.t = X.shape
        self.d = y.shape[1]
        
        # grab sparse design matrix
        X, y = self.make_design(X, y)
        
        # fit model
        super(SparseEncoder, self).fit(X, y)
        
        # reshape coefficients
        self.coef_ = self.coef_.reshape((self.d, self.t, self.f)).swapaxes(1, 2)
    
    def predict(self, X: torch.Tensor, reshape: bool = True, invert_y: bool = False) -> torch.Tensor:
        '''
        Make encoder predictions.
        
        INPUTS:
            X           -   Input data (`trials` x `features` x `timepoints`)
            reshape     -   Whether to reshape the output to the original shape
            invert_y    -   Whether to invert the output to original scale

        OUTPUTS:
            y_h         -   Predicted output (`trials` x `channels` x `timepoints`)
        '''
        
        # grab trial number
        T = X.shape[0]
        
        # grab sparse design matrix
        X = self.make_design(X)
        
        # make predictions
        y_h = super(SparseEncoder, self).predict(X, invert_y = invert_y)
        
        # reshape
        if reshape: y_h = y_h.reshape((T, self.t, self.d)).swapaxes(1, 2)
        
        return y_h

    def score(self, X: torch.Tensor, y: torch.Tensor, method: str = 'corrcoef') -> torch.Tensor:
        '''
        Score the model.
        
        INPUTS:
            X           -   Input data (`trials` x `features` x `timepoints`)
            y           -   Output data (`trials` x `channels` x `timepoints`)
            method      -   Method to use for scoring

        OUTPUTS:
            r           -   Pearson correlation
        '''
        
        # grab prediction
        y_h = self.predict(X, invert_y = True)
        
        # evaluate prediction
        r = torch.nanmean(pearsonr(y_h.swapaxes(0, 2), y.swapaxes(0, 2)))
        
        return r

class B2B(sklearn.base.BaseEstimator):
    '''
    This is a port of rsa.analysis.estimators.B2B but using torch instead of numpy/sklearn.
    '''
    
    def __init__(self, alphas: Union[torch.Tensor, list, int] = torch.Tensor([1]), scaler: Dict = dict(), decoder: Dict = dict(), encoder: Dict = dict()) -> None:
        '''
        Creates new B2B estimator.
        
        INPUTS:
            alphas      -   Alpha values to grid search.
            scaler      -   Options to supply to input scaler (see sklearn.preprocessing.StandardScaler).
            decoder     -   Options to supply to decoder (`fit_intercept`, `scaler`).
            encoder     -   Options to supply to encoder (`fit_intercept`, `scaler`).
        '''
        
        # setup options
        self.alphas = alphas
        self.opts_scaler = scaler
        
        self.opts_decoder = dict(estimator = RidgeCV, alpha_per_target = True, fit_intercept = True, scaler = dict(with_mean = True, with_std = True))
        for k in decoder: self.opts_decoder[k] = decoder[k]
        
        self.opts_encoder = dict(estimator = RidgeCV, alpha_per_target = True, fit_intercept = True, scaler = dict(with_mean = False, with_std = False))
        for k in encoder: self.opts_encoder[k] = encoder[k]
        
        # setup scalers
        self.scaler_ins = StandardScaler(**self.opts_scaler)
        self.scaler_out = StandardScaler(**self.opts_scaler)
        self.scaler_dec = StandardScaler(**self.opts_decoder['scaler'])
        self.scaler_enc = StandardScaler(**self.opts_encoder['scaler'])
        
        # setup models
        self.decoder = self.opts_decoder['estimator'](alphas = self.alphas, fit_intercept = self.opts_decoder['fit_intercept'])
        self.encoder = self.opts_encoder['estimator'](alphas = self.alphas, fit_intercept = self.opts_encoder['fit_intercept'])
    
    def fit_decoder(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
        Fit decoders mapping from `X` to features `y`. Returns
        the patterns extracted from decoders, not decoder weights.
        For more information, please see

            Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.D., Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 96-110. 10.1016/j.neuroimage.2013.10.067
        
        Effectively, this creates the mapping `ß` in `y = ßX + e`, then
        computes `W = cov(X^T) . ß . cov(y^T)^-1`.
        
        INPUTS:
            X   -   Input data (trials x channels).
            y   -   Feature vectors (trials x features).
        
        OUTPUTS:
            W   -   Extracted pattern (channels x features).
        '''
        
        # check data
        if len(y.shape) == 1:
            y = y[:, None]
        
        # fit input scaler
        self.scaler_ins.fit(X)
        
        # fit output scaler
        self.scaler_out.fit(y)
        
        # fit decoder
        self.decoder.fit(self.scaler_ins.transform(X), self.scaler_out.transform(y))
        
        # fit decoder scaler
        self.scaler_dec.fit(self.decoder.predict(X))
        
        # obtain covariance of X
        X = self.scaler_ins.transform(X)
        X = (X - X.mean(0, keepdim = True))
        S_X = torch.cov(X.T)
        
        # obtain covariance of y
        if y.shape[1] > 1:
            y = self.scaler_out.transform(y)
            y = (y - y.mean(axis = 0))
            P_y = torch.linalg.pinv(torch.cov(y.T))
        else:
            P_y = 1.0 / float(y.shape[0] - 1)
        
        # obtain inverse pattern
        if y.shape[1] > 1: W = S_X.mm(self.decoder.coef_.T).mm(P_y)
        else: W = S_X.mm(self.decoder.coef_.T) * P_y
        
        return W
    
    def fit_encoder(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
        Fit encoders mapping from decoded features `X` to real features `y`
        and returns the "true" contribution of each feature on itself `S` 
        where `S = diag(ß)`.
        
        INPUTS:
            X   -   Input data (trials x features).
            y   -   Feature vectors (trials x features).
        
        OUTPUTS:
            S   -   Contribution of each feature.
        '''
        
        # check data
        if len(y.shape) == 1:
            y = y[:, None]
        
        # fit encoder
        self.encoder.fit(self.scaler_dec.transform(X), self.scaler_out.transform(y))
        
        # fit encoder scaler
        self.scaler_enc.fit(self.encoder.predict(self.scaler_dec.transform(X)))
        
        return self.encoder.coef_.diagonal()
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, splits: Union[tuple, None] = None) -> tuple[torch.Tensor]:
        '''
        Fit both decoders and encoders in one convenient step. For more information,
        see `B2B.fit_decoder` and `B2B.fit_encoder`, returning decoded patterns and
        "true" feature contributions.
        
        INPUTS:
            X       -   Input data (trials x channels).
            y       -   Feature vectors (trials x features).
            splits  -   Indices for the two splits required (default = None for first half v. second half).
        
        OUTPUTS:
            W       -   Extracted patterns (channels x features).
            S       -   Contribution of each feature.
        '''
        
        # check data
        if len(y.shape) == 1:
            y = y[:, None]
        
        # obtain splits of data
        if (splits is None) or (isinstance(splits, tuple) and len(splits) != 2):
            split_size = X.shape[0] // 2
            
            split_a = torch.arange(split_size)
            split_b = torch.arange(split_size, 2 * split_size, 1)
        else:
            split_a, split_b = splits
        
        # fit model
        self.W = self.fit_decoder(X[split_a,:], y[split_a,:])
        self.S = self.fit_encoder(self.decoder.predict(X[split_b,:]), y[split_b,:])
        
        return (self.W, self.S)
    
    def predict(self, X) -> tuple[torch.Tensor]:
        '''
        Make predictions for input `X`.
        
        INPUTS:
            X   -   Input data (trials x channels).
        
        OUTPUTS:
            y^  -   Predicted features (trials x features).  
        '''
        
        # obtain predictions
        G = self.scaler_dec.transform(self.decoder.predict(self.scaler_ins.transform(X)))
        H = self.scaler_enc.transform(self.encoder.predict(G))
        
        return (G, H)
    
    def score(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor]:
        '''
        Score model predictions.
        
        INPUTS:
            X           -   Input data (trials x channels).
            y           -   Feature vectors (trials x features).

        OUTPUTS:
            (r_G, r_H)  -   MSE in G (decoder) and H (encoder).
        '''
        
        # check data
        if len(y.shape) == 1:
            y = y[:, None]
        
        # obtain predictions
        G, H = self.predict(X)
        
        return ((G - y) ** 2).sum(0), ((H - y) ** 2).sum(0)

'''
Unit tests
'''

def _unit_tests_get_ch_pos() -> np.ndarray:
    '''
    Returns channel positions as defined in 
    a test subject of mine.
    '''
    
    ch_pos = np.array([[-0.0309026 ,  0.11458518,  0.02786657],
                       [ 0.02840949,  0.11534631,  0.02772126],
                       [-0.05180905,  0.0866879 ,  0.0787141 ],
                       [ 0.05027427,  0.08743838,  0.07727066],
                       [-0.06714873,  0.02335824,  0.10451068],
                       [ 0.06532887,  0.0235731 ,  0.10369243],
                       [-0.05503824, -0.0442103 ,  0.09990898],
                       [ 0.05363601, -0.04433453,  0.10051603],
                       [-0.03157356, -0.08056835,  0.05478965],
                       [ 0.02768309, -0.08048884,  0.05473408],
                       [-0.07187663,  0.07310353,  0.02579046],
                       [ 0.07143527,  0.07450512,  0.02510103],
                       [-0.08598208,  0.01487164,  0.03117337],
                       [ 0.08326136,  0.01525818,  0.03097297],
                       [-0.07445797, -0.04212316,  0.04127363],
                       [ 0.07103246, -0.04225998,  0.04119886],
                       [-0.00122928,  0.09327445,  0.1026393 ],
                       [-0.00137414,  0.02761709,  0.1401995 ],
                       [-0.00170945, -0.04521299,  0.12667292],
                       [-0.00206025, -0.08278299,  0.06073663],
                       [-0.03571587,  0.06171406,  0.11798301],
                       [ 0.03313097,  0.06182848,  0.1167817 ],
                       [-0.03742513, -0.01082424,  0.13344371],
                       [ 0.03647195, -0.01090379,  0.13281228],
                       [-0.07890598,  0.05136739,  0.06296235],
                       [ 0.07784661,  0.05209881,  0.06286711],
                       [-0.08151352, -0.01334569,  0.07313263],
                       [ 0.0814011 , -0.01346205,  0.07336367],
                       [-0.02904395,  0.09144848,  0.09661865],
                       [ 0.02796791,  0.09186997,  0.09577992],
                       [-0.03793783,  0.02633745,  0.12977061],
                       [ 0.03589271,  0.02635814,  0.12841234],
                       [-0.03065362, -0.04492738,  0.11947204],
                       [ 0.02988639, -0.04503254,  0.12074781],
                       [-0.03518602,  0.10912957,  0.05643921],
                       [ 0.03422985,  0.10981127,  0.05711668],
                       [-0.06185234,  0.05713329,  0.09376583],
                       [ 0.06062547,  0.05770742,  0.09379462],
                       [-0.06547224, -0.0118966 ,  0.10777792],
                       [ 0.06469624, -0.01199118,  0.10771287],
                       [-0.03862461, -0.06736158,  0.08241555],
                       [ 0.03466779, -0.06766213,  0.08164653],
                       [-0.06605409,  0.08023978,  0.05377108],
                       [ 0.06633133,  0.08152896,  0.05311628],
                       [-0.0820853 ,  0.01929363,  0.06948967],
                       [ 0.08165316,  0.01969565,  0.0694818 ],
                       [-0.06929985, -0.04322697,  0.0722538 ],
                       [ 0.06586061, -0.04333852,  0.07194131],
                       [-0.05636067,  0.09915152,  0.02514129],
                       [ 0.05422546,  0.09983162,  0.02491507],
                       [-0.08248932,  0.04484874,  0.02768059],
                       [ 0.08010432,  0.04555358,  0.02741239],
                       [-0.08675729, -0.01495097,  0.03515866],
                       [ 0.08362231, -0.01508568,  0.0350577 ],
                       [-0.05694863, -0.06592324,  0.04790744],
                       [ 0.05355731, -0.06641694,  0.04785098],
                       [-0.00133734,  0.11910179,  0.0328899 ],
                       [-0.00123739,  0.11373993,  0.07038366],
                       [-0.00152491, -0.01051838,  0.14154914],
                       [-0.00189983, -0.0680541 ,  0.09591   ]])
    
    return ch_pos

def unit_tests_StandardScaler(device: str = 'cpu') -> bool:
    '''
    Quickly test our scaler versus sklearn.
    '''
    
    import sklearn
    
    # setup data
    x_np = np.random.normal(size = (240, 50))
    x_tr = torch.from_numpy(x_np).to(torch.float64).to(device)
    
    # setup scalers
    y_np = sklearn.preprocessing.StandardScaler().fit(x_np).transform(x_np)
    y_tr = StandardScaler().fit(x_tr).transform(x_tr)
    
    # test
    assert(np.allclose(y_np, y_tr.cpu().numpy()))
    
    return True

def unit_tests_RidgeCV(device: str = 'cpu') -> bool:
    '''
    Quickly test the torch RidgeCV estimator vs sklearn.
    '''
    
    import sklearn
    
    # setup data
    X = np.random.normal(size = (240, 50))
    ß = np.random.normal(size = (50, 1))
    y = X @ ß
    y += np.random.normal(size = y.shape)

    # setup models
    sk = sklearn.linear_model.RidgeCV(alphas = np.logspace(-5, 10, 20)).fit(X, y)
    tr = RidgeCV(alphas = torch.logspace(-5, 10, 20)).fit(torch.from_numpy(X).to(torch.float64).to(device), torch.from_numpy(y).to(torch.float64).to(device))
    
    # grab predictions
    y_sk = sk.predict(X)
    y_tr = tr.predict(torch.from_numpy(X).to(torch.float64).to(device))
    
    # test
    assert(np.allclose(sk.coef_, tr.coef_.cpu().numpy()))
    assert(np.allclose(sk.intercept_, tr.intercept_.cpu().numpy()))
    assert(np.allclose(y_sk, y_tr.cpu().numpy()))
    
    return True

def unit_tests_TimeDelayed_trf(device: str = 'cpu') -> bool:
    '''
    Generates some data with a sine wave response function, then
    fits a model and compares its temporal response function with
    the original one. 
    
    Fails iff Pearsonr(trf, reconstructed) <= .95 or
              Pearsonr(y, reconstructed) <= .95.
    
    NOTE: Here, the target of .95 is "low" because we may experience
    some delay of the fitted TRFs, relative to the real ones.
    '''
    
    # setup dims
    T, F, L, C, W = 240, 10, 200, 60, 50

    # generate sparse design matrix
    X = np.random.choice([0, 1], size = (T, F, L)) * np.random.normal(size = (T, F, L))

    # generate impulse responses
    w = np.arange(W) * 5e-2
    r = np.array([[np.sin(2 * np.pi * w * np.random.uniform(low = 2.5, high = 10.0)) * np.sin(2 * np.pi * w * np.random.uniform(low = 0.5, high = 5.0)) ** 2 for _ in range(C)] for _ in range(F)])

    # convolve for neural data
    y = np.array([[[scipy.signal.convolve(X[i,j,:], r[j,k,:], mode = 'same') for i in range(T)] for j in range(F)] for k in range(C)]).sum(axis = 1).swapaxes(0, 1)

    # add some noise
    y += np.random.normal(size = y.shape)
    
    # move to desired device
    X, y = torch.from_numpy(X).to(torch.float64).to(device), torch.from_numpy(y).to(torch.float64).to(device)
    alphas = torch.logspace(-5, 10, 20).to(torch.float64).to(device)
    r = torch.from_numpy(r.swapaxes(0, 1)).to(torch.float64).to(device)

    # fit model
    trf = TimeDelayed(-0.125, 0.125, 200, alphas = alphas, scaler = dict(with_mean = False, with_std = False))
    trf.fit(X, y)
    
    # grab TRFs
    of = r.clone()
    rf = trf.coef_.clone()
    
    # compute goodness of fit measures
    r_t = pearsonr(of, rf[:,:,1:]).mean()
    r_y = trf.score(X, y).mean()
    
    # test
    assert(r_t > .95)
    assert(r_y > .95)
    
    return True

def unit_tests_TimeDelayed_rec(device: str = 'cpu') -> bool:
    '''
    Generates some continuous stimulus data, then
    computes some wacky temporal response functions
    and generates neural data from them.
    
    Finally, fits stimulus reconstruction model and 
    compares its predictions with real stimulus.
    
    Fails iff Pearsonr(y, y_h) <= .8.
    '''
    
    import warnings
    from .signal import smoothen, boxcar
    
    # setup dims
    T, F, L, C, W = 240, 10, 200, 60, 50

    # generate stimuli
    X = np.random.normal(size = (T, F, L))
    X = smoothen(X, boxcar(10), axis = 2)

    # generate TRFs
    w = np.arange(W) * 5e-2
    r = np.array([[np.sin(2 * np.pi * w * np.random.uniform(low = 2.5, high = 10.0)) * np.sin(2 * np.pi * w * np.random.uniform(low = 0.5, high = 5.0)) ** 2 for _ in range(C)] for _ in range(F)])

    # add some covariance
    r_cov = np.eye(C) + np.random.normal(loc = 0, scale = 0.1)
    r = (r.swapaxes(1, 2) @ r_cov).swapaxes(1, 2)

    # compute neural response
    y = np.array([[[scipy.signal.convolve(X[i,j,:], r[j,k,:], mode = 'same') for i in range(T)] for j in range(F)] for k in range(C)]).sum(axis = 1).swapaxes(0, 1)

    # add some noise
    y += np.random.normal(size = y.shape)
    
    # because we are going from neural -> stimulus, let's relabel
    X, y = y, X

    # move to desired device
    X, y = torch.from_numpy(X).to(torch.float64).to(device), torch.from_numpy(y).to(torch.float64).to(device)
    alphas = torch.logspace(-5, 10, 20).to(torch.float64).to(device)
    
    # fit model
    trf = TimeDelayed(-0.125, 0.125, 200, alphas = alphas)
    trf.fit(X, y)

    # evaluate
    r_y = trf.score(X, y).mean()
    
    # test
    assert(r_y > .8)
    
    return True

def unit_tests_Encoder(device: str = 'cpu') -> bool:
    '''
    Generates some predictor values along with some
    sensitivities per channel. Sensitivities are
    spatially smoothed and patterns are given a 
    correlation structure between features.
    
    Finally, encodes neural data from predictors.
    
    Fails iff spearmanr(y, y_h) < .99.
    '''
    
    from .analysis import compute_rdms
    
    # setup channel positions
    ch_pos = _unit_tests_get_ch_pos()
    
    # setup dimensions
    N, C, F = 1200, ch_pos.shape[0], 15
    temp = 0.25
    
    # generate stimuli
    X = np.random.normal(loc = 0, scale = 1.0, size = (N, F))
    
    # generate sensitivity by channel
    s = np.random.choice([-1, 1], size = (C, F)) * np.random.uniform(low = 0.25, high = 1.0, size = (C, F))
    
    # compute spatial distance matrix
    from .math import euclidean
    d = compute_rdms(np.array([ch_pos, ch_pos]).swapaxes(0, 1).swapaxes(1, 2), f = euclidean)[0,:,:]
    d = 1 - d / d.max()
    d = (np.exp(d / temp) / np.exp(d / temp).sum(axis = 1, keepdims = True))
    
    # compute smoothed sensitivities
    s = (s.T @ d[:,:]).T
    
    # aid pattern dissimilarity through covariance
    s = (s @ (np.eye(F) + np.random.normal(loc = 0, scale = 0.1, size = (F, F))))
    
    # obtain neural patterns
    y = (X[:,None,:] @ s.T[None,:,:]).squeeze()
    
    # add sensor noise
    y = y + np.random.normal(loc = 0, scale = 0.025, size = y.shape)
    
    # move to device
    X, y = torch.from_numpy(X).to(torch.float64).to(device), torch.from_numpy(y).to(torch.float64).to(device)
    alphas = torch.logspace(-5, 10, 20).to(torch.float64).to(device)
    
    # encode
    encoder = Encoder(alphas = alphas)
    encoder.fit(X, y)
    
    # predict
    y_h = encoder.predict(X)
    
    # test predictions
    r_y = spearmanr(y_h.T, y.T).mean()
    assert(r_y > .95)
    
    return True

def unit_tests_Temporal(device: str = 'cpu') -> bool:
    '''
    Generates some predictor values along with some
    sensitivities per channel. Sensitivities are
    spatially smoothed and patterns are given a 
    correlation structure between features.
    
    Finally, decodes predictors from neural data.
    
    Fails iff spearmanr(y, y_h) < .99 or
              spearmanr(sensitivies, patterns) < .80
    '''
    
    from .analysis import compute_rdms
    
    # setup channel positions
    ch_pos = _unit_tests_get_ch_pos()
    
    # setup dimensions
    N, C, F, T = 1200, ch_pos.shape[0], 15, 200
    temp = 0.25
    
    # generate stimuli
    X = np.random.normal(loc = 0, scale = 1.0, size = (N, F, T))
    
    # generate sensitivity by channel
    s = np.random.choice([-1, 1], size = (C, F, 1)) * np.random.uniform(low = 0.25, high = 1.0, size = (C, F, 1))
    
    # compute spatial distance matrix
    from .math import euclidean
    d = compute_rdms(np.array([ch_pos, ch_pos]).swapaxes(0, 1).swapaxes(1, 2), f = euclidean)[0,:,:]
    d = 1 - d / d.max()
    d = (np.exp(d / temp) / np.exp(d / temp).sum(axis = 1, keepdims = True))
    
    # compute smoothed sensitivities
    s = (s.T @ d).T
    
    # aid pattern dissimilarity through covariance
    s = (s.squeeze() @ (np.eye(F) + np.random.normal(loc = 0, scale = 0.1, size = (F, F))))[:,:,None]
    
    # obtain neural patterns
    y = np.array([(X[:,:,t_i] @ s) for t_i in range(T)]).squeeze().swapaxes(0, 2)
    
    # add sensor noise
    y = y + np.random.normal(loc = 0, scale = 0.025, size = y.shape)
    
    # since we are going from neural -> data, let's relabel
    X, y = y, X
    
    # move to device
    X, y = torch.from_numpy(X).to(torch.float64).to(device), torch.from_numpy(y).to(torch.float64).to(device)
    alphas = torch.logspace(-5, 10, 20).to(torch.float64).to(device)
    
    # fit temporal model
    decoder = Temporal(alphas = alphas)
    decoder.fit(X, y)

    # obtain patterns
    P = decoder.collect('pattern_')
    
    # test predictions and patterns
    r_y = decoder.score(X, y)
    r_p = pearsonr(torch.from_numpy((s * np.ones_like(P)).T).to(torch.float64).to(device), P.T)
    
    assert(r_y.mean() > .95)
    assert(r_p.mean() > .80)
    
    return True

def unit_tests_SparseEncoder(device: str = 'cpu') -> bool:
    '''
    Generates some impulse predictors and some
    neural data from temporal response functions.
    
    Finally, fits a sparse encoder model and measures
    how well it can reconstruct the neural data from
    predictors.
    
    Fails iff Spearmanr(y, y_h) <= .95.
    
    NOTE: Here the target of .95 is "low" because we have
    quite a bit of noise in the data.
    '''
    
    import warnings
    
    # setup dimensions
    N, C, T, F, L = 240, 60, 200, 5, 100
    
    # generate impulses
    X = np.zeros((N, F, T))
    X[:,:,100] = 1.0
    X = X * np.random.normal(loc = 0, scale = 1, size = (N, F, T))
    
    # generate temporal response function
    t = np.arange(L) * 1e-3
    trf = np.array([[np.sin(2 * np.pi * t * (1000 / L) * np.random.choice(np.arange(1, 5))) for i in range(C)] for _ in range(F)])
    
    # obtain raw neural response
    y = np.array([[[scipy.signal.convolve(X[i,f,:], trf[f,c,:], mode = 'same') for c in range(C)] for f in range(F)] for i in range(N)])
    y = y.sum(axis = 1)
    y_n = y + np.random.normal(loc = 0, scale = 1, size = y.shape)
    
    # reset X for model
    X = np.ones_like(X) * X[:,:,100,None]
    
    # move to device
    X, y_n = torch.from_numpy(X).to(torch.float64).to(device), torch.from_numpy(y_n).to(torch.float64).to(device)
    alphas = torch.logspace(-5, 10, 20).to(torch.float64).to(device)
    
    # fit model
    encoder = SparseEncoder(alphas = alphas, scaler = dict(with_mean = False, with_std = False))
    encoder.fit(X, y_n)
    
    # get predictions
    y_h = encoder.predict(X, invert_y = True)
    
    # compare
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        r = torch.nanmean(spearmanr(torch.from_numpy(y.swapaxes(0, 2)).to(torch.float64).to(device), y_h.swapaxes(0, 2)))
        assert(r > .95)
    
    return True

def unit_tests_Decoder(device: str = 'cpu') -> bool:
    '''
    Generates some predictor values along with some
    sensitivities per channel. Sensitivities are
    spatially smoothed and patterns are given a 
    correlation structure between features.
    
    Finally, decodes predictors from neural data.
    
    Fails iff spearmanr(y, y_h) < .99 or
              spearmanr(sensitivies, patterns) < .80
    '''
    
    from .analysis import compute_rdms
    
    # setup channel positions
    ch_pos = _unit_tests_get_ch_pos()
    
    # setup dimensions
    N, C, F = 1200, ch_pos.shape[0], 15
    temp = 0.25
    
    # generate stimuli
    X = np.random.normal(loc = 0, scale = 1.0, size = (N, F))
    
    # generate sensitivity by channel
    s = np.random.choice([-1, 1], size = (C, F)) * np.random.uniform(low = 0.25, high = 1.0, size = (C, F))
    
    # compute spatial distance matrix
    from .math import euclidean
    d = compute_rdms(np.array([ch_pos, ch_pos]).swapaxes(0, 1).swapaxes(1, 2), f = euclidean)[0,:,:]
    d = 1 - d / d.max()
    d = (np.exp(d / temp) / np.exp(d / temp).sum(axis = 1, keepdims = True))
    
    # compute smoothed sensitivities
    s = (s.T @ d[:,:]).T
    
    # aid pattern dissimilarity through covariance
    s = (s @ (np.eye(F) + np.random.normal(loc = 0, scale = 0.1, size = (F, F))))
    
    # obtain neural patterns
    y = (X[:,None,:] @ s.T[None,:,:]).squeeze()
    
    # add sensor noise
    y = y + np.random.normal(loc = 0, scale = 0.025, size = y.shape)
    
    # relabel
    X, y = y, X
    
    # move to device
    X, y = torch.from_numpy(X).to(torch.float64).to(device), torch.from_numpy(y).to(torch.float64).to(device)
    alphas = torch.logspace(-5, 10, 20).to(torch.float64).to(device)
    
    # decode
    decoder = Decoder(alphas = alphas)
    decoder.fit(X, y)
    
    # predict
    y_h = decoder.predict(X)
    
    # test predictions
    r_y = spearmanr(y_h.T, y.T)
    assert(r_y.mean() > .95)
    
    # test patterns
    r_p = spearmanr(torch.from_numpy(s.T).to(torch.float64).to(device), decoder.pattern_.T)
    assert(r_p.mean() > .80)
    
    return True

def unit_tests_B2B(n_permutations: int = 25, device: str = 'cpu') -> bool:
    '''
    Generates impulses for features as well as some temporal
    response functions. Makes sure that features covary.
    
    Finally, creates neural data and then decodes the two real
    features along with one correlated but non-causal feature.
    
    We then test the estimated causal contributions of each
    predictor against the ground truth (i.e., [0.5, 0.5, 0.0]).
    
    Fails iff criterion isn't met within +- 0.25. Here 0.25 is
    technically higher than we would need it, but given that
    I don't want to run hundreds of simulations for a unit
    test, I will keep it at that.
    '''
    
    import warnings
    from .signal import smoothen, boxcar
    
    # setup dims
    T, F, L, C, W = 240, 2, 200, 60, 50

    # generate stimuli
    X = np.random.normal(size = (T, F, L))
    X = smoothen(X, boxcar(10), axis = 2)

    # generate TRFs
    w = np.arange(W) * 5e-2
    r = np.array([[np.sin(2 * np.pi * w * np.random.uniform(low = 2.5, high = 10.0)) * np.sin(2 * np.pi * w * np.random.uniform(low = 0.5, high = 5.0)) ** 2 for _ in range(C)] for _ in range(F)])

    # add some covariance
    r_cov = np.eye(C) + np.random.normal(loc = 0, scale = 0.1)
    r = (r.swapaxes(1, 2) @ r_cov).swapaxes(1, 2)
    
    # compute neural response
    y = np.array([[[scipy.signal.convolve(X[i,j,:], r[j,k,:], mode = 'same') for i in range(T)] for j in range(F)] for k in range(C)]).sum(axis = 1).swapaxes(0, 1)

    # add some noise
    y += np.random.normal(size = y.shape)
    
    # add correlated but non-causal feature
    X2 = X[:,[0,1],:].mean(axis = 1, keepdims = True) + np.random.normal(size = (X.shape[0], 1, X.shape[2]))
    X = np.concatenate((X, X2), axis = 1)
    
    # because we are going from neural -> stimulus, let's relabel
    X, y = y, X
    
    # move to device
    X, y = torch.from_numpy(X).to(torch.float64).to(device), torch.from_numpy(y).to(torch.float64).to(device)
    alphas = torch.logspace(-5, 10, 20).to(torch.float64).to(device)
    
    # setup modelling process
    b2b = B2B
    temporal = Temporal(estimator = b2b, alphas = alphas)
    
    n_permutations = 25
    R = np.zeros((n_permutations, L, F + 1))
    
    # loop over permutations
    for p_i in range(n_permutations):
        # permute indices
        indc = torch.arange(T)
        indc = indc[torch.randperm(indc.size()[0])]
        
        # fit
        temporal.fit(X[indc,:,:], y[indc,:,:])
        
        # evaluate
        S = temporal.collect('S')
        R[p_i] = (S / S.sum(0).max()).T.cpu().numpy()
    
    # test that R2 is correct
    assert(np.all(np.isclose(R.mean(axis = (0, 1)), np.array([0.5, 0.5, 0]), atol = 0.25)))
    
    return True