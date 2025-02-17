import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

from .internals import *
from .components import __get_figax

from typing import Union, Callable, Dict

# grab module
__pub = sys.modules['pubplot']

def _bandwidth_botev(Y: np.ndarray) -> float:
    '''
    '''
    
    import scipy
    
    n = 2**14
    mx, mn = Y.max(), Y.min()
    r = mx - mn
    mx, mn = mx + r / 2, mn - r / 2
    
    R = mx - mn
    h, b = np.histogram(Y, bins = n, range = (mn, mx))
    h = h / Y.shape[0]
    
    d = scipy.fftpack.dct(h, norm = None)
    I = np.arange(1, n) ** 2
    d2 = (d[1:] / 2) ** 2
            
    def fp(t, M, I, a2):
        '''
        '''
        
        l = 7
        I = I.astype(np.longdouble)
        a2 = a2.astype(np.longdouble)
        f = 2 * np.pi ** (2 * l) * np.sum(I ** l * a2 * np.exp(-I * np.pi ** 2 * t))
                
        for s in range(l, 1, -1):
            K0 = np.prod(range(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
            const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
            time = (2 * const * K0 / M / f) ** (2 / (3 + 2 * s))
            f = 2 * np.pi ** (2 * s) * np.sum(I ** s * a2 * np.exp(-I * np.pi ** 2 * time))
                
        return t - (2 * M * np.lib.scimath.sqrt(scipy.pi) * f) ** (-2 / 5)
            
    initial = 0.1
    t_star = scipy.optimize.brentq(fp, 0, initial, args = (Y.shape[0], I, d2))
    
    bw = np.lib.scimath.sqrt(t_star) * R
    
    return bw

def violins(Y: np.ndarray, fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None,
                           kernel_type: str = 'gaussian', kernel_bandwidth: Union[float, str] = 'botev', jitter: bool = False, jitter_max: float = 0.5,
                           spacing_x: float = 1, spacing_y: float = 0, density_theta: int = 90, density_max: float = 0.45, 
                           offset_x: float = 0, offset_y: float = 0, violin: Dict = dict(), colours: Union[list, None] = None, 
                           scatter: Union[Dict, bool] = dict(), scatter_dep: Union[Dict, bool] = dict(),
                           mean: Union[Dict, bool] = dict(), mean_dep: Union[Dict, bool] = dict(),
                           labels: Union[list, None] = None,
                           CIs: Union[Dict, bool] = dict(), CI: float = 1.96, CI_type: str = 'se') -> list[matplotlib.lines.Line2D]:
    '''
    '''
    
    # get handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # make Y appropriate shape
    if len(Y.shape) == 1: Y = Y[np.newaxis,:]
    
    # get violin style
    style_violin = dict(alpha = float(__pub.opts['violin_alpha']), linewidth = float(__pub.opts['violin_linewidth']), zorder = 0)
    for k in violin: style_violin[k] = violin[k]
    
    # compute densities
    d_k = []
    d_x = []
    d_y = []
    
    for i in np.arange(0, Y.shape[0], 1):
        # set bandwidth
        bw = kernel_bandwidth
        
        # check if we need to precompute bandwidth
        if bw == 'botev': bw = _bandwidth_botev(Y[i,:])
        
        # compute kernel density
        kde_i = KernelDensity(kernel = kernel_type, bandwidth = bw).fit(Y[i,:,np.newaxis])
        
        # score samples
        x_i = np.linspace(Y[i,:].min(), Y[i,:].max(), 1000)
        y_i = np.exp(kde_i.score_samples(x_i[:,np.newaxis]))
        
        d_x.append(x_i)
        d_y.append(y_i)
        d_k.append(kde_i)
        
    d_x = np.array(d_x)
    d_y = np.array(d_y)
    
    # normalise densities
    d_y /= d_y.sum() * ((d_y.max() / d_y.sum()) * (1 / ((spacing_x + spacing_y) * density_max)))
    
    # setup handle container
    h = []
    
    # make violins
    for i in np.arange(0, Y.shape[0], 1):
        # setup specific style
        style_violin_i = style_violin
        if type(colours) == list and len(np.array(colours).shape) == 2: style_violin_i['color'] = colours[i]
        elif colours is not None: style_violin_i['color'] = colours
        
        # create violin corpus
        cx = np.zeros((2*d_x.shape[1],)); cx[0::2] = d_x[i,:]; cx[1::2] = d_x[i,:]
        cy = np.zeros((2*d_y.shape[1],)); cy[0::2] = d_y[i,:]; cy[1::2] = -d_y[i,:]

        # rotate corpus
        rad = np.radians(density_theta)
        rx = cx * np.cos(rad) - cy * np.sin(rad)
        ry = cy * np.cos(rad) + cx * np.sin(rad)
        
        # plot with offsets
        p, = ax[0].plot(rx + spacing_x*i + offset_x, ry + spacing_y*i + offset_y, **style_violin_i)
        h.append(p)
    
    # get scatter style
    if type(scatter) == dict:
        style_scatter = dict(marker = __pub.opts['violin_scatter_marker'], s = float(__pub.opts['violin_scatter_s']), alpha = float(__pub.opts['violin_scatter_alpha']), zorder = 2)
        for k in scatter: style_scatter[k] = scatter[k]
    else: style_scatter = False
    
    # get scatter dependency style
    if type(scatter_dep) == dict:
        style_scatter_dep = dict(linestyle = __pub.opts['violin_scatter_dep_linestyle'], linewidth = float(__pub.opts['violin_scatter_dep_linewidth']), color = __pub.opts['violin_scatter_dep_color'], alpha = float(__pub.opts['violin_scatter_dep_alpha']), zorder = 1)
        for k in scatter_dep: style_scatter_dep[k] = scatter_dep[k]
    else: style_scatter_dep = False
    
    # plot individual scatter points (if desired)
    if style_scatter:
        pos_x = np.zeros(Y.shape)
        pos_y = np.zeros(Y.shape)
        
        for i in np.arange(0, Y.shape[0], 1):
            # setup specific style
            style_scatter_i = style_scatter
            if type(colours) == list and len(np.array(colours).shape) == 2: style_scatter_i['color'] = colours[i]
            elif colours is not None: style_scatter_i['color'] = colours
            
            # get data
            cx = np.zeros((Y.shape[1],))
            cy = Y[i,:]
            
            # apply jitter
            if jitter: 
                d_j = np.exp(d_k[i].score_samples(Y[i,:,np.newaxis]))
                d_j /= d_y.sum() * ((d_y.max() / d_y.sum()) * (1 / ((spacing_x + spacing_y) * density_max)))
                cx += d_j * np.random.uniform(low = -jitter_max, high = jitter_max, size = (Y.shape[1],))
            
            # rotate data
            rad = np.radians(density_theta-90)
            rx = cx * np.cos(rad) - cy * np.sin(rad)
            ry = cy * np.cos(rad) + cx * np.sin(rad)
            
            # save data
            pos_x[i,:] = rx
            pos_y[i,:] = ry
            
            # scatter data
            ax[0].scatter(rx + (spacing_x*i + offset_x), ry + (spacing_y*i + offset_y), **style_scatter_i)
        
        # plot dependency slopes (if desired)
        if style_scatter_dep:
            for i in np.arange(1, Y.shape[0], 1):
                for j in np.arange(1, Y.shape[1], 1):
                    # obtain positions
                    rx = pos_x[i-1:i+1,j].copy()
                    ry = pos_y[i-1:i+1,j].copy()
                    
                    # fill in spacing
                    rx[0] += spacing_x*(i-1) + offset_x
                    rx[1] += spacing_x*i + offset_x
                    ry[0] += spacing_y*(i-1) + offset_y
                    ry[1] += spacing_y*i + offset_y
                    
                    # connect slopes
                    ax[0].plot(rx, ry, **style_scatter_dep)
    
    # get mean style
    if type(mean) == dict:
        style_mean = dict(marker = __pub.opts['violin_mean_marker'], s = float(__pub.opts['violin_mean_s']), zorder = 4)
        for k in mean: style_mean[k] = mean[k]
    else: style_mean = False
    
    # get CI style
    if type(CIs) == dict:
        style_CI = dict(linestyle = __pub.opts['violin_ci_linestyle'], linewidth = float(__pub.opts['violin_ci_linewidth']), alpha = float(__pub.opts['violin_ci_alpha']), zorder = 3)
        for k in CIs: style_CI[k] = CIs[k]
    else: style_CI = False
    
    # plot means & CIs (if desired)
    if style_mean:
        X_last, Y_last = (0, 0)
        
        for i in np.arange(0, Y.shape[0], 1):
            # setup specific mean styles
            style_mean_i = style_mean
            if type(colours) == list and len(np.array(colours).shape) == 2: style_mean_i['color'] = colours[i]
            elif colours is not None: style_mean_i['color'] = colours
            if labels is not None and len(labels) > i: style_mean_i['label'] = labels[i]
            
            # setup specific CI styles
            if style_CI:
                style_CI_i = style_CI
                if type(colours) == list and len(np.array(colours).shape) == 2: style_CI_i['color'] = colours[i]
                elif colours is not None: style_CI_i['color'] = colours
            
            # get data 
            Y_mu = Y[i,:].mean()
            if CI_type == 'std': Y_sd = Y[i,:].std() # standard deviation of distribution
            else: Y_sd = Y[i,np.random.choice(np.arange(Y.shape[1]), replace = True, size = (10000, Y.shape[1]))].mean(axis = 1).std() # standard error of mean
            Y_lb = Y_mu - CI * Y_sd
            Y_ub = Y_mu + CI * Y_sd
            
            # rotate mean data
            rad = np.radians(density_theta - 90)
            rx = np.zeros((1,)) * np.cos(rad) - Y_mu * np.sin(rad)
            ry = Y_mu * np.cos(rad) + np.zeros((1,)) * np.sin(rad)
            
            # scatter mean
            ax[0].scatter(rx + i*spacing_x + offset_x, ry + i*spacing_y + offset_y, **style_mean_i)
            
            # add dependency slope (if desired)
            if style_scatter_dep and i > 0:
                style_scatter_dep['alpha'] = 1.0
                style_scatter_dep['zorder'] = 2
                ax[0].plot([X_last, rx + i*spacing_x + offset_x], [Y_last, ry + i*spacing_y + offset_y], **style_scatter_dep)
            
            # short cut
            X_last, Y_last = (rx + i*spacing_x + offset_x, ry + i*spacing_y + offset_y)
            
            # rotate CI data (if desired)
            if style_CI:
                rx = np.zeros((2,)) * np.cos(rad) - np.array([Y_lb, Y_ub]) * np.sin(rad)
                ry = np.array([Y_lb, Y_ub]) * np.cos(rad) + np.zeros((2,)) * np.sin(rad)
                ax[0].plot(rx + i*spacing_x + offset_x, ry + i*spacing_y + offset_y, **style_CI_i)
    
    return h

def ridgeline(Y: np.ndarray, fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None,
                             kernel_type: str = 'gaussian', kernel_bandwidth: Union[float, str] = 'botev', jitter: bool = False, jitter_max: float = 0.5, 
                             spacing_x: float = 0, spacing_y: float = 1, density_theta: int = 0, flip_x: bool = False, flip_y: bool = False, density_max: float = 0.45, 
                             offset_x: float = 0, offset_y: float = 0, density: Dict = dict(), colours: Union[list, None] = None, fill: Union[dict, None] = dict(),
                             scatter: Union[Dict, bool] = dict(), scatter_dep: Union[Dict, bool] = dict(),
                             mean: Union[Dict, bool] = dict(), mean_dep: Union[Dict, bool] = dict(),
                             labels: Union[list, None] = None,
                             CIs: Union[Dict, bool] = dict(), CI: float = 1.96, CI_type: str = 'se') -> list[matplotlib.lines.Line2D]:
    '''
    '''
    
    # get handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # make Y appropriate shape
    if len(Y.shape) == 1: Y = Y[np.newaxis,:]
    
    # get basic style
    style_density = dict(linewidth = float(__pub.opts['distributions_linewidth']), alpha = float(__pub.opts['distributions_alpha']))
    for k in density: style_density[k] = density[k]
    
    # get mean style
    if type(mean) == dict:
        style_mean = dict(marker = __pub.opts['violin_mean_marker'], s = float(__pub.opts['violin_mean_s']))
        for k in mean: style_mean[k] = mean[k]
    else: style_mean = False
    
    # get CI style
    if type(CIs) == dict:
        style_CI = dict(linestyle = __pub.opts['violin_ci_linestyle'], linewidth = float(__pub.opts['violin_ci_linewidth']), alpha = float(__pub.opts['violin_ci_alpha']))
        for k in CIs: style_CI[k] = CIs[k]
    else: style_CI = False
    
    # get scatter style
    if type(scatter) == dict:
        style_scatter = dict(marker = __pub.opts['violin_scatter_marker'], s = float(__pub.opts['violin_scatter_s']), alpha = float(__pub.opts['violin_scatter_alpha']))
        for k in scatter: style_scatter[k] = scatter[k]
    else: style_scatter = False
    
    # get scatter dependency style
    if type(scatter_dep) == dict:
        style_scatter_dep = dict(linestyle = __pub.opts['violin_scatter_dep_linestyle'], linewidth = float(__pub.opts['violin_scatter_dep_linewidth']), color = __pub.opts['violin_scatter_dep_color'], alpha = float(__pub.opts['violin_scatter_dep_alpha']))
        for k in scatter_dep: style_scatter_dep[k] = scatter_dep[k]
    else: style_scatter_dep = False
    
    # compute densities
    d_x = []
    d_y = []
    d_k = []
    for i in np.arange(0, Y.shape[0], 1):
        # set bandwidth
        bw = kernel_bandwidth
        
        # check if we need to precompute bandwidth
        if bw == 'botev': bw = _bandwidth_botev(Y[i,:])
        
        # compute kde
        kde_i = KernelDensity(kernel = kernel_type, bandwidth = bw).fit(Y[i,:,np.newaxis])
        
        # score samples
        x_i = np.linspace(Y[i,:].min(), Y[i,:].max(), 1000)
        y_i = np.exp(kde_i.score_samples(x_i[:,np.newaxis]))
        
        d_x.append(x_i)
        d_y.append(y_i)
        d_k.append(kde_i)
        
    d_x = np.array(d_x)
    d_y = np.array(d_y)
    
    # normalise densities
    d_y /= d_y.sum() * ((d_y.max() / d_y.sum()) * (1 / ((spacing_x + spacing_y) * density_max)))
    
    # setup handle container
    h = []
    
    # setup position containers for scatter
    pos_x = np.zeros(Y.shape)
    pos_y = np.zeros(Y.shape)
    
    # setup positions for means
    X_last, Y_last = (0., 0.)
    
    # make distributions (in descending order of view)
    for i in np.arange(Y.shape[0] - 1, -1, -1):
        # fill outline(s) (if desired)
        if type(fill) == dict:
            # setup fill style
            style_fill_i = dict(linewidth = float(__pub.opts['distributions_fill_linewidth']), alpha = float(__pub.opts['distributions_fill_alpha']))
            if type(colours) == list and len(np.array(colours).shape) == 2: style_fill_i['color'] = colours[i]
            elif colours is not None: style_fill_i['color'] = colours
            for k in fill: style_fill_i[k] = fill[k]
            
            # create corpus
            cx = np.zeros((2*d_x.shape[1],)); cx[0::2] = d_x[i,:]; cx[1::2] = d_x[i,:]
            cy = np.zeros((2*d_y.shape[1],)); cy[0::2] = d_y[i,:]; cy[1::2] = 0

            # rotate corpus
            rad = np.radians(density_theta)
            rcx = cx * np.cos(rad) - cy * np.sin(rad)
            rcy = cy * np.cos(rad) + cx * np.sin(rad)
            
            # handle flips
            if flip_x: rcx = -rcx
            if flip_y: rcy = -rcy
            
            # do splits (if desired)
            ax[0].plot(rcx + spacing_x*i + offset_x, rcy + spacing_y*i + offset_y, **style_fill_i)
        
        # produce scatter (if desired)
        if style_scatter:
            # setup specific style
            style_scatter_i = style_scatter
            if type(colours) == list and len(np.array(colours).shape) == 2: style_scatter_i['color'] = colours[i]
            elif colours is not None: style_scatter_i['color'] = colours
            
            # get data
            cx = np.zeros((Y.shape[1],))
            cy = Y[i,:]
            
            # apply jitter
            if jitter: 
                d_j = np.exp(d_k[i].score_samples(Y[i,:,np.newaxis]))
                d_j /= d_y.sum() * ((d_y.max() / d_y.sum()) * (1 / ((spacing_x + spacing_y) * density_max)))
                cx += d_j * np.random.uniform(low = -jitter_max, high = - jitter_max / Y.shape[1], size = (Y.shape[1],))
            
            # rotate data
            rad = np.radians(density_theta-90)
            rx = cx * np.cos(rad) - cy * np.sin(rad)
            ry = cy * np.cos(rad) + cx * np.sin(rad)
                
            # save data
            pos_x[i,:] = rx
            pos_y[i,:] = ry
            
            # scatter data
            ax[0].scatter(rx + (spacing_x*i + offset_x), ry + (spacing_y*i + offset_y), **style_scatter_i)
            
            # plot dependency slopes (if desired)
            if style_scatter_dep:
                if i < (Y.shape[0] - 1):
                    for j in np.arange(Y.shape[1]):
                        # obtain positions
                        rx = pos_x[i:i+2,j].copy()
                        ry = pos_y[i:i+2,j].copy()
                        
                        # fill in spacing
                        rx[0] += spacing_x*(i-1) + offset_x
                        rx[1] += spacing_x*i + offset_x
                        ry[0] += spacing_y*(i-1) + offset_y + 1
                        ry[1] += spacing_y*i + offset_y + 1
                        
                        # connect slopes
                        ax[0].plot(rx, ry, **style_scatter_dep)
        
        # plot means & CIs (if desired)
        if style_mean:
            # setup specific mean styles
            style_mean_i = style_mean.copy()
            if type(colours) == list and len(np.array(colours).shape) == 2: style_mean_i['color'] = colours[i]
            elif colours is not None: style_mean_i['color'] = colours
            if labels is not None and len(labels) > i: style_mean_i['label'] = labels[i]
            
            # setup specific CI styles
            if style_CI:
                style_CI_i = style_CI.copy()
                if type(colours) == list and len(np.array(colours).shape) == 2: style_CI_i['color'] = colours[i]
                elif colours is not None: style_CI_i['color'] = colours
            
            # get data 
            Y_mu = Y[i,:].mean()
            if CI_type == 'std': Y_sd = Y[i,:].std() # standard deviation of distribution
            else: Y_sd = Y[i,np.random.choice(np.arange(Y.shape[1]), replace = True, size = (10000, Y.shape[1]))].mean(axis = 1).std() # standard error of mean
            Y_lb = Y_mu - CI * Y_sd
            Y_ub = Y_mu + CI * Y_sd
                
            # rotate mean data
            rad = np.radians(density_theta - 90)
            rx = np.zeros((1,)) * np.cos(rad) - Y_mu * np.sin(rad)
            ry = Y_mu * np.cos(rad) + np.zeros((1,)) * np.sin(rad)
            
            # scatter mean
            ax[0].scatter(rx + i*spacing_x + offset_x, ry + i*spacing_y + offset_y, **style_mean_i)
            
            # add dependency slope (if desired)
            if style_scatter_dep and (i < (Y.shape[0] - 1)):
                style_mean_dep_i = style_scatter_dep.copy()
                style_mean_dep_i['alpha'] = 1.0
                style_mean_dep_i['zorder'] = 0

                ax[0].plot([X_last, rx + i*spacing_x + offset_x], [Y_last, ry + i*spacing_y + offset_y], **style_mean_dep_i)

            # short cut
            X_last, Y_last = (rx + i*spacing_x + offset_x, ry + i*spacing_y + offset_y)
            
            # add CI data (if desired)
            if style_CI:
                rx = np.zeros((2,)) * np.cos(rad) - np.array([Y_lb, Y_ub]) * np.sin(rad)
                ry = np.array([Y_lb, Y_ub]) * np.cos(rad) + np.zeros((2,)) * np.sin(rad)
                ax[0].plot(rx + i*spacing_x + offset_x, ry + i*spacing_y + offset_y, **style_CI_i)
        
        # rotate density
        rad = np.radians(density_theta)
        rx = d_x[i,:] * np.cos(rad) - d_y[i,:] * np.sin(rad)
        ry = d_y[i,:] * np.cos(rad) + d_x[i,:] * np.sin(rad)
        
        # setup specific style
        style_density_i = style_density
        if type(colours) == list and len(np.array(colours).shape) == 2: style_density_i['color'] = colours[i]
        elif colours is not None: style_density_i['color'] = colours
        if type(labels) == list and len(labels) > i: style_density_i['label'] = labels[i]
        
        # handle flips
        if flip_x: rx = -rx
        if flip_y: ry = -ry
        
        # plot outline(s)
        p, = ax[0].plot(rx + spacing_x*i + offset_x, ry + spacing_y*i + offset_y, **style_density_i)
        h.append(p)
    
    return h

def icebergs(Y: np.ndarray, fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None,
                            kernel_type: str = 'gaussian', kernel_bandwidth: Union[float, str] = 'botev', 
                            spacing_x: float = 1, spacing_y: float = 0, density_theta: int = 90, flip_x: bool = False, flip_y: bool = False, density_max: float = 0.45, 
                            offset_x: float = 0, offset_y: float = 0, density: Dict = dict(), colours: Union[list, None] = None, fill: Union[dict, None] = dict(),
                            splits: Union[bool, None] = None, labels: Union[list, None] = None) -> list[matplotlib.lines.Line2D]:
    '''
    '''
    
    # get handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # make Y appropriate shape
    if len(Y.shape) == 1: Y = Y[np.newaxis,:]
    
    # get violin style
    style_density = dict(linewidth = float(__pub.opts['distributions_linewidth']), alpha = float(__pub.opts['distributions_alpha']), zorder = 1)
    for k in density: style_density[k] = density[k]
    
    # compute densities
    d_x = []
    d_y = []
    for i in np.arange(0, Y.shape[0], 1):
        # set bandwidth
        bw = kernel_bandwidth
        
        # check if we need to precompute bandwidth
        if bw == 'botev': bw = _bandwidth_botev(Y[i,:])
        
        # compute kde
        kde_i = KernelDensity(kernel = kernel_type, bandwidth = bw).fit(Y[i,:,np.newaxis])
        x_i = np.linspace(Y[i,:].min(), Y[i,:].max(), 1000)
        y_i = np.exp(kde_i.score_samples(x_i[:,np.newaxis]))
        d_x.append(x_i)
        d_y.append(y_i)
    d_x = np.array(d_x)
    d_y = np.array(d_y)
    
    # normalise densities
    d_y /= d_y.sum() * ((d_y.max() / d_y.sum()) * (1 / ((spacing_x + spacing_y) * density_max)))
    
    # setup handle container
    h = []
    
    # make distributions
    for i in np.arange(0, Y.shape[0], 1):
        # rotate density
        rad = np.radians(density_theta)
        rx = d_x[i,:] * np.cos(rad) - d_y[i,:] * np.sin(rad)
        ry = d_y[i,:] * np.cos(rad) + d_x[i,:] * np.sin(rad)
        
        # setup specific style and account for splits (if desired)
        style_density_i = style_density
        if type(splits) == dict:
            style_density_i_lb = style_density_i.copy()
            style_density_i_ub = style_density_i.copy()
            cv = splits['critvalue']
            if 'colours' in splits:
                if len(splits['colours']) == 1: 
                    if type(colours) == list and len(np.array(colours).shape) > 1: style_density_i_lb['color'] = colours[i]
                    elif colours is not None: style_density_i_lb['color'] = colours
                    style_density_i_ub['color'] = splits['colours'][0]
                elif len(splits['colours']) == 2: 
                    style_density_i_lb['color'] = splits['colours'][0] 
                    style_density_i_ub['color'] = splits['colours'][1]
            if not 'colours' in splits or len(splits['colours']) not in [1, 2]:
                if type(colours) == list and len(np.array(colours).shape) > 1: 
                    style_density_i_lb['color'] = colours[i]
                    style_density_i_ub['color'] = colours[i]
                elif colours is not None: 
                    style_density_i_lb['color'] = colours
                    style_density_i_ub['color'] = colours
            
            # add remaining styles
            for k in splits:
                if k in ['critvalue', 'colours']: continue
                if k[0:3] == 'lb_': style_density_i_lb[k[3:]] = splits[k]
                if k[0:3] == 'ub_': style_density_i_ub[k[3:]] = splits[k]
            
            # save colours used
            colour_lb = None if not 'color' in style_density_i_lb else style_density_i_lb['color']
            colour_ub = None if not 'color' in style_density_i_ub else style_density_i_ub['color']
        else:
            cv = None
            if type(colours) == list and len(np.array(colours).shape) == 2: style_density_i['color'] = colours[i]
            elif colours is not None: style_density_i['color'] = colours
            if type(labels) == list and len(labels) > i: style_density_i['label'] = labels[i]
        
        # handle flips
        if flip_x: rx = -rx
        if flip_y: ry = -ry
        
        # plot outline(s)
        if cv is not None:
            # plot cutoff outlines
            lb = d_x[i,:] < cv
            ub = d_x[i,:] >= cv
            
            plb, = ax[0].plot(rx[lb] + spacing_x*i + offset_x, ry[lb] + spacing_y*i + offset_y, **style_density_i_lb)
            pub, = ax[0].plot(rx[ub] + spacing_x*i + offset_x, ry[ub] + spacing_y*i + offset_y, **style_density_i_ub)
            h.append(plb)
            h.append(pub)
        else:
            # plot with offsets
            p, = ax[0].plot(rx + spacing_x*i + offset_x, ry + spacing_y*i + offset_y, **style_density_i)
            h.append(p)
        
        # fill outline(s) (if desired)
        if type(fill) == dict:
            # setup fill style
            style_fill_i = dict(linewidth = float(__pub.opts['distributions_fill_linewidth']), alpha = float(__pub.opts['distributions_fill_alpha']), zorder = 0)
            for k in fill: style_fill_i[k] = fill[k]
            
            # create violin corpus
            cx = np.zeros((2*d_x.shape[1],)); cx[0::2] = d_x[i,:]; cx[1::2] = d_x[i,:]
            cy = np.zeros((2*d_y.shape[1],)); cy[0::2] = d_y[i,:]; cy[1::2] = 0

            # rotate corpus
            rad = np.radians(density_theta)
            rcx = cx * np.cos(rad) - cy * np.sin(rad)
            rcy = cy * np.cos(rad) + cx * np.sin(rad)
            
            # handle flips
            if flip_x: rcx = -rcx
            if flip_y: rcy = -rcy
            
            # do splits (if desired)
            if cv is not None:
                style_fill_i_lb = style_fill_i.copy()
                style_fill_i_ub = style_fill_i.copy()
                if colour_lb is not None: style_fill_i_lb['color'] = colour_lb
                if colour_ub is not None: style_fill_i_lb['color'] = colour_ub
                
                # add split styles
                for k in splits:
                    if k in ['critvalue', 'colours']: continue
                    if k[0:3] == 'lb_': style_fill_i_lb[k[3:]] = splits[k]
                    if k[0:3] == 'ub_': style_fill_i_ub[k[3:]] = splits[k]
                
                # reset indices
                lb = cx < cv
                ub = cx >= cv
                
                # plot
                ax[0].plot(rcx[lb] + spacing_x*i + offset_x, rcy[lb] + spacing_y*i + offset_y, **style_fill_i_lb)
                ax[0].plot(rcx[ub] + spacing_x*i + offset_x, rcy[ub] + spacing_y*i + offset_y, **style_fill_i_ub)
            else:
                ax[0].plot(rcx + spacing_x*i + offset_x, rcy + spacing_y*i + offset_y, **style_fill_i)
                
    
    return h