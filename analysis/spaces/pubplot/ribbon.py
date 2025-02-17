import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .internals import *
from .components import __get_figax
from .model import lm

from typing import Union, Callable, Dict

# grab module
__pub = sys.modules['pubplot']

def from_data(x: Union[list, np.ndarray], lb: Union[list, np.ndarray], ub: Union[list, np.ndarray], mu: Union[list, np.ndarray],
                fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None,
                ribbon: Union[Dict, None] = dict(), mean: Union[Dict, None, bool] = dict()) -> matplotlib.lines.Line2D:
    '''
    '''
    
    # setup styles
    style_ribbon = dict(alpha = float(__pub.opts['ribbon_ci_alpha']), edgecolor = __pub.opts['ribbon_ci_edgecolor'], lw = 0, zorder = 1)
    for k in ribbon: style_ribbon[k] = ribbon[k]
    if type(mean) == dict:
        style_mu = dict(alpha = float(__pub.opts['ribbon_mu_alpha']), linestyle = __pub.opts['ribbon_mu_linestyle'], zorder = 2)
        for k in mean: style_mu[k] = mean[k]
    else: style_mu = False
    
    # get handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # plot ribbon and mean (if desired)
    p = ax[0].fill_between(x, lb, ub, **style_ribbon)
    if style_mu: p, = ax[0].plot(x, mu, **style_mu)
    
    return p

def from_coef(x, parameters, fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None, 
                             model: Callable = lm, link: Union[Callable, None] = None, CI: float = 1.96,
                             ribbon: Union[Dict, None] = dict(), mean: Union[Dict, None, bool] = dict()) -> matplotlib.lines.Line2D:
    '''
    '''
    
    # get parameters
    lbp = [(parameters[i][0] - CI*parameters[i][1], parameters[i][2]) for i in np.arange(0, len(parameters), 1)]
    ubp = [(parameters[i][0] + CI*parameters[i][1], parameters[i][2]) for i in np.arange(0, len(parameters), 1)]
    mup = [(parameters[i][0], parameters[i][2]) for i in np.arange(0, len(parameters), 1)]
    
    # compute bounds and mu
    lb = model(lbp, link = link)
    ub = model(ubp, link = link)
    mu = model(mup, link = link)
    
    # hand off to ribbon function
    return from_data(x, lb, ub, mu, fig = fig, ax = ax, ribbon = ribbon, mean = mean)