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

def from_coef(parameters, spacing_x: float = 1, spacing_y: float = 0, offset_x: float = 0, offset_y: float = 0,
                          fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None, 
                          model: Callable = lm, link: Union[Callable, None] = None, CI: float = 1.96,
                          mean: Dict = dict(), CIs: Union[Dict, None, bool] = dict(), mean_dep: Union[Dict, bool] = dict(),
                          colours: Union[list, str, None] = None, labels: Union[list, None] = None) -> list[matplotlib.collections.PathCollection]:
    '''
    '''
    
    # get parameters
    lbp = [(parameters[i][0] - CI*parameters[i][1], 1) for i in np.arange(0, len(parameters), 1)]
    ubp = [(parameters[i][0] + CI*parameters[i][1], 1) for i in np.arange(0, len(parameters), 1)]
    mup = [(parameters[i][0], 1) for i in np.arange(0, len(parameters), 1)]
    
    # compute bounds and mu
    lb = np.array([model(np.array(lbp[i])[np.newaxis,:], link = link) for i in np.arange(0, len(lbp), 1)])
    ub = np.array([model(np.array(ubp[i])[np.newaxis,:], link = link) for i in np.arange(0, len(lbp), 1)])
    mu = np.array([model(np.array(mup[i])[np.newaxis,:], link = link) for i in np.arange(0, len(lbp), 1)])
    
    # get handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # setup mean styles
    style_mean = dict(marker = __pub.opts['means_marker'], s = float(__pub.opts['means_s']), alpha = float(__pub.opts['means_alpha']), zorder = 3)
    for k in mean: style_mean[k] = mean[k]
    
    # setup CI styles
    if type(CIs) == dict:
        style_CIs = dict(linestyle = __pub.opts['means_ci_linestyle'], linewidth = float(__pub.opts['means_ci_linewidth']), alpha = float(__pub.opts['means_ci_alpha']), zorder = 2)
        for k in CIs: style_CIs[k] = CIs[k]
    else: style_CIs = False
    
    # setup dependency styles
    if type(mean_dep) == dict:
        style_mean_dep = dict(linestyle = __pub.opts['means_dep_linestyle'], linewidth = float(__pub.opts['means_dep_linewidth']), color = __pub.opts['means_dep_color'], alpha = float(__pub.opts['means_dep_alpha']), zorder = 1)
        for k in mean_dep: style_mean_dep[k] = mean_dep[k]
    else: style_mean_dep = False
    
    # setup handles container
    h = []
    
    # loop over data
    last_X, last_Y = (0, 0)
    for i, (lbi, ubi, mui) in enumerate(zip(lb, ub, mu)):
        # get specific mean styles
        style_mean_i = style_mean
        if type(colours) == list and len(np.array(colours).shape) == 2: style_mean_i['color'] = colours[i]
        elif colours is not None: style_mean_i['color'] = colours
        if labels is not None and len(labels) > i: style_mean_i['label'] = labels[i]
        
        # scatter mean
        p = ax[0].scatter(i*spacing_x + offset_x, mui + i*spacing_y + offset_y, **style_mean_i)
        h.append(p)
        
        # add CI (if desired)
        if style_CIs:
            # get specific CI styles
            style_CIs_i = style_CIs
            if type(colours) == list and len(np.array(colours).shape) == 2: style_CIs_i['color'] = colours[i]
            elif colours is not None: style_mean_i['color'] = colours
            
            # plot CI
            ax[0].plot([i*spacing_x + offset_x, i*spacing_x + offset_x], [lbi + i*spacing_y + offset_y, ubi + i*spacing_y + offset_y], **style_CIs_i)
        
        # add dependency (if desired)
        if style_mean_dep and i > 0:
            ax[0].plot([last_X, i*spacing_x + offset_x], [last_Y, mui + i*spacing_y + offset_y], **style_mean_dep)
        
        last_X = i*spacing_x + offset_x
        last_Y = mui + i*spacing_y + offset_y
    
    return h