import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .internals import *
from .components import __get_figax

from typing import Union, Callable, Dict

# grab module
__pub = sys.modules['pubplot']

def legend(fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None, **kwargs) -> matplotlib.legend.Legend:
    '''
    '''
    
    # get handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # setup style
    style_legend = dict(ncols = int(__pub.opts['legend_ncols']), 
                        handlelength = float(__pub.opts['legend_handlelength']), 
                        handletextpad = float(__pub.opts['legend_handletextpad']), 
                        frameon = bool(int(__pub.opts['legend_frameon'])), 
                        loc = __pub.opts['legend_loc'])
    for k in kwargs: style_legend[k] = kwargs[k]
    
    # add legend
    return ax[0].legend(**style_legend)

def figlabels(labels: list[str], fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None, 
                                 x_offset: float = -0.20625, y_offset: float = 0, **kwargs):
    '''
    '''
    
    # get handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # setup style
    style_labels = dict(annotation_clip = False, fontweight = __pub.opts['figlabels_fontweight'], fontsize = int(__pub.opts['figlabels_fontsize']), ha = __pub.opts['figlabels_ha'], va = __pub.opts['figlabels_va'])
    for k in kwargs: style_labels[k] = kwargs[k]
    
    # loop over axes
    for i, ax_i in enumerate(ax):
        if i >= len(labels): break
        
        # get limits
        x = np.min(ax_i.get_xlim()) + x_offset * np.sum(np.abs(ax_i.get_xlim()))
        y = np.max(ax_i.get_ylim()) + y_offset * np.sum(np.abs(ax_i.get_ylim()))
        
        # setup individual style
        style_labels_i = style_labels
        style_labels_i['xy'] = (x, y)
        style_labels_i['xytext'] = (x, y)
        
        # add label
        ax_i.annotate(labels[i], **style_labels_i)

def finish(despine: bool = True, tight_layout: bool = True):
    '''
    '''
    
    # despine (if desired)
    if despine: sns.despine()
    
    # tight layout (if desired)
    if tight_layout: plt.tight_layout()