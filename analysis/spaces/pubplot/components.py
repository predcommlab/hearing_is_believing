import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .internals import *

from typing import Union, Callable, Dict

# grab module
__pub = sys.modules['pubplot']

def figure(**kwargs):
    '''
    '''
    
    # check rows, cols and figsize
    if 'nrows' not in kwargs: kwargs['nrows'] = int(1)
    if 'ncols' not in kwargs: kwargs['ncols'] = int(1)
    if 'figsize' not in kwargs: kwargs['figsize'] = (int(kwargs['ncols']) * float(__pub.opts['figure_col_width']),
                                                     int(kwargs['nrows']) * float(__pub.opts['figure_col_height']))
    
    # create figure and return handles
    fig, ax = plt.subplots(**kwargs)
    return (fig, ax)

def __get_figax(fig = None, ax = None):
    '''
    '''
    
    # check if figure/axes exist
    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        axes = [ax]
    
    return (fig, axes)