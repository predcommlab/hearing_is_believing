import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .internals import *
from .components import __get_figax

from typing import Union, Callable, Dict

# grab module
__pub = sys.modules['pubplot']

def __sig_from_p(p: Union[str, float]) -> str:
    '''
    '''
    
    sig = p if type(p) is str else 'n.s.' if p > float(__pub.opts['significance_sigma1']) else '*' if p > float(__pub.opts['significance_sigma2']) else '**' if p > float(__pub.opts['significance_sigma3']) else '***'
    
    return sig

def single(p: Union[str, float], fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None,
                                 coords: tuple[float, float] = (0, 0), **kwargs):
    '''
    '''
    
    # get significance
    sig = __sig_from_p(p)
    
    # get handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # setup style
    style_significance = dict(xy = coords, xytext = coords, ha = __pub.opts['significance_ha'], va = __pub.opts['significance_va'])
    for k in kwargs: style_significance[k] = kwargs[k]
    
    # add significance
    ax[0].annotate(sig, **style_significance)

def pair(p: Union[str, float], fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None,
                               coords_a: tuple[float, float] = (0, 1), coords_b: tuple[float, float] = (1, 1), 
                               bar: Dict = dict(), significance: Dict = dict(), theta: int = 0,
                               sig_shift_x: float = 0., sig_shift_y: float = .0375, pro_shift_x: float = 0, pro_shift_y: float = -.0125):
    '''
    '''
    
    # get significance
    sig = __sig_from_p(p)
    
    # get handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # grab available space
    x_sum = np.sum(np.abs(ax[0].get_xlim()))
    y_sum = np.sum(np.abs(ax[0].get_ylim()))
    
    # compute shift
    coords_sig = np.array([x_sum * sig_shift_x, y_sum * sig_shift_y])
    coords_bar = np.array([[coords_a[0], coords_b[0]], [coords_a[1], coords_b[1]]])
    coords_lbar = np.array([[0, x_sum * pro_shift_x], [0, y_sum * pro_shift_y]])
    coords_rbar = np.array([[0, x_sum * pro_shift_x], [0, y_sum * pro_shift_y]])
    
    # rotate shift
    rad = np.radians(theta)
    rot_sig = coords_sig; rot_sig[0] = coords_sig[0] * np.cos(rad) - coords_sig[1] * np.sin(rad); rot_sig[1] = coords_sig[1] * np.cos(rad) + coords_sig[0] * np.sin(rad)
    rot_lbar = coords_lbar; rot_lbar[0,:] = coords_lbar[0,:] * np.cos(rad) - coords_lbar[1,:] * np.sin(rad); rot_lbar[1,:] = coords_lbar[1,:] * np.cos(rad) + coords_lbar[0,:] * np.sin(rad)
    rot_rbar = coords_rbar; rot_rbar[0,:] = coords_rbar[0,:] * np.cos(rad) - coords_rbar[1,:] * np.sin(rad); rot_rbar[1,:] = coords_rbar[1,:] * np.cos(rad) + coords_rbar[0,:] * np.sin(rad)
    
    # add bar positioning
    rot_sig[0] += coords_bar[0,0]; rot_sig[1] += coords_bar[1,0]; rot_sig[0] += (coords_b[0] - coords_a[0]) / 2; rot_sig[1] += (coords_b[1] - coords_a[1]) / 2
    if pro_shift_x < 0: rot_lbar[0,:] += coords_bar[0,:]; rot_lbar[1,:] += coords_bar[1,0]
    else: rot_lbar[0,:] += coords_bar[0,0]; rot_lbar[1,:] += coords_bar[1,:]
    if pro_shift_x < 0: rot_rbar[0,:] += coords_bar[0,:]; rot_rbar[1,:] += coords_bar[1,1]
    else: rot_rbar[0,:] += coords_bar[0,1]; rot_rbar[1,:] += coords_bar[1,:]
    
    # setup bar style
    style_bar = dict(linestyle = __pub.opts['significance_bar_linestyle'], linewidth = float(__pub.opts['significance_bar_linewidth']), color = __pub.opts['significance_bar_color'], alpha = float(__pub.opts['significance_bar_alpha']), zorder = 999, clip_on = False)
    for k in bar: style_bar[k] = bar[k]
    
    # setup sig style
    style_significance = dict(xy = (rot_sig[0], rot_sig[1]), xytext = (rot_sig[0], rot_sig[1]), ha = __pub.opts['significance_ha'], va = __pub.opts['significance_va'])
    for k in significance: style_significance[k] = significance[k]
    
    # plot top line
    ax[0].plot(coords_bar[0,:], coords_bar[1,:], **style_bar)
    ax[0].plot(rot_lbar[0,:], rot_lbar[1,:], **style_bar)
    ax[0].plot(rot_rbar[0,:], rot_rbar[1,:], **style_bar)
    ax[0].annotate(sig, **style_significance)

def grouped_pair(p: Union[str, float], fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None,
                                       coords_a: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (1, 1)), coords_b: tuple[tuple[float, float], tuple[float, float]] = ((2, 1), (3, 1)), 
                                       a_shift_x: float = 0, a_shift_y: float = 0, b_shift_x: float = 0, b_shift_y: float = 0,
                                       ca_shift_x: float = 0, ca_shift_y: float = 0, cb_shift_x: float = 0, cb_shift_y: float = 0,
                                       bar: Dict = dict(), significance: Dict = dict(), theta: int = 0,
                                       sig_shift_x: float = 0., sig_shift_y: float = .0375, pro_shift_x: float = 0, pro_shift_y: float = -.0125):
    '''
    '''
    
    # get handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # make group coords
    coords_bar1 = np.array([[coords_a[0][0], coords_a[1][0]], [coords_a[0][1], coords_a[1][1]]])
    coords_bar2 = np.array([[coords_b[0][0], coords_b[1][0]], [coords_b[0][1], coords_b[1][1]]])
    
    # grab available space
    x_sum = np.sum(np.abs(ax[0].get_xlim()))
    y_sum = np.sum(np.abs(ax[0].get_ylim()))
    
    # make comparison coords
    coords_comp_a = [coords_a[0][0] + (coords_a[1][0] - coords_a[0][0]) / 2, coords_a[0][1] + (coords_a[1][1] - coords_a[0][1]) / 2]
    coords_comp_b = [coords_b[0][0] + (coords_b[1][0] - coords_b[0][0]) / 2, coords_b[0][1] + (coords_b[1][1] - coords_b[0][1]) / 2]
    
    # grab maximum shift along bars
    max_x_bar1 = np.abs(coords_bar1[0,1] - coords_bar1[0,0]); max_y_bar1 = np.abs(coords_bar1[1,1] - coords_bar1[1,0])
    max_x_bar2 = np.abs(coords_bar2[0,1] - coords_bar2[0,0]); max_y_bar2 = np.abs(coords_bar2[1,1] - coords_bar2[1,0])
    
    # relative shift along bars
    coords_comp_a[0] += max_x_bar1 * a_shift_x; coords_comp_a[1] += max_y_bar1 * a_shift_y
    coords_comp_b[0] += max_x_bar2 * b_shift_x; coords_comp_b[1] += max_y_bar2 * b_shift_y
    
    # absolute shift of bars for 2nd set of points
    coords_comp_c = np.copy(coords_comp_a) + np.array([ca_shift_x, ca_shift_y])
    coords_comp_d = np.copy(coords_comp_b) + np.array([cb_shift_x, cb_shift_y])
    
    # extend l-/r-bars for 2nd set of points
    coords_lbar = np.array([[0, x_sum * -pro_shift_x], [0, y_sum * -pro_shift_y]])
    coords_rbar = np.array([[0, x_sum * -pro_shift_x], [0, y_sum * -pro_shift_y]])
    rad = np.radians(theta)
    rot_lbar = coords_lbar; rot_lbar[0,:] = coords_lbar[0,:] * np.cos(rad) - coords_lbar[1,:] * np.sin(rad); rot_lbar[1,:] = coords_lbar[1,:] * np.cos(rad) + coords_lbar[0,:] * np.sin(rad)
    rot_rbar = coords_rbar; rot_rbar[0,:] = coords_rbar[0,:] * np.cos(rad) - coords_rbar[1,:] * np.sin(rad); rot_rbar[1,:] = coords_rbar[1,:] * np.cos(rad) + coords_rbar[0,:] * np.sin(rad)
    
    # retrieve final set of 2nd points
    coords_comp_e = tuple(np.copy(coords_comp_c) + rot_lbar.sum(axis = 1))
    coords_comp_f = tuple(np.copy(coords_comp_d) + rot_rbar.sum(axis = 1))
    
    # setup bar style
    style_bar = dict(linestyle = __pub.opts['significance_bar_linestyle'], linewidth = float(__pub.opts['significance_bar_linewidth']), color = __pub.opts['significance_bar_color'], alpha = float(__pub.opts['significance_bar_alpha']), zorder = 999, clip_on = False)
    for k in bar: style_bar[k] = bar[k]
    
    # plot group bars
    ax[0].plot(coords_bar1[0,:], coords_bar1[1,:], **style_bar)
    ax[0].plot(coords_bar2[0,:], coords_bar2[1,:], **style_bar)
    
    # plot extended l-/r-bars
    ax[0].plot([coords_comp_a[0], coords_comp_c[0]], [coords_comp_a[1], coords_comp_c[1]], **style_bar)
    ax[0].plot([coords_comp_b[0], coords_comp_d[0]], [coords_comp_b[1], coords_comp_d[1]], **style_bar)
    
    # yield to pair
    pair(p, fig = fig, ax = ax[0], 
            coords_a = coords_comp_e, coords_b = coords_comp_f, 
            bar = bar, significance = significance, theta = theta, 
            sig_shift_x = sig_shift_x, sig_shift_y = sig_shift_y, 
            pro_shift_x = pro_shift_x, pro_shift_y = pro_shift_y)