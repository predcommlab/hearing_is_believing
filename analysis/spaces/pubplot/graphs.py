import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np

from .internals import *
from .components import __get_figax

from typing import Union, Any, Dict

# grab module
__pub = sys.modules['pubplot']

def directed(nodes: Dict[str, tuple[list[float], str]], edges: list[tuple[str, str, str]], 
             fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None,
             node: Union[Dict, None] = dict(), edge: Union[Dict, None] = dict(), direction: Union[Dict, None] = dict(), 
             autapse: Union[Dict, None] = dict(), node_label: Union[Dict, None] = dict(), edge_label: Union[Dict, None] = dict(),
             auta_shift: tuple[float, float] = (0., 0.07), edge_shift: float = 0.035):
    '''
    '''
    
    # get handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # cosmetic setup of nodes
    style_node = dict(marker = __pub.opts['graphs_directed_node_marker'],
                      s = float(__pub.opts['graphs_directed_node_s']),
                      color = __pub.opts['graphs_directed_node_color'],
                      edgecolor = __pub.opts['graphs_directed_node_edgecolor'],
                      linewidth = float(__pub.opts['graphs_directed_node_linewidth']),
                      alpha = float(__pub.opts['graphs_directed_node_alpha']))
    if type(node) == dict:
        for k in node: style_node[k] = node[k]
    
    # cosmetic setup of edges
    style_edge = dict(linewidth = float(__pub.opts['graphs_directed_edge_linewidth']),
                      linestyle = __pub.opts['graphs_directed_edge_linestyle'],
                      color = __pub.opts['graphs_directed_edge_color'],
                      alpha = float(__pub.opts['graphs_directed_edge_alpha']))
    if type(edge) == dict:
        for k in edge: style_edge[k] = edge[k]
    
    # cosmetic setup of directions
    style_direction = dict(arrowprops = dict(arrowstyle = '->', **style_edge),
                           fontsize = int(__pub.opts['graphs_directed_direction_fontsize']))
    if type(direction) == dict:
        for k in direction: style_direction[k] = direction[k]
    
    # cosmetic setp of autapses
    style_autapse = dict(**style_node); style_autapse['s'] = style_node['s']*(2/3); style_autapse['color'] = 'None'
    if type(autapse) == dict:
        for k in autapse: style_autapse[k] = autapse[k]
    
    # cosmetic setup of node labels
    style_node_label = dict(ha = __pub.opts['graphs_directed_node_label_ha'],
                            va = __pub.opts['graphs_directed_node_label_va'],
                            fontsize = int(__pub.opts['graphs_directed_node_label_fontsize']))
    if type(node_label) == dict:
        for k in node_label: style_node_label[k] = node_label[k]
    
    # cosmetic setup of edge labels
    style_edge_label = dict(**style_node_label); style_edge_label['fontsize'] = int(__pub.opts['graphs_directed_edge_label_fontsize'])
    if type(edge_label) == dict:
        for k in edge_label: style_edge_label[k] = edge_label[k]
    
    # generate nodes
    for node in nodes: 
        ax[0].scatter(*nodes[node][0], **style_node, zorder = 2)
        ax[0].annotate(nodes[node][1], xy = (nodes[node][0][0], nodes[node][0][1]),
                                       xytext = (nodes[node][0][0], nodes[node][0][1]),
                       **style_node_label, zorder = 5)

    # generate edges
    for edge in edges:
        if edge[0] == edge[1]:
            # autapse
            ax[0].scatter([nodes[edge[0]][0][0] + auta_shift[0]], 
                          [nodes[edge[0]][0][1] + auta_shift[1]], **style_autapse, zorder = 1)
            ax[0].annotate(edge[2], xy = (nodes[edge[0]][0][0] + auta_shift[0], nodes[edge[0]][0][1] + auta_shift[1]),
                                    xytext = (nodes[edge[0]][0][0] + auta_shift[0], nodes[edge[0]][0][1] + auta_shift[1]),
                           **style_edge_label, zorder = 2)
        else:
            # regular edge
            x0, y0 = nodes[edge[0]][0][0], nodes[edge[0]][0][1]
            x1, y1 = nodes[edge[1]][0][0], nodes[edge[1]][0][1]
            dx, dy = (x1 - x0), (y1 - y0)
            rx, ry = (0.6, 0.8), (0.6, 0.8)
            ax[0].plot([x0, x1], [y0, y1], **style_edge, zorder = 1)
            ax[0].annotate('', xy = (x0+rx[1]*dx, y0+ry[1]*dy), 
                               xytext = (x0+rx[0]*dx, y0+ry[0]*dy), 
                               ha = 'center', va = 'center', 
                           **style_direction, zorder = 5)
            
            theta = np.arctan2(y0+ry[1]*dy - y0+ry[0]*dy, x0+rx[1]*dx - x0+rx[0]*dx)
            cx = x0 + .5*(rx[1]+rx[0])*dx + edge_shift*np.sin(theta)
            cy = y0 + .5*(ry[1]+ry[0])*dy + edge_shift*np.cos(theta)
            ax[0].annotate(edge[2], xy = (cx, cy), 
                                    xytext = (cx, cy), 
                           **style_edge_label, zorder = 2)

def procedure(components: list[Dict[str, str]], fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None,
              offset_x: float = 0, offset_y: float = 0, spacing_x: float = 0.6, spacing_y: float = 1.8, dim_x: float = 2.0, dim_y: float = 2.4, 
              duration_offset_x: float = 1.15, duration_offset_y: float = 0.5,
              monitor: Dict[str, Any] = dict(), text: Dict[str, Any] = dict(), image: Dict[str, Any] = dict(),
              delay: Dict[str, Any] = dict(), show_delay: bool = True):
    '''
    '''
    
    # grab handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # prepare monitor style
    style_monitor = dict(facecolor = 'gray', edgecolor = 'black', linewidth = .5, clip_on = False)
    for k in monitor: style_monitor[k] = monitor[k]
    
    # prepare text style
    style_text = dict(fontsize = 6, color = 'black', ha = 'center', va = 'center', annotation_clip = False)
    for k in text: style_text[k] = text[k]
    
    # prepare image style
    style_image = dict(scale_max = 50, max_x = dim_y*0.75, offset_x = dim_x*0.2, max_y = dim_x*0.75, offset_y = dim_y*0.03)
    for k in image: style_image[k] = image[k]
    
    # prepare delay style
    style_delay = dict(arrowprops = dict(arrowstyle = '->', linewidth = 0.5), **style_text)
    for k in delay: style_delay[k] = delay[k]
    
    # setup internal function for monitor
    def __add_monitor(x: float, y: float, style: Dict[str, Any] = dict()):
        # setup current style
        style_monitor_i = style_monitor.copy()
        for k in style: style_monitor_i[k] = style[k]
        
        # draw monitor
        ax[0].add_patch(Rectangle((x, y), dim_y, dim_x, **style_monitor_i))
    
    # setup internal function for text
    def __add_text(txt: str, xy: tuple[float, float], style: Dict[str, Any] = dict()):
        # setup current style
        style_text_i = style_text.copy()
        for k in style: style_text_i[k] = style[k]
        
        # draw text
        ax[0].annotate(txt, xy = xy, xytext = xy, **style_text_i)
    
    # setup internal fucntion for image
    def __add_image(fin: str, x: float, y: float, style: Dict[str, Any] = dict()):
        # setup current style
        style_image_i = style_image.copy()
        for k in style: style_image_i[k] = style[k]
        
        # load image
        img = Image.open(fin)
        w, h = img.size
        
        # resize image
        F = [w / style_image_i['scale_max'], h / style_image_i['scale_max']]
        F_w, F_h = int(w / F[1]), int(h / F[0])
        img = img.resize((F_w, F_h))
        
        # as np, with rescaled colours
        imx = np.asarray(img) / 255
        
        # create position meshgrid
        xs = x + np.linspace(0, F_w / style_image_i['scale_max'], F_w) * style_image_i['max_x']
        xs += style_image_i['offset_x'] * (F_w / style_image_i['scale_max'] / (dim_x / dim_y))
        ys = y + np.linspace(F_h / style_image_i['scale_max'], 0, F_h) * style_image_i['max_y']
        ys += style_image_i['offset_y'] * (F_h / style_image_i['scale_max'])
        xx, yy = np.meshgrid(xs, ys)
        
        # draw image as scatter
        ax[0].scatter(xx, yy, marker = '.', s = 1, c = imx.reshape((F_w * F_h, imx.shape[2])), clip_on = False)
    
    # setup save states
    x_last = 0
    y_last = 0
    d_last = 0
    
    # loop over components
    for i, c_i in enumerate(components):
        # get current bottom coords
        x, y = i*spacing_x + offset_x, -spacing_y*(i+1) + offset_y
        
        # add defaults
        if 'T' not in c_i: c_i['T'] = 'text'
        if 'stimulus' not in c_i: c_i['T'] = ''
        if 'duration' not in c_i: c_i['duration'] = ''
        if 'delay' not in c_i: c_i['delay'] = ''
        if 'style' not in c_i: c_i['style'] = dict()
        
        # grab parameters
        T = c_i['T']
        stimulus = c_i['stimulus']
        duration = c_i['duration']
        delay = c_i['delay']
        style = c_i['style']
        
        # what component do we need?
        if T == 'text':
            __add_monitor(x, y)
            __add_text(stimulus, (x + dim_y / 2, y + dim_x / 2), style = style)
        elif T == 'image':
            __add_monitor(x, y)
            __add_image(stimulus, x, y, style = style)
        elif T == 'image-text':
            __add_monitor(x, y)
            __add_image(stimulus[0], x, y)
            __add_text(stimulus[1], (x + dim_y / 2, y + dim_x / 2), style = style)
        elif T == 'image-only':
            __add_image(stimulus, x, y)
        elif T == 'text-only':
            __add_text(stimulus, (x + dim_y / 2, y + dim_x / 2), style = style)
        elif T == 'image-text-only':
            __add_image(stimulus[0], x, y)
            __add_text(stimulus[1], (x + dim_y / 2, y + dim_x / 2), style = style)
        
        # add duration text
        style_duration = style_text.copy()
        style_duration['ha'] = 'left'
        ax[0].annotate(duration, xy = (x + dim_y * duration_offset_x, y + dim_x * duration_offset_y), **style_duration)
        
        # add time arrows + delay (if desired)
        if i > 0 and show_delay:
            ax[0].annotate('', xy = (x, y), xytext = (x_last, y_last), **style_delay)
            ax[0].annotate(d_last, xy = (x_last + (x - x_last) / 2, y_last + (y - y_last) / 2),
                                   xytext = (x_last + (x - x_last) / 2 - 0.3, y_last + (y - y_last) / 2),
                                   rotation = np.arctan2([y-y_last], [x-x_last])[0] * 180 / np.pi,
                                   **style_text)
        
        # store relevant variables for future
        x_last, y_last = x, y
        d_last = delay

def chord(nodes, edges, r0: float = 1., r1: float = 0.9, fig: Union[matplotlib.figure.Figure, None] = None, ax: Union[matplotlib.axes.Axes, None] = None,
                        xy: tuple[float, float] = (0, 0), sample_node: int = 50, sample_edge: int = 50, node = dict(), edge = dict(),
                        sin_scale: float = 0.125, disable_edges: bool = False):
    '''
    '''
    
    # grab handles
    fig, ax = __get_figax(fig = fig, ax = ax)
    
    # setup node style
    style_node = dict(alpha = .5)
    for k in node: style_node[k] = node[k]
    
    # setup edge style
    style_edge = dict(alpha = .5, linestyle = ':', linewidth = 1)
    for k in edge: style_edge[k] = edge[k]
    
    # add empty data structures for nodes
    for k in nodes:
        nodes[k]['E_l'] = []
        nodes[k]['E_w'] = []
        nodes[k]['E_a'] = dict()
        if 'colour' not in nodes[k]: nodes[k] = 'gray'
    
    # make edges available in nodes
    for k in edges:
        e_i = edges[k]
        
        if e_i['t'] not in nodes[e_i['f']]['E_l']:
            nodes[e_i['f']]['E_l'].append(e_i['t'])
            nodes[e_i['f']]['E_w'].append(e_i['w'])
        if e_i['f'] not in nodes[e_i['t']]['E_l']:
            nodes[e_i['t']]['E_l'].append(e_i['f'])
            nodes[e_i['t']]['E_w'].append(e_i['w'])
    
    # get mass of nodes
    M = np.sum([nodes[k]['N'] for k in nodes])
    
    # setup rolling theta
    rtheta = 0
    
    # grab offsets
    x, y = xy
    
    # loop over edges
    for k in nodes:
        # grab node
        n_i = nodes[k]
        M_i = n_i['N'] / M
        
        # anti-clockwise ordering of edges
        n_i_indx = np.where(np.array(list(nodes.keys())) == k)[0][0]
        n_ordr = np.hstack((np.arange(n_i_indx, len(nodes), 1), np.arange(0, n_i_indx, 1)))
        n_olab = np.array(list(nodes.keys()))[n_ordr]
        
        # collect newly ordered data (labels + weights)
        E_ol = []
        E_ow = []
        for e_k in n_olab:
            if e_k in n_i['E_l']:
                E_ol.append(e_k)
                E_ow.append(np.array(n_i['E_w'])[np.where(np.array(n_i['E_l']) == e_k)[0][0]])
        
        # delta, running theta and delta
        delta = 2 * np.pi * M_i
        rtheta += delta
        rdelta = 0
        
        # loop over ordered edges of node
        for j in np.arange(0, len(E_ol), 1):
            # setup specific style
            style_node_i = style_node.copy()
            style_node_i['color'] = n_i['colour']
            style_node_j = style_node_i.copy()
            style_node_j['alpha'] = 1.0
            if j == 0: style_node_i['label'] = k
            
            # setup mass of edge
            M_e_j = E_ow[j] / n_i['N']
            
            # setup current delta
            delta_j = -delta * M_e_j
            
            # generate part of outer ring
            x0 = x + r0 * np.cos(np.linspace(rtheta + rdelta, rtheta + rdelta + delta_j, sample_node))
            y0 = y + r0 * np.sin(np.linspace(rtheta + rdelta, rtheta + rdelta + delta_j, sample_node))
            
            # generate part of inner ring
            x1 = x + r1 * np.cos(np.linspace(rtheta + rdelta, rtheta + rdelta + delta_j, sample_node))
            y1 = y + r1 * np.sin(np.linspace(rtheta + rdelta, rtheta + rdelta + delta_j, sample_node))
            
            # generate corpus of ring
            rx = np.zeros((2*x0.shape[0],)); rx[0::2] = x0; rx[1::2] = x1
            ry = np.zeros((2*y0.shape[0],)); ry[0::2] = y0; ry[1::2] = y1

            # store inner coordinates (for connecting edges later)
            nodes[k]['E_a'][E_ol[j]] = (x1, y1)
            
            # plot node, outlines & separators
            ax[0].plot(rx, ry, **style_node_i, zorder = 2)
            ax[0].plot(x0, y0, **style_node_j, zorder = 3)
            ax[0].plot(x1, y1, **style_node_j, zorder = 3)
            if j == len(E_ol)-1: ax[0].plot(rx[::-1][0:2], ry[::-1][0:2], color = 'black', zorder = 4)
            
            # finally, roll delta
            rdelta += delta_j
    
    # flag for faster plots without edges (useful for trying to fit this in another plot)
    if not disable_edges:
        # loop over edges
        for k in edges:
            # grab nodes of edge
            n_f = nodes[edges[k]['f']]
            n_t = nodes[edges[k]['t']]
            
            # interpolate colour gradient
            cf = n_f['colour']
            ct = n_t['colour']
            cxy = np.array([np.linspace(cf_i, ct_i, sample_edge) for cf_i, ct_i in zip(cf, ct)]).T
            
            # retrieve inner coordinates
            x0, y0 = n_f['E_a'][edges[k]['t']]
            x1, y1 = n_t['E_a'][edges[k]['f']]
            
            # loop over coordinate points (inverting x1 and y1)
            for x0_i, x1_i, y0_i, y1_i in zip(x0, x1[::-1], y0, y1[::-1]):
                # generate raw points on edge
                dx = np.linspace(x0_i, x1_i, sample_edge) - x
                dy = np.linspace(y0_i, y1_i, sample_edge) - y
                
                # add scaled transform
                rx = dx + dx * np.sin(np.linspace(0, -np.pi, sample_edge)) * sin_scale + x
                ry = dy + dy * np.sin(np.linspace(0, -np.pi, sample_edge)) * sin_scale + y
                
                # loop over shifted pairs of transformed data points for individual colouring
                for j, (rx_i, ry_i, rx_j, ry_j) in enumerate(zip(rx[0:sample_edge-1], ry[0:sample_edge-1], rx[1:sample_edge], ry[1:sample_edge])):
                    # setup individual style + use colour gradient
                    style_edge_i = style_edge.copy()
                    style_edge_i['color'] = cxy[j,:]
                    
                    # draw edge points
                    ax[0].plot([rx_i, rx_j], [ry_i, ry_j], **style_edge_i)
    