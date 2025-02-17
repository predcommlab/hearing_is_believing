'''
embeddings::plot.py

General purpose plotting functions that may be useful for working with
the embeddings we are creating. Available as embedding.plot.*
'''

from .internal import *
from .func import compute_dissimilarity_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from matplotlib.patches import Polygon
import umap.umap_ as umap
from scipy.stats import gaussian_kde, ttest_rel, ttest_ind
from typing import Any, Optional, Union

def single_cluster(G: np.ndarray, K: sklearn.cluster._kmeans.KMeans, C: list[str], k: int = 0, s_max: int = 100, figsize: tuple[int, int] = (8, 8)):
    '''
    Creates a plot of the `k`-th cluster in `K`, plotting the labels from `C`
    over the UMAP projection of `G`. Plot size can be changed by supplying a
    tuple for `figsize`, number of words to be plotted can be changed through
    `s_max`.
    '''
    
    assert(type(G) is np.ndarray) or critical(ref = 'embeddings::plot_single_cluster()', msg = f'`G` must be of type np.ndarray.')
    assert(type(K) is sklearn.cluster._kmeans.KMeans) or critical('embeddings::plot_single_cluster()', msg = f'`K` must be of type sklearn.cluster._kmeans.KMeans.')
    assert(type(C) is list) or critical(ref = 'embeddings::plot_single_cluster()', msg = f'`C` must be of type list.')
    assert(k in np.unique(K.labels_)) or critical(ref = 'embeddings::plot_single_cluster()', msg = f'`k` must have at least one labelled element.')
    assert(s_max > 0) or critical(ref = 'embeddings::plot_single_cluster()', msg = f'`s_max` must be > 0.')
    
    # prepare data and fit UMAP
    C_k = C[k]
    G_k = np.vstack((G[np.where(K.labels_ == k)[0].astype(int)], K.cluster_centers_[k]))
    R_k = umap.UMAP()
    E_k = R_k.fit_transform(G_k)
    
    # start plotting
    fig, ax = plt.subplots(figsize = figsize)
    
    # mark centre
    ax.scatter(E_k[-1, 0], E_k[-1, 1], s = 30, marker = 'x')
    
    # select words (all if num < s_max, otherwise random sample of s_max)
    indcs = np.arange(0, E_k.shape[1] - 1, 1) if E_k.shape[0] - 1 <= s_max else np.random.choice(np.arange(0, E_k.shape[0] - 1, 1), size = (s_max,), replace = False)
    
    # annotate words
    for indx in indcs: ax.annotate(C_k[indx], E_k[indx,:], xytext = E_k[indx,:])
    
    # set axes for view of full cluster
    ax.set_xlim([E_k[:,0].min(), E_k[:,0].max()])
    ax.set_ylim([E_k[:,1].min(), E_k[:,1].max()])
    
    # disable axis (in UMAP-space it's arbitrary anyway)
    ax.axis('off')


def space_by_context(M: pd.core.frame.DataFrame, g: Any, include_distractors: bool = False, annotate: bool = False, cmap: str = 'viridis', figsize: tuple[int, int] = (8, 8), ax: Any = None):
    '''
    Create a plot of the UMAP projection of the semantic space in `g`, colouring in
    the items in the stimulus data frame `M` by their context. To also show the dis-
    tractors, supply `include_distractors`. Colour map may be altered through `cmap`
    and figure size may be altered through `figsize`.
    '''
    
    assert(type(M) is pd.core.frame.DataFrame) or critical(ref = 'embeddings::plot_space_by_context()', msg = f'`M` must be a pandas dataframe of stimuli.')
    assert(hasattr(g, 'key_to_index')) or critical(ref = 'embedding::plot_space_by_context()', msg = f'`g` must be word2vec glove model.')
    
    # collect items
    C = M.context.tolist()
    X = M.target.str.lower().tolist()
    
    if include_distractors:
        for d1 in M.d1.str.lower().tolist():
            C.append('D1')
            X.append(d1)
    
    # get embeddings and project
    E = np.array(g[X])
    R = umap.UMAP()
    E = R.fit_transform(E)
    
    # plot data points
    if ax is None: fig, ax = plt.subplots(figsize = figsize)
    
    cmap = plt.cm.get_cmap(cmap, np.unique(C).shape[0])
    cid = {np.unique(C)[i]:i for i in np.arange(0, np.unique(C).shape[0])}
    
    for C_i in np.unique(C):
        indcs = np.where(np.array(C) == C_i)[0]
        ax.scatter(E[indcs,0], E[indcs,1], marker = '.', color = cmap(cid[C_i]), label = C_i)
    
    if annotate is True:
        for i, x in enumerate(X):
            ax.annotate(x, (E[i,0], E[i,1]), xytext = (E[i,0], E[i,1]), alpha = 0.5)
    
    plt.legend()
    ax.axis('off')

def space_by_context_redesign(M: pd.core.frame.DataFrame, g: Any, annotate: bool = False, cmap: str = 'viridis', figsize: tuple[int, int] = (8, 8), ax: Any = None):
    '''
    Create a plot of the UMAP projection of the semantic space in `g`, colouring in
    the items in the stimulus data frame `M` by their context. Note that this should
    be used only for target1/target2*context1/context2 data structures (the redesigned
    version of the expeirment). Colour map may be altered through `cmap` and figure 
    size may be altered through `figsize`.
    '''
    
    assert(type(M) is pd.core.frame.DataFrame) or critical(ref = 'embeddings::plot_space_by_context()', msg = f'`M` must be a pandas dataframe of stimuli.')
    assert(hasattr(g, 'key_to_index')) or critical(ref = 'embedding::plot_space_by_context()', msg = f'`g` must be word2vec glove model.')
    
    # collect items
    C = [*M.context1.tolist(), *M.context2.tolist()]
    X = [*M.target1.str.lower().tolist(), *M.target2.str.lower().tolist()]
    
    # get embeddings and project
    E = np.array(g[X])
    R = umap.UMAP()
    E = R.fit_transform(E)
    
    # plot data points
    if ax is None: fig, ax = plt.subplots(figsize = figsize)
    
    cmap = plt.cm.get_cmap(cmap, np.unique(C).shape[0])
    cid = {np.unique(C)[i]:i for i in np.arange(0, np.unique(C).shape[0])}
    
    for C_i in np.unique(C):
        indcs = np.where(np.array(C) == C_i)[0]
        ax.scatter(E[indcs,0], E[indcs,1], marker = '.', color = cmap(cid[C_i]), label = C_i)
    
    if annotate is True:
        for i, x in enumerate(X):
            ax.annotate(x, (E[i,0], E[i,1]), xytext = (E[i,0], E[i,1]), alpha = 0.5)
    
    plt.legend()
    ax.axis('off')

def rainclouds(M: np.ndarray, L: list[str] = None, ax: Any = None, xlab: str = None, ylab: str = None, 
                              title: str = None, alpha_dot: float = 0.4, figsize: tuple[int, int] = (8, 8), 
                              cov_factor: float = 0.5, tests: list[list[int]] = None, is_ind: bool = True, 
                              adjust: str = 'Bonferroni-Holm', cmap: str = 'viridis', bar_sig: bool = False) -> Optional[tuple[np.ndarray, np.ndarray]]:
    '''
    Creates a raincloud visualisation for the matrix `M` that is conditionXvalues. Condition
    labels can be supplied via `L` as type list. `xlab`, `ylab` and `title` are available
    as optional string arguments. `alpha_dot` sets the scatter visibility, `figsize` sets
    the figure size, IFF `ax` was not supplied, `cov_factor` is the covarianve factor to use
    for estimating the density, `tests` is an optional test structure that contains a list of
    lists where each list should have two entries that indicate the index in `M` of the groups
    to be compared using an independent or dependent samples t-test (as set through `is_ind`).
    Adjustments for multiple comparisons `adjust` can be 'Bonferroni-Holm', 'Bonferroni' or 
    None. Finally, colour is given through `cmap`.
    
    If `tests` was specified, this will return a tuple of (`t-values`, `p-values`).
    '''
    
    assert(type(M) is np.ndarray) or critical(ref = 'embeddings::plot_rainclouds()', msg = f'`M` must be of type np.ndarray.')
    assert(len(M.shape) == 2) or critical(ref = 'embeddings::plot_rainclouds()', msg = f'`M` must have two dimensions.')
    if L is not None: assert(len(L) == M.shape[0]) or critical(ref = 'embeddings::plot_rainclouds()', msg = f'`L` must be of equal size to M.shape[0].')
    assert(np.all([(type(lab) in [str, None]) or (lab == None) for lab in [xlab, ylab, title]])) or critical(ref = 'embeddings::plot_rainclouds()', msg = f'All labels must be of either type str or None.')
    
    Ys = M.shape[0]
    cmap = plt.cm.get_cmap(cmap, M.shape[0]+2)
    
    if ax is None: fig, ax = plt.subplots(figsize = figsize)
    if tests is None: tests = []
    
    # pre-compute densities
    v = np.zeros((M.shape[0], M.shape[1] + 2, 2))
    for i in np.arange(0, Ys, 1):
        D = gaussian_kde(M[i,:].flatten())
        D.covariance_factor = lambda: cov_factor
        D._compute_covariance()
        
        XD = np.linspace(M[i,:].min(), M[i,:].max(), M.shape[1])
        v[i,:] = [(M[i,:].min(), D(XD).min()), *zip(XD, D(XD)), (M[i,:].max(), D(XD).min())]
    
    # normalise densities, plot
    for i in np.arange(0, Ys, 1):
        L_i = None if L is None else L[i]
        
        v[i,:,1] /= v[:,:,1].max()
        v[i,:,1] = v[i,:,1] * 0.3 + i
        p = Polygon(v[i,:], facecolor = cmap(i), edgecolor = cmap(i), alpha = alpha_dot)
        
        ax.add_patch(p)
        ax.scatter(M[i,:], np.random.uniform(i - 0.5, i - 0.05, size = (M.shape[1],)), marker = '.', s = 2, alpha = alpha_dot, color = cmap(i))
        ax.scatter(M[i,:].mean(), i - 0.275, marker = 'o', s = 35, color = cmap(i), alpha = 1.0)
        ax.hlines(i - 0.275, xmin = M[i,:].mean() - M[i,:].std(), xmax = M[i,:].mean() + M[i,:].std(), linestyle = '-', linewidth = 1.0, color = cmap(i), alpha = 1.0)
    
    # statistics
    p = np.ones((len(tests,)))
    t = np.zeros((len(tests,)))
    
    for i, test in enumerate(tests):
        f = ttest_ind if is_ind else ttest_rel
        t[i], p[i] = f(M[test[0],:], M[test[1],:])
    
    px = np.copy(p)
    for i, _ in enumerate(px):
        if adjust == 'Bonferroni-Holm': p[i] *= 1 if len(np.where(px > px[i])) < 1 else np.where(px > px[i])[0].shape[0] + 1
        elif adjust == 'Bonferroni': p[i] *= len(p)
    
    # show stats
    if bar_sig is False:
        for i, p_i in enumerate(p):
            xmin, xmax = (M[tests[i][0],:].mean(), M[tests[i][1],:].mean()) if M[tests[i][0],:].mean() < M[tests[i][1],:].mean() else (M[tests[i][1],:].mean(), M[tests[i][0],:].mean())
            y1, y2 = (tests[i][0]-0.275, tests[i][1]-0.275) if M[tests[i][0],:].mean() < M[tests[i][1],:].mean() else (tests[i][1]-0.275, tests[i][0]-0.275)
            s = 1 if p[i] <= 1e-2 else 2 if p[i] <= 1e-1 else 3
            ax.plot([xmin, xmax], [y1, y2], linewidth = 1, color = 'black', alpha = 1 / s, label = '***' if s == 1 else '**' if s == 2 else '*')

        ax.legend()
    else:
        # plot stats
        ip = 0
        d = np.argsort(np.abs(np.array(tests)[:,0] - np.array(tests)[:,1]).squeeze())
        for n in np.arange(d.shape[0]-1, -1, -1):
            i = d[n]
            test = tests[i]
            if p[i] > .05: continue
            s = 1 if p[i] <= 1e-2 else 2 if p[i] <= 1e-1 else 3
            v = 1
            u = (M.max() - M.min()) / 100
            u2 = (Ys-1+0.6+1.1)/100
            ymin, ymax = (test[0], test[1]) if test[0] < test[1] else (test[1], test[0])
            ymin, ymax = (ymin-u2, ymax+u2)
            xmin, xmax = (M.min() + u/2 + v*u*ip, M.min() + u/2 + v*u*ip + v*u)
            ax.fill_between([xmin+u/5, xmax-u/5], [ymin-0.275, ymin-0.275], [ymax-0.275, ymax-0.275], facecolor = 'black', alpha = 0.5/s)
            ip += 1

        # stats legend
        if len(tests) > 0:
            u = (M.max() - M.min()) / 100
            u2 = (Ys-1+0.6+1.1)/100
            um = Ys-1+0.6

            ax.fill_between([M.min()+3*u,M.min()+8*u], [um-1*u2-1*u2,um-1*u2-1*u2], [um-1*u2,um-1*u2], facecolor = 'black', alpha = 0.5/1)
            ax.annotate('***', (M.min()+3.75*u,um-5*u2), xytext = (M.min()+3.75*u,um-5*u2))

            ax.fill_between([M.min()+11*u,M.min()+16*u], [um-1*u2-1*u2,um-1*u2-1*u2], [um-1*u2,um-1*u2], facecolor = 'black', alpha = 0.5/2)
            ax.annotate('**', (M.min()+12.5*u,um-5*u2), xytext = (M.min()+12.5*u,um-5*u2))

            ax.fill_between([M.min()+19*u,M.min()+24*u], [um-1*u2-1*u2,um-1*u2-1*u2], [um-1*u2,um-1*u2], facecolor = 'black', alpha = 0.5/3)
            ax.annotate('*', (M.min()+21.0*u,um-5*u2), xytext = (M.min()+21.0*u,um-5*u2))
    
    # cosmetic bits
    ax.set_ylim([-1.1, (Ys-1)+0.6])
    ax.set_xlim([M.min(), M.max()])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yticks(np.arange(-0.275, Ys-0.275, 1))
    ax.set_yticklabels(L, rotation = 90, va = 'center')
    
    if len(tests) > 0:
        return (t, p)

def dissimilarity_matrix(X: np.ndarray, C: Union[list[int], np.ndarray] = None, ax: Any = None, cmap: str = 'RdBu', 
                                        figsize: tuple[int, int] = (16, 16), vbounds: str = 'val', method: str = 'cd', 
                                        methodlab: str = r'cosine dissimilarity') -> np.ndarray:
    '''
    Create and plot a dissimilarity matrix from itemsXfeatures matrix `M`. Category
    identities can be supplied in `C`. Dissimilarity is computed as (1 - CS(x, y)).
    Optionally, supply name of colour map `cmap`, figure size tuple `figsize` unless `ax`
    was supplied, method of setting colour bounds `vbounds` (either val for [min, max] 
    or bnd for [0, 2] or sym for symmetrical val). Note that the method may also be 
    altered by supplying `method`, so long as it is compatible with parameter `f` from 
    embeddings::compute_dissimilarity_matrix(). Note that if `method` is not one of the
    defaults of parameter `f`, `methodlab` should be supplied to label the colourbar.
    
    Returns dissimilartiy matrix `Z`.
    '''
    
    assert(type(X) is np.ndarray) or critical(ref = 'embeddings::plot_dissimilarity_matrix()', msg = f'`X` must be of type np.ndarray.')
    assert(len(X.shape) == 2) or critical(ref = 'embeddings::plot_dissimilarity_matrix()', msg = f'`X` must have two dimensions (itemsXfeatures).')
    assert((C is None) or (type(C) in [list, np.ndarray])) or critical(ref = 'embeddings::plot_dissimilarity_matrix()', msg = f'`C` must be of type None, list or np.ndarray.')
    assert((C is None) or (len(C) == X.shape[0])) or critical(ref = 'embeddings::plot_dissimilarity_matrix()', msg = f'`C` must equal dimensions zero of `X` in length.')
    assert(vbounds in ['val', 'bnd', 'sym']) or critical(ref = 'embeddings::plot_dissimilarity_matrix()', msg = f'`vbounds` must be either `val`, `bnd` or `sym`.')
    
    # sort by category, if supplied
    C = C if C is not None else np.arange(0, X.shape[0], 1)
    indcs = np.argsort(np.array(C))
    C, X = np.array(C)[indcs], X[indcs,:]
    
    # compute cosine dissimilarity
    Z = compute_dissimilarity_matrix(X, f = method)
    
    # plot matrix
    if ax is None: fig, ax = plt.subplots(figsize = figsize)
    vmin, vmax = (Z.min(), Z.max()) if vbounds == 'val' else (0, 2) if vbounds == 'bnd' else (-max(abs(Z.min()), abs(Z.max())), max(abs(Z.min()), abs(Z.max())))
    mat = ax.imshow(Z, cmap = 'RdBu', vmin = vmin, vmax = vmax)
    cbar = plt.colorbar(mat, ax = ax, shrink = 0.825)
    cbar.set_label(r'cosine dissimilarity' if method == 'cd' else r'cosine similarity' if method == 'cs' else r'euclidean distance' if method == 'ed' else r'$z$-score(euclidean distance)' if method == 'zed' else methodlab)
    
    # extract tick labels and positions
    xticks, xtickslabs = [], []
    for cntxt in np.unique(C):
        xticks.append(np.mean([np.max(np.where(C == cntxt)[0]), np.min(np.where(C == cntxt)[0])]))
        xtickslabs.append(cntxt)
    
    # set cosmetics
    ax.set_xticks(xticks)
    ax.set_xticks(np.arange(0.5, X.shape[0] + 0.5, 1), minor = True)
    ax.set_xticklabels(xtickslabs)
    ax.set_yticks(xticks)
    ax.set_yticks(np.arange(0.5, X.shape[0] + 0.5, 1), minor = True)
    ax.set_yticklabels(xtickslabs)

    return Z