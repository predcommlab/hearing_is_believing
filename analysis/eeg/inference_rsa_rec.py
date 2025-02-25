'''
Quick script to aggregate results from reconstruction models
across subjects and perform statistical inference tests.

All options supported by `subject_rsa_rec.py` are 
available here.

NOTE: The output report created by this script isn't
necessarily going to include the prettiest figures; it's
mostly to get a quick overview & to get all the exact
statistical results in a convenient place.
'''

import mne, json, warnings
import sys, os, data, rsa
import auxiliary as aux
sys.path.append('../spaces/')
import pubplot as pub
pub.styles.set()
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import gzip, pickle, time
import scipy
import numpy as np

def _plot_eeg_layout(info, ax, mod = 1.1):
    '''
    '''
    
    # obtain channel positions
    ch_pos = np.array([info['chs'][i]['loc'][0:3] for i in range(60)])
    ch_pos_2d = mne.channels.layout._find_topomap_coords(info, 'eeg')

    # obtain convex hull
    hull = scipy.spatial.ConvexHull(ch_pos_2d)
    ch_hull = ch_pos_2d[hull.vertices,:]
    
    # compute decomposition
    mu = ch_hull.mean(axis = 0)
    ch_hull -= mu
    U, S, V = np.linalg.svd(ch_hull.T)
    
    # compute ellipse
    i = np.linspace(0, 2 * np.pi, 1000)
    circle = np.stack((np.cos(i), np.sin(i)))
    transform = np.sqrt(2 / ch_hull.shape[0]) * U.dot(np.diag(S))
    fit = mod * transform.dot(circle) + mu[:,np.newaxis]
    
    # plot ellipse
    xmax, ymax = mod**3 * np.abs(fit[0,:]).max(), mod * np.abs(fit[1,:]).max()
    ax.plot(fit[0,:], fit[1,:], color = 'black', clip_on = False)
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-ymax, ymax])
    
    # plot ears
    ear_arg = dict(clip_on = False, fill = None, facecolor = None, edgecolor = 'black')
    s_x, s_y = 0.05, 0.15
    x_L, y_L = 2 * xmax, 2 * ymax
    
    xy_l = (fit[0,:].min() - s_x * x_L / 2, fit[1,:].mean())
    ear_l = Ellipse(xy = xy_l, width = s_x * x_L, height = s_y * y_L, **ear_arg)
    ax.add_patch(ear_l)
    
    xy_r = (fit[0,:].max() + s_x * x_L / 2, fit[1,:].mean())
    ear_r = Ellipse(xy = xy_r, width = s_x * x_L, height = s_y * y_L, **ear_arg)
    ax.add_patch(ear_r)
    
    # plot nose
    phi = np.pi / 30
    p = np.array([phi, 0, -phi])
    p = mod * transform.dot(np.stack((np.cos(p), np.sin(p)))) + mu[:,np.newaxis]
    p[0,:] -= p[0,:].mean()
    ax.plot(p[0,0:2], p[1,0:2] * np.array([1, mod]), color = 'black', clip_on = False)
    ax.plot(p[0,1:3], p[1,1:3] * np.array([mod, 1]), color = 'black', clip_on = False)
    
    return fit

def _plot_topo_inlay(w, info, ax, s = 1000, ds = 10, 
                     method = 'cubic', levels = 100, 
                     sensors = True, highlight = [],
                     cmap = 'RdBu_r', alpha = 1.0,
                     vmax = None):
    '''
    '''
    
    # add layout
    f = _plot_eeg_layout(info, ax)
    
    # obtain channel positions
    ch_pos = np.array([info['chs'][i]['loc'][0:3] for i in range(60)])
    ch_pos_2d = mne.channels.layout._find_topomap_coords(info, 'eeg')
    
    # create grid
    x, y = f
    grid_x, grid_y = np.linspace(x.min(), x.max(), s), np.linspace(y.min(), y.max(), s)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    
    # interpolate grid
    f_s = np.arange(0, f.shape[1], ds)
    f_indx = np.array([rsa.math.euclidean(f[np.newaxis,:,i] * np.ones_like(ch_pos_2d), ch_pos_2d).argsort()[0] for i in f_s])
    ch_xy = np.concatenate((ch_pos_2d, f[:,f_s].T), axis = 0)
    ch_z = np.concatenate((w, w[f_indx]))
    grid_z = scipy.interpolate.griddata((ch_xy[:,0], ch_xy[:,1]), ch_z, (grid_x, grid_y), method = method)
    
    # plot
    vmax = vmax if vmax is not None else np.abs(grid_z[np.isnan(grid_z) == False]).max()
    ax.contourf(grid_x, grid_y, grid_z, vmin = -vmax, vmax = vmax, cmap = cmap, levels = levels, alpha = alpha)
    
    # plot sensors
    if sensors: 
        indc = np.setdiff1d(np.arange(ch_pos_2d.shape[0]), highlight)
        ax.scatter(ch_pos_2d[indc,0], ch_pos_2d[indc,1], color = 'grey', marker = '.', s = 0.1)
    
    # plot highlights
    if len(highlight) > 0:
        ax.scatter(ch_pos_2d[highlight,0], ch_pos_2d[highlight,1], color = 'black', marker = 'x', s = 0.1)

if __name__ == '__main__':
    # meta options
    start_from = aux.get_opt('from', cast = int, default = 0)       # start at sid no. N

    fs = aux.get_opt('fs', default = 200, cast = int)               # frequency to downsample signal to (through binnning)
    n_k1 = aux.get_opt('n_k1', default = 1, cast = int)
    n_k2 = aux.get_opt('n_k2', default = 20, cast = int)
    off_L = aux.get_opt('off_L', default = 100, cast = int)
    max_L = aux.get_opt('max_L', default = 300, cast = int)
    b_clear = bool(aux.get_opt('b_clear', default = 0, cast = int))
    n_folds = aux.get_opt('n_folds', default = 5, cast = int)
    n_permutations = aux.get_opt('n_permutations', default = 100, cast = int)
    n_workers = aux.get_opt('n_workers', default = 4, cast = int)
    n_alphas = aux.get_opt('n_alphas', default = 20, cast = int)
    t_min = aux.get_opt('t_min', default = 0.0, cast = float)
    t_max = aux.get_opt('t_max', default = 0.25, cast = float)
    
    # dump opts
    print(f'-------- RSA: REC --------')
    print(f'[OPTS]\tfs={fs}\tb_clear={b_clear}')
    print(f'[OPTS]\tn_k1={n_k1}\tn_k2={n_k2}')
    print(f'[OPTS]\toff_L={off_L}\tmax_L={max_L}')
    print(f'[OPTS]\tfolds={n_folds}\tpermutations={n_permutations}')
    print(f'[OPTS]\tworkers={n_workers}\talphas={n_alphas}')
    print(f'[OPTS]\tt_min={t_min}\tt_max={t_max}')
    print(f'--------------------------')
    
    # load subjects and create containers
    subjects = data.Subjects.trim()
    r = []
    p = []
    
    print(f'[REC] Loading data...')
    n = 0
    N = len(subjects)
    ts = time.time()
    
    # loop over subjects
    for s_i, subject in enumerate(subjects):
        # skip if desired
        if int(subject.sid) < start_from: 
            continue
        
        # update
        aux.progressbar(n, N, ts, msg = f'[REC]')
        
        if s_i == 0: 
            eeg = mne.read_epochs(f'./data/preprocessed/eeg/sub{subject.sid}/rerp-MT2-epo.fif', verbose = False)
            info = eeg.info
        
        # find file
        dir_out = f'./data/processed/eeg/sub{subject.sid}/'
        if not b_clear: f = f'{dir_out}rec-data.pkl.gz'
        else: f = f'{dir_out}rec-data-clear.pkl.gz'
        
        # load data
        with gzip.open(f, 'rb') as f:
            r_i, p_i, _, _, _, _, _, _, _, _, _ = pickle.load(f)
        
        r.append(r_i)
        p.append(p_i)
        
        n += 1
    
    r = np.array(r)
    p = np.array(p)
    
    # update
    print(f'')
    print(f'[REC] Computing statistics...')
    
    # compute 1samp tests
    tests = [scipy.stats.ttest_1samp(r[:,i], popmean = 0) for i in range(r.shape[1])]
    tests.append(scipy.stats.ttest_1samp(r.mean(axis = 1), popmean = 0))

    # adjust p-values
    pvalues = np.array([test.pvalue for test in tests])
    pvalues = rsa.stats.bonferroni_holm(pvalues)

    # concat the global r
    all_r = np.concatenate((r, r.mean(axis = 1, keepdims = True)), axis = 1)

    # compute effect sizes
    cohensd = rsa.stats.cohens_d(all_r).squeeze()

    # compute descriptives
    mu = all_r.mean(axis = 0)
    sd = all_r.std(axis = 0)
    se = rsa.stats.bootstrap_se(all_r)
    lb, ub = mu - 1.96 * se, mu + 1.96 * se

    # summarise
    outputs = np.array([dict(mu = mu[i], sd = sd[i], se = se[i], lb = lb[i], ub = ub[i], t = tests[i].statistic, df = tests[i].df, p_cor = pvalues[i], p_unc = tests[i].pvalue, d = cohensd[i]) for i in range(len(tests))])

    '''
    Start report
    '''
    
    report = mne.Report(title = 'Stimulus reconstruction', verbose = False)
    
    C = pub.colours.equidistant('tab20c', k = 20)
    
    # add average plot
    fig, ax = pub.figure()
    avg = r.mean(axis = 1)
    mu = avg.mean()
    se = rsa.stats.bootstrap_se(avg)
    pub.dist.violins(avg, CI = False, kernel_bandwidth = 0.01, colours = [C[0]], ax = ax)
    ax.plot([0, 0], [mu - 1.96 * se, mu + 1.96 * se], color = C[0])
    ax.plot([-0.5, 0.5], [0, 0], color = 'black')
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel('Reconstruction score\n' + r'(Pearson $r$)')

    ax_topo = ax.inset_axes([0.6, 0.6, 0.4, 0.3]); ax_topo.axis('off')
    _plot_topo_inlay(p.mean(axis = (0, 1, 3)), info, ax = ax_topo, sensors = False)
    pub.cosmetics.finish()
    
    report.add_figure(fig = fig, title = 'Overview', caption = '', image_format = 'png')
    plt.close()
    
    # add plot of frequency-bands
    fig, ax = pub.figure(nrows = 4, ncols = 7, sharex = True, sharey = True)

    for i in range(r.shape[1]):
        j, k = i // 7, i % 7
        
        avg = r[:,i]
        mu = avg.mean()
        se = rsa.stats.bootstrap_se(avg)
        pub.dist.violins(avg, CI = False, kernel_bandwidth = 0.01, colours = [C[0]], ax = ax[j,k])
        ax[j,k].plot([0, 0], [mu - 1.96 * se, mu + 1.96 * se], color = C[0])
        ax[j,k].plot([-0.5, 0.5], [0, 0], color = 'black')
        ax[j,k].set_xlim([-0.5, 1.5])
        ax[j,k].set_xticks([])
        ax[j,k].set_xticklabels([])
        ax[j,k].set_ylabel('Reconstruction score\n' + r'(Pearson $r$)')
        ax[j,k].set_title(fr'Band$_{{{i}}}$')

        ax_topo = ax[j,k].inset_axes([0.6, 0.6, 0.4, 0.3]); ax_topo.axis('off')
        _plot_topo_inlay(p[:,i,:,:].mean(axis = (0, 2)), info, ax = ax_topo, sensors = False)
    pub.cosmetics.finish()
    
    report.add_figure(fig = fig, title = 'Frequency bands', caption = '', image_format = 'png')
    plt.close()
    
    # add statistics
    html = '<b>AVERAGE</b><pre>'
    output = outputs[-1]
    html += f'mu={output["mu"]}, sd={output["sd"]}, se={output["se"]}, lb={output["lb"]}, ub={output["ub"]}, t={output["t"]}, df={output["df"]}, p_cor={output["p_cor"]}, p_unc={output["p_unc"]}, d={output["d"]}'
    html += '</pre><br /><b>FREQUENCY BANDS</b><pre>'
    for i, output in enumerate(outputs[:-1]):
        html += f'b{i}: mu={output["mu"]}, sd={output["sd"]}, se={output["se"]}, lb={output["lb"]}, ub={output["ub"]}, t={output["t"]}, df={output["df"]}, p_cor={output["p_cor"]}, p_unc={output["p_unc"]}, d={output["d"]}<br />'
    html += '</pre>'
    
    report.add_html(title = 'Statistics', html = html)
    
    # export data
    print(f'[REC] Exporting results...')
    
    dir_out = f'./data/results/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    if not b_clear: f = f'{dir_out}reconstruction.pkl.gz'
    else: f = f'{dir_out}reconstruction_clear.pkl.gz'
    
    with gzip.open(f, 'wb') as f:
        pickle.dump((r, p, outputs), f)
    
    dir_out += 'reports/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    if not b_clear: f = f'{dir_out}reconstruction.html'
    else: f = f'{dir_out}reconstruction_clear.html'
    
    report.save(f, overwrite = True, verbose = False)