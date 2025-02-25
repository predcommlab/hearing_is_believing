'''
Quick script to collect data from rERP knock-outs in the learning task
and perform statistical inference.

All options supported by `subject_rerp_mt1_knockout.py` are
also available here, of course.

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

import gzip, pickle, time
import scipy
import numpy as np

from inference_rsa_rec import _plot_topo_inlay

if __name__ == '__main__':
    # meta options
    start_from = aux.get_opt('from', cast = int, default = 0)       # start at sid no. N
    
    # script options
    n_cbp = aux.get_opt('n_cbp', default = 10000, cast = int)                   # how many permutations should we run for cluster-based permutation tests?
    n_cbt = aux.get_opt('n_cbt', default = 2.0322, cast = float)                # what alpha level to use for cluster formation? (here, alpha = 0.05, two-tailed)
    
    # processing options
    fs = aux.get_opt('fs', default = 200, cast = int)                           # frequency to downsample signal to (through binnning)
    b_zsc = bool(aux.get_opt('b_zsc', default = 0, cast = int))                 # should we normalise all data and predictors?
    b_spa = bool(aux.get_opt('b_spa', default = 0, cast = int))                 # should we enforce sparsity? (i.e., are we interested in TRF-like events or more temporally distributed computations?)
    b_bsl = bool(aux.get_opt('b_bsl', default = 0, cast = int))                 # should we baselien correct?
    s_mod = aux.get_opt('s_mod', default = 'spc', cast = str)                   # which hypotheses do we test here? (spc = specific, inv = invariant)
    s_bsl = aux.get_opt('s_bsl', default = 'llm', cast = str)                   # which baseline models to use? (tar/alt/llm for target/alternative/large language model).
    n_features = aux.get_opt('n_features', default = 10, cast = int)            # how many features to use from reduced llm? (default = 10)
    n_mod = aux.get_opt('n_mod', default = 2, cast = int)                       # which model to test (default = 2)?
    a_coefs = aux.get_opt('a_coefs', default = 'all', cast = str)               # which coefficients to test (default = -1)?
    n_workers = aux.get_opt('n_workers', default = 4, cast = int)               # how many jobs to use
    n_alphas = aux.get_opt('n_alphas', default = 20, cast = int)                # how many alphas to test
    n_folds = aux.get_opt('n_folds', default = 5, cast = int)                   # how many folds to use for CV
    n_permutations = aux.get_opt('n_permutations', default = 50, cast = int)    # how many permutations to perform
    n_topk = aux.get_opt('n_topk', default = 5, cast = int)                     # how many top words to use for acoustic surprisal
    n_edge_L = aux.get_opt('n_edge_l', default = 10, cast = int)                # how many samples to consider in acoustic edge computation
    n_maxL = aux.get_opt('n_maxl', default = 2000, cast = int)                  # total duration after onset (in epochs)
    n_tmin = aux.get_opt('n_tmin', default = -0.8, cast = float)                # minimum to consider for time delaying ridge
    n_tmax = aux.get_opt('n_tmax', default = 0.1, cast = float)                 # maximum to consider for time delaying ridge
    backend = aux.get_opt('backend', default = 'torch', cast = str)             # what backend to use for model fitting? (torch/numpy)
    device = aux.get_opt('device', default = 'cuda', cast = str)                # what device to use for model fitting? (cpu/gpu, gpu only available in torch)
    maxtask = aux.get_opt('maxtask', default = 1, cast = int)                   # for multiprocessing, when do we kill a child? this is for efficient garbage collection (and keeping memory required to a minimum)
    
    # prepare coefficients
    if a_coefs != 'all': a_coefs = np.array(a_coefs.split(',')).astype(int)
    
    # make a readable label for coefs
    if a_coefs != 'all': a_coefs = ','.join(list(a_coefs.astype(str)))
    
    # dump opts
    print(f'------- rERP: MT1 --------')
    print(f'[OPTS]\tfs={fs}\t\tb_zsc={b_zsc}')
    print(f'[OPTS]\tb_spa={b_spa}\tb_bsl={b_bsl}')
    print(f'[OPTS]\ts_mod={s_mod}\ts_bsl={s_bsl}\tn_features={n_features}')
    print(f'[OPTS]\tn_mod={n_mod}\ta_coefs={a_coefs}')
    print(f'[OPTS]\tn_workers={n_workers}\tn_alphas={n_alphas}')
    print(f'[OPTS]\tn_folds={n_folds}\tn_permutations={n_permutations}')
    print(f'[OPTS]\tn_topk={n_topk}\tn_edge_l={n_edge_L}')
    print(f'[OPTS]\tn_maxl={n_maxL}\tn_tmin={n_tmin}\tn_tmax={n_tmax}')
    print(f'[OPTS]\tbackend={backend}\tdevice={device}')
    print(f'--------------------------')
    
    # load subjects and create containers
    subjects = data.Subjects.trim()
    
    R = []
    r = []
    
    print(f'[ENC] Loading data...')
    n = 0
    N = len(subjects)
    ts = time.time()
    
    # loop over subjects
    for s_i, subject in enumerate(subjects):
        # skip if desired
        if int(subject.sid) < start_from: 
            continue
        
        aux.progressbar(n, N, ts, msg = f'[ENC]')
        
        if s_i == 0: 
            eeg = mne.read_epochs(f'./data/preprocessed/eeg/sub{subject.sid}/rerp-MT2-epo.fif', verbose = False)
            info = eeg.info
        
        # find file
        dir_out = f'./data/processed/eeg/sub{subject.sid}/'
        f = f'{dir_out}rerp-mt1-ko-n{n_mod}-c{a_coefs}-k{n_topk}-z{int(b_zsc)}-s{int(b_spa)}-b{int(b_bsl)}-{s_mod}-{s_bsl}.pkl.gz'
        
        # load data
        with gzip.open(f, 'rb') as f:
            R_i, _, _, _ = pickle.load(f)
        
        R.append(R_i)
        
        # load data for delta
        f = f'{dir_out}rerp-mt1-k{n_topk}-z{int(b_zsc)}-s{int(b_spa)}-b{int(b_bsl)}-{s_mod}-{s_bsl}.pkl.gz'
        
        with gzip.open(f, 'rb') as f:
            _, _, r_i = pickle.load(f)
        
        r.append(r_i)
        
        n += 1
    
    R = np.array(R)
    r = np.array(r)
    
    '''
    Compute delta and R^2
    '''
    
    if n_mod > 0: delta = r[:,n_mod,:].mean(axis = 1) - r[:,0,:].mean(axis = 1)
    else: delta = r[:,n_mod,:].mean(axis = 1)
    
    R2 = R / delta[...,None,None,None].mean()

    '''
    Compute adjacency
    '''
    
    print(f'')
    print(f'[ENC] Computing adjacency...')
    adjacency, names = mne.channels.find_ch_adjacency(info, 'eeg')
    adj0, n0 = mne.channels.read_ch_adjacency('easycapM11')
    indc = [np.where(np.array(n0) == ch)[0][0] for ch in names]
    adj1, n1 = mne.channels.read_ch_adjacency('easycapM11', picks = indc)
    
    # obtain colours, labels quickly
    ch_pos = np.array([info['chs'][i]['loc'][0:3] for i in range(60)])
    ch_lab = np.array([info['chs'][i]['ch_name'] for i in range(60)])
    ch_col = ch_pos.copy()
    ch_col -= ch_col.min(axis = 0)
    ch_col /= ch_col.max(axis = 0)
    
    '''
    Begin report
    '''
    
    report = mne.Report(title = f'rERP-ko: mt1 {s_bsl} {s_mod} m{n_mod}', verbose = False)
    
    '''
    Cluster-based permutation tests
    '''

    print(f'[ENC] Cluster-based permutation tests...')
    
    # setup progressbar
    n = 0
    N = R.shape[2]
    ts = time.time()
    
    # setup dummy
    cluster_based = {f'ß_{c_i}': [] for c_i in range(N)}
    
    # setup colour
    C = pub.colours.equidistant('tab20c', k = 20)
    
    # loop over coefficients
    for coef in range(R.shape[2]):
        # update
        aux.progressbar(n, N, ts, msg = f'[ENC]')
        
        # perform stats
        # NOTE: We wrap this in a catch warnings block
        # because sometimes coefficients may simply
        # be zeros (because they were not used in the
        # model) which causes mne to throw a warning
        # that we can safely ignore.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            R2_c = R2[:,:,coef,:]
            t, c, p, _ = mne.stats.spatio_temporal_cluster_1samp_test(R2_c.swapaxes(1, 2), 
                                                                      n_permutations = n_cbp, n_jobs = n_workers, 
                                                                      tail = 0, threshold = n_cbt, adjacency = adj1, verbose = False)
            mask = np.zeros((len(p),))
        
        if len(p) > 0:
            # NOTE: Because we cannot easily perform a 
            # Bonferroni-Holm correction here, we simply
            # perform a regular Bonferroni over coefs.
            # 
            # I think it's debateable wether or not this
            # truly is required, given that we only really
            # ever care about a handful of coefficients at
            # most (and, most of the time, it'll just be one),
            # but I am doing it to be safe.
            p = np.array(p)
            pcor = np.clip(N * p, 0, 1)
            mask = (pcor <= .05)
        
        if mask.sum() == 0:
            # no significant effects, plot all channels
            fig, ax = pub.figure()
            for ch in range(60): ax.plot(R2_c[:,ch,:].mean(axis = 0), color = ch_col[ch])
            ax.plot(np.zeros((R2_c.shape[2],)), color = 'black')
            ax.set_ylabel(r'Variance explained ($R^2$)')
            ax.set_xticks(np.arange(0, R2_c.shape[2], 40))
            ax.set_xticklabels(np.round(np.arange(-20, R2_c.shape[2] - 20, 40)*5e-3, 2))
            pub.cosmetics.finish()
            
            report.add_figure(fig = fig, title = f'ß_{coef}', caption = '', image_format = 'png')
            plt.close()
        else:
            # plot clusters
            n_eff = mask.sum()
            nrows = np.ceil(n_eff / 3).astype(int)
            ncols = np.min([n_eff, 3])
            
            # setup figure
            fig, ax = pub.figure(nrows = nrows, ncols = ncols)
            if nrows == 1: ax = np.array([ax])
            if ncols == 1: ax = np.array([ax])
            
            # setup cluster dummy
            clusters = []
            
            # loop over masked clusters
            for jj, ii in enumerate(np.where(mask)[0]):
                # grab axis indices
                j, k = jj // 3, jj % 3
                
                # grab timepoints and channels
                c_ii = c[ii]
                tps = np.unique(c_ii[0])
                chs = np.unique(c_ii[1])
                
                # compute descriptives
                # NOTE: Here we compute a biased Cohen's d statistic, as
                # per Eric Maris's suggestion in:
                #   
                #   https://mailman.science.ru.nl/pipermail/fieldtrip/2017-September/024644.html
                # 
                # This is not ideal, but it's the best we can do.
                R2_cc = R2_c[:,chs,:].mean(axis = 1)
                mu = R2_cc.mean(axis = 0)
                se = rsa.stats.bootstrap_se(R2_cc)
                t = np.arange(se.shape[0])
                absmax = np.abs(np.array([mu - 1.96 * se, mu + 1.96 * se])).max()
                cohensd = rsa.stats.cohens_d(R2_c[:,c_ii[1],c_ii[0]].mean(axis = 1, keepdims = True))[0,0]
                
                # plot cluster and topography
                ax[j,k].fill_between(t, mu - 1.96 * se, mu + 1.96 * se, edgecolor = None, facecolor = C[0], alpha = 0.25)
                ax[j,k].plot(mu, color = C[0])
                ax[j,k].plot([tps.min(), tps.max()], [1.1 * absmax, 1.1 * absmax], color = 'black')
                ax[j,k].plot(np.zeros_like(mu), color = 'black')
                ax_topo = ax[j,k].inset_axes([0.05, 0.65, 0.2, 0.2])
                ax_topo.axis('off')
                _plot_topo_inlay(R2_c[:,:,t.min():t.max()].mean(axis = (0, 2)), info, ax = ax_topo, sensors = False, highlight = chs)
                
                # add labels
                ax[j,k].set_title(fr'Cluster $o_{{{ii}}} i_{{{jj}}}$')
                ax[j,k].set_ylabel(r'Variance explained ($R^2$)')
                ax[j,k].set_xlabel(r'Time since onset (s)')
                ax[j,k].set_xticks(np.arange(0, R2_c.shape[2], 40))
                ax[j,k].set_xticklabels(np.round(np.arange(-20, R2_c.shape[2] - 20, 40)*5e-3, 2))
                
                # add cluster to list
                clusters.append(dict(i = jj, o = ii, tmin = tps.min(), tmax = tps.max(), N_ch = chs.shape[0], p_cor = pcor[ii], p_unc = p[ii], d = cohensd, chs_i = chs, chs_l = ch_lab[chs], times = c_ii[0], channels = c_ii[1]))
            
            pub.cosmetics.finish()
            
            report.add_figure(fig = fig, title = f'ß_{coef}', caption = '', image_format = 'png')
            plt.close()
            
            # create stats report for coefficient
            caption = '<b>Clusters:</b><br /><pre>'
            for cluster in clusters:
                caption += f'ß_{coef} i_{cluster["i"]} o_{cluster["o"]}: {cluster["tmin"]}-{cluster["tmax"]} (ch = {cluster["N_ch"]}), p_cor = {cluster["p_cor"]}, p_unc = {cluster["p_unc"]}, d={cluster["d"]}, chs_i={cluster["chs_i"]}, chs_l={cluster["chs_l"]}\n'
            caption += '</pre>'
            
            report.add_html(title = f'stats: ß_{coef}', html = caption)
            
            # add to stack
            cluster_based[f'ß_{coef}'] = clusters
        
        # tally
        n += 1
    
    '''
    Export data
    '''
    print(f'')
    print(f'[ENC] Exporting data...')
    
    dir_out = './data/results/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    f = f'{dir_out}rerp-mt1-ko-n{n_mod}-c{a_coefs}-k{n_topk}-z{int(b_zsc)}-s{int(b_spa)}-b{int(b_bsl)}-{s_mod}-{s_bsl}.pkl.gz'
    with gzip.open(f, 'wb') as f:
        pickle.dump((R2, cluster_based), f)
    
    dir_out += 'reports/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    f = f'{dir_out}rerp-mt1-ko-n{n_mod}-c{a_coefs}-k{n_topk}-z{int(b_zsc)}-s{int(b_spa)}-b{int(b_bsl)}-{s_mod}-{s_bsl}.html'
    report.save(f, overwrite = True, verbose = False)