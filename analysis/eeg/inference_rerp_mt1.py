'''
Quick script to collect data from rERP models in learning task
and perform statistical inference.

All options supported by `subject_rerp_mt1.py` are
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

if __name__ == '__main__':
    # meta options
    start_from = aux.get_opt('from', cast = int, default = 0)       # start at sid no. N
    
    # processing options
    fs = aux.get_opt('fs', default = 200, cast = int)                           # frequency to downsample signal to (through binnning)
    b_zsc = bool(aux.get_opt('b_zsc', default = 0, cast = int))                 # should we normalise all data and predictors?
    b_spa = bool(aux.get_opt('b_spa', default = 0, cast = int))                 # should we enforce sparsity? (i.e., are we interested in TRF-like events or more temporally distributed computations?)
    b_bsl = bool(aux.get_opt('b_bsl', default = 0, cast = int))                 # should we baselien correct?
    s_mod = aux.get_opt('s_mod', default = 'spc', cast = str)                   # which hypotheses do we test here? (spc = specific, inv = invariant)
    s_bsl = aux.get_opt('s_bsl', default = 'llm', cast = str)                   # which baseline models to use? (tar/alt/llm for target/alternative/large language model).
    n_features = aux.get_opt('n_features', default = 10, cast = int)            # how many features to use from reduced llm? (default = 10)
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
    
    # dump opts
    print(f'------- rERP: MT1 --------')
    print(f'[OPTS]\tfs={fs}\t\tb_zsc={b_zsc}')
    print(f'[OPTS]\tb_spa={b_spa}\tb_bsl={b_bsl}')
    print(f'[OPTS]\ts_mod={s_mod}\ts_bsl={s_bsl}\tn_features={n_features}')
    print(f'[OPTS]\tn_workers={n_workers}\tn_alphas={n_alphas}')
    print(f'[OPTS]\tn_folds={n_folds}\tn_permutations={n_permutations}')
    print(f'[OPTS]\tn_topk={n_topk}\tn_edge_l={n_edge_L}')
    print(f'[OPTS]\tn_maxl={n_maxL}\tn_tmin={n_tmin}\tn_tmax={n_tmax}')
    print(f'[OPTS]\tbackend={backend}\tdevice={device}')
    print(f'--------------------------')
    
    # quickly make it so that the LLM also specifies dims
    if s_bsl == 'llm': s_bsl = f'llm{n_features}'
    
    # load subjects and create containers
    subjects = data.Subjects.trim()
    
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
        f = f'{dir_out}rerp-mt1-k{n_topk}-z{int(b_zsc)}-s{int(b_spa)}-b{int(b_bsl)}-{s_mod}-{s_bsl}.pkl.gz'
        
        # load data
        with gzip.open(f, 'rb') as f:
            _, _, r_i = pickle.load(f)
        
        r.append(r_i)
        
        n += 1
    
    r = np.array(r)
    
    # make this a whole-brain analysis
    r_wb = np.mean(r, axis = 2)

    '''
    One sample tests
    '''

    print(f'')
    print(f'[ENC] One sample tests...')

    # compute 1samp tests
    tests = [scipy.stats.ttest_1samp(r_wb[:,i], popmean = 0) for i in range(r_wb.shape[1])]
    
    # adjust p-values
    pvalues = np.array([test.pvalue for test in tests])
    pvalues = rsa.stats.bonferroni_holm(pvalues)

    # compute effect sizes
    cohensd = rsa.stats.cohens_d(r_wb).squeeze()

    # compute descriptives
    mu = r_wb.mean(axis = 0)
    sd = r_wb.std(axis = 0)
    se = rsa.stats.bootstrap_se(r_wb)
    lb, ub = mu - 1.96 * se, mu + 1.96 * se

    # summarise
    one_samp = {f't{i}': dict(mu = mu[i], sd = sd[i], se = se[i], lb = lb[i], ub = ub[i], t = tests[i].statistic, df = tests[i].df, p_cor = pvalues[i], p_unc = tests[i].pvalue, d = cohensd[i]) for i in range(len(tests))}
    
    '''
    Contrasts
    '''

    print(f'[ENC] Contrasts...')

    # setup dummies
    name = []
    tests = []
    cohensd = []
    mu = []
    sd = []
    se = []
    lb, ub = [], []
    pct = []
    pct_sd = []
    pct_se = []
    pct_lb, pct_ub = [], []
    
     # run contrasts in wb
    for i in range(r_wb.shape[1]):
        for j in range(r_wb.shape[1]):
            if i <= j: continue
            
            name.append(f'wb: {i}-{j}')
            tests.append(scipy.stats.ttest_rel(r_wb[:,i], r_wb[:,j]))
            cohensd.append(rsa.stats.cohens_d(r_wb[:,i], r_wb[:,j], paired = True).squeeze())
            
            # over differences
            x = r_wb[:,i] - r_wb[:,j]
            mu_x = x.mean()
            sd_x = x.std()
            se_x = rsa.stats.bootstrap_se(x)
            lb_x, ub_x = mu_x - 1.96 * se_x, mu_x + 1.96 * se_x

            mu.append(mu_x)
            sd.append(sd_x)
            se.append(se_x)
            lb.append(lb_x)
            ub.append(ub_x)

            # over percentage differences
            x = ((r_wb[:,i] / r_wb[:,j]) - 1) * 100
            mu_x = x.mean()
            sd_x = x.std()
            se_x = rsa.stats.bootstrap_se(x)
            lb_x, ub_x = mu_x - 1.96 * se_x, mu_x + 1.96 * se_x

            pct.append(mu_x)
            pct_sd.append(sd_x)
            pct_se.append(se_x)
            pct_lb.append(lb_x)
            pct_ub.append(ub_x)

    
    # correct p-values
    pvalues = np.array([test.pvalue for test in tests])
    pvalues = rsa.stats.bonferroni_holm(pvalues)

    # summarise
    contrasts = {name[i]: dict(contrast = name[i], mu = mu[i], sd = sd[i], se = se[i], lb = lb[i], ub = ub[i], t = tests[i].statistic, df = tests[i].df, p_cor = pvalues[i], p_unc = tests[i].pvalue, d = float(cohensd[i]), pct = pct[i], pct_sd = pct_sd[i], pct_se = pct_se[i], pct_lb = pct_lb[i], pct_ub = pct_ub[i]) for i in range(len(tests))}
    
    '''
    Begin report
    '''
    
    report = mne.Report(title = f'rERP: mt1 {s_mod}', verbose = False)
    
    fig, ax = pub.figure()
    C = pub.colours.equidistant('tab20c', k = 20)

    cols = [C[16], C[0], C[4], C[12]]
    pub.dist.violins(r_wb.T, kernel_bandwidth = 0.01, CI = False, ax = ax, colours = cols)
    for i in range(4):
        mu = r_wb[:,i].mean()
        se = rsa.stats.bootstrap_se(r_wb[:,i])
        
        ax.plot([i, i], [mu - 1.96 * se, mu + 1.96 * se], color = cols[i])
    
    ax.set_ylabel('Encoding performance\n' + r'(Pearson $r$)')
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['baseline', 'acoustic', 'semantic', 'combined'], rotation = 45)

    pub.cosmetics.finish()
    
    report.add_figure(fig = fig, title = 'Overview', caption = '', image_format = 'png')
    plt.close()
    
    html = '<pre>'
    for samp_n in one_samp: 
        samp = one_samp[samp_n]
        html += f'{samp_n}: mu={samp["mu"]}, sd={samp["sd"]}, se={samp["se"]}, lb={samp["lb"]}, ub={samp["ub"]}, t={samp["t"]}, df={samp["df"]}, p_cor={samp["p_cor"]}, p_unc={samp["p_unc"]}, d={samp["d"]}<br />'
    html += '</pre>'
    report.add_html(title = 'One-sample tests', html = html)

    html = '<pre>'
    for contrast in contrasts: 
        contrast = contrasts[contrast]
        html += f'{contrast["contrast"]}: mu={contrast["mu"]}, sd={contrast["sd"]}, se={contrast["se"]}, lb={contrast["lb"]}, ub={contrast["ub"]}, t={contrast["t"]}, df={contrast["df"]}, p_cor={contrast["p_cor"]}, p_unc={contrast["p_unc"]}, d={contrast["d"]}, pct_mu={contrast["pct"]}, pct_sd={contrast["pct_sd"]}, pct_se={contrast["pct_se"]}, pct_lb={contrast["pct_lb"]}, pct_ub={contrast["pct_ub"]}<br />'
    html += '</pre>'
    report.add_html(title = 'Contrasts', html = html)
    
    '''
    Export data
    '''
    print(f'[ENC] Exporting data...')
    
    dir_out = './data/results/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    f = f'{dir_out}rerp-mt1-k{n_topk}-z{int(b_zsc)}-s{int(b_spa)}-b{int(b_bsl)}-{s_mod}-{s_bsl}.pkl.gz'
    with gzip.open(f, 'wb') as f:
        pickle.dump((r_wb, one_samp, contrasts), f)
    
    dir_out += 'reports/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    f = f'{dir_out}rerp-mt1-k{n_topk}-z{int(b_zsc)}-s{int(b_spa)}-b{int(b_bsl)}-{s_mod}-{s_bsl}.html'
    report.save(f, overwrite = True, verbose = False)