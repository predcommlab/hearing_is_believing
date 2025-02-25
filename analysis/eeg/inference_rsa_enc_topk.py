'''
Quick script to aggregate results from increasing k
analysis in encoders.

All options supported by `subject_rsa_enc.py` are
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
    fs = aux.get_opt('fs', default = 200, cast = int)                           # frequency of reconstructed gammatones (must match subject_rsa_rec.py)
    n_k1 = aux.get_opt('n_k1', default = 10, cast = int)                        # length of smoothing kernel (in samples)
    pad_L = aux.get_opt('pad_L', default = n_k1, cast = int)                    # padding to apply before smoothing (should be >= n_k1)
    b_brain = bool(aux.get_opt('b_brain', default = 0, cast = int))             # should we use brain data for targets/alternatives?
    b_clear = bool(aux.get_opt('b_clear', default = 0, cast = int))             # should we load clears from clear training?
    b_morph = bool(aux.get_opt('b_morph', default = 0, cast = int))             # should we load morphs from clear training?
    n_mod = aux.get_opt('n_mod', default = 3, cast = int)                      # should we fit _only a specific_ model? (give index)
    n_folds = aux.get_opt('n_folds', default = 5, cast = int)                   # number of folds for cross-validation
    n_permutations = aux.get_opt('n_permutations', default = 50, cast = int)    # number of permutations to run
    n_workers = aux.get_opt('n_workers', default = 5, cast = int)               # number of parallel workers to use
    n_alphas = aux.get_opt('n_alphas', default = 20, cast = int)                # number of alphas to consider
    b_acoustic = aux.get_opt('b_acoustic', default = 1, cast = int)             # should we include acoustic predictors in models?
    b_semantic = bool(aux.get_opt('b_semantic', default = 0, cast = int))       # should we include semantic predictors in models?
    
    # dump opts
    print(f'-------- RSA: ENC --------')
    print(f'[OPTS]\tfs={fs}\tn_k1={n_k1}')
    print(f'[OPTS]\tpad_L={pad_L}')
    print(f'[OPTS]\tb_brain={b_brain}\tb_semantic={b_semantic}')
    print(f'[OPTS]\tb_clear={b_clear}\tb_morph={b_morph}')
    print(f'[OPTS]\tn_folds={n_folds}\tn_permutations={n_permutations}')
    print(f'[OPTS]\tn_workers={n_workers}\tn_alphas={n_alphas}')
    print(f'--------------------------')
    
    # load subjects and create containers
    subjects = data.Subjects.trim()
    ks = np.arange(1, 20, 2)
    
    r, ß = [[] for i in range(len(subjects))], [[] for i in range(len(subjects))]
    
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
        
        for n_topk in ks:
            f = f'{dir_out}rec-enc-b{int(b_brain)}-m{int(b_morph)}-c{int(b_clear)}-a{int(b_acoustic)}-s{int(b_semantic)}-k{int(n_topk)}-m{n_mod}.pkl.gz'
        
            # load data
            with gzip.open(f, 'rb') as f:
                _, ß_i, r_i = pickle.load(f)

            r[s_i].append(r_i)
            ß[s_i].append(ß_i)
        
        n += 1
    
    r, ß = np.array(r), np.array(ß)
    
    '''
    One sample tests
    '''
    print(f'')
    print(f'[ENC] One sample tests...')
    
    # compute tests
    tests = [scipy.stats.ttest_1samp(r[:,i,0], popmean = 0) for i in range(ks.shape[0])]
    
    # adjust p-values
    pvalues = np.array([test.pvalue for test in tests])
    pvalues = rsa.stats.bonferroni_holm(pvalues)
    
    # compute effect size
    cohensd = rsa.stats.cohens_d(r.squeeze()).squeeze()
    
    # compute descriptives
    mu = r.mean(axis = (0, 2))
    sd = r.std(axis = (0, 2))
    se = rsa.stats.bootstrap_se(r.squeeze())
    lb, ub = mu - 1.96 * se, mu + 1.96 * se
    
    # summarise
    one_samp = {f'k{k}': dict(mu = mu[i], sd = sd[i], se = se[i], lb = lb[i], ub = ub[i], t = tests[i].statistic, df = tests[i].df, p_cor = pvalues[i], p_unc = tests[i].pvalue, d = cohensd[i]) for i, k in enumerate(ks)}
    
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
    
    # run contrasts
    for i in range(1, ks.shape[0]):
        name.append(f'{i}-{i-1}')
        tests.append(scipy.stats.ttest_rel(r[:,i,0], r[:,i-1,0]))
        cohensd.append(rsa.stats.cohens_d(r[:,i,0], r[:,i-1,0], paired = True))
        
        x = r[:,i,0] - r[:,i-1,0]
        mu_x = x.mean()
        sd_x = x.std()
        se_x = rsa.stats.bootstrap_se(x)
        lb_x, ub_x = mu_x - 1.96 * se_x, mu_x + 1.96 * se_x
        
        mu.append(mu_x)
        sd.append(sd_x)
        se.append(se_x)
        lb.append(lb_x)
        ub.append(ub_x)
    
    # correct p-values
    pvalues = np.array([test.pvalue for test in tests])
    pvalues = rsa.stats.bonferroni_holm(pvalues)
    
    # summarise
    contrasts = {name[i]: dict(contrast = name[i], mu = mu[i], sd = sd[i], se = se[i], lb = lb[i], ub = ub[i], t = tests[i].statistic, df = tests[i].df, p_cor = pvalues[i], p_unc = tests[i].pvalue, d = float(cohensd[i])) for i in range(len(tests))}
    
    '''
    Begin report
    '''
    
    report = mne.Report(title = 'top-k RSA Encoders', verbose = False)
    
    C = pub.colours.equidistant('tab20c', k = 20)
    
    # add overview
    fig, ax = pub.figure()
    pub.dist.violins(r[:,:,0].T, CI = False, kernel_bandwidth = 0.01, colours = [C[0]] * ks.shape[0], ax = ax)
    for i in range(ks.shape[0]):
        mu = r[:,i,0].mean()
        se = rsa.stats.bootstrap_se(r[:,i,0])
        lb, ub = mu - 1.96 * se, mu + 1.96 * se
        ax.plot([i, i], [lb, ub], color = C[0])
    ax.set_ylabel('Encoding performance\n' + r'(Pearson $r$)')
    ax.set_xlabel(r'$k$ predictions')
    ax.set_xticks(np.arange(ks.shape[0]))
    ax.set_xticklabels(ks)
    pub.cosmetics.finish()
    
    report.add_figure(title = 'Overview', fig = fig, caption = '', image_format = 'png')
    plt.close()
    
    # add statistics
    html = '<pre>'
    for samp_n in one_samp: 
        samp = one_samp[samp_n]
        html += f'{samp_n}: mu={samp["mu"]}, sd={samp["sd"]}, se={samp["se"]}, lb={samp["lb"]}, ub={samp["ub"]}, t={samp["t"]}, df={samp["df"]}, p_cor={samp["p_cor"]}, p_unc={samp["p_unc"]}, d={samp["d"]}<br />'
    html += '</pre>'
    report.add_html(title = 'One-sample tests', html = html)

    html = '<pre>'
    for contrast in contrasts: 
        contrast = contrasts[contrast]
        html += f'{contrast["contrast"]}: mu={contrast["mu"]}, sd={contrast["sd"]}, se={contrast["se"]}, lb={contrast["lb"]}, ub={contrast["ub"]}, t={contrast["t"]}, df={contrast["df"]}, p_cor={contrast["p_cor"]}, p_unc={contrast["p_unc"]}, d={contrast["d"]}<br />'
    html += '</pre>'
    report.add_html(title = 'Contrasts', html = html)
    
    # add coef plot
    fig, ax = pub.figure(nrows = 2, ncols = 4)
    
    c_k = np.linspace(C[3], C[0], ks.shape[0])
    
    for i in range(8):
        for ii, k_i in enumerate(ks):
            j, k = i // 4, i % 4
            
            ß_i = ß[:,ii,0,i,:]
            mu = ß_i.mean(axis = 0)
            se = rsa.stats.bootstrap_se(ß_i)
            t = np.arange(se.shape[0])
            
            label = None
            if (ii == 0) or (ii == (len(ks) - 1)): label = fr'$k={k_i}$'
            ax[j,k].plot(np.zeros((t.shape[0],)), color = 'black')
            ax[j,k].fill_between(t, mu - 1.96 * se, mu + 1.96 * se, edgecolor = None, facecolor = c_k[ii], alpha = 0.25)
            ax[j,k].plot(mu, color = c_k[ii], label = label)
        
        ax[j,k].set_title(fr'$\beta_{{{i}}}$')
        ax[j,k].set_xticks(np.arange(0, 201, 50))
        ax[j,k].set_xticklabels(np.round(np.arange(0, 201, 50)*5e-3, 2))
        ax[j,k].set_xlabel(r'Stimulus time (s)')
        ax[j,k].set_ylabel(r'$\beta$-coefficient (a.u.)')
        pub.cosmetics.legend(ax = ax[j,k], loc = 'lower left')
    
    pub.cosmetics.finish()
    
    report.add_figure(title = 'Coefficients', fig = fig, caption = '', image_format = 'png')
    plt.close()
    
    '''
    Export data
    '''
    print(f'[ENC] Exporting data...')
    
    dir_out = './data/results/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    f = f'{dir_out}encoding_topk_b{int(b_brain)}-m{int(b_morph)}-c{int(b_clear)}.pkl.gz'
    with gzip.open(f, 'wb') as f:
        pickle.dump((r, ß, one_samp, contrasts), f)
    
    dir_out += 'reports/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    f = f'{dir_out}encoding_topk_b{int(b_brain)}-m{int(b_morph)}-c{int(b_clear)}.html'
    report.save(f, overwrite = True, verbose = False)