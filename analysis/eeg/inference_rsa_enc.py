'''
Quick script to collect data from RSA encoder and perform
statistical inference.

All options supported by `subject_rsa_enc.py` are
also available here, of course.

NOTE: The output report created by this script isn't
necessarily going to include the prettiest figures; it's
mostly to get a quick overview & to get all the exact
statistical results in a convenient place.
'''

import mne, json, warnings
import sys, os, aux, data, rsa
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
    
    # script options
    n_cbp = aux.get_opt('n_cbp', default = 10000, cast = int)                   # how many permutations should we run for cluster-based permutation tests?

    # processing options
    fs = aux.get_opt('fs', default = 200, cast = int)                           # frequency of reconstructed gammatones (must match subject_rsa_rec.py)
    n_topk = aux.get_opt('n_topk', default = 5, cast = int)                     # number of top-k predictions to consider
    n_k1 = aux.get_opt('n_k1', default = 10, cast = int)                        # length of smoothing kernel (in samples)
    pad_L = aux.get_opt('pad_L', default = n_k1, cast = int)                    # padding to apply before smoothing (should be >= n_k1)
    b_brain = bool(aux.get_opt('b_brain', default = 0, cast = int))             # should we use brain data for targets/alternatives?
    b_clear = bool(aux.get_opt('b_clear', default = 0, cast = int))             # should we load clears from clear training?
    b_morph = bool(aux.get_opt('b_morph', default = 0, cast = int))             # should we load morphs from clear training?
    n_mod = aux.get_opt('n_mod', default = -1, cast = int)                      # should we fit _only a specific_ model? (give index)
    n_folds = aux.get_opt('n_folds', default = 5, cast = int)                   # number of folds for cross-validation
    n_permutations = aux.get_opt('n_permutations', default = 50, cast = int)    # number of permutations to run
    n_workers = aux.get_opt('n_workers', default = 5, cast = int)               # number of parallel workers to use
    n_alphas = aux.get_opt('n_alphas', default = 20, cast = int)                # number of alphas to consider
    b_semantic = bool(aux.get_opt('b_semantic', default = 0, cast = int))       # should we include semantic predictors in models?
    b_acoustic = bool(aux.get_opt('b_acoustic', default = 1, cast = int))       # should we include acoustic predictors in models?
    
    # dump opts
    print(f'-------- RSA: ENC --------')
    print(f'[OPTS]\tfs={fs}\tn_k1={n_k1}')
    print(f'[OPTS]\tpad_L={pad_L}\tn_topk={n_topk}')
    print(f'[OPTS]\tb_brain={b_brain}')
    print(f'[OPTS]\tb_clear={b_clear}\tb_morph={b_morph}')
    print(f'[OPTS]\tn_folds={n_folds}\tn_permutations={n_permutations}')
    print(f'[OPTS]\tn_workers={n_workers}\tn_alphas={n_alphas}')
    print(f'--------------------------')
    
    # load subjects and create containers
    subjects = data.Subjects.trim()
    
    ß_a = []
    ß_s = []
    ß_as = []
    
    r_a = []
    r_s = []
    r_as = []
    
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
        f = f'{dir_out}rec-enc-b{int(b_brain)}-m{int(b_morph)}-c{int(b_clear)}-a1-s0-k{int(n_topk)}-m{n_mod}.pkl.gz'
        
        # load data
        with gzip.open(f, 'rb') as f:
            _, ß_r, r_i = pickle.load(f)
        
        ß_a.append(ß_r)
        r_a.append(r_i)
        
        # find file
        dir_out = f'./data/processed/eeg/sub{subject.sid}/'
        f = f'{dir_out}rec-enc-b{int(b_brain)}-m{int(b_morph)}-c{int(b_clear)}-a0-s1-k{int(n_topk)}-m{n_mod}.pkl.gz'
        
        # load data
        with gzip.open(f, 'rb') as f:
            _, ß_r, r_i = pickle.load(f)
        
        ß_s.append(ß_r)
        r_s.append(r_i)

        # find file
        dir_out = f'./data/processed/eeg/sub{subject.sid}/'
        f = f'{dir_out}rec-enc-b{int(b_brain)}-m{int(b_morph)}-c{int(b_clear)}-a1-s1-k{int(n_topk)}-m{n_mod}.pkl.gz'
        
        # load data
        with gzip.open(f, 'rb') as f:
            _, ß_r, r_i = pickle.load(f)
        
        ß_as.append(ß_r)
        r_as.append(r_i)
        
        n += 1
    
    ß_a = np.array(ß_a)
    r_a = np.array(r_a)
    
    ß_s = np.array(ß_s)
    r_s = np.array(r_s)

    ß_as = np.array(ß_as)
    r_as = np.array(r_as)

    '''
    Quickly re-organise order;
    this just makes it a bit more convenient to plot later
    '''
    
    r_a = r_a[...,[0, 2, 1, 3]]
    r_s = r_s[...,[0, 2, 1, 3]]
    r_as = r_as[...,[0, 2, 1, 3]]

    ß_a = ß_a[:,[0, 2, 1, 3]]
    ß_s = ß_s[:,[0, 2, 1, 3]]
    ß_as = ß_as[:,[0, 2, 1, 3]]

    '''
    One sample tests
    '''

    print(f'')
    print(f'[ENC] One sample tests...')

    # compute 1samp tests
    tests = [scipy.stats.ttest_1samp(r_a[:,i], popmean = 0) for i in range(r_a.shape[1])]
    tests += [scipy.stats.ttest_1samp(r_s[:,i], popmean = 0) for i in range(r_s.shape[1])]
    tests += [scipy.stats.ttest_1samp(r_as[:,i], popmean = 0) for i in range(r_as.shape[1])]

    # adjust p-values
    pvalues = np.array([test.pvalue for test in tests])
    pvalues = rsa.stats.bonferroni_holm(pvalues)

    # compute effect sizes
    cohensd = np.concatenate((rsa.stats.cohens_d(r_a).squeeze(),
                              rsa.stats.cohens_d(r_s).squeeze(),
                              rsa.stats.cohens_d(r_as).squeeze()), axis = 0)

    # compute descriptives
    all_r = np.concatenate((r_a, r_s, r_as), axis = 1)
    mu = all_r.mean(axis = 0)
    sd = all_r.std(axis = 0)
    se = rsa.stats.bootstrap_se(all_r)
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

    # run contrasts in a
    for i in range(r_a.shape[1]):
        for j in range(r_a.shape[1]):
            if i <= j: continue
            
            name.append(f'a: {i}-{j}')
            tests.append(scipy.stats.ttest_rel(r_a[:,i], r_a[:,j]))
            cohensd.append(rsa.stats.cohens_d(r_a[:,i], r_a[:,j], paired = True).squeeze())
            
            x = r_a[:,i] - r_a[:,j]
            mu_x = x.mean()
            sd_x = x.std()
            se_x = rsa.stats.bootstrap_se(x)
            lb_x, ub_x = mu_x - 1.96 * se_x, mu_x + 1.96 * se_x
            
            mu.append(mu_x)
            sd.append(sd_x)
            se.append(se_x)
            lb.append(lb_x)
            ub.append(ub_x)
    
    # run contrasts in s
    for i in range(r_s.shape[1]):
        for j in range(r_s.shape[1]):
            if i <= j: continue
            
            name.append(f's: {i}-{j}')
            tests.append(scipy.stats.ttest_rel(r_s[:,i], r_s[:,j]))
            cohensd.append(rsa.stats.cohens_d(r_s[:,i], r_s[:,j], paired = True).squeeze())
            
            x = r_s[:,i] - r_s[:,j]
            mu_x = x.mean()
            sd_x = x.std()
            se_x = rsa.stats.bootstrap_se(x)
            lb_x, ub_x = mu_x - 1.96 * se_x, mu_x + 1.96 * se_x
            
            mu.append(mu_x)
            sd.append(sd_x)
            se.append(se_x)
            lb.append(lb_x)
            ub.append(ub_x)

    # run contrasts in as
    for i in range(r_as.shape[1]):
        for j in range(r_as.shape[1]):
            if i <= j: continue
            
            name.append(f'as: {i}-{j}')
            tests.append(scipy.stats.ttest_rel(r_as[:,i], r_as[:,j]))
            cohensd.append(rsa.stats.cohens_d(r_as[:,i], r_as[:,j], paired = True).squeeze())
            
            x = r_as[:,i] - r_as[:,j]
            mu_x = x.mean()
            sd_x = x.std()
            se_x = rsa.stats.bootstrap_se(x)
            lb_x, ub_x = mu_x - 1.96 * se_x, mu_x + 1.96 * se_x
            
            mu.append(mu_x)
            sd.append(sd_x)
            se.append(se_x)
            lb.append(lb_x)
            ub.append(ub_x)

    # run contrasts between a, s
    for i in range(r_as.shape[1]):
        name.append(f's-a: {i}')
        tests.append(scipy.stats.ttest_rel(r_s[:,i], r_a[:,i]))
        cohensd.append(rsa.stats.cohens_d(r_s[:,i], r_a[:,i], paired = True).squeeze())
        
        x = r_s[:,i] - r_a[:,i]
        mu_x = x.mean()
        sd_x = x.std()
        se_x = rsa.stats.bootstrap_se(x)
        lb_x, ub_x = mu_x - 1.96 * se_x, mu_x + 1.96 * se_x
        
        mu.append(mu_x)
        sd.append(sd_x)
        se.append(se_x)
        lb.append(lb_x)
        ub.append(ub_x)
    
    # run contrasts between a, as
    for i in range(r_as.shape[1]):
        name.append(f'as-a: {i}')
        tests.append(scipy.stats.ttest_rel(r_as[:,i], r_a[:,i]))
        cohensd.append(rsa.stats.cohens_d(r_as[:,i], r_a[:,i], paired = True).squeeze())
        
        x = r_as[:,i] - r_a[:,i]
        mu_x = x.mean()
        sd_x = x.std()
        se_x = rsa.stats.bootstrap_se(x)
        lb_x, ub_x = mu_x - 1.96 * se_x, mu_x + 1.96 * se_x
        
        mu.append(mu_x)
        sd.append(sd_x)
        se.append(se_x)
        lb.append(lb_x)
        ub.append(ub_x)
    
    # run contrasts between s, as
    for i in range(r_as.shape[1]):
        name.append(f'as-s: {i}')
        tests.append(scipy.stats.ttest_rel(r_as[:,i], r_s[:,i]))
        cohensd.append(rsa.stats.cohens_d(r_as[:,i], r_s[:,i], paired = True).squeeze())
        
        x = r_as[:,i] - r_s[:,i]
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
    
    report = mne.Report(title = 'RSA Encoders', verbose = False)
    
    fig, ax = pub.figure()
    C = pub.colours.equidistant('tab20c', k = 20)

    pub.dist.violins(r_a.T, kernel_bandwidth = 0.005, ax = ax, colours = [C[0]] * 4)
    pub.dist.violins(r_s.T, offset_x = 4, kernel_bandwidth = 0.005, ax = ax, colours = [C[4]] * 4)
    pub.dist.violins(r_as.T, offset_x = 8, kernel_bandwidth = 0.005, ax = ax, colours = [C[12]] * 4)
    ax.plot([0, 0], [r_a.mean(), r_a.mean()], color = C[0], label = r'acoustic')
    ax.plot([0, 0], [r_s.mean(), r_s.mean()], color = C[4], label = r'semantic')
    ax.plot([0, 0], [r_as.mean(), r_as.mean()], color = C[12], label = r'acoustic+semantic')
    
    ax.set_ylabel('Encoding performance\n' + r'(Pearson $r$)')
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(['baseline', 'invariant', 'specific', 'combined'] * 2, rotation = 45)

    pub.cosmetics.legend(ax = ax, loc = 'lower right')
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
        html += f'{contrast["contrast"]}: mu={contrast["mu"]}, sd={contrast["sd"]}, se={contrast["se"]}, lb={contrast["lb"]}, ub={contrast["ub"]}, t={contrast["t"]}, df={contrast["df"]}, p_cor={contrast["p_cor"]}, p_unc={contrast["p_unc"]}, d={contrast["d"]}<br />'
    html += '</pre>'
    report.add_html(title = 'Contrasts', html = html)
    
    '''
    Cluster-based permutation tests
    '''

    print(f'[ENC] Cluster-based permutation tests...')

    # setup progressbar
    n = 0
    N = ß_a.shape[1] * ß_a.shape[2]
    ts = time.time()

    # setup dummy
    cluster_based = {x: {f'm{m_i}': {f'ß{c_i}': [] for c_i in range(ß_a.shape[2])} for m_i in range(ß_a.shape[1])} for x in ['a', 's', 'as']}

    # loop over models
    for m_i in range(ß_a.shape[1]):
        fig, ax = pub.figure(nrows = 2, ncols = 4)
        
        caption_a = ''
        caption_s = ''
        caption_as = ''
        
        for c_i in range(ß_a.shape[2]):
            # update
            aux.progressbar(n, N, ts, msg = f'[ENC]')
            
            j, k = c_i // 4, c_i % 4
            
            ß_i = ß_a[:,m_i,c_i,:]
            mu = ß_i.mean(axis = 0)
            se = rsa.stats.bootstrap_se(ß_i)
            t = np.arange(mu.shape[0])
            
            ax[j,k].plot(np.zeros((t.shape[0],)), color = 'black', linewidth = 0.5)
            ax[j,k].fill_between(t, mu - 1.96 * se, mu + 1.96 * se, edgecolor = None, facecolor = C[0], alpha = 0.25)
            ax[j,k].plot(mu, color = C[0], label = r'acoustic')
            ax[j,k].set_title(fr'$ß_{{{c_i}}}$')
            
            sigs = 0
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                
                # compute stats
                t, c, p_unc, _ = mne.stats.permutation_cluster_1samp_test(ß_i, n_permutations = n_cbp, verbose = False)
                
                # NOTE: Since we are chosing only one model anyway,
                # here we correct within that model. Given that we 
                # cannot easily do a BH correction here, we're also
                # switching to Bonferroni.
                p_cor = np.clip(p_unc * ß_a.shape[2], 0, 1)
                
                # grab maximum
                absmax = np.abs([mu - 1.96 * se, mu + 1.96 * se]).max()
                
                # loop over clusters
                for ii, (cc, pp) in enumerate(zip(c, p_cor)):
                    if pp <= .05:
                        # compute descriptives
                        # NOTE: Here we compute a biased Cohen's d statistic, as
                        # per Eric Maris's suggestion in:
                        #   
                        #   https://mailman.science.ru.nl/pipermail/fieldtrip/2017-September/024644.html
                        # 
                        # This is not ideal, but it's the best we can do.
                        tmin, tmax = cc[0].min(), cc[0].max()
                        mu = ß_i[:,tmin:tmax].mean()
                        sd = ß_i[:,tmin:tmax].mean(axis = 1).std()
                        se = rsa.stats.bootstrap_se(ß_i[:,tmin:tmax].mean(axis = 1))
                        lb, ub = mu - 1.96 * se, mu + 1.96 * se
                        d = rsa.stats.cohens_d(ß_i[:,tmin:tmax].mean(axis = 1))[0]
                        
                        caption_a += f'ß_{c_i}: {tmin}-{tmax}, p_cor = {p_cor[ii]}, p_unc = {p_unc[ii]}, mu={mu}, sd={sd}, se={se}, lb={lb}, ub={ub}, d={d}\n'
                        cluster_based['a'][f'm{m_i}'][f'ß{c_i}'].append(dict(tmin = tmin, tmax = tmax, p_cor = p_cor[ii], p_unc = p_unc[ii], mu = mu, sd = sd, se = se, lb = lb, ub = ub, d = d, cluster = c))
                        
                        ax[j,k].plot([tmin, tmax], [absmax * (1.1 + sigs * 0.05), absmax * (1.1 + sigs * 0.05)], color = C[0])
                        
                        sigs += 1
            
            ß_i = ß_s[:,m_i,c_i,:]
            mu = ß_i.mean(axis = 0)
            se = rsa.stats.bootstrap_se(ß_i)
            t = np.arange(mu.shape[0])
            
            ax[j,k].plot(np.zeros((t.shape[0],)), color = 'black', linewidth = 0.5)
            ax[j,k].fill_between(t, mu - 1.96 * se, mu + 1.96 * se, edgecolor = None, facecolor = C[4], alpha = 0.25)
            ax[j,k].plot(mu, color = C[4], label = r'semantic')
            ax[j,k].set_title(fr'$ß_{{{c_i}}}$')
            
            sigs = 0
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                
                # compute stats
                t, c, p_unc, _ = mne.stats.permutation_cluster_1samp_test(ß_i, n_permutations = n_cbp, verbose = False)
                
                # NOTE: Since we are chosing only one model anyway,
                # here we correct within that model. Given that we 
                # cannot easily do a BH correction here, we're also
                # switching to Bonferroni.
                p_cor = np.clip(p_unc * ß_a.shape[2], 0, 1)
                
                # grab maximum
                absmax = np.abs([mu - 1.96 * se, mu + 1.96 * se]).max()
                
                # loop over clusters
                for ii, (cc, pp) in enumerate(zip(c, p_cor)):
                    if pp <= .05:
                        # compute descriptives
                        # NOTE: Here we compute a biased Cohen's d statistic, as
                        # per Eric Maris's suggestion in:
                        #   
                        #   https://mailman.science.ru.nl/pipermail/fieldtrip/2017-September/024644.html
                        # 
                        # This is not ideal, but it's the best we can do.
                        tmin, tmax = cc[0].min(), cc[0].max()
                        mu = ß_i[:,tmin:tmax].mean()
                        sd = ß_i[:,tmin:tmax].mean(axis = 1).std()
                        se = rsa.stats.bootstrap_se(ß_i[:,tmin:tmax].mean(axis = 1))
                        lb, ub = mu - 1.96 * se, mu + 1.96 * se
                        d = rsa.stats.cohens_d(ß_i[:,tmin:tmax].mean(axis = 1))[0]
                        
                        caption_s += f'ß_{c_i}: {tmin}-{tmax}, p_cor = {p_cor[ii]}, p_unc = {p_unc[ii]}, mu={mu}, sd={sd}, se={se}, lb={lb}, ub={ub}, d={d}\n'
                        cluster_based['s'][f'm{m_i}'][f'ß{c_i}'].append(dict(tmin = tmin, tmax = tmax, p_cor = p_cor[ii], p_unc = p_unc[ii], mu = mu, sd = sd, se = se, lb = lb, ub = ub, d = d, cluster = c))
                        
                        ax[j,k].plot([tmin, tmax], [absmax * (1.1 + sigs * 0.05), absmax * (1.1 + sigs * 0.05)], color = C[4])
                        
                        sigs += 1

            ß_i = ß_as[:,m_i,c_i,:]
            mu = ß_i.mean(axis = 0)
            se = rsa.stats.bootstrap_se(ß_i)
            t = np.arange(mu.shape[0])
            
            ax[j,k].plot(np.zeros((t.shape[0],)), color = 'black', linewidth = 0.5)
            ax[j,k].fill_between(t, mu - 1.96 * se, mu + 1.96 * se, edgecolor = None, facecolor = C[12], alpha = 0.25)
            ax[j,k].plot(mu, color = C[12], label = r'acoustic+semantic')
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                
                # compute stats
                t, c, p_unc, _ = mne.stats.permutation_cluster_1samp_test(ß_i, n_permutations = n_cbp, verbose = False)
                
                # NOTE: Since we are chosing only one model anyway,
                # here we correct within that model. Given that we 
                # cannot easily do a BH correction here, we're also
                # switching to Bonferroni.
                p_cor = np.clip(p_unc * ß_a.shape[2], 0, 1)
                
                # grab maximum
                absmax = np.abs([mu - 1.96 * se, mu + 1.96 * se]).max()
                
                # loop over clusters
                for ii, (cc, pp) in enumerate(zip(c, p_cor)):
                    if pp <= .05:
                        # compute descriptives
                        tmin, tmax = cc[0].min(), cc[0].max()
                        mu = ß_i[:,tmin:tmax].mean()
                        sd = ß_i[:,tmin:tmax].mean(axis = 1).std()
                        se = rsa.stats.bootstrap_se(ß_i[:,tmin:tmax].mean(axis = 1))
                        lb, ub = mu - 1.96 * se, mu + 1.96 * se
                        d = rsa.stats.cohens_d(ß_i[:,tmin:tmax].mean(axis = 1))[0]
                        
                        caption_as += f'ß_{c_i}: {tmin}-{tmax}, p_cor = {p_cor[ii]}, p_unc = {p_unc[ii]}, mu={mu}, sd={sd}, se={se}, lb={lb}, ub={ub}, d={d}\n'
                        cluster_based['as'][f'm{m_i}'][f'ß{c_i}'].append(dict(tmin = tmin, tmax = tmax, p_cor = p_cor[ii], p_unc = p_unc[ii], mu = mu, sd = sd, se = se, lb = lb, ub = ub, d = d, cluster = c))
                        
                        ax[j,k].plot([tmin, tmax], [absmax * (1.1 + sigs * 0.05), absmax * (1.1 + sigs * 0.05)], color = C[12])
                        
                        sigs += 1
            
            ax[j,k].set_xticks(np.arange(0, ß_a.shape[3]+1, 50))
            ax[j,k].set_xticklabels(np.round(np.arange(0, ß_a.shape[3]+1, 50)*5e-3, 2))
            ax[j,k].set_xlabel(r'Stimulus time (s)')
            ax[j,k].set_ylabel(r'$\beta$-coefficient (a.u.)')
            pub.cosmetics.legend(ax = ax[j,k], loc = 'lower left')
            
            n += 1
        
        pub.cosmetics.finish()
        
        report.add_figure(
            fig = fig,
            title = f'Cluster-based permutation tests in M={m_i}',
            caption = '',
            image_format = 'png',
        )
        
        plt.close()
        
        report.add_html(title = f'Cluster-based permutation results in M={m_i}', html = f'<b>ACOUSTIC</b><br /><pre>{caption_a}</pre><br /><b>SEMANTIC</b><br /><pre>{caption_s}</pre><br /><b>ACOUSTIC-SEMANTIC</b><br /><pre>{caption_as}</pre>')

    '''
    Export data
    '''
    print(f'')
    print(f'[ENC] Exporting data...')
    
    dir_out = './data/results/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    f = f'{dir_out}encoding_b{int(b_brain)}-m{int(b_morph)}-c{int(b_clear)}-k{int(n_topk)}.pkl.gz'
    with gzip.open(f, 'wb') as f:
        pickle.dump((r_a, ß_a, r_s, ß_s, r_as, ß_as, one_samp, contrasts, cluster_based), f)
    
    dir_out += 'reports/'
    if os.path.isdir(dir_out) == False: os.mkdir(dir_out)
    
    f = f'{dir_out}encoding_b{int(b_brain)}-m{int(b_morph)}-c{int(b_clear)}-k{int(n_topk)}.html'
    report.save(f, overwrite = True, verbose = False)