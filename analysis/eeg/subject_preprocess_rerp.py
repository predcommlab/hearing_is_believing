'''
Script to run a relatively bare preprocessing procedure for a
given subject. Essentially, this will read the raw data, set the
relevant info, apply a notch filter for line noise (and harmonics),
apply demeaning to channels, remove excessively noisy single trials
that might poorly condition the ICA, apply a bandpass filter, epoch 
the data, select bad channels (and interpolate them), and compute
an ICA over the full data set that is used to unmix passive listening
and learning tasks. Finally, the resulting trials are inspected for 
excessive remaining noise (and flagged).

For mininmal usage, please supply participant id `pid` and session
id `sid`. All other parameters are optional (and defaults are set 
appropriately).
'''

import numpy as np
import pandas as pd
import mne
import os, sys
import auxiliary as aux
import data, rsa
import matplotlib.pyplot as plt

sys.path.append('../spaces/')
import pubplot as pub
pub.styles.set()
C = pub.colours.equidistant('tab20c', k = 20)

from typing import Any, Union

def get_bad_trials(Z: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    '''
    Visual routine to help identify excessively
    noisy trials. Essentially, this takes the
    absolute maximum Z score values over time,
    plots the average per trial and lets you
    chose a threshold to use for flagging.
    
    INPUTS:
        Z           -   Matrix from rsa.signal.maxZ
        threshold   -   Base level threshold (default = 5.0)
    
    OUTPUTS:
        bads        -   Indices of bad trials
    '''
    
    while True:
        fig, ax = pub.components.figure()
        indc = np.arange(0, Z.shape[0], 1)
        bads = Z.mean(axis = 1) >= threshold
        goods = Z.mean(axis = 1) < threshold
        
        ax.scatter(indc[goods], Z.mean(axis = 1)[goods], color = C[8], marker = '.', s = 3)
        ax.scatter(indc[bads], Z.mean(axis = 1)[bads], color = C[4], marker = '.', s = 3)
        ax.set_title(f'Removing {bads.sum()}/{Z.shape[0]} with thr={threshold}')
        pub.cosmetics.finish()
        plt.show()
        
        inp = input('...')
        if inp == 'exit': break
        
        try: 
            cast = float(inp)
            threshold = cast
        except:
            print(f'...`{inp}` cannot be converted to float threshold.')
    
    bads = Z.mean(axis = 1) >= threshold
    
    return np.where(bads)[0]

if __name__ == '__main__':
    '''
    Start preprocessing
    '''
    
    # grab participant
    id = aux.get_opt('id', cast = str)      # identifier (either sid or pid)
    subject = data.Subjects[id]             # load subject
    sid, pid = subject.sid, subject.pid     # get session & participant id
    
    # filter options
    bpl = aux.get_opt('bpl', default = 0.5, cast = float)           # bandpass lower cut-off frequency
    bph = aux.get_opt('bph', default = 15.0, cast = float)          # bandpass upper cut-off frequency
    bpl_m = aux.get_opt('bpl_m', default = 110.0, cast = float)     # bandpass lower cut-off frequency (muscle)
    bph_m = aux.get_opt('bph_m', default = 140.0, cast = float)     # bandpass upper cut-off frequency (muscle)
    bpm = aux.get_opt('bpm', default = 'iir', cast = str)           # bandpass filter method
    bpo = aux.get_opt('bpo', default = 1, cast = int)               # bandpass filter order
    bpf = aux.get_opt('bpf', default = 'butter', cast = str)        # bandpass filter type
    notch = aux.get_opt('notch', default = 50.0, cast = float)      # notch filter frequency
    harmonics = aux.get_opt('harmonics', default = 10, cast = int)  # number of notch filter harmonics
    
    # ICA options
    ica_tmin = aux.get_opt('ica_tmin', default = -0.5, cast = float)    # epoched segments beginning (relative to event)
    ica_tmax = aux.get_opt('ica_tmax', default = 2.5, cast = float)     # epoched segments end (relative to event)
    
    # dump opts
    print(f'------ PREPROCESSING ------')
    print(f'[OPTS]\tsid={sid}\tpid={pid}')
    print(f'[OPTS]\tnotch={notch}\tharmonics={harmonics}')
    print(f'[OPTS]\tbandpass=[{bpm} {bpo}-order {bpf} {bpl}-{bph}Hz]')
    print(f'[OPTS]\tmuscle=[{bpm} {bpo}-order {bpf} {bpl_m}-{bph_m}Hz]')
    print(f'[OPTS]\tica=[{ica_tmin}-{ica_tmax}s]')
    print(f'--------------------------')
    
    # setup paths
    path_in = f'./data/raw/eeg/sub{sid}/'
    path_out = f'./data/preprocessed/eeg/sub{sid}/'
    
    # load raw eeg data
    print('[RAW] Loading and preparing data...')
    raw = mne.io.read_raw_brainvision(path_in + f'sempriors{sid}.vhdr', verbose = False).load_data()
    
    # apply fibre correction (if required)
    if subject.fibre_correction:
        # as explained elsewhere, one of the researchers did a poor job rewiring the
        # fibre cables, leading to the two fibres being swapped. here, we correct
        # for these cases by switching them back.
        
        print('[FIB] Correcting rewired fibres...')
        desired_chs = np.array(raw.info['ch_names'])
        current_chs = [raw.info['ch_names'][i] for i in np.hstack((np.arange(32, 64, 1), np.arange(0, 32, 1)))]
        raw.rename_channels({k: v for k, v in zip(desired_chs, current_chs)})
        raw = raw.reorder_channels(desired_chs)
    
    # define EEG/EOG channels
    eogs = ['VEOG1', 'VEOG2', 'HEOG1', 'HEOG2']
    types = ['eeg'] * len(raw.info['ch_names'])
    for i, name in enumerate(raw.info['ch_names']):
        if name in eogs: types[i] = 'eog'
    raw.set_channel_types(dict(zip(raw.info['ch_names'], types)), verbose = False)
    
    # set 10-20 montage
    raw.set_montage('standard_1020', verbose = False)
    
    # apply notch filter for line noise and harmonics
    print('[FIL] Applying notch filter(s)...')
    raw = raw.notch_filter(freqs = [i * notch for i in np.arange(1, harmonics, 1)], verbose = False)
    
    # apply channel-wise demeaning
    print('[FIL] Applying demeaning procedure...')
    data = raw.copy().apply_function(lambda x: x - x.mean(), verbose = False)
    
    # apply bandpass filter
    print('[FIL] Applying bandpass filters...')
    data_m = data.copy().filter(l_freq = bpl_m, h_freq = bph_m, method = bpm, iir_params = dict(order = bpo, ftype = bpf), verbose = False)
    data_f = data.copy().filter(l_freq = bpl, h_freq = bph, method = bpm, iir_params = dict(order = bpo, ftype = bpf), verbose = False)
    
    # grab event timings across data
    print('[EVT] Finding event data...')
    triggers, desc = mne.events_from_annotations(data_f, verbose = False)
    onsets = np.where(triggers[:,2] == 1)[0]
    
    # find triggers for relevant tasks
    mt2 = triggers[onsets[523]:,:]
    
    # create event tuples
    events_mt2 = (mt2, desc)
    
    # create event transcriptions
    event_names = {'fixation': 1, 'cue': 2, 'audio_on': 3, 'audio_off': 4, 'response_on': 5, 'response_off': 6}

    # cut epochs over full data for ICA
    print('[EPS] Preparing epochs for ICA...')
    epochs_ica = mne.Epochs(data_f, mt2, tmin = ica_tmin, tmax = ica_tmax, event_id = event_names, baseline = None, preload = False, on_missing = 'ignore', verbose = False)
    
    # identify bad channels (requires user input!)
    epochs_ica.load_data()['audio_on'].plot()
    print(f'[CHS] Please jot down bad channels (by name). `exit` to finish.')
    bads, _ = aux.multi_input(allowed_inputs = epochs_ica.info['ch_names'])
    bads = np.array(bads)
    
    # make sure out path exists
    if os.path.isdir(path_out) == False: os.mkdir(path_out)
    
    # save bad channels
    with open(path_out + 'rerp_bads.npy', 'wb') as f:
        np.save(f, bads)
    
    # identify bad segments (requires user input!)
    print(f'[MSC] Please jot down the desired muscle threshold. `exit` to finish.')
    epochs_mt2 = mne.Epochs(data_m, mt2, tmin = ica_tmin, tmax = ica_tmax, event_id = event_names, baseline = None, preload = False, on_missing = 'ignore', verbose = False)
    
    epochs_mt2.info['bads'] = list(bads)
    
    epochs_mt2 = epochs_mt2.load_data().copy().pick(picks = 'eeg', verbose = False)
    if len(bads) > 0: epochs_mt2 = epochs_mt2.interpolate_bads(reset_bads = True, verbose = False)
    
    Z_mt2 = rsa.signal.maxZ(epochs_mt2['audio_on'].get_data())
    bads_mt2 = get_bad_trials(Z_mt2)
    
    with open(path_out + 'rerp_badtrials_ica.npy', 'wb') as f:
        np.save(f, bads_mt2)
    
    # interpolate bad channels before ICA
    print('[CHS] Interpolating bad channels for ICA data...')
    epochs_ica.info['bads'] = list(bads)
    interp_ica = epochs_ica.load_data().copy().pick(picks = 'eeg', verbose = False)
    if len(bads) > 0: interp_ica = interp_ica.interpolate_bads(reset_bads = True, verbose = False)
    
    # remove bad trials before ICA
    print(f'[MSC] Removing bad trials from ICA data...')
    interp_ica = interp_ica['audio_on']
    bad_trials = bads_mt2
    interp_ica = interp_ica.drop(bad_trials)
    
    # compute ICA
    print('[ICA] Fitting ICA...')
    ica = mne.preprocessing.ICA(n_components = 60 - len(bads), max_iter = 'auto')
    ica.fit(interp_ica)
    
    # save ICA
    ica.save(path_out + 'rerp-ica.fif', overwrite = True)
    
    # identify ICA components
    ica.plot_sources(interp_ica)
    print('[ICA] Please jot down the bad components (by no). `exit` to finish.')
    bad_components, descriptions = aux.multi_input(allowed_inputs = [str(i) for i in np.arange(0, ica.n_components_, 1)], follow_up = '...Reasoning: ')
    bad_components = np.array([int(comp) for comp in bad_components])
    
    # save bad components
    with open(path_out + 'rerp_badcomponents.npy', 'wb') as f:
        np.save(f, bad_components)
        np.save(f, descriptions)
    
    # epoch individual task data
    print('[EPS] Epoching task data...')
    epochs_mt2 = mne.Epochs(data_f, mt2, tmin = -0.5, tmax = 2.0, event_id = event_names, baseline = None, preload = False, on_missing = 'ignore', verbose = False)

    # set bad channel info
    print('[EPS] Interpolating bad channels in task data...')
    epochs_mt2.info['bads'] = list(bads)

    # interpolate channels
    interp_mt2 = epochs_mt2.load_data().copy().pick(picks = 'eeg', verbose = False)
    if len(bads) > 0: interp_mt2 = interp_mt2.interpolate_bads(reset_bads = True, verbose = False)

    # setup ICA mixing
    print('[EPS] Removing bad ICA components from task data...')
    ica.exclude = list(bad_components)
    
    # apply ICA
    pp_mt2 = ica.apply(interp_mt2.copy()['audio_on'], verbose = False)

    # identify bad trials
    print('[EPS] Please jot down the desired threshold for bad trials...')
    Z_mt2 = rsa.signal.maxZ(pp_mt2.get_data())
    bads_mt2 = get_bad_trials(Z_mt2)
    
    with open(path_out + 'rerp_badtrials.npy', 'wb') as f:
        np.save(f, bads_mt2)
    
    # save cleaned data
    print('[EPS] Exporting data...')
    pp_mt2.save(path_out + 'rerp-MT2-epo.fif', overwrite = True, verbose = False)