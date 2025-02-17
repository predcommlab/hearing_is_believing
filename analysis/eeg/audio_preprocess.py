'''
Script to preprocess audio data in several ways:
In any case, this will compute gammatone spectro-
grams over all audio data (at a desired sampling
frequency `fs`). Depending on the method, it will
then either compute the speech envelope or simply
the full spectrum (`method`).

Note that this relies on `gammatone` which ports
MATLAB's GT filterbank to python. For more in-
formation, please refer to:

    https://github.com/detly/gammatone

OPTIONS
    f_min       -   Minimum frequency to consider (default = 50)
    chns        -   Number of channels (default = 28)
    fs          -   Output sampling frequency (default = 1e3)
    fsn         -   Output sampling frequency name (default = '1k')
    method      -   Should we compute envelopes or spectrograms? (envelope/spectrogram, default = spectrogram)

EXAMPLE USAGE:
    python audio_preprocess.py fs=200 fsn=200 method=spectrogram
'''

import aux
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../spaces/')

import rsa
import gammatone as gt
import gammatone.gtgram
import soundfile as sf
import librosa
import glob

from typing import Any, Union

def spectrogram(f_in: str, f_out: str, fs: float, f_min: int, chns: int):
    '''
    Extracts the spectrogram from `f_in` with a minimum
    frequency of `f_min`, `chns` number of channels, and
    an output sampling frequency of `fs` to `f_out`.
    
    INPUTS:
        f_in    -   Input file.
        f_out   -   Output file.
        fs      -   Desired sampling frequency.
        f_min   -   Lowest frequency.
        chns    -   Number of channels.
    '''
    
    x, orig_fs = librosa.load(f_in)
    y = gammatone.gtgram.gtgram(x, orig_fs, 1 / fs, 1 / fs, chns, f_min)
    
    with open(f_out, 'wb') as f:
        np.save(f, y)

def envelope(f_in: str, f_out: str, fs: float, f_min: int, chns: int):
    '''
    Extracts the speech envelope from `f_in`. To do this,
    we compute the spectrogram (using lowest frequency
    `f_min`, number of channels `chns` and desired out-
    put frequency `fs`) and then exponentiate the abs-
    olute value with 0.6 before averaging over channels.
    
    INPUTS:
        f_in    -   Input file.
        f_out   -   Output file.
        fs      -   Desired sampling frequency.
        f_min   -   Lowest frequency.
        chns    -   Number of channels.
    '''
    
    x, orig_fs = librosa.load(f_in)
    y = gammatone.gtgram.gtgram(x, orig_fs, 1 / fs, 1 / fs, chns, f_min)
    y = (np.abs(y) ** 0.6).mean(axis = 0)
    
    with open(f_out, 'wb') as f:
        np.save(f, y)

if __name__ == '__main__':
    '''
    Start processing the audio.
    '''
    
    # get options
    f_min = aux.get_opt('fmin', default = 50, cast = int)                   # minimum frequency
    chns = aux.get_opt('chns', default = 28, cast = int)                    # number of channels
    fs = aux.get_opt('fs', default = 1e3, cast = float)                     # output sampling frequency
    fsn = aux.get_opt('fsn', default = '1k', cast = str)                    # output sampling frequency name
    method = aux.get_opt('method', default = 'spectrogram', cast = str)     # should we compute envelopes or spectrograms? (envelope/spectrogram)
    
    # dump opts
    print(f'------- GTG: audio -------')
    print(f'[OPTS]\tf_min={f_min}\tchns={chns}')
    print(f'[OPTS]\tfs={fs}\tfsn={fsn}')
    print(f'[OPTS]\tmethod={method}')
    print(f'--------------------------')
    
    # setup target folders
    path_in = './data/raw/audio/'
    path_out = './data/preprocessed/audio/'
    paths = ['clear', 'morphed', 'control', 'narrative', 'vocoded']
    f = spectrogram if method == 'spectrogram' else envelope
    
    # make sure output folder exists
    print(f'[GTG] Computing gammatone {method}...')
    fout0 = f'{path_out}gt-{fsn}Hz/'
    if os.path.isdir(fout0) == False: os.mkdir(fout0)
    
    # loop over folders
    for folder in paths:
        # make sure sub-folder exists
        fout1 = f'{fout0}{folder}/'
        if os.path.isdir(fout1) == False: os.mkdir(fout1)
        
        # loop over items
        for i, path in enumerate(glob.glob(f'{path_in}{folder}/*.wav')):
            print(f'{folder}: {i}.\t\t', end = '\r')
            out = fout1 + '.'.join(path.split('/')[-1].split('.')[:-1]) + '.npy'
            f(path, out, fs, f_min, chns)