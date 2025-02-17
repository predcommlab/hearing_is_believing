'''
Script to perform dimensionality reduction
over activation values from Wav2Vec2.0.

OPTIONS
    n_features  -   Number of features to use (default = 50)

EXAMPLE USAGE:
    python audio_w2v2_pca.py n_features=10
'''

import os, sys, glob, gzip, pickle
from typing import Any, Union, Callable

import numpy as np
import pandas as pd
import librosa
from sklearn.decomposition import PCA

import aux

if __name__ == '__main__':
    '''
    Start processing the audio.
    '''
    
    # get options
    n_features = aux.get_opt('n_features', default = 50, cast = int)    # number of dimensions in subspace
    
    # dump opts
    print(f'------- PCA: w2v2 --------')
    print(f'[OPTS]\tn_features={n_features}')
    print(f'--------------------------')
    
    # setup target folders
    path = f'./data/preprocessed/audio/w2v2/'
    folders = ['clear', 'morphed', 'narrative_12ch']
    
    print(f'[PCA] Subspace projection...')
    # load narrative folder
    for i, layer in enumerate(glob.glob(f'{path}narrative/*.pkl.gz')):
        suffix = '_'.join(layer.split('/')[-1].split('_')[-2:]).split('.')[0]
        layer_type = suffix.split('_')[0]
        if layer_type == 'narrative': 
            suffix = 'decoder'
            layer_type = 'decoder'
        
        print(f'{suffix}: PCA...\t\t\t', end = '\r')
        
        # load original activations
        with gzip.open(layer, 'rb') as f:
            z = pickle.load(f)

        # fit PCA
        eff_features = np.min([36, n_features]) if layer_type == 'decoder' else n_features
        pca = PCA(n_components = eff_features)
        if layer_type == 'conv': z_dec = pca.fit_transform(z.squeeze().T)
        else: z_dec = pca.fit_transform(z.squeeze())
        
        # save narrative data
        path_out = f'{path}reduced{n_features}_narrative/'
        if os.path.isdir(path_out) == False: os.mkdir(path_out)
        
        with open(f'{path_out}narrative_{suffix}.npy', 'wb') as f:
            np.save(f, z_dec)
        
        # loop over folders
        for folder in folders:
            path_out = f'{path}reduced{n_features}_{folder}/'
            if os.path.isdir(path_out) == False: os.mkdir(path_out)
            
            # loop over items
            for j, item in enumerate(glob.glob(f'{path}{folder}/*_{suffix}.pkl.gz')):
                print(f'{suffix}: {folder}{j}...\t\t\t', end = '\r')
                out = '.'.join(item.split('/')[-1].split('.')[0:-2])
                
                # load item
                with gzip.open(item, 'rb') as f:
                    z = pickle.load(f)
                
                # transform
                if layer_type == 'conv': z_dec = pca.transform(z.squeeze().T)
                else: z_dec = pca.transform(z.squeeze())
                
                # save data
                with open(path_out + out + '.npy', 'wb') as f:
                    np.save(f, z_dec)