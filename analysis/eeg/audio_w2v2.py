'''
Script to obtain layer-specific activations
from Wav2Vec2.0-large-xlsr-53-german for all audio
files in a given directory.

NOTE: Running this model is quite memory intensive
if you want to run it over the full narrative data.
At the very least, you should have 20GB+ of memory
available (preferably 40GB).

NOTE: Here, we use the CTC version of the model to
also obtain the final decoder activations because
this model was trained on German audio.

OPTIONS
    folder      -   Input folder
    device      -   Device to use (cpu/mps/cuda, default = None)

EXAMPLE USAGE:
    python audio_w2v2.py folder=narrative device=cuda
'''

import os, sys, glob, gzip, pickle
from typing import Any, Union, Callable

import librosa
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import aux

if __name__ == '__main__':
    '''
    Start processing the audio.
    '''
    
    # get options
    folder = aux.get_opt('folder', default = 'clear', cast = str)           # folder name
    device = aux.get_opt('device', default = None, cast = lambda x: x)      # device to use
    
    # dump opts
    print(f'------- ACT: w2v2 --------')
    print(f'[OPTS]\tfolder={folder}\tdevice={device}')
    print(f'--------------------------')
    
    # setup target folders
    path_in = f'./data/raw/audio/{folder}/'
    path_out = f'./data/preprocessed/audio/w2v2/{folder}/'
    
    # make sure output folder exists
    assert(os.path.isdir(path_in))
    if os.path.isdir(path_out) == False: os.mkdir(path_out)
    
    # load model
    print(f'[PRE] Loading model...')
    if device == None: 
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[PRE] Switching to `{device}`.')

    model_name = "facebook/wav2vec2-large-xlsr-53-german"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    
    # add hooks to model to extract activations
    print(f'[PRE] Creating hooks...')
    out = ''
    def make_hook(name: str) -> Callable:
        def hook(model, input, output):
            f_out = f'{path_out}{out}_{name}.pkl.gz'
            
            with gzip.open(f_out, 'wb') as f:
                pickle.dump(output.cpu().numpy(), f)
            
        return hook

    for i, layer in enumerate(model.wav2vec2.feature_extractor.conv_layers):
        layer.register_forward_hook(make_hook(f'conv_L{i}'))
    
    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        layer.final_layer_norm.register_forward_hook(make_hook(f'transformer_L{i}'))
    
    model.lm_head.register_forward_hook(make_hook('decoder'))
    
    print(f'[ACT] Processing data...')
     # loop over items
    for i, path in enumerate(glob.glob(f'{path_in}/*.wav')):
        print(f'{folder}: {i}.\t\t', end = '\r')
        out = '.'.join(path.split('/')[-1].split('.')[0:-1])
        
        # prepare data
        y, sr = librosa.load(path, sr = 16000) # Wav2Vec2 requires 16kHz
        inputs = processor(y, sampling_rate = sr, do_normalize = True, return_tensors = 'pt').to(device) # convert inputs
        
        # perform forward pass
        with torch.no_grad():
            outputs = model(**inputs)