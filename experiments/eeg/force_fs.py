import librosa
import soundfile as sf
import sys, os, glob

from typing import Union

def get_opt(opt: str, default: Union[bool, str] = False) -> Union[bool, str]:
    '''
    '''

    for k in sys.argv[1:]:
        if(k[0:len(opt)] == opt): return k[len(opt)+1:]
    
    return default

if __name__ == '__main__':
    # setup
    sr = int(get_opt('sr', default = 48000))
    folder = str(get_opt('folder', default = None))
    if folder is None: exit()
    else: folder = folder + '*.wav'
    targets = glob.glob(folder)

    # loop over folder
    for i, target in enumerate(targets):
        print(f'{i+1}/{len(targets)}')
        x, fs = librosa.load(target, sr = sr)
        sf.write(target, x, fs)