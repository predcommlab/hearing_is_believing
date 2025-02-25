'''
Collects data from exemplar task and outputs
it to:

    ./data/preprocessed/beh/all_mt2.csv

OPTIONS:
    None

EXAMPLE USAGE:
    python beh_preprocess_mt2.py
'''

import numpy as np
import pandas as pd
import mne, os, sys, copy
import auxiliary as aux
import data, rsa
import gzip, pickle, time
import warnings
import re

sys.path.append('../spaces/')
import embeddings as emb

from typing import Union, Dict

if __name__ == '__main__':
    '''
    Start data collection.
    '''
    
    # load included subjects
    subjects = data.Subjects.trim()
    
    # load glove
    G = emb.glove.load_embedding(f_in = f'./data/preprocessed/misc/glove/w2v_50D.txt') # load mini version for speed
    
    # prepare dataframe
    dfn = {'sid': [], 'trial_no': [], 'block': [], 'context': [], 'face': [], 'feature': [], 'stimulus': [], 'target': [], 'alternative': [], 'fit_t': [], 'fit_a': [], 'ufit_t': [], 'ufit_a': [], 'target_pos': [], 'is_bad': [], 'correct': [], 'rt': []}
    
    done = 0
    N = len(subjects) * 120
    ts = time.time()
    
    # loop over all subjects
    for sub in subjects:
        # grab id
        sid, pid = sub.sid, sub.pid
        
        # load data
        df = pd.read_csv(f'./data/raw/beh/sub{sid}/{pid}.csv')
        mt1 = df.loc[(df.type == data.defs.TRIAL_MT_MAIN) | (df.type == data.defs.TRIAL_MT_CONTROL) | (df.type == data.defs.TRIAL_MT_MAIN)].reset_index(drop = True)
        mt2 = df.loc[(df.type == data.defs.TRIAL_ET_GOOD) | (df.type == data.defs.TRIAL_ET_BAD)].reset_index(drop = True)
        
        # load last RTFE positions
        contexts = np.array(['essen', 'fashion', 'outdoor', 'technik', 'politik', 'unterhaltung'])
        context_pos = np.zeros((6, 50))
        last_con = mt1.context.tolist()[-1]

        with gzip.open(f'./data/raw/rtfe/sub{sid}/{pid}_t263_{last_con}.pkl.gz', 'rb') as f:
            _, _, _, posterior, _, _ = pickle.load(f)
            context_pos = posterior.T
        
        with gzip.open(f'./data/raw/rtfe-unspecific/sub{sid}/{pid}_t263_essen.pkl.gz', 'rb') as f:
            _, _, _, posterior, _, _ = pickle.load(f)
            context_pos_uns = posterior.T
        
        # loop over trials
        for i in range(len(mt2)):
            aux.progressbar(done, N, ts, msg = f'[{sid}]')
            
            trial = mt2.loc[i]
            
            # gather data
            no, block = i, trial.block
            context = trial.context
            pos = np.where(contexts == context)[0]
            true_speaker = trial.preload_speaker
            ts_re = re.compile(r'image=\'([^\']+)\'')
            true_speaker = ts_re.search(true_speaker).groups()[0].split('/')[-1].split('.')[0]
            face, feature = true_speaker.split('_')
            stimulus = trial.stimulus.split('/')[-1]
            target, alternative = trial.options_0, trial.options_1
            fit_t, fit_a = rsa.math.cosine(G[target.lower()][np.newaxis,:], context_pos[pos,:]).squeeze(), rsa.math.cosine(G[alternative.lower()][np.newaxis,:], context_pos[pos,:]).squeeze()
            ufit_t, ufit_a = rsa.math.cosine(G[target.lower()], context_pos_uns[0,:]).squeeze(), rsa.math.cosine(G[alternative.lower()], context_pos_uns[0,:])
            target_pos = trial.target_position
            is_bad = trial.type == data.defs.TRIAL_ET_BAD
            correct, rt = trial.correct, trial.rt
            
            # append to dataframe
            dfn['sid'].append(sid)
            dfn['trial_no'].append(no)
            dfn['block'].append(block)
            dfn['context'].append(context)
            dfn['face'].append(face)
            dfn['feature'].append(feature)
            dfn['stimulus'].append(stimulus)
            dfn['target'].append(target)
            dfn['alternative'].append(alternative)
            dfn['fit_t'].append(fit_t)
            dfn['fit_a'].append(fit_a)
            dfn['ufit_t'].append(ufit_t)
            dfn['ufit_a'].append(ufit_a)
            dfn['target_pos'].append(target_pos)
            dfn['is_bad'].append(is_bad)
            dfn['correct'].append(correct)
            dfn['rt'].append(rt)
            
            done += 1
    
    dfn = pd.DataFrame.from_dict(dfn)
    dfn.to_csv('./data/preprocessed/beh/all_mt2.csv')