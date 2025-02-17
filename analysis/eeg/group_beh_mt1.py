'''
Collects all behavioural data from the
learning task and outputs it to:

    ./data/preprocessed/beh/all_mt1.csv

OPTIONS:
    None

EXAMPLE USAGE:
    python beh_preprocess_mt1.py
'''

import numpy as np
import pandas as pd
import mne, os, sys, copy
import aux, data, rsa
import gzip, pickle, time
import warnings
import re

sys.path.append('../spaces/')
import embeddings as emb

from typing import Union, Dict

if __name__ == '__main__':
    '''
    Start data  collection.
    '''
    
    # load included subjects
    subjects = data.Subjects.trim()
        
    # load glove
    G = emb.glove.load_embedding(f_in = f'./data/preprocessed/misc/glove/w2v_50D.txt') # load mini version for speed
    
    # prepare dataframe
    dfn = {'sid': [], 'trial_no': [], 'block': [], 'context': [], 'face': [], 'feature': [], 'stimulus': [], 'word_a': [], 'word_b': [], 'kappa': [], 'fit_a': [], 'fit_b': [], 'a_pos': [], 'a_is_target': [], 'chose_a': [], 'chose_context': [], 'rt': []}
    
    done = 0
    N = len(subjects) * 240
    ts = time.time()
    
    # loop over all subjects
    for sub in subjects:
        # grab id
        sid, pid = sub.sid, sub.pid
        
        # load data
        dfk = pd.read_excel(f'../spaces/dumps/stimuli_pairs_from_old.xlsx')
        df = pd.read_csv(f'./data/raw/beh/sub{sid}/{pid}.csv')
        mt1 = df.loc[(df.type == data.defs.TRIAL_MT_MAIN) | (df.type == data.defs.TRIAL_MT_CONTROL) | (df.type == data.defs.TRIAL_MT_PRACTICE)].reset_index(drop = True)
        
        # load last RTFE positions
        contexts = np.array(['essen', 'fashion', 'outdoor', 'technik', 'politik', 'unterhaltung'])
        context_pos = np.zeros((6, 50))
        
        # loop over trials
        for i in range(len(mt1)):
            aux.progressbar(done, N, ts, msg = f'[{sid}]')
            
            trial = mt1.loc[i]
            
            # gather basics
            no, block = i, trial.block
            context = trial.context
            pos = np.where(contexts == context)[0]
            pos_rtfe = context_pos[pos,:]
            
            # update rtfe values
            with gzip.open(f'./data/raw/rtfe/sub{sid}/{pid}_t{i-6}_{context}.pkl.gz', 'rb') as f:
                _, _, _, posterior, _, _ = pickle.load(f)
                context_pos[pos[0],:] = posterior[:,pos[0]].T
            
            # do we need to skip this trial?
            if trial.type in [data.defs.TRIAL_MT_CONTROL, data.defs.TRIAL_MT_PRACTICE]: continue
            
            # gather stimulus features
            speaker = trial.speaker
            face, feature = speaker.split('_')
            stimulus = trial.stimulus.split('/')[-1]
            target, alternative = trial.options_0, trial.options_1
            
            # find word_a & word_b
            word_a, word_b = stimulus.split('_')[0].split('-')
            word_a = word_a if word_a != 'warten' else 'waten'
            word_b = word_b if word_b != 'warten' else 'waten'
            r = int(stimulus.split('_')[4][1])
            a_is_target = word_a.lower() == target.lower()
            
            # find kappa
            indx = np.where((dfk.target1.str.lower() == word_a.lower()) & (dfk.target2.str.lower() == word_b.lower()))[0]
            kappa = dfk.loc[indx].f1_k if r == 1 else dfk.loc[indx].f2_k
            kappa = kappa.tolist()[0]
            
            # find other descriptors
            fit_a, fit_b = rsa.math.cosine(G[word_a.lower()][np.newaxis,:], pos_rtfe).squeeze(), rsa.math.cosine(G[word_b.lower()][np.newaxis,:], pos_rtfe).squeeze()
            a_pos = trial.target_position if a_is_target else 'down' if (trial.target_position == 'up') else 'up'
            correct, rt = trial.correct, trial.rt
            chose_a = correct if a_is_target else correct == False
            
            # append to dataframe
            dfn['sid'].append(sid)
            dfn['trial_no'].append(no)
            dfn['block'].append(block)
            dfn['context'].append(context)
            dfn['face'].append(face)
            dfn['feature'].append(feature)
            dfn['stimulus'].append(stimulus)
            dfn['word_a'].append(word_a)
            dfn['word_b'].append(word_b)
            dfn['kappa'].append(kappa)
            dfn['fit_a'].append(fit_a)
            dfn['fit_b'].append(fit_b)
            dfn['a_pos'].append(a_pos)
            dfn['a_is_target'].append(a_is_target)
            dfn['chose_a'].append(chose_a)
            dfn['chose_context'].append(correct)
            dfn['rt'].append(rt)
            
            done += 1
    
    dfn = pd.DataFrame.from_dict(dfn)
    dfn.to_csv('./data/preprocessed/beh/all_mt1.csv')