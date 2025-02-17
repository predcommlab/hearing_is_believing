import numpy as np
import pandas as pd
import os

from trial import Trial

class Dataset:
    '''
    '''

    def __init__(self, pid: str, path: str):
        '''
        '''

        # set paths
        self.pid = pid
        self.path = path
        self.out = os.path.join(self.path, pid + '.csv')
        
        # set trial structure
        self.trials = {}
    
    def write(self, T: Trial):
        '''
        '''

        # loop over trial keys and add data
        for k in T.__dict__.keys():
            # get value
            v = T.__getattribute__(k)

            # unpack, if required
            if type(v) in [list, tuple]:
                for i in np.arange(0, len(v), 1):
                    if f'{k}_{i}' not in self.trials: self.trials[f'{k}_{i}'] = []
                    v_i = v[i] if v[i] is not None else 'N/A'
                    self.trials[f'{k}_{i}'].append(v_i)
            else:
                if k not in self.trials: self.trials[k] = []
                v_i = v if v is not None else 'N/A'
                self.trials[k].append(v_i)
        
        # add index
        self.trials['index'] = np.arange(0, len(self.trials[k]), 1)
        
        # write to csv
        df = pd.DataFrame.from_dict(self.trials)
        df.to_csv(self.out, encoding = 'utf-8')
