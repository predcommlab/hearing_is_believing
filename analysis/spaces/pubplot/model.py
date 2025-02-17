import numpy as np
import pandas as pd
from typing import Union, Callable
from types import SimpleNamespace
import re

def read(f_in: str, index_by: int = 0, properties: Union[list, None] = None, terms: Union[list, None] = None) -> SimpleNamespace:
    '''
    '''
    
    # helper function for legalisation of labels
    def __rewrite__(label: str) -> str:
        '''
        '''
        
        rules = [dict(p = re.compile('[\*\:]+'), t = '_by_'),
                 dict(p = re.compile('[\|]+'), t = '_given_'),
                 dict(p = re.compile('[^a-zA-Zß_]'), t = '')]
        
        for rule in rules: label = rule['p'].sub(rule['t'], label)
        
        return label
    
    # read file
    if f_in.split('.')[-1] == 'csv': T = pd.read_csv(f_in)
    elif f_in.split('.')[-1] in ['xlsx', 'xls']: T = pd.read_excel(f_in)
    else: return {}
    
    # prepare output
    out = {}
    
    # prepare search
    cols = T.keys()
    key = cols[index_by]
    rows = T[key].tolist()
    
    # loop over rows we want as cols
    for j, row in enumerate(rows):
        # find label, rename if desired and make appropriate
        if terms is not None and j < len(terms): rowlab = terms[j]
        else: rowlab = row
        rowlab = __rewrite__(rowlab)
        
        # add to output dict if necessary
        if not hasattr(out, rowlab): out[rowlab] = {}
        
        # loop over cols we want as rows
        i = 0
        for col in cols:
            # if same, move on
            if col == key: continue
            
            # replace current label with desired label
            if properties is not None and i < len(properties): label = properties[i]
            else: label = col
            
            # make valid label and entry
            label = __rewrite__(label)
            out[rowlab][label] = T.loc[np.where(np.array(rows) == row)][col].tolist()[0]
            
            i += 1
    
    # convert to simple name spaces for nice dot syntax
    for key in out: out[key] = SimpleNamespace(**out[key])
    out = SimpleNamespace(**out)
    
    return out

def lm(parameters: list[tuple[float, Union[np.ndarray, float]]], link: Union[Callable, None] = None) -> np.ndarray:
    '''
    '''
    
    # compute linear model
    y_hat = np.array([np.dot(parameters[i][0], parameters[i][1]) for i in np.arange(0, len(parameters), 1)]).sum(axis = 0)
    
    # transform using link function if required
    y_hat = y_hat if not callable(link) else link(y_hat)
    
    return y_hat
    