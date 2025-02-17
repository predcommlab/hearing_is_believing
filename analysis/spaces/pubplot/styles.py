import sys
import matplotlib
import pandas as pd
import pkgutil
from io import StringIO
from .internals import *

# grab module
__pub = sys.modules['pubplot']

def set(style: str = 'default', encoding: str = 'utf-8', sep: str = ';'):
    '''
    '''
    
    assert(type(style) is str) or critical(ref = 'pubplot::styles::set()', msg = f'`style` must be of type string.')
    
    # load data
    fdat = pkgutil.get_data(__name__, './resources/' + style + '.csv').decode(encoding)
    
    # set within module
    __pub.style = style
    
    # read style sheet
    S = pd.read_csv(StringIO(fdat), sep = sep)
    __pub.opts = dict()
    
    # add defaults
    for k, v in zip(S.parameter.tolist(), S.value.tolist()):
        if k[0:6] == '__pub_': __pub.opts[k[6:]] = v
        else: matplotlib.rcParams[k] = v
    

def reset():
    '''
    '''
    
    matplotlib.rcdefaults()