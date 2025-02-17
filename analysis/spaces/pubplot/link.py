import numpy as np
from typing import Any

def logit(x: np.ndarray) -> np.ndarray:
    '''
    '''
    
    return 1 / (1 + np.exp(-x))