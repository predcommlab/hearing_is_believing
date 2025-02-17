import matplotlib.pyplot as plt
import numpy as np
from typing import Union

def equidistant(cmap: Union[str, object] = 'tab20c', k: int = 20) -> np.ndarray:
    '''
    '''
    
    return plt.get_cmap(cmap)(np.linspace(0, 1, k))[:,0:3]