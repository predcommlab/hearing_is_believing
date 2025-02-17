'''
embeddings::model_auxiliary.py

Auxiliary functions for model fitting (e.g., time steppers
for solving ODEs). Available as embeddings.model.*
'''

from .internal import *
import numpy as np
from typing import Any, Callable, Union

def rk4(dydt: Callable, y: np.ndarray, dt: float, *args: Union[Any, None]) -> np.ndarray:
    '''
    '''

    k1 = dydt(y, dt, *args)
    k2 = dydt(y + dt/2*k1, dt, *args)
    k3 = dydt(y + dt/2*k2, dt, *args)
    k4 = dydt(y + dt*k3, dt, *args)

    return y + dt*6**-1 * (k1 + 2*k2 + 2*k3 + k4)

def euler(dydt: Callable, y: np.ndarray, dt: float, *args:  Union[Any, None]) -> np.ndarray:
    '''
    '''

    return y + dt*dydt(y, dt, *args)