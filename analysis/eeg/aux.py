'''
Utilities for the console.
'''

import sys, time
import numpy as np

from typing import Any, Union, Callable

def timestamp(s: float) -> str:
    '''
    '''
    
    hrs = int(s // 3600)
    mns = int((s % 3600) / 60)
    scs = int((s % 3600) % 60)
    
    return f'{hrs:02d}h:{mns:02d}m:{scs:02d}s'

def progressbar(i: int, N: int, ts: float, bars: int = 40, bar_chr: str = '-', emp_chr: str = ' ', msg: str = '') -> None:
    '''
    '''
    
    i += 1
    p = np.round(i / N * 100, 2)
    b = np.round(p / 100 * bars).astype(int)
    p1 = np.round((N - i) / N * 100, 2)
    f = p1 / p
    te = (time.time() - ts) * f
    
    bstr = ''.join([(bar_chr if i < b else emp_chr) for i in np.arange(0, bars, 1)])
    tstr = timestamp(np.round(time.time() - ts).astype(int))
    estr = timestamp(np.round(te).astype(int))
    cstr = ''.join([' '] * bars)
    
    print(f'{msg} [{bstr}] {p:03.02f}% in {tstr} (ETA: {estr}).{cstr}', end = '\r')

def get_opt(opt: str, default: Any = False, cast: Callable = bool) -> Any:
    '''
    Retrieve arguments from console.
    
    INPUTS:
        opt     -   Key to look for.
        default -   Default value if missing.
    
    OUTPUTS:
        value   -   Value of key (or default).
    '''
    
    for k in sys.argv[1:]:
        if (k[0:len(opt)] == opt) and ((len(k) == len(opt)) or (k[len(opt)] == '=')): return cast(k[len(opt)+1:])
    
    return cast(default)

def multi_input(exit: str = 'exit', follow_up: Union[None, str] = None, allowed_inputs: Union[None, list[str]] = None) -> tuple[list[str], list[str]]:
    '''
    Prompt for input (with potential follow up question to prompt).
    
    INPUTS:
        exit            -   Which term to use to exit?
        follow_up       -   What, if any, follow-up question should be asked?
        allowed_inputs  -   What are the allowed input strings?
    
    OUTPUTS:
        (selection,     -   Array of items selected.
         answers)       -   Array of follow-up answers, if any.
    '''
    
    selection = []
    answers = []
    
    while True:
        inp = input('...')
        if inp == exit: break
        if allowed_inputs is not None and inp not in allowed_inputs: print(f'...illegal input `{inp}`.'); continue
        if inp not in selection: selection.append(inp)
        if follow_up is not None: answers.append(input(follow_up))

    return (selection, answers)