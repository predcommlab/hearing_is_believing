'''
embeddings::internal.py

Internal functions that embeddings.py depends on. Should not be called
directly.
'''

def critical(ref: str = 'unknown::unknown', msg: str = 'Unspecified critical error encountered.'):
    '''
    Presents a critical error described in `msg` that occurred in `ref`.
    '''
    
    print(f'\nError in {ref}: {msg}')

def warning(ref: str = 'unknown::unknown', msg: str = 'Unspecified warning encountered.'):
    '''
    Presents a warning described in `msg` that occurred in `ref`.
    '''
    
    print(f'\nWarning in {ref}: {msg}')

def message(ref: str = 'unknown::unknown', msg: str = 'Unspecified warning encountered.', terminate: bool = True):
    '''
    Presents a message described in `msg` that occurred in `ref`.
    '''
    
    if terminate: print(f'\nMessage in {ref}: {msg}')
    else: print(f'Message in {ref}: {msg}', end = '\r')