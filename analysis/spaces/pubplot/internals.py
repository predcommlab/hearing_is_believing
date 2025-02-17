import sys

def critical(ref: str = 'pubplot::Unknown', msg: str = 'An unknown critical error was encountered.'):
    '''
    '''
    
    sys.exit(f'[{ref}]: {msg}')

def warning(ref: str = 'pubplot::Unknown', msg: str = 'An unknown warning arose.'):
    '''
    '''
    
    print(f'[{ref}]: {msg}')
    