import os, collections
import numpy as np

from dataclasses import dataclass
from typing import Any, Union, Callable, Dict

# enum definitions
SEX_MALE: str = 'm'
SEX_FEMALE: str = 'f'
SEX_NONBINARY: str = 'nb'
SEX_UNDISCLOSED: int = '-'

# enum definitions
TYPE_PILOT: int = 0
TYPE_REGULAR: int = 1

# enum definitions
STATUS_EXCLUDE: int = 0
STATUS_INCLUDE: int = 1

# enum definitions
ELECTRODES_S1: int = 1
ELECTRODES_S2: int = 2

# define single subject data class
@dataclass
class Subject:
    # identifiers
    sid: str                    # session identifier (eeg)
    pid: str                    # participant identifier (behaviour)
    
    # descriptives
    age: int                    # age in years
    sex: str                    # sex (see SEX_*)
    set: int                    # electrode set used (see ELECTRODES_*)
    
    # functional settings
    type: int                   # type of session (see TYPE_*)
    status: int                 # status of session (see STATUS_*)
    fibre_correction: bool      # does this session require the fibre correction? (i.e., someone had rewired the fibre cables improperly, accidentally swapping chns 1-32 for 33-64, for a brief period in February)

    # misc
    notes: str                  # any notes about the session
    
    def __post_init__(self):
        '''
        Set a few path variables for convenience.
        '''
        
        # raw data
        self.d_raw_beh = f'./data/raw/beh/sub{self.sid}/'
        self.d_raw_eeg = f'./data/raw/eeg/sub{self.sid}/'
        
        # preprocessed data
        self.d_preprocessed_beh = f'./data/preprocessed/beh/sub{self.sid}/'
        self.d_preprocessed_eeg = f'./data/preprocessed/eeg/sub{self.sid}/'
        
        # processed data
        self.d_processed_beh = f'./data/processed/beh/sub{self.sid}/'
        self.d_processed_eeg = f'./data/processed/eeg/sub{self.sid}/'
    
    def is_available(self, path: str, requires: list[str]) -> bool:
        '''
        All-purpose method to check existence of directory
        and specified list of files.
        '''
        
        return os.path.isdir(path) and np.all([os.path.isfile(os.path.join(path, require)) for require in requires])
    
    def preprocessed(self, requires: Union[None, list[str]] = None) -> bool:
        '''
        Function to make sure preprocessed data
        are available.
        '''
        
        # setup checks to run
        requires = requires if requires is not None else ['badcomponents.npy', 'bads.npy', 'full-ica.fif', 
                                                          'processed-PL1-epo.fif', 'processed-PL2-epo.fif', 
                                                          'processed-MT1-epo.fif', 'processed-MT2-epo.fif']
        
        return self.is_available(self.d_preprocessed_eeg, requires)

# define container class
class SubjectContainer:
    info: list[Subject]
    
    def __init__(self, info: list[Subject]):
        '''
        Make info available
        '''
        
        self.info = info
    
    def __getitem__(self, id: str) -> Union[Subject, bool]: 
        '''
        Make class subscriptable by sid and pid.
        '''
        
        for subject in self.info:
            if id in [subject.pid, subject.sid]: return subject
        
        return False
    
    def __len__(self) -> int:
        '''
        Enable len function over class.
        '''
        
        return len(self.info)
    
    def __iter__(self) -> collections.abc.Iterator[Subject]:
        '''
        Make class iterable.
        '''
        
        for subject in self.info:
            yield subject

    def select(self, f: Callable) -> Any:
        '''
        Obtain a new subject container from the
        selection function `f`. Note that this
        is applied subject-wise.
        '''
        
        return SubjectContainer([subject for subject in self.info if f(subject)])

    def drop(self, f: Callable) -> Any:
        '''
        Wrapper for .select (but inverting f).
        '''
        
        return self.select(lambda sub: f(sub) == False)
    
    def trim(self) -> Any:
        '''
        Wrapper for .select (for good data sets)
        '''
        
        return self.select(lambda sub: (sub.type == TYPE_REGULAR) and (sub.status == STATUS_INCLUDE))

    def summarise(self, var: str, f: Callable = lambda x: dict(mu = x.mean(), sigma = x.std(), min = x.min(), max = x.max())) -> Any:
        '''
        Compute some kind of summary descriptor function `f`
        over all current subjects w.r.t. `var`.
        '''
        
        return f(np.array([getattr(subject, var) for subject in self.info]))

    def tabulate(self, var: str, f: Callable = lambda x: np.unique(x, return_counts = True)) -> Dict:
        '''
        Compute count summaries over `var`. (Wrapper for .summarise())
        '''
        
        # grab array descriptor
        nd = self.summarise(var, f = f)
        
        return {k: v for k, v in zip(nd[0], nd[1])}

# define all subjects
Subjects = SubjectContainer([
    Subject(sid = '0000', pid = '60H9FB', age = 27, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_PILOT, status = STATUS_EXCLUDE, fibre_correction = False, notes = '20kO recording; amplifier battery pack had to be replaced'),
    Subject(sid = '0001', pid = 'KWFHP2', age = 26, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_PILOT, status = STATUS_EXCLUDE, fibre_correction = False, notes = '20kO recording'),
    Subject(sid = '0002', pid = 'ZS6I40', age = 33, sex = SEX_MALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = True, notes = ''),
    Subject(sid = '0003', pid = '81ZZVB', age = 23, sex = SEX_MALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0004', pid = '8O1AHW', age = 26, sex = SEX_MALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0005', pid = 'M2SXO0', age = 33, sex = SEX_FEMALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0006', pid = 'BXR71I', age = 26, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0007', pid = 'BCVJYN', age = 32, sex = SEX_MALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0008', pid = '81J6YK', age = 20, sex = SEX_FEMALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0009', pid = 'HI9LYM', age = 27, sex = SEX_MALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'construction work on building; subject reported not having heard it, however'),
    Subject(sid = '0010', pid = '1MI7K6', age = 26, sex = SEX_FEMALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'construction work on building; subject reported not having heard it, however'),
    Subject(sid = '0011', pid = 'DC1RSR', age = 39, sex = SEX_MALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes  = 'very challenging to reduce kO; recording at 4-18kO; absolutely triple check data quality'),
    Subject(sid = '0012', pid = '81E0JU', age = 38, sex = SEX_FEMALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0013', pid = '76UKDB', age = 24, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0014', pid = '5RNB7T', age = 37, sex = SEX_FEMALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0015', pid = '8Z3MND', age = 24, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'construction work on building; subject reported having heard it slightly'),
    Subject(sid = '0016', pid = 'Z2IZCV', age = 22, sex = SEX_FEMALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'very challenging to reduce kO in some channels; recording at 2-20kO; absolutely triple check data quality'),
    Subject(sid = '0017', pid = 'NP1YZQ', age = 27, sex = SEX_MALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'may be the chosen one; ive never seen lower impedences, i think'),
    Subject(sid = '0018', pid = 'AMHE3M', age = 30, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0019', pid = 'MCCKUW', age = 25, sex = SEX_FEMALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0020', pid = 'K5DHX0', age = 26, sex = SEX_MALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0021', pid = '8TCO5S', age = 27, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'construction work MT1 block 7 through MT2 block 1'),
    Subject(sid = '0022', pid = '5O7E3J', age = 28, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'challenging to reduce kO in some channels along the edge of the cap; recording at 2-20kO; construction work on building; subject reported having heard it slightly'),
    Subject(sid = '0023', pid = 'IUQNHS', age = 30, sex = SEX_MALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_EXCLUDE, fibre_correction = False, notes = 'subject could not sit still; data quality looks poor; subject misunderstood task?'),
    Subject(sid = '0024', pid = 'SC7S8M', age = 26, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0025', pid = 'UUJB0Y', age = 22, sex = SEX_FEMALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'right CPs difficult to get right in kO measures'),
    Subject(sid = '0026', pid = '85NIU5', age = 21, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'right F2 high kO measurement before test but good signal during test'),
    Subject(sid = '0027', pid = '8PEYLW', age = 24, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0028', pid = '2L7KN6', age = 31, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0029', pid = 'K4X5ZL', age = 35, sex = SEX_FEMALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'F2 impedance readings still not working properly'),
    Subject(sid = '0030', pid = 'MWEA0N', age = 25, sex = SEX_FEMALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'C2/C4 impedance slightly off but good signal, monitor situation'),
    Subject(sid = '0031', pid = '9B7Q5Q', age = 34, sex = SEX_MALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'subject was quite tense; double check quality'),
    Subject(sid = '0032', pid = 'TMI2Q8', age = 32, sex = SEX_MALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0033', pid = 'Y9LYNH', age = 24, sex = SEX_MALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'native but raised in VN'),
    Subject(sid = '0034', pid = 'ZHXM7J', age = 22, sex = SEX_MALE, set = ELECTRODES_S2, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = 'native but raised in IN'),
    Subject(sid = '0035', pid = 'XVUNQY', age = 31, sex = SEX_MALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0036', pid = 'MMDZUL', age = 24, sex = SEX_MALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = ''),
    Subject(sid = '0037', pid = '7MI1TF', age = 39, sex = SEX_MALE, set = ELECTRODES_S1, type = TYPE_REGULAR, status = STATUS_INCLUDE, fibre_correction = False, notes = '')
])