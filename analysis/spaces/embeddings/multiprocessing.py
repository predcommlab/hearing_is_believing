'''
embeddings::multiprocessing.py

Functions and classes utilised in multiprocessing.
'''

from .internal import *
from dataclasses import dataclass, field
from typing import Any, Union, Callable
from multiprocessing import Pool, get_context

@dataclass
class Processor:
    '''
    This is a prototype class for multiprocessing. To be used by 
    supplying a list of `tasks` that N number of `workers` should
    process. To complete tasks, call ::run() whereupon results
    will be available from `results` and returned value.
    '''
    
    workers: int = 8
    maxtasksperchild: int = 3 # because of some kind of memory leak somewhere, this may be necessary; see https://stackoverflow.com/questions/76137433
    tasks: list[Any] = field(default_factory = lambda: [])
    results: list[Any] = field(default_factory = lambda: [])

    def run(self, tasks: list[Any], external: Union[Callable, None] = None, timeout: Union[int, None] = None, **kwargs: Any) -> list[Any]:
        '''
        Runs the `tasks` from dataclass using N `workers` and stores `results`. Note
        that either `external` must be callable or `worker` must be implemented in
        the data class to specify the task logic. Optionally, `timeout` and further
        `kwargs` may be specified for the worker call. Returns list `results`.
        '''

        with get_context("spawn").Pool(processes = self.workers, maxtasksperchild = self.maxtasksperchild) as pool:
            f = self.worker if not callable(external) else external
            procs = [pool.apply_async(f, (load,), kwargs) for load in tasks]
            self.tasks = tasks
            self.results = [res.get(timeout = timeout) for res in procs]
        
        return self.results
    
    def worker(self, load: Any, **kwargs: Any) -> None:
        '''
        This is a placeholder function. To make use of this, a new subclass
        should be implemented that overrides this function. For example, a
        class like this could be implemented:
        
            @dataclass
            class Browser(Processor):
                def worker(self, load: str) -> str:
                    return requests.get(load).text
            
            if __name__ == '__main__':
                freeze_support()
                
                browser = Browser()
                text = browser.run(['https://wikipedia.org', 'https://old.reddit.com'])
        
        It should be noted that if Processor::worker() is not reimplemented, then 
        an `external` function must be supplied for Processor::run(). Otherwise
        this will result in a NotImplementedError being thrown by the workers. This
        could be done like so:
            
            def MatMul(tuple_matrices: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
                return np.dot(tuple_matrices[0], tuple_matrices[1])
            
            if __name__ == '__main__':
                freeze_support()

                processor = Processor()
                Y = processor.run(RetrieveHugeMatricesAsTuples(), external = MatMul)
        '''

        raise NotImplementedError