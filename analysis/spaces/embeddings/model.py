'''
embeddings::model.py

Functions and classes utilised in modelling the trial-wise semantic
space estimates per participant.
'''

from .internal import *
from . import model_auxiliary as aux
import numpy as np
import arviz as az
import pymc as pm
import pandas as pd
import os.path
import pickle
import gzip
from typing import Any, Callable, Union

'''
Parameters for readability
'''

PRIOR_LOCAL: int = 1    # Indicates that talker-specific priors should be modelled.
PRIOR_GLOBAL: int = 2   # Indicates that talker-agnostic priors should be modelled.

POSTERIOR_CORRECT: int = 1      # Indicates that posteriors should be estimated from the correct choice at any given trial.
POSTERIOR_CHOICE: int = 2       # Indicates that posteriors should be estimates from the choice of the participant at any given trial.
POSTERIOR_SIMULATED: int = 3    # Indicates that posteriors should be estimated from a simulated choice (3 of 6 correct).
POSTERIOR_RANDOM: int = 4       # Indicates that posteriors should be estimated from a random choice.
POSTERIOR_FEEDBACK_NEG: int = 5 # Indicates that posteriors should be estimated from the correct choice at any given trial, but that negative feedback should be valued more strongly.
POSTERIOR_FEEDBACK_OPP: int = 6 # Indicates that posteriors should be estimated from the choice given at any trial, but that steps should be taken in the opposite direction of the wrong answer (or correct direction if correct).

GAMMA_BASELINE: float = 1e-3                                # baseline learning rate (for free energy models)
GAMMA_CONSTANT = lambda x: GAMMA_BASELINE                   # lambda function for constant learning rate
GAMMA_LINEAR = lambda x: (x+1)*GAMMA_BASELINE               # lambda function for a linearly increasing learning rate
GAMMA_EXPONENTIAL = lambda x: np.exp(x)*GAMMA_BASELINE      # lambda function for an exponentially increasing learning rate
GAMMA_LOG = lambda x: np.log(x+1)*GAMMA_BASELINE            # lambda function for a logarithmically increasing learning rate

class Session:
    def __init__(self, f_in: str = None, f_T: str = None, f_C: str = None, f_P: str = None, G: Any = None, 
                       type_priors: int = PRIOR_LOCAL, type_posteriors: int = POSTERIOR_CORRECT, verbose: bool = True):
        '''
        This is a super class for individual model specifications that helps track
        the relevant data for any subject session. Specifically, this will first
        load the trial file `f_T`, the control trial file `f_C` and the practice
        trial file `f_P` as well as the raw subject data file `f_in`. It will then
        find all targets and contexts and retrieve the GloVe vectors from `G`.

        It should be noted that the process by which trials and contexts are
        selected can be manipulated in two ways. Firstly, `type_priors` offers
        either the option `PRIOR_LOCAL` to tag individual items by their
        individual context or `PRIOR_GLOBAL` to use a single global context
        for fitting space estimates. Secondly, `type_posteriors` offers either
        the option `POSTERIOR_CORRECT` to use whichever target was labelled
        as correct for the trial for fitting the space or, alternatively, the
        option `POSTERIOR_CHOICE` to use the target that the participant selected
        for any given trial.

        Further, for debugging (or general keeping tabs) purposes, `verbose` may 
        be enabled to receive updates of what's happening, given that fitting
        models may be a lengthy process. It should perhaps also be noted here
        that, given the processing time, it may be worth considering running
        sessions in parallel. This can be achieved using the mp wrapper that is
        provided in embeddings::mp.

        Note that this class has no business with actual model fitting; All this
        does is provide data for that purpose. Individual model classes should
        inherit this type, super()-initialise and then fit the individual model.
        '''

        assert(f_in != None and type(f_in) is str) or critical(ref = f'embeddings::model::Session()', msg = f'`f_in` is required and must be of type string.')
        assert(os.path.isfile(f_in)) or critical(ref = f'embeddings::model::Session()', msg = f'File under `f_in` must be existing participant data.')
        assert(f_T != None and type(f_T) is str) or critical(ref = f'embeddings::model::Session()', msg = f'`f_T` is required and must be of type string.')
        assert(os.path.isfile(f_T)) or critical(ref = f'embeddings::model::Session()', msg = f'File under `f_T` must be existing target data.')
        assert(f_C != None and type(f_C) is str) or critical(ref = f'embeddings::model::Session()', msg = f'`f_C` is required and must be of type string.')
        assert(os.path.isfile(f_C)) or critical(ref = f'embeddings::model::Session()', msg = f'File under `f_C` must be existing control data.')
        assert(f_P != None and type(f_P) is str) or critical(ref = f'embeddings::model::Session()', msg = f'`f_P` is required and must be of type string.')
        assert(os.path.isfile(f_P)) or critical(ref = f'embeddings::model::Session()', msg = f'File under `f_P` must be existing practice data.')
        assert(hasattr(G, 'most_similar')) or critical(ref = f'embeddings::model::Session()', msg = f'`G` must be word2vec gensim model.')
        assert(type(type_priors) is int and type_priors in [PRIOR_LOCAL, PRIOR_GLOBAL]) or critical(ref = f'embeddings::model::Session()', msg = f'`type_priors` must be of type int and either `PRIOR_LOCAL` or `PRIOR_GLOBAL`.')
        assert(type(type_posteriors) is int and type_posteriors in [POSTERIOR_CORRECT, POSTERIOR_CHOICE, POSTERIOR_SIMULATED, POSTERIOR_RANDOM, POSTERIOR_FEEDBACK_NEG, POSTERIOR_FEEDBACK_OPP]) or critical(ref = f'embeddings::model::Session()', msg = f'`type_posteriors` must be of type int and either `POSTERIOR_CORRECT`, `POSTERIOR_CHOICE`, `POSTERIOR_SIMULATED`, `POSTERIOR_RANDOM`, `POSTERIOR_FEEDBACK_NEG`, or `POSTERIOR_FEEDBACK_OPP`.')

        # set states
        self.verbose = verbose
        self.type_priors = type_priors
        self.type_posteriors = type_posteriors

        # load trial data
        if verbose: message(ref = f'embeddings::model::Session()', msg = f'Loading trial data...')
        self.G = G
        self.T = pd.read_excel(f_T)
        self.C = pd.read_excel(f_C)
        self.P = pd.read_excel(f_P)
        self.M = pd.read_csv(f_in)
        self.pid = np.unique(self.M.pid.tolist())[0]
        self.fid = f'{self.pid}_prior{type_priors}_posterior{type_posteriors}'

        # code trial data
        if verbose: message(ref = f'embeddings::model::Session()', msg = f'Coding trial data...')
        self.trials_raw = self.M.loc[np.where((self.M.trial_type == '2afc_audio_visual'))[0]]
        self.targets = []
        self.alternatives = []
        self.contexts = []
        self.contexts2 = []
        self.y_g = []
        
        # if model is simulated, pick contexts to perform well in
        if type_posteriors == POSTERIOR_SIMULATED:
            self.rng = np.random.default_rng(seed = int(''.join(x for x in self.pid if x.isdigit())))
            self.fav_contexts = self.rng.choice(np.unique(self.T.context1.tolist()), size = (3,), replace = False)

        for indx in self.trials_raw.index[np.arange(0, self.trials_raw.shape[0], 1)]:
            trial = self.trials_raw.loc[indx]

            if trial.no < 0:
                # practice trial
                pair = self.P.loc[((self.P.target == trial.option_left) & (self.P.popular == trial.option_right)) |
                                  ((self.P.target == trial.option_right) & (self.P.popular == trial.option_left))]
                main_context = pair.context.tolist()[0]
                alt_context = main_context
                main_target = pair.target.tolist()[0]
                alt_target = pair.popular.tolist()[0]
            elif not trial.is_control:
                # trial
                pair = self.T.loc[((self.T.target1 == trial.option_left) & (self.T.target2 == trial.option_right)) |
                                  ((self.T.target1 == trial.option_right) & (self.T.target2 == trial.option_left))]
                is_main = (((trial.option_left == pair.target1) & (trial.target_position == 'left')) |
                           ((trial.option_right == pair.target1) & (trial.target_position == 'right'))).tolist()[0]
                main_context = pair.context1.tolist()[0] if is_main else pair.context2.tolist()[0]
                alt_context = pair.context2.tolist()[0] if is_main else pair.context1.tolist()[0]
                main_target = pair.target1.tolist()[0] if is_main else pair.target2.tolist()[0]
                alt_target = pair.target2.tolist()[0] if is_main else pair.target1.tolist()[0]
            elif trial.is_control:
                # practice trial
                pair = self.C.loc[((self.C.target == trial.option_left) & (self.C.popular == trial.option_right)) |
                                  ((self.C.target == trial.option_right) & (self.C.popular == trial.option_left))]
                main_context = pair.context.tolist()[0]
                alt_context = main_context
                main_target = pair.target.tolist()[0]
                alt_target = pair.target.tolist()[0]
            else:
                # some issue
                warning(ref = f'embeddings::model::Session()', msg = f'Trial `{indx}` was neither practice, control nor real trial. Skipping.')
                continue
            
            if type_priors == PRIOR_GLOBAL: 
                self.contexts.append('global')
                self.contexts2.append('global')
            else:
                self.contexts.append(main_context)
                self.contexts2.append(alt_context)
            
            if type_posteriors == POSTERIOR_CORRECT or trial.choice_option not in ['left', 'right']:
                self.targets.append(main_target.lower())
                self.alternatives.append(alt_target.lower())
            elif type_posteriors == POSTERIOR_CHOICE:
                self.targets.append(trial.option_left.lower() if trial.choice_option == 'left' else trial.option_right.lower())
                self.alternatives.append(trial.option_right.lower() if trial.choice_option == 'left' else trial.option_left.lower())
            elif type_posteriors == POSTERIOR_SIMULATED:
                if main_context in self.fav_contexts:
                    self.targets.append(main_target.lower())
                    self.alternatives.append(alt_target.lower())
                else:
                    self.targets.append(alt_target.lower())
                    self.alternatives.append(main_target.lower())
            elif type_posteriors == POSTERIOR_RANDOM:
                self.rng = np.random.default_rng(seed = int(''.join(x for x in self.pid if x.isdigit())))
                c = self.rng.choice([main_target.lower(), alt_target.lower()], size = (2,), replace = False)
                self.targets.append(c[0])
                self.alternatives.append(c[1])
            elif type_posteriors == POSTERIOR_FEEDBACK_NEG:
                self.targets.append(main_target.lower())
                self.alternatives.append(alt_target.lower())
            elif type_posteriors == POSTERIOR_FEEDBACK_OPP:
                self.targets.append(main_target.lower())
                self.alternatives.append(alt_target.lower())
            else:
                # some issue
                warning(ref = f'embeddings::model::Session()', msg = f'Trial `{indx} could not be associated with any known posterior (`{type_posteriors}`). Skipping.')
                continue
            
            self.y_g.append(trial.choice_is_target)
        
        # cast y_g
        self.y_g = np.array(self.y_g).astype(int)
        
        # load vectors
        if verbose: message(ref = f'embeddings::model::Session()', msg = f'Loading vectors...')
        self.vectors = np.array(self.G[self.targets])
        self.vectors2 = np.array(self.G[self.alternatives])
        
        # rotate vectors where necessary
        if type_posteriors == POSTERIOR_FEEDBACK_OPP:
            self.vectors[self.y_g == 0] = -self.vectors2[self.y_g == 0]

    def export(self):
        '''
        This is a placeholder function that individual models
        should reimplement with their own specific export functions.
        '''
        
        raise NotImplementedError
    
    def load(self):
        '''
        This is a placeholder function that individual models
        should reimplement with their own specific import functions.
        '''
    
        raise NotImplementedError
    
    def load_eval(self):
        '''
        This is a placeholder function that individual models
        should reimplement with their own specific load eval functions.
        '''

        raise NotImplementedError
    
    def fit(self):
        '''
        This is a placeholder function that individual models
        should reimplement with their own specific fitting functions.
        '''

        raise NotImplementedError

    def evaluate(self):
        '''
        This is a placeholder function that individual models
        should reimplement with their own specific evaluation functions.
        '''

        raise NotImplementedError

class IdealObserver(Session):
    def __init__(self, f_in: str = None, f_T: str = None, f_C: str = None, f_P: str = None, G: Any = None, 
                 type_priors: int = PRIOR_LOCAL, type_posteriors: int = POSTERIOR_CORRECT, verbose: bool = True,
                 path: str = './dumps/models/IdealObserver/'):
        '''
        This class is used for fitting and evaluating an ideal 
        observer. It inherits from ::Session(), so please find
        further instructions for parameters there.
        '''

        # initialise session
        super().__init__(f_in = f_in, f_T = f_T, f_C = f_C, f_P = f_P, G = G, type_priors = type_priors, type_posteriors = type_posteriors, verbose = verbose)

        # setup empty vars
        self.model = None
        self.trace = None
        self.posteriors = None

        # setup path
        self.path = path
    
    def export(self, f_out: str = None):
        '''
        Reimplementation of Session::export(). This will take the current
        trace & model as a tuple and dump a gzipped pickle of the two to
        `f_out`. Note that, if `f_out` exists already, this function will
        not overwrite it, but abort instead.
        '''

        assert(f_out != None and type(f_out) == str) or critical(ref = f'embeddings::model::IdealObserver::export()', msg = f'`f_out` is required and must be of type string.')
        assert(not os.path.isfile(os.path.join(self.path, f'{f_out}.pkl.gz'))) or critical(ref = f'embeddings::model::IdealObserver::export()', msg = f'Output file `{f_out}` already exists. Aborting.')
        
        # gzip open the file and pickle dump trace & model
        with gzip.open(os.path.join(self.path, f'{f_out}.pkl.gz'), 'wb') as f:
            pickle.dump((self.trace, self.model), f)

    def load(self, f_in: str = None) -> tuple[pm.backends.base.MultiTrace, pm.model.Model]:
        '''
        Reimplementation of Session::load(). This will simply
        load trace & model from the file specified under `f_in`
        and return them in a tuple (and set them locally).
        '''

        assert(f_in != None and type(f_in) == str) or critical(ref = f'embeddings::model::IdealObserver::load()', msg = f'`f_in` is required and must be of type string.')
        assert(os.path.isfile(os.path.join(self.path, f'{f_in}.pkl.gz'))) or critical(ref = f'embeddings::model::IdealObserver::load()', msg = f'Input file `{f_in}` does not exist. Aborting.')

        # gzip open the file and pickle load trace & model
        with gzip.open(os.path.join(self.path, f'{f_in}.pkl.gz'), 'rb') as f:
            (self.trace, self.model) = pickle.load(f)
        
        return (self.trace, self.model)
    
    def load_eval(self, f_in: str = None) -> np.ndarray:
        '''
        Reimplementation of Session::load_eval(). This will
        simply load and return the evaluation specified in 
        `f_in`. For details on the posteriors, please see
        the description in ::evaluate().
        '''

        assert(f_in != None and type(f_in) == str) or critical(ref = f'embeddings::moidel::IdealObserver::load_eval()', msg = f'`f_in` is required and must be of type string.')
        assert(os.path.isfile(os.path.join(self.path, f'{f_in}.npy.gz'))) or critical(ref = f'embeddings::model::IdealObserver::load_eval()', msg = f'Input file `{f_in}` does not exist. Aborting.')

        # gzip open the file and numpy load posteriors
        with gzip.open(os.path.join(self.path, f'{f_in}.npy.gz'), 'rb') as f:
            self.posteriors = np.load(f)
        
        return self.posteriors
        
    def fit(self, start_from: int = 0, stop_at: int = 0, T: int = 1e3):
        '''
        Reimplementation of Session::fit(). This will loop over all
        trials for the current participant and fit a very simple
        Gaussian for the current estimate of the speaker's position:

            P(t) = N(mu, sigma)

            where

                mu = U(min(v), max(v))
                sigma = U(1e-3, 1)
            
            and

                v = G(targets)
        
        For details on how this function may be manipulated, please
        see parameters `type_priors` and `type_posteriors` from
        model::Session(). Note also that `start_from` and `stop_at`
        may be supplied if you wish to fit only certain trials.

        Note also that this will export (traces, model) after every
        trial that can later be loaded and evaluated.
        '''

        # set stopping condition
        if stop_at == 0: stop_at = len(self.contexts)

        # loop over trials
        for i, context in enumerate(self.contexts):
            if i < start_from or i >= stop_at: continue
            if self.verbose: message(ref = f'embeddings::model::IdealObserver::fit()', msg = f'Fitting model at trial{i}.')

            # get current indices of context
            indcs = np.where(np.array(self.contexts) == context)[0]
            current_indcs = indcs[indcs <= i]

            # fit model over current trials
            with pm.Model() as self.model:
                mu = pm.Uniform('mu', lower = self.vectors[current_indcs,:].min(), upper = self.vectors[current_indcs].max(), shape = (self.vectors.shape[1],))
                sigma = pm.Uniform('sigma', lower = 1e-3, upper = 1.0, shape = (self.vectors.shape[1],))
                y_hat = pm.Normal('y_hat', mu = mu, sigma = sigma, observed = self.vectors[current_indcs,:], shape = (self.vectors.shape[1],))

                self.trace = pm.sample(int(T), return_inferencedata = False, progressbar = False)
            
            # export
            self.export(f_out = f'{self.fid}_t{i}_{context}_{len(current_indcs)}')
        
        if self.verbose: message(ref = f'embeddings::model::IdealObserver::fit()', msg = f'Completed.')
    
    def evaluate(self) -> np.ndarray:
        '''
        Reimplementation of Session::evaluate(). This will walk all
        trials defined in the session, load the corresponding trace
        and model and sample the posterior distribution. Mean and
        standard deviations are computed for mu and sigma parameters
        and saved to the `posteriors` that are saved and returned.

        `posteriors` are trial_noXdimsXpar where par0 = mean(mu),
        par1 = std(mu), par2 = mean(sigma) & par3 = std(sigma).
        '''

        # allocate memory
        self.posteriors = np.zeros((len(self.targets), self.vectors.shape[1], 4))

        # loop over trials
        for i, context in enumerate(self.contexts):
            if self.verbose: message(ref = f'embeddings::model::IdealObserver::evaluate()', msg = f'Evaluating trial{i}...', terminate = False)

            # find current indices
            indcs = np.where(np.array(self.contexts) == context)[0]
            current_indcs = indcs[indcs <= i]

            # load trace & model
            self.load(f_in = f'{self.fid}_t{i}_{context}_{len(current_indcs)}')

            with self.model:
                # sample posterior
                ppc = pm.sample_posterior_predictive(self.trace, var_names = ["mu", "sigma", "y_hat"], progressbar = False)

                # mu descriptives
                self.posteriors[i,:,0] = ppc["mu"].mean(axis = 0)
                self.posteriors[i,:,1] = ppc['mu'].std(axis = 0)

                # sigma descriptives
                self.posteriors[i,:,2] = ppc["sigma"].mean(axis = 0)
                self.posteriors[i,:,3] = ppc["sigma"].std(axis = 0)
        if self.verbose: message(ref = f'embeddings::model::IdealObserver::evaluate()', msg = f'Evaluated all trials...')

        # save posteriors
        if self.verbose: message(ref = f'embeddings::model::IdealObserver::evaluate()', msg = f'Saving posteriors...')
        with gzip.open(os.path.join(self.path, f'{self.fid}.npy.gz'), 'wb') as f:
            np.save(f, self.posteriors)

        return self.posteriors

class FreeEnergy(Session):
    def __init__(self, f_in: str = None, f_T: str = None, f_C: str = None, f_P: str = None, G: Any = None, 
                 type_priors: int = PRIOR_LOCAL, type_posteriors: int = POSTERIOR_CORRECT, verbose: bool = True,
                 path: str = './dumps/models/FreeEnergy/'):
        '''
        This class is used for fitting and evaluating a free energy
        model. It inherits from ::Session(), so please find
        further instructions for parameters there.
        '''

        # initialise session
        super().__init__(f_in = f_in, f_T = f_T, f_C = f_C, f_P = f_P, G = G, type_priors = type_priors, type_posteriors = type_posteriors, verbose = verbose)

        # setup empty vars
        self.model = None

        # setup path
        self.path = path
    
    def export(self, f_out: str = None):
        '''
        Reimplementation of Session::export(). This will take the current
        model as a tuple and dump a gzipped pickle of its states to
        `f_out`. Note that, if `f_out` exists already, this function will
        not overwrite it, but abort instead.
        '''

        assert(f_out != None and type(f_out) == str) or critical(ref = f'embeddings::model::FreeEnergy::export()', msg = f'`f_out` is required and must be of type string.')
        assert(not os.path.isfile(os.path.join(self.path, f'{f_out}.pkl.gz'))) or critical(ref = f'embeddings::model::FreeEnergy::export()', msg = f'Output file `{f_out}` already exists. Aborting.')
        
        # gzip open the file and pickle dump trace & model
        with gzip.open(os.path.join(self.path, f'{f_out}.pkl.gz'), 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, f_in: str = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Reimplementation of Session::load(). This will simply
        load a model (in states) from the file specified under `f_in`
        and return them in a tuple (and set them locally).
        '''

        assert(f_in != None and type(f_in) == str) or critical(ref = f'embeddings::model::FreeEnergy::load()', msg = f'`f_in` is required and must be of type string.')
        assert(os.path.isfile(os.path.join(self.path, f'{f_in}.pkl.gz'))) or critical(ref = f'embeddings::model::FreeEnergy::load()', msg = f'Input file `{f_in}` does not exist. Aborting.')

        # gzip open the file and pickle load trace & model
        with gzip.open(os.path.join(self.path, f'{f_in}.pkl.gz'), 'rb') as f:
            self.model = pickle.load(f)
        
        return self.model
    
    def load_eval(self, f_in: str = None) -> np.ndarray:
        '''
        Reimplementation of Session::load_eval(). This will
        simply load and return the evaluation specified in 
        `f_in`. For details on the posteriors, please see
        the description in ::evaluate().
        '''

        assert(f_in != None and type(f_in) == str) or critical(ref = f'embeddings::moidel::FreeEnergy::load_eval()', msg = f'`f_in` is required and must be of type string.')
        assert(os.path.isfile(os.path.join(self.path, f'{f_in}.npy.gz'))) or critical(ref = f'embeddings::model::FreeEnergy::load_eval()', msg = f'Input file `{f_in}` does not exist. Aborting.')

        # gzip open the file and numpy load posteriors
        with gzip.open(os.path.join(self.path, f'{f_in}.npy.gz'), 'rb') as f:
            self.posteriors = np.load(f)
        
        return self.posteriors
    
    def fit(self, start_from: int = 0, stop_at: int = 0, gamma: Union[Callable, None] = None, steps: int = 100):
        '''
        Reimplementation of Session::fit(). This will loop over all
        trials for the current participant and fit the data using a
        free energy-based approximation of what neurons might be
        computing. Specifically, we use

            F = ln f(phi; p_mu_s, p_sigma_s) + ln f(M; p_mu_m, p_sigma_m)
        
        and want to minimise this free energy. Effectively, this yields
        a relatively simple set of PDEs for an input, two error, a phi
        and sensation node. Please see the paper for more details of
        how this works (or read the code).
        
        For details on how this function may be manipulated, please
        see parameters `type_priors` and `type_posteriors` from
        model::Session(). Note also that `start_from` and `stop_at`
        may be supplied if you wish to fit only certain trials.

        Note that this will also save priors after every step of
        the model for later inspection.
        '''

        # set priors from full glove space
        X = np.zeros((len(self.G), len(self.G['kayak'])))
        V = self.G.key_to_index.keys()
        for i, v in enumerate(V): X[i,:] = self.G[v]
        mu = X.mean(axis = 0)
        del X, V

        # setup identities
        C = np.unique(self.contexts)
        I = np.eye(len(C))

        # setup priors of identities
        p_mu_s = np.tile(mu, (I.shape[0], 1)).T
        p_sigma_s = np.random.uniform(low = 1, high = 5, size = (self.vectors.shape[1],))

        # setup prior of sensations
        p_sigma_m = np.random.uniform(low = 1, high = 5, size = (self.vectors.shape[1],))

        # setup gamma
        if gamma is None: gamma = lambda x: 1e-2

        # set stopping condition
        if stop_at == 0: stop_at = len(self.contexts)

        # loop over trials
        for i, context in enumerate(self.contexts):
            if i < start_from or i >= stop_at: continue
            if self.verbose: message(ref = f'embeddings::model::FreeEnergy::fit()', msg = f'Fitting model at trial{i}.')

            # current sensations and initialise phi
            I_t = I[np.array(C) == context,:].squeeze().reshape((len(C),))
            M = self.vectors[i,:]
            phi = np.dot(p_mu_s, I_t)

            # current error representations
            epsilon_s = phi*0
            epsilon_m = M*0

            # compute N model steps
            for _ in np.arange(0, steps, 1):
                # effective gamma
                delta = gamma(i)
                if self.type_posteriors == POSTERIOR_FEEDBACK_NEG: delta = delta / (self.y_g[i] + 1)

                # get derivatives
                d_epsilon_s = phi - np.dot(p_mu_s, I_t) - p_sigma_s*epsilon_s
                d_epsilon_m = M - phi - p_sigma_m*epsilon_m
                d_phi = epsilon_m - epsilon_s
                d_p_mu_s = epsilon_s
                d_p_sigma_s = .5 * (epsilon_s**2 - 1/p_sigma_s)
                d_p_sigma_m = .5 * (epsilon_m**2 - 1/p_sigma_m)

                # update model
                epsilon_s = epsilon_s + delta*d_epsilon_s
                epsilon_m = epsilon_m + delta*d_epsilon_m
                phi = phi + delta*d_phi
                p_mu_s = p_mu_s + delta*(I_t[:,np.newaxis] @ d_p_mu_s[np.newaxis,:]).T
                p_sigma_s = np.maximum(p_sigma_s + delta*d_p_sigma_s, np.ones_like(p_sigma_s))
                p_sigma_m = np.maximum(p_sigma_m + delta*d_p_sigma_m, np.ones_like(p_sigma_m))
            
            # get current indices of context
            indcs = np.where(np.array(self.contexts) == context)[0]
            current_indcs = indcs[indcs <= i]

            # make tuple & export
            self.model = (epsilon_s, epsilon_m, phi, p_mu_s, p_sigma_s, p_sigma_m)
            self.export(f_out = f'{self.fid}_t{i}_{context}_{len(current_indcs)}')
        
        if self.verbose: message(ref = f'embeddings::model::FreeEnergy::fit()', msg = f'Completed.')
    
    def evaluate(self) -> np.ndarray:
        '''
        Reimplementation of Session::evaluate(). This will walk all
        trials defined in the session, load the corresponding data
        and collect the full matrix including all model parameters,
        to be dumped into one file. Also returns the `posteriors`.

        `posteriors` are trial_noXdimsXpar where parameters are (in
        order): epsilon_s, epsilon_m, phi, p_sigma_s, p_sigma_m, and
        p_mu_s. Note that p_mu_s extends over N parameters (each 
        defining one speaker/prior.)
        '''

        # allocate memory
        self.posteriors = np.zeros((len(self.targets), self.vectors.shape[1], 5+len(np.unique(self.contexts))))

        # loop over trials
        for i, context in enumerate(self.contexts):
            if self.verbose: message(ref = f'embeddings::model::FreeEnergy::evaluate()', msg = f'Evaluating trial{i}...', terminate = False)

            # find current indices
            indcs = np.where(np.array(self.contexts) == context)[0]
            current_indcs = indcs[indcs <= i]

            # load trace & model
            self.load(f_in = f'{self.fid}_t{i}_{context}_{len(current_indcs)}')
            (epsilon_s, epsilon_m, phi, p_mu_s, p_sigma_s, p_sigma_m) = self.model

            # place in posterior structure
            self.posteriors[i,:,0] = epsilon_s
            self.posteriors[i,:,1] = epsilon_m
            self.posteriors[i,:,2] = phi
            self.posteriors[i,:,3] = p_sigma_s
            self.posteriors[i,:,4] = p_sigma_m
            self.posteriors[i,:,5:] = p_mu_s

        if self.verbose: message(ref = f'embeddings::model::FreeEnergy::evaluate()', msg = f'Evaluated all trials...')

        # save posteriors
        if self.verbose: message(ref = f'embeddings::model::FreeEnergy::evaluate()', msg = f'Saving posteriors...')
        with gzip.open(os.path.join(self.path, f'{self.fid}.npy.gz'), 'wb') as f:
            np.save(f, self.posteriors)

        return self.posteriors