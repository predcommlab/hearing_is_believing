'''
embeddings::func.py

General purpose functions for semantic embeddings we are creating as well
as stimulus creation functions (primarily math & classification related).
Available as embedding.*
'''

from .internal import *
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.stats import gaussian_kde, ttest_rel, ttest_ind
from bs4 import BeautifulSoup
import requests as r
from typing import Any, Callable, Union

def cluster_full_space(L: np.ndarray, G: np.ndarray, k: int = 25, f: bool = False) -> tuple[sklearn.cluster._kmeans.KMeans, list[list[str]]]:
    '''
    Takes labels `L` and full matrix `G` to compute `k` number of clusters. Returns
    the KMeans structure from sklearn `K` and an array of an array of labels per 
    cluster `C_l` in a tuple. Note that, to compute for k > 1e4, please supply `f`
    as True to force the operation.
    '''
    
    assert(type(L) is np.ndarray) or critical(ref = 'embeddings::cluster_full_space()', msg = f'`L` must be of type np.ndarray.')
    assert(type(G) is np.ndarray) or critical(ref = 'embeddings::cluster_full_space()', msg = f'`G` must be of type np.ndarray.')
    assert((k > 1) and ((k < 1e4) or (f is True))) or critical(ref = 'embeddings::cluster_full_space()', msg = f'This requires 1 < k < 1e4, but got k = {k}.')
    
    # compute k-means, get labels
    K = KMeans(n_clusters = k, random_state = 0).fit(G)
    C_l = [L[np.where(K.labels_ == n)[0].astype(int)] for n in np.arange(0, k, 1)]
    
    return (K, C_l)

def nchoosek_equidistant_samples(O: np.ndarray, g: Any, k: int, ignore: list[str] = ['None', None]) -> tuple[np.ndarray, np.ndarray]:
    '''
    Takes groupXitems matrix `O`, glove `g` and number `k` options
    that are roughly equidistant from zero in semantic space, while 
    ignoring items in `ignore`. Returns a matrix `S` that resembles
    `groupXindices`.
    '''
    
    assert(hasattr(O, 'shape') and (type(O) is np.ndarray)) or critical(ref = 'embeddings::nchoosek_equidistant_samples()', msg = f'`O` must be np.ndarray.')
    assert(hasattr(g, 'key_to_index')) or critical(ref = 'embedding::nchoosek_equidistant_samples()', msg = f'`g` must be word2vec glove model.')
    assert((k > 1) and O.shape[1] > k) or critical(ref = 'embedding::nchoosek_equidistant_samples()', msg = f'Operation requires 1 < k < N (1 < ?{k} < ?{O.shape[1]}).')
    assert(type(ignore) is list) or critical(ref = 'embedding::nchoosek_equidistant_samples()', msg = f'`ignore` must be of type list.')
    
    # distance matrix
    D = np.zeros(O.shape) * np.nan
    
    # calculate ED except for bad options
    for i in np.arange(0, O.shape[0], 1):
        for j in np.arange(0, O.shape[1], 1):
            if O[i,j] in ignore:
                continue
            else:
                D[i,j] = distance.euclidean(g[O[i,j]], np.zeros_like(g[O[i,j]]))
    
    # abs(z-transform + N), estimate probabilities
    mu, sigma = (np.nanmean(D), np.nanstd(D))
    D = np.abs(((D - mu) / sigma) + np.random.normal(size = D.shape)*1e-3)
    p = ((1 / D).T / np.nansum((1 / D), axis = 1)).T
    
    # sample without NaN
    S = np.array([np.random.choice(np.where(np.isnan(p[i,:]) == False)[0], p = p[i,np.where(np.isnan(p[i,:]) == False)[0]], size = (1, k), replace = False) for i in np.arange(0, O.shape[0], 1)]).squeeze()
    
    # safety check (mainly for smaller n/k)
    DS = np.array([D[i,S[i,:]] for i in np.arange(0, O.shape[0], 1)])
    if np.any(DS.mean(axis = 1) > (DS.mean() + DS.std())) or np.any(DS.mean(axis = 1) < (DS.mean() - DS.std())):
        warning(ref = 'embeddings::nchoosek_equidistant_samples()', msg = f'One group mean is (more than) one standard deviation away from the others.')
    
    return (S, np.array([DS.mean(axis = 1), DS.std(axis = 1)]))

def IPA_from_words(L: list[str]) -> list[str]:
    '''
    Returns a list `d` of pronunciations (IPA), as requested
    in `L`. Individual entries are collected from wiktionary.
    '''
    
    assert(type(L) in [list, str]) or critical(ref = 'embeddings::IPA_from_words()', msg = f'`L` must be a list of strings or a single string.')
    if type(L) == list: assert(np.all([type(L_i) in [str, None] for L_i in L])) or critical(ref = 'embeddings::IPA_from_words()', msg = f'`L` must be a list of strings or a single string.')
    
    if type(L) == str: L = [L]
    
    d = []
    
    for L_i in L:
        # make wiktionary request
        X = r.get(f'https://de.wiktionary.org/wiki/{L_i}')
        
        if X.status_code != 200:
            warning(ref = 'embeddings::IPA_from_words()', msg = f'Request failed. Skipping `{L_i}`.')
            d.append('-')
            continue
        
        # convert to bs4 and read as html to find IPA
        S = BeautifulSoup(X.text, 'html.parser')
        S_i = S.find('span', {'class': 'ipa'})
        
        if S_i is None:
            warning(ref = 'embeddings::IPA_from_words()', msg = f'No IPA pronunciation found for `{L_i}`.')
            d.append('-')
            continue
        
        d.append(S_i.getText())
    
    return d

def frequency_from_words(L: Union[list[str], str], F: pd.core.frame.DataFrame, fb: int = np.nan, warnings: bool = True) -> list[float]:
    '''
    Returns a list `d` frequencies (in SUBTLEX), as requested
    in `L` and retrieved from pandas subtlex object `F`. Note that the fallback value
    (in case an entry cannot be found) is `fb` = np.nan.
    '''
    
    assert(type(L) in [list, str]) or critical(ref = 'embeddings::frequency_from_words()', msg = f'`L` must be a list of strings or a single string.')
    if type(L) == list: assert(np.all([type(L_i) in [str, None] for L_i in L])) or critical(ref = 'embeddings::frequency_from_words()', msg = f'`L` must be a list of strings or a single string.')
    assert(type(F) is pd.core.frame.DataFrame) or critical(ref = 'embeddings::frequency_from_words()', msg = f'`F` must be a pandas dataframe of SUBTLEX.')
    
    if type(L) == str: L = [L]
        
    d = []
    
    for L_i in L:
        # find word
        X = F.loc[F['Word'].str.lower() == L_i.lower()]
        
        if X.SUBTLEX.shape[0] < 1:
            if warnings: warning(ref = 'embeddings::frequency_from_word()', msg = f'Could not find L_i = `{L_i}` in `F`.')
            d.append(fb)
            continue
        
        # enter frequency
        d.append(X.SUBTLEX.tolist()[0])
    
    return d

def levenshtein_distance_from_words(x: Union[list[str], str], y: Union[list[str], str]) -> list[int]:
    '''
    Returns a list of levenshtein distances `LD` between words in `x`
    and `y`. Note that `x` and `y` may be supplied as lists or single
    strings, but that the output `LD` is always type list.
    '''
    
    assert(type(x) in [list, str]) or critical(ref = 'embeddings::lavenshtein_distance_from_words()', msg = f'`x` must be a string or list of strings.')
    assert(type(y) in [list, str]) or critical(ref = 'embeddings::levenshtein_distance_from_words()', msg = f'`y` must be a string or list of strings.')
    if type(x) == list: assert(np.all([type(x_i) in [str, None] for x_i in x])) or critical(ref = 'embeddings::levenshtein_distance_from_words()', msg = f'`x` must be a string or list of strings.')
    if type(y) == list: assert(np.all([type(y_i) in [str, None] for y_i in y])) or critical(ref = 'embeddings::levenshtein_distance_from_words()', msg = f'`y` must be a string or list of strings.')
    assert(type(x) == type(y)) or critical(ref = 'embeddings::levenshtein_distance_from_words()', msg = f'`x` and `y` must be of the same type.')
    if type(x) == list: assert(len(x) == len(y)) or critical(ref = 'embeddings::levenshtein_distance_from_words()', msg = f'`x` and `y` must be of equal shape.')
    
    if type(x) == str: x, y = ([x], [y])
    
    LD = []
    
    for (x_i, y_i) in zip(x, y):
        # get size and initialise matrix
        x_s, y_s = (len(x_i), len(y_i))
        D = np.zeros((x_s + 1, y_s + 1))
        
        # fill range vectors
        D[:,0] = np.arange(0, x_s + 1, 1)
        D[0,:] = np.arange(0, y_s + 1, 1)
        
        # fill remaining elements
        for x_j in np.arange(1, x_s + 1, 1):
            for y_j in np.arange(1, y_s + 1, 1):
                D[x_j, y_j] = min(D[x_j - 1, y_j] + 1,
                                  D[x_j - 1, y_j - 1] + (0 if x_i[x_j - 1] == y_i[y_j - 1] else 1),
                                  D[x_j, y_j - 1] + 1)
        
        LD.append(D[x_s, y_s])
    
    return LD

def cosine_similarity_between_words(x: Union[list[str], str], y: Union[list[str], str], g: Any) -> list[float]:
    '''
    Computes the cosine similarities between words in `x` and `y` found
    in the semantic space defined by glove model `g`. Returns a list of
    similarities. Note that while `x` and `y` may be string or list of
    strings, the similarity returned `CS` is always of type list.
    '''
    
    assert(type(x) in [list, str]) or critical(ref = 'embeddings::cosine_similarity_between_words()', msg = f'`x` must be a string or list of strings.')
    assert(type(y) in [list, str]) or critical(ref = 'embeddings::cosine_similarity_between_words()', msg = f'`y` must be a string or list of strings.')
    if type(x) == list: assert(np.all([type(x_i) in [str, None] for x_i in x])) or critical(ref = 'embeddings::cosine_similarity_between_words()', msg = f'`x` must be a string or list of strings.')
    if type(y) == list: assert(np.all([type(y_i) in [str, None] for y_i in y])) or critical(ref = 'embeddings::cosine_similarity_between_words()', msg = f'`y` must be a string or list of strings.')
    assert(type(x) == type(y)) or critical(ref = 'embeddings::cosine_similarity_between_words()', msg = f'`x` and `y` must be of the same type.')
    if type(x) == list: assert(len(x) == len(y)) or critical(ref = 'embeddings::cosine_similarity_between_words()', msg = f'`x` and `y` must be of equal shape.')
    assert(hasattr(g, 'key_to_index')) or critical(ref = 'embeddings::cosine_similarity_between_words()', msg = f'`g` must be word2vec glove model.')
    
    if type(x) == str: x, y = ([x], [y])
        
    CS = []
    
    for (x_i, y_i) in zip(x, y):
        x_g, y_g = (g[x_i], g[y_i])
        CS.append(np.dot(x_g, y_g) / (np.linalg.norm(x_g) * np.linalg.norm(y_g)))
    
    return CS

def euclidean_distance_between_words(x: Union[list[str], str], y: Union[list[str], str], g: Any) -> list[float]:
    '''
    Computes the euclidean distance between words in `x` and `y` found
    in the semantic space defined by glove model `g`. Returns a list of
    distances. Note that while `x` and `y` may be string or list of
    strings, the distances returned `D` are always of type list.
    '''
    
    assert(type(x) in [list, str]) or critical(ref = 'embeddings::euclidean_distance_between_words()', msg = f'`x` must be a string or list of strings.')
    assert(type(y) in [list, str]) or critical(ref = 'embeddings::euclidean_distance_between_words()', msg = f'`y` must be a string or list of strings.')
    if type(x) == list: assert(np.all([type(x_i) in [str, None] for x_i in x])) or critical(ref = 'embeddings::euclidean_distance_between_words()', msg = f'`x` must be a string or list of strings.')
    if type(y) == list: assert(np.all([type(y_i) in [str, None] for y_i in y])) or critical(ref = 'embeddings::euclidean_distance_between_words()', msg = f'`y` must be a string or list of strings.')
    assert(type(x) == type(y)) or critical(ref = 'embeddings::euclidean_distance_between_words()', msg = f'`x` and `y` must be of the same type.')
    if type(x) == list: assert(len(x) == len(y)) or critical(ref = 'embeddings::euclidean_distance_between_words()', msg = f'`x` and `y` must be of equal shape.')
    assert(hasattr(g, 'key_to_index')) or critical(ref = 'embeddings::euclidean_distance_between_words()', msg = f'`g` must be word2vec glove model.')
    
    if type(x) == str: x, y = ([x], [y])
        
    D = []
    
    for (x_i, y_i) in zip(x, y):
        x_g, y_g = (g[x_i], g[y_i])
        D.append(distance.euclidean(x_g, y_g))
    
    return D

def compute_dissimilarity_matrix(X: np.ndarray, f: Union[Callable[[np.ndarray, np.ndarray], np.ndarray], str] = None) -> np.ndarray:
    '''
    Computes and returns (dis-)similartiy matrix `Z` from itemsXfeatures matrix
    `X` using measure `f` (default: cosine dissimilarity, alternatives: cosine
    similarity `cs`, euclidean distance `ed`, z-score(ed) `zed`).
    '''
    
    assert(type(X) is np.ndarray) or critical(ref = 'embeddings::compute_dissimilarity_matrix()', msg = f'`X` must be of type np.ndarray.')
    assert(len(X.shape) == 2) or critical(ref = 'embeddings::compute_dissimilarity_matrix()', msg = f'`X` must have two dimensions (itemsXfeatures).')
    assert(callable(f) or (f is None) or (f.lower() in ['cd', 'cs', 'ed', 'zed'])) or critical(ref = 'embeddings::compute_dissimilarity_matrix', msg = f'If supplied, `f` must be a callable function or in [`cd`, `cs`, `ed`, `zed`].')
    
    def cs(x, y):
        # compute cosine similarity
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    
    def cd(x, y):
        # compute cosine dissimilarity
        return 1 - cs(x, y)
    
    def ed(x, y):
        # compute euclidean distance
        return distance.euclidean(x, y)
    
    zs = f == 'zed'
    f = f if callable(f) else cd if (f is None) or (f == 'cd') else cs if f == 'cs' else ed
    Z = np.zeros((X.shape[0], X.shape[0]))
    for i in np.arange(0, X.shape[0], 1):
        for j in np.arange(0, X.shape[0], 1):
            Z[i,j] = f(X[i,:], X[j,:])
    
    # if requested as in z(ed), z-score
    if zs: Z = (Z - Z.mean()) / Z.std()
    
    return Z