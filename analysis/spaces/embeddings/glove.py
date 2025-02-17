'''
embeddings::glove.py

Wrapper functions to transform, load and sample GloVe.
Available as embeddings.glove.*
'''

from .internal import *
import os
import numpy as np
from sklearn.decomposition import PCA
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from typing import Any


def GloVe_to_word2vec(f_in: str = './glove-german/vectors.txt', f_out: str = './glove-german/gensim_glove_vectors.txt'):
    '''
    Takes GloVe-structured data at `f_in` and transforms it to a word2vec gensim format.
    Note that, as a safe guard, this will not allow overwriting of the specified output 
    file `f_out`.
    '''
    
    assert(os.path.isfile(f_in)) or critical(ref = 'embeddings::GloVe_to_word2vec()', msg = f'File `{f_in}` does not exist.')
    assert(not os.path.isfile(f_out)) or critical(ref = 'embeddings::GloVe_to_word2vec()', msg = f'File `{f_out}` already exists.')
    
    glove2word2vec(glove_input_file = f_in, word2vec_output_file = f_out)

def load_embedding(f_in: str = './glove-german/gensim_glove_vectors.txt') -> Any:
    '''
    Loads a word2vec embedding in gensim format from `f_in` and returns
    the KeyedVectors.
    '''
    
    assert(os.path.isfile(f_in)) or critical(ref = 'embeddings::load_embedding()', msg = f'File {f_in} does not exist.')
    
    return KeyedVectors.load_word2vec_format(f_in, binary = False)

def collect_full_space(M: Any) -> tuple[np.ndarray, np.ndarray]:
    '''
    Takes word2vec input `M` and transforms into one big matrix of item x feature.
    Returns both `L` labels and `G` matrix in a tuple.
    '''
    
    assert(hasattr(M, "key_to_index")) or critical(ref = 'embeddings::collect_full_space()', msg = f'`M` must be in word2vec gensim format.')
    
    # get all keys, select all vectors
    L = np.array([k for k in M.key_to_index])
    G = np.array(M[L])
    
    return (L, G)

def sample_from_topics(X: list[str], g: Any, n: int = 250, ignore: list[str] = ['None', None]) -> np.ndarray:
    '''
    Returns `n` most similar items from glove `g` per topic 
    specified in `X`, while ignoring items in `ignore` as a
    `groupXitems` matrix.
    '''
    
    assert(type(X) is list and len(X) > 1) or critical(ref = 'embeddings::sample_from_topics()', msg = f'`X` must be of type list and have two or more items.') 
    assert(hasattr(g, 'most_similar')) or critical(ref = 'embeddings::sample_from_topics()', msg = f'`g` must be word2vec glove model.')
    assert(n > 1) or critical(ref = 'embeddings::sample_from_topics()', msg = f'`n` number of samples must be greater than one.')
    assert(type(ignore) is list) or critical(ref = 'embeddings::sample_from_topics()', msg = f'`ignore` must be of type list.')
    
    # select similar candidates per topic
    Xs = np.array([np.array(g.most_similar(x, topn = n))[:,0] for x in X])
    
    # weed out some bad ones
    for i, xs in enumerate(X):
        for j, x in enumerate(Xs[i,:]):
            Xs[i,j] = Xs[i,j].replace(X[i], '') if (len(Xs[i,j].replace(X[i], '')) > 3) and (Xs[i,j] not in ignore) else None
    
    return Xs

def word2vec_to_numpy(g: Any, f_out: str = None, verbose: bool = True) -> tuple[str, str]:
    '''
    Converts word2vec `g` to numpy dump in `f_out`. Note that
    this will generate two files, so f_out should simply be a
    template without file ending. Files generated are the
    vocabulary as well as vector files. File names are
    returned in a tuple (vocab, vectors).
    '''
    
    assert(hasattr(g, 'most_similar')) or critical(ref = 'embeddings::word2vec_to_numpy()', msg = f'`g` must be word2vec gensim model.')
    assert(type(f_out) is str) or critical(ref = 'embeddings::word2vec_to_numpy()', msg = f'`f_out` must be of type string.')

    # allocate memory & get vocab
    if verbose: message(ref = 'embeddings::word2vec_to_numpy()', msg = f'Allocating memory...')
    X = np.zeros((len(g), 300))
    V = g.key_to_index.keys()

    # fill in vectors
    if verbose: message(ref = 'embeddings::word2vec_to_numpy()', msg = f'Filling matrix...')
    for i, v in enumerate(V):
        X[i,:] = g[v]
    
    # save vectors
    if verbose: message(ref = 'embeddings::word2vec_to_numpy()', msg = f'Saving matrix...')
    f_vec = f_out + '_vec.npy'
    with open(f_vec, 'wb') as f:
        np.save(f, X)
    
    # save vocab
    if verbose: message(ref = 'embeddings::word2vec_to_numpy()', msg = f'Saving vocabulary...')
    f_voc = f_out + '_voc.npy'
    with open(f_voc, 'wb') as f:
        np.save(f, np.array(list(V)))

    return (f_voc, f_vec)

def numpy_to_word2vec(f_in_voc: str = None, f_in_vec: str = None, f_out: str = None, verbose: bool = True):
    '''
    Converts numpy data from `f_in_voc` (vocabulary) and `f_in_vec` (vectors)
    to word2vec format in `f_out` readable by gensim (see load_embedding).
    '''

    assert(type(f_in_voc) is str) or critical(ref = 'embeddings::numpy_to_word2vec()', msg = f'`f_in_voc` must be of type string.')
    assert(type(f_in_vec) is str) or critical(ref = 'embeddings::numpy_to_word2vec()', msg = f'`f_in_vec` must be of type string.')
    assert(type(f_out) is str) or critical(ref = 'embeddings::numpy_to_word2vec()', msg = f'`f_out` must be of type string.')

    # load vocab
    if verbose: message(ref = 'embeddings::numpy_to_word2vec()', msg = f'Loading vocabulary...')
    with open(f_in_voc, 'rb') as f:
        V = np.load(f)
    
    # load vectors
    if verbose: message(ref = 'embeddings::numpy_to_word2vec()', msg = f'Loading vectors...')
    with open(f_in_vec, 'rb') as f:
        X = np.load(f)

    # export as w2v (readable by KeyedVectors.load_word2vec_format)
    if verbose: message(ref = 'embeddings::numpy_to_word2vec()', msg = f'Exporting as word2vec...')
    if os.path.exists(f_out): os.remove(f_out)

    with open(f_out, 'a+') as f:
        f.write(f'{V.shape[0]} {X.shape[1]}\n') # header
        for i in np.arange(0, X.shape[0], 1):
            if verbose: print(str(i), end = '\r')
            f_str = V[i]
            for j in np.arange(0, X.shape[1], 1): f_str += ' ' + X[i,j].astype(str)
            f_str += '\n'
            f.write(f_str)
    
    # report
    if verbose: message(ref = 'embeddings::numpy_to_word2vec()', msg = f'Completed.')

def PPA(f_in: str = None, f_out: str = None, d: int = 20, D: int = 7, verbose: bool = True, save_memory: bool = True):
    '''
    Performs the post-processing algorithm originally described in Mu & Viswanath (2018). Essentially, this
    will remove the top-`D` (of `d`) components from demeaned PCA of `X` found in `f_in`. Note that `f_in`
    must be a numpy dump (see word2vec_to_numpy).
    '''

    assert(type(f_in) is str) or critical(ref = 'embeddings::PPA()', msg = f'`f_in` must be of type string.')
    assert(type(f_out) is str) or critical(ref = 'embeddings::PPA()', msg = f'`f_out` must be of type string.')
    assert(type(d) is int and d > 0) or critical(ref = 'embeddings::PPA()', msg = f'`d` must be of type int and greater than zero.')
    assert(type(D) is int and D > 0) or critical(ref = 'embeddings::PPA()', msg = f'`D` must be of type int and greater than zero.')
    assert(d > D) or critical(ref = 'embeddings::PPA()', msg = f'`d` must be greater than `D`.')

    # load vectors
    if verbose: message(ref = 'embeddings::PPA()', msg = f'Loading vectors...')
    with open(f_in, 'rb') as f:
        X = np.load(f)

    # subtract mean embedding
    if verbose: message(ref = 'embeddings::PPA()', msg = f'Removing mean embedding...')
    X = X - X.mean(axis = 0)

    # perform PCA for top-d
    if verbose: message(ref = 'embeddings::PPA()', msg = f'Computing top-d PCA...')
    pca = PCA(n_components = d, copy = (save_memory == False))
    pca.fit(X)

    # reload (because of PCA overwriting), if necessary
    if save_memory:
        if verbose: message(ref = 'embeddings::PPA()', msg = f'Restoring `X`...')
        with open(f_in, 'rb') as f:
            X = np.load(f)
        X = X - X.mean(axis = 0)
    
    # compute top-D component contribution
    Y = np.zeros_like(X)
    for D_i in np.arange(0, D, 1):
        if verbose: message(ref = 'embeddings::PPA()', msg = f'Computing top-{D_i} contribution...')
        Y = Y + (pca.components_[D_i].T * X) * pca.components_[D_i]
    
    # remove top-D
    if verbose: message(ref = 'embeddings::PPA()', msg = f'Removing top-D components...')
    X = X - Y

    # save data
    if verbose: message(ref = 'embeddings::PPA()', msg = f'Saving results...')
    with open(f_out, 'wb') as f:
        np.save(f, X)

def word2vec_reduction(f_in: str = None, f_out: str = None, pca_d: int = 50, d: int = 20, D: int = 7, verbose: bool = True, save_memory: bool = True):
    '''
    Implements the dimensionality reduction technique described in Raunak et al. (2019)
    and Raunak (2017). Essentially, this performs PPA, then projects into lower PCA
    space of dimensionality `pca_d` and then performs PPA again. This uses the
    numpy representation stored in `f_in` (see word2vec_to_numpy) as well as
    parameters total N for PPA `d` and top-`D` (see PPA).
    '''

    assert(type(f_in) is str) or critical(ref = 'embeddings::word2vec_reduction()', msg = f'`f_in` must be of type string.')
    assert(type(f_out) is str) or critical(ref = 'embeddings::word2vec_reduction()', msg = f'`f_out` must be of type string.')
    assert(type(pca_d) is int and pca_d > 0) or critical(ref = 'embeddings::word2vec_reduction()', msg = f'`pca_d` must be of type int and greater than zero.')
    assert(type(d) is int and d > 0) or critical(ref = 'embeddings::word2vec_reduction()', msg = f'`d` must be of type int and greater than zero.')
    assert(type(D) is int and D > 0) or critical(ref = 'embeddings::word2vec_reduction()', msg = f'`D` must be of type int and greater than zero.')
    assert(d > D) or critical(ref = 'embeddings::word2vec_reduction()', msg = f'`d` must be greater than `D`.')
    
    # compute first PPA
    if verbose: message(ref = 'embeddings::word2vec_reduction()', msg = f'Starting first PPA...')
    tmp_PPA = f_out + '_tmp'
    PPA(f_in = f_in, f_out = tmp_PPA, d = d, D = D, verbose = verbose, save_memory = save_memory)

    # load temp
    if verbose: message(ref = 'embeddings::word2vec_reduction()', msg = f'Loading first PPA...')
    with open(tmp_PPA, 'rb') as f:
        X = np.load(f)

    # compute PCA
    if verbose: message(ref = 'embeddings::word2vec_reduction()', msg = f'Computing PCA...')
    pca = PCA(n_components = pca_d, copy = (save_memory == False))
    pca.fit(X)

    # reload (because of PCA overwriting), if necessary
    if save_memory:
        if verbose: message(ref = 'embeddings::word2vec_reduction()', msg = f'Restoring `X`...')
        with open(tmp_PPA, 'rb') as f:
            X = np.load(f)
    
    # transform vectors
    if verbose: message(ref = 'embeddings::word2vec_reduction()', msg = f'Projecting vectors...')
    X = pca.transform(X)

    # save vectors
    if verbose: message(ref = 'embeddings::word2vec_reduction()', msg = f'Saving PCA...')
    tmp_PCA = f_out + '_tmp2'
    with open(tmp_PCA, 'wb') as f:
        np.save(f, X)
    
    # compute final PPA
    if verbose: message(ref = 'embeddings::word2vec_reduction()', msg = f'Starting second PPA...')
    PPA(f_in = tmp_PCA, f_out = f_out, d = d, D = D, verbose = verbose, save_memory = save_memory)

    # clean up
    if verbose: message(ref = 'embeddings::word2vec_reduction()', msg = f'Removing temporary files...')
    os.remove(tmp_PPA)
    os.remove(tmp_PCA)