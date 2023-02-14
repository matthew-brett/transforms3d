''' Utilities for transforms3d '''

import math
from itertools import permutations

import numpy as np

# Numpy default random number generator, allowing for older Numpy
try:
    np_default_rng = np.random.default_rng
except AttributeError:
    np_default_rng = np.random.RandomState



def normalized_vector(vec):
    ''' Return vector divided by Euclidean (L2) norm

    See :term:`unit vector` and :term:`Euclidean norm`

    Parameters
    ----------
    vec : array-like shape (3,)

    Returns
    -------
    nvec : array shape (3,)
       vector divided by L2 norm

    Examples
    --------
    >>> vec = [1, 2, 3]
    >>> l2n = np.sqrt(np.dot(vec, vec))
    >>> nvec = normalized_vector(vec)
    >>> np.allclose(np.array(vec) / l2n, nvec)
    True
    >>> vec = np.array([[1, 2, 3]])
    >>> vec.shape
    (1, 3)
    >>> normalized_vector(vec).shape
    (3,)
    '''
    vec = np.asarray(vec).squeeze()
    return vec / math.sqrt((vec**2).sum())


def vector_norm(vec):
    ''' Return vector Euclidaan (L2) norm

    See :term:`unit vector` and :term:`Euclidean norm`

    Parameters
    ----------
    vec : array-like shape (3,)

    Returns
    -------
    norm : scalar

    Examples
    --------
    >>> vec = [1, 2, 3]
    >>> l2n = np.sqrt(np.dot(vec, vec))
    >>> nvec = vector_norm(vec)
    >>> np.allclose(nvec, np.sqrt(np.dot(vec, vec)))
    True
    '''
    vec = np.asarray(vec)
    return math.sqrt((vec**2).sum())


def inique(iterable):
    ''' Generate unique elements from `iterable`

    Parameters
    ----------
    iterable : iterable

    Returns
    -------
    gen : generator
       generator that yields unique elements from `iterable`
    
    Examples
    --------
    >>> tuple(inique([0, 1, 2, 0, 2, 3]))
    (0, 1, 2, 3)
    '''
    history = []
    for val in iterable:
        if val not in history:
            history.append(val)
            yield val


def permuted_signs(seq):
    ''' Generate permuted signs for sequence `seq`

    Parameters
    ----------
    seq : sequence

    Returns
    -------
    gen : generator
       generator returning `seq` with signs permuted

    Examples
    --------
    >>> tuple(permuted_signs([1, -2, 0]))
    ((1, -2, 0), (1, -2, 0), (1, 2, 0), (1, 2, 0), (-1, -2, 0), (-1, -2, 0), (-1, 2, 0), (-1, 2, 0))
    '''
    seq = tuple(seq)
    n = len(seq)
    for fs in inique(permutations([1]*n + [-1]*n, n)):
        yield tuple(e * f for e, f in zip(seq, fs))


def permuted_with_signs(seq):
    ''' Return all permutations of `seq` with all sign permutations

    Parameters
    ----------
    seq : sequence

    Returns
    -------
    gen : generator
       generator returning permutations and sign permutations

    Examples
    --------
    >>> tuple(permuted_with_signs((1,2)))
    ((1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1))
    '''
    for pseq in permutations(seq):
        for sseq in permuted_signs(pseq):
            yield sseq


def random_unit_vector(rng=None):
    """ Return random normalized 3D unit vector

    Parameters
    ----------
    rng : None or random number generator, optional
        `rng` must have function / method `normal` that allows `size=` keyword.

    Returns
    -------
    vec : shape (3,) array
        Vector at random on unit sphere.

    Notes
    -----
    https://mathworld.wolfram.com/SpherePointPicking.html
    """
    if rng is None:
        rng = np_default_rng()
    return normalized_vector(rng.normal(size=3))


def unpack(array, axis=-1, keepdim=False):
    ''' Unpack an array along the given axis

    Parameters
    ----------
    array : array-like with arbitrary shape (..., N, ...)
    axis : axis to unpack
    keepdim: keep a dimension of length 1 at `axis` position

    Returns
    -------
    out : np.ndarray with shape (N, ...)
       array with the given axis moved to the front, meant to be passed to Python unpacking

    Examples
    --------
    >>> a, b = unpack([[1, 2], [3, 4], [5, 6]], axis=-1)
    >>> a
    array([1, 3, 5])
    >>> b
    array([2, 4, 6])
    '''
    array = np.asarray(array)
    array = np.rollaxis(array, axis)
    if keepdim:
        array = np.expand_dims(array, axis)
    return array


def diag_2d(v):
    ''' Construct batches of diagonal matrices

    Parameters
    ----------
    v : array-like batch of vectors with shape (..., N)

    Returns
    -------
    out : np.ndarray with shape (..., N, N)
       batch of matrices with v on the diagonal
       
    Examples
    --------
    >>> a = np.ones((128, 3))
    >>> b = diag_2d(a)
    >>> b.shape
    (128, 3, 3)
    >>> np.allclose(b[0], np.diag(a[0]))
    True
    '''
    *batches, n = v.shape
    diag_indices = tuple([..., *np.diag_indices(n)])
    out = np.zeros((*batches, n, n))
    out[diag_indices] = v
    return out


def transpose_2d(array):
    ''' Transpose batches of 2D arrays

    Parameters
    ----------
    array : array-like batch of 2D arrays with shape (..., N, M)

    Returns
    -------
    out : np.ndarray with shape (..., M, N)
       array with the last two dimensions transposed
       
    Examples
    --------
    >>> a = np.zeros((128, 2, 3))
    >>> b = transpose_2d(a)
    >>> b.shape
    (128, 3, 2)
    '''
    return np.transpose(array, (*range(array.ndim - 2), -1, -2))
