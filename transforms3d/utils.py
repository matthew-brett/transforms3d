''' Utilities for tranforms3d '''

import math

import numpy as np

def normalized_vector(data):
    ''' Return vector divided by Euclidian (L2) norm

    See :term:`unit vector` and :term:`Euclidian norm`

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
    '''
    data = np.asarray(data)
    return data / math.sqrt((data**2).sum())
