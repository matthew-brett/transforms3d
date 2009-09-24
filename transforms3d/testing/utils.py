'''  Utilities for generating sample distributions

To get nose to run the doctests:

nosetests --doctest-tests

or

doctest-tests=1

in your .noserc file
'''

import numpy as np


def permutations(iterable, r=None):
    ''' Generate all permutations of `iterable`, of length `r`

    From Python docs, expressing 2.6 ``itertools.permutations``
    algorithm.

    If the elements are unique, then the resulting permutations will
    also be unique.

    Parameters
    ----------
    iterable : iterable
       returning elements that will be permuted
    r : None or int
       length of sequence to return, if None (default) use length of
       `iterable`

    Returns
    -------
    gen : generator
       generator that yields permutations
    
    Examples
    --------
    >>> tuple(permutations(range(3)))
    ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    >>> tuple(permutations(range(3), 2))
    ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
    '''
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = range(n)
    cycles = range(n, n-r, -1)
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return


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
            
    
