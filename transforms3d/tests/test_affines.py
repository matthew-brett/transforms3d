''' Test module for affines '''

import numpy as np

from numpy.testing import assert_array_equal, assert_raises, dec, \
    assert_array_almost_equal
    
from nose.tools import assert_true, assert_false

from transforms3d.affines import shears2matrix, compose, decompose, \
    decompose44
from transforms3d.taitbryan import euler2mat


def test_shears():
    for n, N in ((1, 2),
                 (3, 3),
                 (6, 4),
                 (10, 5),
                 (15, 6),
                 (21, 7),
                 (78, 13)):
        shears = np.arange(n)
        M = shears2matrix(shears)
        e = np.eye(N)
        inds = np.triu(np.ones((N,N)), 1).astype(bool)
        e[inds] = shears
        yield assert_array_equal, M, e
    for n in (2, 4, 5, 7, 8, 9):
        shears = np.zeros(n)        
        yield assert_raises, ValueError, shears2matrix, shears


def permute(seq):
    # Return list of unique permutations of 3 element sequence
    seq = list(seq)
    indlist = (
        (0,2,1),
        (1,2,0),
        (1,0,2),
        (2,0,1),
        (2,1,0))
    permuted = [seq]
    for inds in indlist:
        res = [seq[inds[0]], seq[inds[1]], seq[inds[2]]]
        if res not in permuted:
            permuted.append(res)
    return permuted


def permute_signs(seq):
    # Permute signs on all non-zero elements in 3 element sequence
    signs = np.array(((1, 1, 1),
                      (1, 1, -1),
                      (1, -1, 1),
                      (1, -1, -1),
                      (-1, 1, 1),
                      (-1, 1, -1),
                      (-1, -1, 1),
                      (-1, -1, -1)
                      ))
    permuted = []
    aseq = np.array(seq)
    snz = aseq != 0
    if not np.any(snz):
        return [seq]
    for s in signs:
        sseq = aseq.copy()
        sseq[snz] = sseq[snz] * s[snz]
        sseq = list(sseq)
        if sseq not in permuted:
            permuted.append(sseq)
    return permuted


def permute_with_signs(seq):
    seqs = permute(seq)
    res = []
    for s in seqs:
        res += permute_signs(s)
    return res

    
_r13 = np.sqrt(1/3.0)
_r12 = np.sqrt(0.5)
sphere_points = (
        permute_with_signs([1, 0, 0]) + 
        permute_with_signs([_r12, _r12, 0]) + 
        permute_signs([_r13, _r13, _r13])
    )


def test_compose():
    # Test that rotation vector raises error
    T = np.ones(3)
    R = np.ones(3)
    Z = np.ones(3)
    yield assert_raises, ValueError, compose, T, R, Z


@dec.slow
def test_de_compose():
    # Make a series of translations etc, compose and decompose
    for trans in permute([10,20,30]):
        for rots in permute([0.2,0.3,0.4]):
            for zooms in permute([1.1,1.9,2.3]):
                for shears in permute([0.01, 0.04, 0.09]):
                    Rmat = euler2mat(*rots)
                    M = compose(trans, Rmat, zooms, shears)
                    for func in decompose, decompose44:
                        T, R, Z, S = func(M)
                        yield (assert_true,
                               np.allclose(trans, T) and
                               np.allclose(Rmat, R) and
                               np.allclose(zooms, Z) and
                               np.allclose(shears, S))


def test_decompose_shears():
    # Check that zeros shears are also returned
    T, R, Z, S = decompose(np.eye(4))
    yield assert_array_equal, S, np.zeros(3)


def test_rand_de_compose():
    # random matrices
    for i in range(50):
        M = np.random.normal(size=(4,4))
        M[-1] = [0, 0, 0, 1]
        for func in decompose, decompose44:
            T, R, Z, S = func(M)
            M2 = compose(T, R, Z, S)
            yield assert_array_almost_equal, M, M2
