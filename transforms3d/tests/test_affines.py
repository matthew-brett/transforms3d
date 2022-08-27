''' Test module for affines '''

import numpy as np
from itertools import permutations

from numpy.testing import (assert_array_equal, assert_array_almost_equal)

from transforms3d.affines import (compose, decompose, decompose44)
from transforms3d.taitbryan import euler2mat
from transforms3d.testing import assert_raises


def test_compose():
    # Test that rotation vector raises error
    T = np.ones(3)
    R = np.ones(3)
    Z = np.ones(3)
    assert_raises(ValueError, compose, T, R, Z)


def test_de_compose():
    # Make a series of translations etc, compose and decompose
    for trans in permutations([10,20,30]):
        for rots in permutations([0.2,0.3,0.4]):
            for zooms in permutations([1.1,1.9,2.3]):
                for shears in permutations([0.01, 0.04, 0.09]):
                    Rmat = euler2mat(*rots)
                    M = compose(trans, Rmat, zooms, shears)
                    for func in decompose, decompose44:
                        T, R, Z, S = func(M)
                        assert (
                            np.allclose(trans, T) and
                            np.allclose(Rmat, R) and
                            np.allclose(zooms, Z) and
                            np.allclose(shears, S))


def test_decompose_shears():
    # Check that zeros shears are also returned
    T, R, Z, S = decompose(np.eye(4))
    assert_array_equal(S, np.zeros(3))


def test_rand_de_compose():
    # random matrices
    for i in range(50):
        M = np.random.normal(size=(4,4))
        M[-1] = [0, 0, 0, 1]
        for func in decompose, decompose44:
            T, R, Z, S = func(M)
            M2 = compose(T, R, Z, S)
            assert_array_almost_equal(M, M2)
