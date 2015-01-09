""" Testing zooms and shears 

"""
import warnings

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

import transforms3d.zooms as tzs
import transforms3d.shears as tss


def test_zdir_zmat_aff():
    # test zdir to zmat and back
    for i in range(10):
        factor = np.random.random() * 10 - 5
        direct = np.random.random(3) - 0.5
        origin = np.random.random(3) - 0.5
        S0 = tzs.zfdir2mat(factor, None)
        f2, d2 = tzs.mat2zfdir(S0)
        S1 = tzs.zfdir2mat(f2, d2)
        yield assert_array_almost_equal, S0, S1
        direct = np.random.random(3) - 0.5
        S0 = tzs.zfdir2mat(factor, direct)
        f2, d2 = tzs.mat2zfdir(S0)
        S1 = tzs.zfdir2mat(f2, d2)
        yield assert_array_almost_equal, S0, S1
        # affine versions of same
        S0 = tzs.zfdir2aff(factor)
        f2, d2, o2 = tzs.aff2zfdir(S0)
        yield assert_array_almost_equal, S0, tzs.zfdir2aff(f2, d2, o2)
        S0 = tzs.zfdir2aff(factor, direct)
        f2, d2, o2 = tzs.aff2zfdir(S0)
        yield assert_array_almost_equal, S0, tzs.zfdir2aff(f2, d2, o2)
        S0 = tzs.zfdir2aff(factor, direct, origin)
        f2, d2, o2 = tzs.aff2zfdir(S0)
        yield assert_array_almost_equal, S0, tzs.zfdir2aff(f2, d2, o2)


def test_striu():
    # Shears encoded as vector from triangle above diagonal of shear mat
    S = [0.1, 0.2, 0.3]
    assert_array_equal(tss.striu2mat(S),
                       [[ 1. ,  0.1,  0.2],
                        [ 0. ,  1. ,  0.3],
                        [ 0. ,  0. ,  1. ]])
    assert_array_equal(tss.striu2mat([1]),
                       [[ 1.,  1.],
                        [ 0.,  1.]])
    for n, N in ((1, 2),
                 (3, 3),
                 (6, 4),
                 (10, 5),
                 (15, 6),
                 (21, 7),
                 (78, 13)):
        shears = np.arange(n)
        M = tss.striu2mat(shears)
        e = np.eye(N)
        inds = np.triu(np.ones((N,N)), 1).astype(bool)
        e[inds] = shears
        assert_array_equal(M, e)
    for n in (2, 4, 5, 7, 8, 9):
        shears = np.zeros(n)
        assert_raises(ValueError, tss.striu2mat, shears)


def test_aff2sadn():
    # test aff2sadn function
    # This function can be very unstable.
    # Test with known random numbers to make sure we don't hit an unstable
    # spot.
    rng = np.random.RandomState(12)
    for i in range(10):
        angle = rng.random_sample() * np.pi
        direct = rng.random_sample(3) - 0.5
        vect = rng.random_sample(3)  # random vector
        normal = np.cross(direct, vect) # orthogonalize against direct
        point = rng.random_sample(3) - 0.5
        # Make shear affine from angle, direction, normal and point
        S0 = tss.sadn2aff(angle, direct, normal, point)
        # Reconstruct angle, direction, normal, point from affine
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a2, d2, n2, p2 = tss.aff2sadn(S0)
        # Confirm the shear affines are equivalent
        S1 = tss.sadn2aff(a2, d2, n2, p2)
        assert_array_almost_equal(S0, S1)
