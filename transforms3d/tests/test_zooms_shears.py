""" Testing zooms and shears 

"""
import warnings

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

import transforms3d.zooms_shears as tzs


def test_zdir_zmat_aff():
    # test zdir to zmat and back
    for i in range(10):
        factor = np.random.random() * 10 - 5
        direct = np.random.random(3) - 0.5
        origin = np.random.random(3) - 0.5
        S0 = tzs.zdir2zmat(factor, None)
        f2, d2 = tzs.zmat2zdir(S0)
        S1 = tzs.zdir2zmat(f2, d2)
        yield assert_array_almost_equal, S0, S1
        direct = np.random.random(3) - 0.5
        S0 = tzs.zdir2zmat(factor, direct)
        f2, d2 = tzs.zmat2zdir(S0)
        S1 = tzs.zdir2zmat(f2, d2)
        yield assert_array_almost_equal, S0, S1
        # affine versions of same
        S0 = tzs.zdir2aff(factor)
        f2, d2, o2 = tzs.aff2zdir(S0)
        yield assert_array_almost_equal, S0, tzs.zdir2aff(f2, d2, o2)
        S0 = tzs.zdir2aff(factor, direct)
        f2, d2, o2 = tzs.aff2zdir(S0)
        yield assert_array_almost_equal, S0, tzs.zdir2aff(f2, d2, o2)
        S0 = tzs.zdir2aff(factor, direct, origin)
        f2, d2, o2 = tzs.aff2zdir(S0)
        yield assert_array_almost_equal, S0, tzs.zdir2aff(f2, d2, o2)


def test_aff2shear_adn():
    # test aff2shear_adn function
    # This function can be very unstable.
    # Test with known random numbers to make sure we don't hit an unstable
    # spot.
    rng = np.random.RandomState(42)
    for i in range(10):
        angle = rng.random_sample() * np.pi
        direct = rng.random_sample(3) - 0.5
        vect = rng.random_sample(3)  # random vector
        normal = np.cross(direct, vect) # orthogonalize against direct
        point = rng.random_sample(3) - 0.5
        # Make shear affine from angle, direction, normal and point
        S0 = tzs.shear_adn2aff(angle, direct, normal, point)
        # Reconstruct angle, direction, normal, point from affine
        with warnings.catch_warnings():
            a2, d2, n2, p2 = tzs.aff2shear_adn(S0)
        # Confirm the shear affines are equivalent
        S1 = tzs.shear_adn2aff(a2, d2, n2, p2)
        assert_array_almost_equal(S0, S1)
