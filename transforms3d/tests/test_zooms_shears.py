""" Testing zooms and shears 

"""

import math

import numpy as np

import transforms3d.zooms as tzs
import transforms3d.shears as tss
from transforms3d.utils import vector_norm, random_unit_vector

from numpy.testing import assert_array_equal, assert_array_almost_equal
from transforms3d.testing import assert_raises


def test_zfdir_zmat_aff():
    # test zfdir to zmat and back
    for i in range(10):
        factor = np.random.random() * 10 - 5
        direct = np.random.random(3) - 0.5
        origin = np.random.random(3) - 0.5
        S0 = tzs.zfdir2mat(factor, None)
        f2, d2 = tzs.mat2zfdir(S0)
        S1 = tzs.zfdir2mat(f2, d2)
        assert_array_almost_equal(S0, S1)
        direct = np.random.random(3) - 0.5
        S0 = tzs.zfdir2mat(factor, direct)
        f2, d2 = tzs.mat2zfdir(S0)
        S1 = tzs.zfdir2mat(f2, d2)
        assert_array_almost_equal(S0, S1)
        # affine versions of same
        S0 = tzs.zfdir2aff(factor)
        f2, d2, o2 = tzs.aff2zfdir(S0)
        assert_array_almost_equal(S0, tzs.zfdir2aff(f2, d2, o2))
        S0 = tzs.zfdir2aff(factor, direct)
        f2, d2, o2 = tzs.aff2zfdir(S0)
        assert_array_almost_equal(S0, tzs.zfdir2aff(f2, d2, o2))
        S0 = tzs.zfdir2aff(factor, direct, origin)
        f2, d2, o2 = tzs.aff2zfdir(S0)
        assert_array_almost_equal(S0, tzs.zfdir2aff(f2, d2, o2))


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


def ref_mat2sadn(mat):
    # Original (unstable) implementation)
    mat = np.asarray(mat)
    # normal: cross independent eigenvectors corresponding to the eigenvalue 1
    l, V = np.linalg.eig(mat)
    near_1, = np.nonzero(abs(np.real(l.squeeze()) - 1.0) < 1e-4)
    if near_1.size < 2:
        raise ValueError("no two linear independent eigenvectors found %s" % l)
    V = np.real(V[:, near_1]).squeeze().T
    lenorm = -1.0
    for i0, i1 in ((0, 1), (0, 2), (1, 2)):
        n = np.cross(V[i0], V[i1])
        l = vector_norm(n)
        if l > lenorm:
            lenorm = l
            normal = n
    normal /= lenorm
    # direction and angle
    direction = np.dot(mat - np.eye(3), normal)
    angle = vector_norm(direction)
    direction /= angle
    angle = math.atan(angle)
    return angle, direction, normal


def ref_aff2sadn(aff):
    # Original (unstable) implementation)
    aff = np.asarray(aff)
    angle, direction, normal = ref_mat2sadn(aff[:3,:3])
    # point: eigenvector corresponding to eigenvalue 1
    l, V = np.linalg.eig(aff)
    near_1, = np.nonzero(abs(np.real(l.squeeze()) - 1.0) < 1e-8)
    if near_1.size == 0:
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    point = np.real(V[:, near_1[-1]]).squeeze()
    point = point[:3] / point[3]
    return angle, direction, normal, point


def test_ref_aff2sadn():
    # test aff2sadn and reference function
    # This reference function can be very unstable.
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
        a2, d2, n2, p2 = ref_aff2sadn(S0)
        # Confirm the shear affines are equivalent
        S1 = tss.sadn2aff(a2, d2, n2, p2)
        assert_array_almost_equal(S0, S1)
        # Confirm similar to actual implementation
        a, d, n, p = tss.aff2sadn(S0)
        S_actual = tss.sadn2aff(a, d, n, p)
        assert_array_almost_equal(S0, S_actual)


def random_normal(direct, rng):
    # Make another random vector to form cross-product.
    vect = random_unit_vector(rng)
    # Cross-product is orthogonal to direct.
    return np.cross(direct, vect)


def test_aff2sadn():
    # Test actual implemtation
    rng = np.random.RandomState()
    for i in range(10000):
        angle = rng.uniform(-1, 1) * np.pi
        direct = random_unit_vector(rng)
        rnorm = random_normal(direct, rng)
        point = random_unit_vector(rng)
        # Make shear affine from angle, direction, normal and point
        S0 = tss.sadn2aff(angle, direct, rnorm, point)
        # Reconstruct angle, direction, normal, point from affine
        a, d, n, p = tss.aff2sadn(S0)
        S_actual = tss.sadn2aff(a, d, n, p)
        assert_array_almost_equal(S0, S_actual, decimal=5)


def test_inverse_outer():
    rng = np.random.RandomState()
    for i in range(10000):
        in_t = np.tan(rng.uniform(-1, 1) * np.pi)
        direct = random_unit_vector(rng)
        rnorm = random_normal(direct, rng)
        M = in_t * np.outer(direct, rnorm)
        t, a, b = tss.inverse_outer(M)
        assert np.allclose(M, t * np.outer(a, b))
