''' Test Gohlke implementations against newer transforms3d

These tests should shrink as the Gohlke transforms get incorporated
'''

import math
import warnings

import numpy as np

from transforms3d import quaternions as tq
from transforms3d import taitbryan as ttb
from transforms3d import zooms as tzs
from transforms3d import shears as tss
from transforms3d import reflections as trf

from transforms3d.tests.samples import euler_tuples

from transforms3d import _gohlketransforms as tg

from numpy.testing import assert_array_almost_equal


def test_quaternion_imps():
    for x, y, z in euler_tuples:
        M = ttb.euler2mat(z, y, x)
        quat = tq.mat2quat(M)
        # Against transformations code
        tM = tg.quaternion_matrix(quat)
        assert_array_almost_equal(M, tM[:3,:3])
        M44 = np.eye(4)
        M44[:3,:3] = M
        tQ = tg.quaternion_from_matrix(M44)
        assert tq.nearly_equivalent(quat, tQ)


def test_euler_imps():
    for x, y, z in euler_tuples:
        M1 = tg.euler_matrix(z, y, x,'szyx')[:3,:3]
        M2 = ttb.euler2mat(z, y, x)
        assert_array_almost_equal(M1, M2)
        q1 = tg.quaternion_from_euler(z, y, x, 'szyx')
        q2 = ttb.euler2quat(z, y, x)
        assert tq.nearly_equivalent(q1, q2)


def test_zooms_shears():
    for i in range(5):
        factor = np.random.random() * 10 - 5
        direct = np.random.random(3) - 0.5
        origin = np.random.random(3) - 0.5
        # factor, etc to matrices
        S0 = tzs.zfdir2aff(factor, None, None)
        S1 = tg.scale_matrix(factor, None, None)
        assert_array_almost_equal(S0, S1, 8)
        S0 = tzs.zfdir2aff(factor, direct, None)
        S1 = tg.scale_matrix(factor, None, direct)
        assert_array_almost_equal(S0, S1, 8)
        S0 = tzs.zfdir2aff(factor, direct, origin)
        S1 = tg.scale_matrix(factor, origin, direct)
        assert_array_almost_equal(S0, S1, 8)
        # matrices to factor, etc
        S0 = tzs.zfdir2aff(factor, direct, origin)
        f1, d1, o1 = tzs.aff2zfdir(S0)
        f2, o2, d2 = tg.scale_from_matrix(S0)
        assert_array_almost_equal(f1, f2)
        if d1 is None:
            assert d2 is None
        else:
            assert_array_almost_equal(d1, d2, 8)
        assert_array_almost_equal(o1, o2[:3], 8)


def test_reflections():
    for i in range(5):
        v0 = np.random.random(3) - 0.5
        v1 = np.random.random(3) - 0.5
        M0 = trf.rfnorm2aff(v0)
        M1 = tg.reflection_matrix([0,0,0], v0)
        assert_array_almost_equal(M0, M1, 8)
        M0 = trf.rfnorm2aff(v0, v1)
        M1 = tg.reflection_matrix(v1, v0)
        assert_array_almost_equal(M0, M1, 8)
        n0, p0 = trf.aff2rfnorm(M0)
        p1, n1 = tg.reflection_from_matrix(M0)
        assert_array_almost_equal(n0, n1)
        assert_array_almost_equal(p0, p1[:3])


def test_shears():
    angle = (np.random.random() - 0.5) * 4*math.pi
    direct = np.random.random(3) - 0.5
    normal = np.cross(direct, np.random.random(3))
    S0 = tss.sadn2aff(angle, direct, normal)
    S1 = tg.shear_matrix(angle, direct, [0,0,0], normal)
    assert_array_almost_equal(S0, S1, 8)
    point = np.random.random(3) - 0.5
    S0 = tss.sadn2aff(angle, direct, normal, point)
    S1 = tg.shear_matrix(angle, direct, point, normal)
    assert_array_almost_equal(S0, S1, 8)
    # Confirm our own implementation does a correct round trip.
    a0, d0, n0, p0 = tss.aff2sadn(S0)
    S0_back = tss.sadn2aff(a0, d0, n0, p0)
    assert_array_almost_equal(S0, S0_back)
    # We no longer test tg.shear_from_matrix, because it is so unstable.
    # We aren't using that implementation.
