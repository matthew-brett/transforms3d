''' Test Gohlke implementations against newer transforms3d

These tests should shrink as the Gohlke transforms get incorporated
'''
import numpy as np

import transforms3d.quaternions as tq
import transforms3d.taitbryan as ttb
import transforms3d.zooms_shears as tzs

from transforms3d.testing import euler_tuples

from transforms3d.gohlketransforms import quaternion_matrix, \
    quaternion_from_matrix, euler_matrix, \
    quaternion_from_euler, scale_matrix, scale_from_matrix


from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_almost_equal


def test_quaternion_imps():
    for x, y, z in euler_tuples:
        M = ttb.euler2mat(z, y, x)
        quat = tq.mat2quat(M)
        # Against transformations code
        tM = quaternion_matrix(quat)
        yield assert_array_almost_equal, M, tM[:3,:3]
        M44 = np.eye(4)
        M44[:3,:3] = M
        tQ = quaternion_from_matrix(M44)
        yield assert_true, tq.nearly_equivalent(quat, tQ)


def test_euler_imps():
    for x, y, z in euler_tuples:
        M1 = euler_matrix(z, y, x,'szyx')[:3,:3]
        M2 = ttb.euler2mat(z, y, x)
        yield assert_array_almost_equal, M1, M2
        q1 = quaternion_from_euler(z, y, x, 'szyx')
        q2 = ttb.euler2quat(z, y, x)
        yield assert_true, tq.nearly_equivalent(q1, q2)


def test_zooms_shears():
    for i in range(5):
        factor = np.random.random() * 10 - 5
        direct = np.random.random(3) - 0.5
        origin = np.random.random(3) - 0.5
        # factor, etc to matrices
        S0 = tzs.zdir2aff(factor, None, None)
        S1 = scale_matrix(factor, None, None)
        yield assert_array_almost_equal, S0, S1, 8
        S0 = tzs.zdir2aff(factor, direct, None)
        S1 = scale_matrix(factor, None, direct)
        yield assert_array_almost_equal, S0, S1, 8
        S0 = tzs.zdir2aff(factor, direct, origin)
        S1 = scale_matrix(factor, origin, direct)
        yield assert_array_almost_equal, S0, S1, 8
        # matrices to factor, etc
        S0 = tzs.zdir2aff(factor, direct, origin)
        f1, d1, o1 = tzs.aff2zdir(S0)        
        f2, o2, d2 = scale_from_matrix(S0)
        yield assert_array_almost_equal, f1, f2
        if d1 is None:
            yield assert_true, d2 is None
        else:
            yield assert_array_almost_equal, d1, d2
        yield assert_array_almost_equal, o1, o2[:3]
