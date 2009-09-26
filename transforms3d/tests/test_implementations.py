''' Test Gohlke implementations against newer transforms3d

These tests should shrink as the Gohlke transforms get incorporated
'''
import numpy as np

import transforms3d.quaternions as tq
import transforms3d.taitbryan as ttb

from transforms3d.testing import euler_tuples

from transforms3d.gohlketransforms import quaternion_matrix, \
    quaternion_from_matrix, euler_matrix, \
    quaternion_from_euler


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


