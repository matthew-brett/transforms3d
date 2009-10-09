''' Test rotation matrix implementations against affine implementations

'''

import numpy as np

import transforms3d.quaternions as tq
import transforms3d.taitbryan as ttb

from transforms3d.testing import euler_tuples

from transforms3d.affines import from_angle_axis_point, to_angle_axis_point

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_almost_equal


def test_angle_axis_imps():
    for x, y, z in euler_tuples:
        M = ttb.euler2mat(z, y, x)
        q = tq.mat2quat(M)
        vec, theta = tq.quat2axangle(q)
        M1 = tq.axangle2rmat(vec, theta)
        M2 = from_angle_axis_point(theta, vec)[:3,:3]
        yield assert_array_almost_equal, M1, M2
        M3 = np.eye(4)
        M3[:3,:3] = M
        t2, v2, point = to_angle_axis_point(M3)
        M4 = tq.axangle2rmat(t2, v2)
        yield assert_array_almost_equal, M, M4



