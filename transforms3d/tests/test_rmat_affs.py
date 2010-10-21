''' Test rotation matrix implementations against affine implementations

'''

import numpy as np

from .. import quaternions as tq
from .. import taitbryan as ttb

from .samples import euler_tuples

from ..affines import axangle2aff, aff2axangle

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_almost_equal


def test_angle_axis_imps():
    for x, y, z in euler_tuples:
        M = ttb.euler2mat(z, y, x)
        q = tq.mat2quat(M)
        vec, theta = tq.quat2axangle(q)
        M1 = tq.axangle2rmat(vec, theta)
        M2 = axangle2aff(vec, theta)[:3,:3]
        yield assert_array_almost_equal, M1, M2
        M3 = np.eye(4)
        M3[:3,:3] = M
        v2, t2, point = aff2axangle(M3)
        M4 = tq.axangle2rmat(v2, t2)
        yield assert_array_almost_equal, M, M4



