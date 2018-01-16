import math

import numpy as np

from transforms3d.axangles import axangle2aff, aff2axangle, axangle2mat, mat2axangle

from transforms3d import quaternions as tq
from transforms3d import taitbryan as ttb

from transforms3d.tests.samples import euler_tuples

from numpy.testing import assert_array_almost_equal, assert_almost_equal

from transforms3d.testing import assert_raises


def test_aa_points():
    i3 = math.sqrt(1/3.0)
    for theta in (-0.2, 0.5):
        for vec in np.r_[np.eye(3), [[i3, i3, i3]]]:
            for point in [[0.3, 0.4, 0.5],[-0.2, 0, 4.0]]:
                R = axangle2aff(vec, theta)
                v2, t2, p2 = aff2axangle(R)
                assert_array_almost_equal(vec, v2)
                assert_array_almost_equal(theta, t2)
                assert_array_almost_equal(p2[:3], 0)
                # recovering a point
                point = [0.3, 0.4, 0.5]
                RP = axangle2aff(vec, theta, point)
                v3, t3, p3 = aff2axangle(RP)
                assert_array_almost_equal(vec, v3)
                assert_array_almost_equal(theta, t3)
                # doing the whole thing by hand
                T = np.eye(4)
                T[:3,3] = point
                iT = T.copy()
                iT[:3,3] *= -1
                M_hand = np.dot(T, np.dot(R, iT))
                assert_array_almost_equal(RP, M_hand)
                # do round trip
                RP_back = axangle2aff(v3, t3, p3)
                assert_array_almost_equal(RP, RP_back)


def test_mat2axangle_thresh():
    # Test precision threshold to mat2axangle
    axis, angle = mat2axangle(np.eye(3))
    assert_almost_equal(axis, [0, 0, 1])
    assert_almost_equal(angle, 0)
    offset = 1e-6
    mat = np.diag([1 + offset] * 3)
    axis, angle = mat2axangle(mat)
    assert_almost_equal(axis, [0, 0, 1])
    assert_almost_equal(angle, 0)
    offset = 1e-4
    mat = np.diag([1 + offset] * 3)
    assert_raises(ValueError, mat2axangle, mat)
    axis, angle = mat2axangle(mat, 1e-4)
    assert_almost_equal(axis, [0, 0, 1])
    assert_almost_equal(angle, 0)


def test_angle_axis_imps():
    for x, y, z in euler_tuples:
        M = ttb.euler2mat(z, y, x)
        q = tq.mat2quat(M)
        vec, theta = tq.quat2axangle(q)
        M1 = axangle2mat(vec, theta)
        M2 = axangle2aff(vec, theta)[:3,:3]
        assert_array_almost_equal(M1, M2)
        v1, t1 = mat2axangle(M1)
        M3 = axangle2mat(v1, t1)
        assert_array_almost_equal(M, M3)
        A = np.eye(4)
        A[:3,:3] = M
        v2, t2, point = aff2axangle(A)
        M4 = axangle2mat(v2, t2)
        assert_array_almost_equal(M, M4)


def test_errors():
    M = np.ones((3, 3))
    assert_raises(ValueError, mat2axangle, M)
    A = np.eye(4)
    A[:3, :3] = M
    assert_raises(ValueError, aff2axangle, A)
