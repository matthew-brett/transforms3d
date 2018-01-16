""" Testing reflections
"""

import numpy as np

from transforms3d.reflections import rfnorm2mat, rfnorm2aff, mat2rfnorm, aff2rfnorm
from transforms3d.utils import normalized_vector

from transforms3d.testing import assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from transforms3d.tests.samples import euler_mats


def assert_almost_equal_sign(v1, v2):
    # Assert vectors are almost equal or v1 * -1 ~= v2
    if np.all(np.sign(v1) == np.sign(v2)):
        assert_array_almost_equal(v1, v2)
    else:
        assert_array_almost_equal(v1 * -1, v2)


def test_rfnorm_unit():
    # Test reflections from unit normals
    rng = np.random.RandomState()
    points = rng.normal(size=(3, 20))
    h_points = np.ones((4, 20))
    h_points[:3] = points
    for i, normal in enumerate(np.eye(3)):  # unit vector normals
        rfmat = rfnorm2mat(normal)
        pts2 = rfmat.dot(points)
        flip_vec = np.ones((4, 1))
        flip_vec[i] = -1
        assert_array_equal(pts2, points * flip_vec[:3])
        rfaff = rfnorm2aff(normal)
        h_pts2 = rfaff.dot(h_points)
        assert_array_equal(h_pts2, h_points * flip_vec)


def test_no_reflection():
    assert_raises(ValueError, mat2rfnorm, np.eye(3))
    assert_raises(ValueError, aff2rfnorm, np.eye(4))
    # Rotations are not reflections
    for mat in euler_mats:
        assert_raises(ValueError, mat2rfnorm, mat)
        aff = np.eye(4)
        aff[:3, :3] = mat
        assert_raises(ValueError, aff2rfnorm, aff)


def test_list_input():
    # Test sequences for input to routines
    assert_array_equal(rfnorm2mat([1, 0, 0]),
                       [[-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    assert_array_equal(rfnorm2aff([1, 0, 0]),
                       [[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])


def test_rfnorm_round_trip():
    rng = np.random.RandomState()
    vecs = rng.normal(size=(20, 3))
    pts = rng.normal(size=(20, 3))
    for vec, pt in zip(vecs, pts):
        normal = normalized_vector(vec)
        rfmat = rfnorm2mat(normal)
        n2 = mat2rfnorm(rfmat)
        assert_almost_equal_sign(normal, n2)
        rfaff = rfnorm2aff(normal)
        n2, p2 = aff2rfnorm(rfaff)
        assert_almost_equal_sign(normal, n2)
        assert_array_almost_equal(p2, 0)
        rfaff = rfnorm2aff(normal, pt)
        n2, p2 = aff2rfnorm(rfaff)
        assert_almost_equal_sign(normal, n2)
        back_aff = rfnorm2aff(n2, p2)
        assert_array_almost_equal(rfaff, back_aff)
