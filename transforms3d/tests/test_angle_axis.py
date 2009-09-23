import math

import numpy as np

from transforms3d.gohlketransforms import rotation_matrix, \
    rotation_from_matrix

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_almost_equal, dec


def test_aa_points():
    i3 = math.sqrt(1/3.0)
    for theta in (-0.2, 0.5):
        for vec in np.r_[np.eye(3), [[i3, i3, i3]]]:
            for point in [[0.3, 0.4, 0.5],[-0.2, 0, 4.0]]:
                R = rotation_matrix(theta, vec)
                t2, v2, p2 = rotation_from_matrix(R)
                yield assert_array_almost_equal, theta, t2
                yield assert_array_almost_equal, vec, v2
                yield assert_array_almost_equal, p2[:3], 0
                # recovering a point
                point = [0.3, 0.4, 0.5]
                RP = rotation_matrix(theta, vec, point)
                t3, v3, p3 = rotation_from_matrix(RP)
                yield assert_array_almost_equal, theta, t3
                yield assert_array_almost_equal, vec, v3
                # doing the whole thing by hand
                T = np.eye(4)
                T[:3,3] = point
                iT = T.copy()
                iT[:3,3] *= -1
                M_hand = np.dot(T, np.dot(R, iT))
                yield assert_array_almost_equal, RP, M_hand
                # do round trip
                RP_back = rotation_matrix(t3, v3, p3)
                yield assert_array_almost_equal, RP, RP_back
