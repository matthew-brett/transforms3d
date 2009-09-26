import math

import numpy as np

import transforms3d.affines as taf

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_almost_equal, dec


def test_aa_points():
    i3 = math.sqrt(1/3.0)
    for theta in (-0.2, 0.5):
        for vec in np.r_[np.eye(3), [[i3, i3, i3]]]:
            for point in [[0.3, 0.4, 0.5],[-0.2, 0, 4.0]]:
                R = taf.from_angle_axis_point(theta, vec)
                t2, v2, p2 = taf.to_angle_axis_point(R)
                yield assert_array_almost_equal, theta, t2
                yield assert_array_almost_equal, vec, v2
                yield assert_array_almost_equal, p2[:3], 0
                # recovering a point
                point = [0.3, 0.4, 0.5]
                RP = taf.from_angle_axis_point(theta, vec, point)
                t3, v3, p3 = taf.to_angle_axis_point(RP)
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
                RP_back = taf.from_angle_axis_point(t3, v3, p3)
                yield assert_array_almost_equal, RP, RP_back
