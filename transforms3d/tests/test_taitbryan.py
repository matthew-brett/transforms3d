''' Tests for Euler angles using Tait-Bryan ZYX convention '''

import math
import numpy as np
from numpy import pi

from transforms3d import quaternions as tq
from transforms3d import taitbryan as ttb
from transforms3d import axangles as taa

from numpy.testing import assert_array_equal, assert_array_almost_equal

from transforms3d.tests.samples import euler_tuples

FLOAT_EPS = np.finfo(np.float64).eps


def x_only(x):
    cosx = np.cos(x)
    sinx = np.sin(x)
    return np.array(
        [[1, 0, 0],
         [0, cosx, -sinx],
         [0, sinx, cosx]])


def y_only(y):
    cosy = np.cos(y)
    siny = np.sin(y)
    return np.array(
        [[cosy, 0, siny],
         [0, 1, 0],
         [-siny, 0, cosy]])


def z_only(z):
    cosz = np.cos(z)
    sinz = np.sin(z)
    return np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]])


def sympy_euler(z, y, x):
    # The whole matrix formula for z,y,x rotations from Sympy
    cos = math.cos
    sin = math.sin
    # the following copy / pasted from Sympy - see derivations subdirectory
    return [
        [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
        [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
        [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
        ]


def is_valid_rotation(M):
    if not np.allclose(np.linalg.det(M), 1):
        return False
    return np.allclose(np.eye(3), np.dot(M, M.T))


def test_basic_euler():
    # some example rotations, in radians
    zr = 0.05
    yr = -0.4
    xr = 0.2
    # Rotation matrix composing the three rotations
    M = ttb.euler2mat(zr, yr, xr)
    # Corresponding individual rotation matrices
    M1 = ttb.euler2mat(zr, 0, 0)
    M2 = ttb.euler2mat(0, yr, 0)
    M3 = ttb.euler2mat(0, 0, xr)
    # which are all valid rotation matrices
    for rot in (M, M1, M2, M3):
        assert is_valid_rotation(rot)
    # Full matrix is composition of three individual matrices
    assert np.allclose(M, np.dot(M3, np.dot(M2, M1)))
    # Applying an opposite rotation same as inverse (the inverse is
    # the same as the transpose, but just for clarity)
    assert np.allclose(
        ttb.euler2mat(0, 0, -xr), np.linalg.inv(ttb.euler2mat(0, 0, xr)))


def test_euler_mat():
    M = ttb.euler2mat(0, 0, 0)
    assert_array_equal(M, np.eye(3))
    for x, y, z in euler_tuples:
        M1 = ttb.euler2mat(z, y, x)
        M2 = sympy_euler(z, y, x)
        assert_array_almost_equal(M1, M2)
        M3 = np.dot(x_only(x), np.dot(y_only(y), z_only(z)))
        assert_array_almost_equal(M1, M3)
        zp, yp, xp = ttb.mat2euler(M1)
        # The parameters may not be the same as input, but they give the
        # same rotation matrix
        M4 = ttb.euler2mat(zp, yp, xp)
        assert_array_almost_equal(M1, M4)


def sympy_euler2quat(z, y, x):
    # direct formula for z,y,x quaternion rotations using sympy
    # see derivations subfolder
    cos = math.cos
    sin = math.sin
    # the following copy / pasted from Sympy output
    return (cos(0.5*x)*cos(0.5*y)*cos(0.5*z) - sin(0.5*x)*sin(0.5*y)*sin(0.5*z),
            cos(0.5*x)*sin(0.5*y)*sin(0.5*z) + cos(0.5*y)*cos(0.5*z)*sin(0.5*x),
            cos(0.5*x)*cos(0.5*z)*sin(0.5*y) - cos(0.5*y)*sin(0.5*x)*sin(0.5*z),
            cos(0.5*x)*cos(0.5*y)*sin(0.5*z) + cos(0.5*z)*sin(0.5*x)*sin(0.5*y))


def crude_mat2euler(M):
    ''' The simplest possible - ignoring atan2 instability '''
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    return math.atan2(-r12, r11), math.asin(r13), math.atan2(-r23, r33)


def test_euler_instability():
    # Test for numerical errors in mat2euler
    # problems arise for cos(y) near 0
    po2 = pi / 2
    zyx = po2, po2, po2
    M = ttb.euler2mat(*zyx)
    # Round trip
    M_back = ttb.euler2mat(*ttb.mat2euler(M))
    assert np.allclose(M, M_back)
    # disturb matrix slightly
    M_e = M - FLOAT_EPS
    # round trip to test - OK
    M_e_back = ttb.euler2mat(*ttb.mat2euler(M_e))
    assert np.allclose(M_e, M_e_back)
    # not so with crude routine
    M_e_back = ttb.euler2mat(*crude_mat2euler(M_e))
    assert not np.allclose(M_e, M_e_back)


def test_quats():
    for x, y, z in euler_tuples:
        M1 = ttb.euler2mat(z, y, x)
        quatM = tq.mat2quat(M1)
        quat = ttb.euler2quat(z, y, x)
        assert tq.nearly_equivalent(quatM, quat)
        quatS = sympy_euler2quat(z, y, x)
        assert tq.nearly_equivalent(quat, quatS)
        zp, yp, xp = ttb.quat2euler(quat)
        # The parameters may not be the same as input, but they give the
        # same rotation matrix
        M2 = ttb.euler2mat(zp, yp, xp)
        assert_array_almost_equal(M1, M2)


def test_axangle_euler():
    # Conversion between axis, angle and euler
    for x, y, z in euler_tuples:
        M1 = ttb.euler2mat(z, y, x)
        ax, angle = ttb.euler2axangle(z, y, x)
        M2 = taa.axangle2mat(ax, angle)
        assert_array_almost_equal(M1, M2)
        zp, yp, xp = ttb.axangle2euler(ax, angle)
        M3 = ttb.euler2mat(zp, yp, xp)
        assert_array_almost_equal(M1, M3)
