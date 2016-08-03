''' Test quaternion calculations '''

import math

import numpy as np

from nose.tools import (assert_raises, assert_true, assert_equal)

from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal)

from .. import quaternions as tq
from .. import axangles as taa

from .samples import euler_mats

# Example quaternions (from rotations)
euler_quats = []
for M in euler_mats:
    euler_quats.append(tq.mat2quat(M))
# M, quaternion pairs
eg_pairs = list(zip(euler_mats, euler_quats))

# Set of arbitrary unit quaternions
unit_quats = set()
params = range(-2,3)
for w in params:
    for x in params:
        for y in params:
            for z in params:
                q = (w, x, y, z)
                Nq = np.sqrt(np.dot(q, q))
                if not Nq == 0:
                    q = tuple([e / Nq for e in q])
                    unit_quats.add(q)


def test_fillpos():
    # Takes np array
    xyz = np.zeros((3,))
    w,x,y,z = tq.fillpositive(xyz)
    assert_equal(w, 1)
    # Or lists
    xyz = [0] * 3
    w,x,y,z = tq.fillpositive(xyz)
    assert_equal(w, 1)
    # Errors with wrong number of values
    assert_raises(ValueError, tq.fillpositive, [0, 0])
    assert_raises(ValueError, tq.fillpositive, [0]*4)
    # Errors with negative w2
    assert_raises(ValueError, tq.fillpositive, [1.0]*3)
    # Test corner case where w is near zero
    wxyz = tq.fillpositive([1, 0, 0])
    assert_equal(wxyz[0], 0.0)
    eps = np.finfo(float).eps
    wxyz = tq.fillpositive([1 + eps, 0, 0])
    assert_equal(wxyz[0], 0.0)
    # Bump up the floating point error - raises error
    assert_raises(ValueError, tq.fillpositive, [1 + eps * 3, 0, 0])
    # Increase threshold, happy again
    wxyz = tq.fillpositive([1 + eps * 3, 0, 0], w2_thresh=eps * -10)
    assert_equal(wxyz[0], 0.0)


def test_qconjugate():
    # Takes sequence
    cq = tq.qconjugate((1, 0, 0, 0))
    # Returns float type
    assert_true(cq.dtype.kind == 'f')


def test_quat2mat():
    # also tested in roundtrip case below
    M = tq.quat2mat([1, 0, 0, 0])
    yield assert_array_almost_equal, M, np.eye(3)
    # Non-unit quaternion
    M = tq.quat2mat([3, 0, 0, 0])
    yield assert_array_almost_equal, M, np.eye(3)
    M = tq.quat2mat([0, 1, 0, 0])
    yield assert_array_almost_equal, M, np.diag([1, -1, -1])
    # Non-unit quaternion, same result as normalized
    M = tq.quat2mat([0, 2, 0, 0])
    yield assert_array_almost_equal, M, np.diag([1, -1, -1])
    yield assert_array_almost_equal, M, np.diag([1, -1, -1])
    M = tq.quat2mat([0, 0, 0, 0])
    yield assert_array_almost_equal, M, np.eye(3)


def test_qinverse():
    # Takes sequence
    iq = tq.qinverse((1, 0, 0, 0))
    # Returns float type
    yield assert_true, iq.dtype.kind == 'f'
    for M, q in eg_pairs:
        iq = tq.qinverse(q)
        iqM = tq.quat2mat(iq)
        iM = np.linalg.inv(M)
        yield assert_true, np.allclose(iM, iqM)


def test_qeye():
    qi = tq.qeye()
    yield assert_true, qi.dtype.kind == 'f'
    yield assert_true, np.all([1,0,0,0]==qi)
    yield assert_true, np.allclose(tq.quat2mat(qi), np.eye(3))


def test_qnorm():
    qi = tq.qeye()
    yield assert_true, tq.qnorm(qi) == 1
    yield assert_true, tq.qisunit(qi)
    qi[1] = 0.2
    yield assert_true, not tq.qisunit(qi)


def test_qmult():
    # Test that quaternion * same as matrix * 
    for M1, q1 in eg_pairs[0::4]:
        for M2, q2 in eg_pairs[1::4]:
            q21 = tq.qmult(q2, q1)
            yield assert_array_almost_equal, np.dot(M2,M1), tq.quat2mat(q21)


def test_qrotate():
    for vec in np.eye(3):
        for M, q in eg_pairs:
            vdash = tq.rotate_vector(vec, q)
            vM = np.dot(M, vec.reshape(3,1))[:,0]
            yield assert_array_almost_equal, vdash, vM


def test_quaternion_reconstruction():
    # Test reconstruction of arbitrary unit quaternions
    for q in unit_quats:
        M = tq.quat2mat(q)
        qt = tq.mat2quat(M)
        # Accept positive or negative match
        posm = np.allclose(q, qt)
        negm = np.allclose(q, -qt)
        yield assert_true, posm or negm


def test_angle_axis2quat():
    q = tq.axangle2quat([1, 0, 0], 0)
    assert_array_equal(q, [1, 0, 0, 0])
    q = tq.axangle2quat([1, 0, 0], np.pi)
    assert_array_almost_equal(q, [0, 1, 0, 0])
    q = tq.axangle2quat([1, 0, 0], np.pi, True)
    assert_array_almost_equal(q, [0, 1, 0, 0])
    q = tq.axangle2quat([2, 0, 0], np.pi, False)
    assert_array_almost_equal(q, [0, 1, 0, 0])


def test_quat2axangle():
    ax, angle = tq.quat2axangle([1, 0, 0, 0])
    assert_array_equal(ax, [1, 0, 0])
    assert_array_equal(angle, 0)
    # Non-normalized quaternion, unit quaternion
    ax, angle = tq.quat2axangle([5, 0, 0, 0])
    assert_array_equal(ax, [1, 0, 0])
    assert_array_equal(angle, 0)
    # Rotation by 90 degrees around x
    r2d2 = np.sqrt(2) / 2.
    quat_x_90 = np.array([r2d2, r2d2, 0, 0])
    ax, angle = tq.quat2axangle(quat_x_90)
    assert_almost_equal(ax, [1, 0, 0])
    assert_almost_equal(angle, np.pi / 2)
    # Not-normalized version of same, gives same output
    ax, angle = tq.quat2axangle(quat_x_90 * 7)
    assert_almost_equal(ax, [1, 0, 0])
    assert_almost_equal(angle, np.pi / 2)
    # Any non-finite value gives nan angle
    for pos in range(4):
        for val in np.nan, np.inf, -np.inf:
            q = [1, 0, 0, 0]
            q[pos] = val
            ax, angle = tq.quat2axangle(q)
            assert_almost_equal(ax, [1, 0, 0])
            assert_true(np.isnan(angle))
    # Infinite length likewise, because of length overflow
    f64info = np.finfo(np.float64)
    ax, angle = tq.quat2axangle([2, f64info.max, 0, 0])
    assert_almost_equal(ax, [1, 0, 0])
    assert_true(np.isnan(angle))
    # Very small values give indentity transformation
    ax, angle = tq.quat2axangle([0, f64info.eps / 2, 0, 0])
    assert_almost_equal(ax, [1, 0, 0])
    assert_equal(angle, 0)


def sympy_aa2mat(vec, theta):
    # sympy expression derived from quaternion formulae
    v0, v1, v2 = vec # assumed normalized
    sin = math.sin
    cos = math.cos
    return np.array([
            [      1 - 2*v1**2*sin(0.5*theta)**2 - 2*v2**2*sin(0.5*theta)**2, -2*v2*cos(0.5*theta)*sin(0.5*theta) + 2*v0*v1*sin(0.5*theta)**2,  2*v1*cos(0.5*theta)*sin(0.5*theta) + 2*v0*v2*sin(0.5*theta)**2],
            [ 2*v2*cos(0.5*theta)*sin(0.5*theta) + 2*v0*v1*sin(0.5*theta)**2,       1 - 2*v0**2*sin(0.5*theta)**2 - 2*v2**2*sin(0.5*theta)**2, -2*v0*cos(0.5*theta)*sin(0.5*theta) + 2*v1*v2*sin(0.5*theta)**2],
            [-2*v1*cos(0.5*theta)*sin(0.5*theta) + 2*v0*v2*sin(0.5*theta)**2,  2*v0*cos(0.5*theta)*sin(0.5*theta) + 2*v1*v2*sin(0.5*theta)**2,       1 - 2*v0**2*sin(0.5*theta)**2 - 2*v1**2*sin(0.5*theta)**2]])


def sympy_aa2mat2(vec, theta):
    # sympy expression derived from direct formula
    v0, v1, v2 = vec # assumed normalized
    sin = math.sin
    cos = math.cos
    return np.array([
            [v0**2*(1 - cos(theta)) + cos(theta),
             -v2*sin(theta) + v0*v1*(1 - cos(theta)),
             v1*sin(theta) + v0*v2*(1 - cos(theta))],
            [v2*sin(theta) + v0*v1*(1 - cos(theta)),
             v1**2*(1 - cos(theta)) + cos(theta),
             -v0*sin(theta) + v1*v2*(1 - cos(theta))],
            [-v1*sin(theta) + v0*v2*(1 - cos(theta)),
              v0*sin(theta) + v1*v2*(1 - cos(theta)),
              v2**2*(1 - cos(theta)) + cos(theta)]])


def test_axis_angle():
    for M, q in eg_pairs:
        vec, theta = tq.quat2axangle(q)
        q2 = tq.axangle2quat(vec, theta)
        yield tq.nearly_equivalent, q, q2
        aa_mat = taa.axangle2mat(vec, theta)
        yield assert_array_almost_equal, aa_mat, M
        aa_mat2 = sympy_aa2mat(vec, theta)
        yield assert_array_almost_equal, aa_mat, aa_mat2
        aa_mat22 = sympy_aa2mat2(vec, theta)
        yield assert_array_almost_equal, aa_mat, aa_mat22
