''' Test quaternion calculations '''

import math
from itertools import product
from os.path import dirname, join as pjoin

import numpy as np

from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal)

from transforms3d import quaternions as tq
from transforms3d import axangles as taa
from transforms3d.testing import assert_raises

from transforms3d.tests.samples import euler_mats


DATA_DIR = pjoin(dirname(__file__), 'data')

# Example quaternions (from rotations)
euler_quats = []
for M in euler_mats:
    euler_quats.append(tq.mat2quat(M))
# M, quaternion pairs
eg_pairs = list(zip(euler_mats, euler_quats))

# Sets of arbitrary unit and not-unit quaternions
quats = set()
unit_quats = set()
params = np.arange(-2, 3, 0.5)
for w, x, y, z in product(params, params, params, params):
    q = (w, x, y, z)
    Nq = np.sqrt(np.dot(q, q))
    if Nq == 0:
        continue
    quats.add(q)
    q_n = tuple([e / Nq for e in q])
    unit_quats.add(q_n)


def test_fillpos():
    # Takes np array
    xyz = np.zeros((3,))
    w,x,y,z = tq.fillpositive(xyz)
    assert w == 1
    # Or lists
    xyz = [0] * 3
    w,x,y,z = tq.fillpositive(xyz)
    assert w == 1
    # Errors with wrong number of values
    assert_raises(ValueError, tq.fillpositive, [0, 0])
    assert_raises(ValueError, tq.fillpositive, [0]*4)
    # Errors with negative w2
    assert_raises(ValueError, tq.fillpositive, [1.0]*3)
    # Test corner case where w is near zero
    wxyz = tq.fillpositive([1, 0, 0])
    assert wxyz[0] == 0.0
    eps = np.finfo(float).eps
    wxyz = tq.fillpositive([1 + eps, 0, 0])
    assert wxyz[0] == 0.0
    # Bump up the floating point error - raises error
    assert_raises(ValueError, tq.fillpositive, [1 + eps * 3, 0, 0])
    # Increase threshold, happy again
    wxyz = tq.fillpositive([1 + eps * 3, 0, 0], w2_thresh=eps * -10)
    assert wxyz[0] == 0.0


def test_qconjugate():
    # Takes sequence
    cq = tq.qconjugate((1, 0, 0, 0))
    # Returns float type
    assert cq.dtype.kind == 'f'


def test_quat2mat():
    # also tested in roundtrip case below
    M = tq.quat2mat([1, 0, 0, 0])
    assert_array_almost_equal(M, np.eye(3))
    # Non-unit quaternion
    M = tq.quat2mat([3, 0, 0, 0])
    assert_array_almost_equal(M, np.eye(3))
    M = tq.quat2mat([0, 1, 0, 0])
    assert_array_almost_equal(M, np.diag([1, -1, -1]))
    # Non-unit quaternion, same result as normalized
    M = tq.quat2mat([0, 2, 0, 0])
    assert_array_almost_equal(M, np.diag([1, -1, -1]))
    assert_array_almost_equal(M, np.diag([1, -1, -1]))
    M = tq.quat2mat([0, 0, 0, 0])
    assert_array_almost_equal(M, np.eye(3))


def test_qinverse():
    # Takes sequence
    iq = tq.qinverse((1, 0, 0, 0))
    # Returns float type
    assert iq.dtype.kind == 'f'
    for M, q in eg_pairs:
        iq = tq.qinverse(q)
        iqM = tq.quat2mat(iq)
        iM = np.linalg.inv(M)
        assert np.allclose(iM, iqM)


def test_qeye():
    qi = tq.qeye()
    assert qi.dtype.kind == 'f'
    assert np.all([1,0,0,0]==qi)
    assert np.allclose(tq.quat2mat(qi), np.eye(3))


def test_qexp():
    angular_velocity_pure_quaterion = np.array([0., math.pi, 0, 0])
    dt = 1.0
    q_integrate_angular_vel = tq.qexp(angular_velocity_pure_quaterion * dt/2)
    # See https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/ near the end. 
    # The formula q(t) = qexp(q_w * t / 2), where q_w is [0 w_x, w_y, w_z]
    # represents angular velocity in x,y,z, produces a quaternion that
    # represents the integration of angular velocity w during time t  so this
    # test rotate the y vector [0 1 0], at math.pi ras/s around the x axis for
    # 1 sec. This is the main use case for using qexp
    assert np.allclose(tq.rotate_vector(np.array([0,1,0]), q_integrate_angular_vel), np.array([0,-1,0]))

    # from https://www.mathworks.com/help/aerotbx/ug/quatexp.html
    assert np.allclose(tq.qexp(np.array([0, 0, 0.7854, 0])), np.array([0.7071, 0., 0.7071, 0.]), atol=1e-05)


def test_qlog():
    # From https://www.mathworks.com/help/aerotbx/ug/quatlog.html?s_tid=doc_ta
    assert np.allclose(tq.qlog(np.array([0.7071, 0, 0.7071, 0])), np.array([0., 0., 0.7854, 0.]), atol=1e-05)


def test_qexp_qlog():
    # Test round trip
    for unit_quat in unit_quats:
        assert tq.nearly_equivalent(tq.qlog(tq.qexp(unit_quat)), unit_quat)
        assert tq.nearly_equivalent(tq.qexp(tq.qlog(unit_quat)), unit_quat)


def test_qpow():
    # https://www.mathworks.com/help/aerotbx/ug/quatpower.html?searchHighlight=quaternion%20power&s_tid=doc_srchtitle
    assert np.allclose(tq.qpow(np.array([0.7071, 0, 0.7071, 0]), 2), np.array([0, 0, 1, 0]), atol=1e-05)   


def test_qexp_matlab():
    from scipy.io import loadmat
    ml_quats = loadmat(pjoin(DATA_DIR, 'processed_quats.mat'))
    o_quats, o_unit_quats, quat_e, quat_p = [
        ml_quats[k] for k in ['quats', 'unit_quats', 'quat_e', 'quat_p']]
    for i in range(len(o_quats)):
        assert np.allclose(tq.qexp(o_quats[i]), quat_e[i])
    for i in range(len(o_unit_quats)):
        for p_i, p in enumerate(np.arange(1, 4, 0.5)):
            assert np.allclose(tq.qpow(o_unit_quats[i], p), quat_p[0, p_i][i])


def test_qnorm():
    qi = tq.qeye()
    assert tq.qnorm(qi) == 1
    assert tq.qisunit(qi)
    qi[1] = 0.2
    assert not tq.qisunit(qi)
    # Test norm is sqrt of scalar for multiplication with conjugate.
    # https://en.wikipedia.org/wiki/Quaternion#Conjugation,_the_norm,_and_reciprocal
    for q in quats:
        q_c = tq.qconjugate(q)
        exp_norm = np.sqrt(tq.qmult(q, q_c)[0])
        assert np.allclose(tq.qnorm(q), exp_norm)


def test_qmult():
    # Test that quaternion * same as matrix * 
    for M1, q1 in eg_pairs[0::4]:
        for M2, q2 in eg_pairs[1::4]:
            q21 = tq.qmult(q2, q1)
            assert_array_almost_equal(np.dot(M2,M1), tq.quat2mat(q21))


def test_qrotate():
    for vec in np.eye(3):
        for M, q in eg_pairs:
            vdash = tq.rotate_vector(vec, q)
            vM = np.dot(M, vec.reshape(3,1))[:,0]
            assert_array_almost_equal(vdash, vM)


def test_quaternion_reconstruction():
    # Test reconstruction of arbitrary unit quaternions
    for q in unit_quats:
        M = tq.quat2mat(q)
        qt = tq.mat2quat(M)
        # Accept positive or negative match
        posm = np.allclose(q, qt)
        negm = np.allclose(q, -qt)
        assert posm or negm


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
            assert np.isnan(angle)
    # Infinite length likewise, because of length overflow
    f64info = np.finfo(np.float64)
    ax, angle = tq.quat2axangle([2, f64info.max, 0, 0])
    assert_almost_equal(ax, [1, 0, 0])
    assert np.isnan(angle)
    # Very small values give indentity transformation
    ax, angle = tq.quat2axangle([0, f64info.eps / 2, 0, 0])
    assert_almost_equal(ax, [1, 0, 0])
    assert angle == 0


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
        assert tq.nearly_equivalent(q, q2)
        aa_mat = taa.axangle2mat(vec, theta)
        assert_array_almost_equal(aa_mat, M)
        aa_mat2 = sympy_aa2mat(vec, theta)
        assert_array_almost_equal(aa_mat, aa_mat2)
        aa_mat22 = sympy_aa2mat2(vec, theta)
        assert_array_almost_equal(aa_mat, aa_mat22)


def test_rotate_normalize():
    # From: https://github.com/matthew-brett/transforms3d/issues/16
    q = np.array([1 ,0 ,1, 0])
    r = np.array([1,1,1])
    R = tq.quat2mat(q)
    mat_rot_vec = np.dot(R,r)
    # Using Trans3d library directly to rotate the vector.
    # This q is not normalized.
    non_norm_rv = tq.rotate_vector(r, q)
    assert not np.allclose(non_norm_rv, mat_rot_vec)
    # This q normalized.
    norm_rv = tq.rotate_vector(r, q / tq.qnorm(q))
    assert np.allclose(norm_rv, mat_rot_vec)
    # Specify normalization.
    norm_rv2 = tq.rotate_vector(r, q, is_normalized=False)
    assert np.allclose(norm_rv2, mat_rot_vec)
    norm_rv3 = tq.rotate_vector(r, q, False)
    assert np.allclose(norm_rv3, mat_rot_vec)
