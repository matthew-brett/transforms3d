import math

import numpy as np

import transforms3d.quaternions as tq
import transforms3d.taitbryan as ttb

from transforms3d.gohlketransforms import quaternion_matrix, \
    quaternion_from_matrix, euler_matrix, \
    quaternion_from_euler, rotation_matrix, \
    rotation_from_matrix

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_almost_equal, dec

# Set of arbitrary unit quaternions
unit_quats = set()
params = (-3,4)
for w in range(*params):
    for x in range(*params):
        for y in range(*params):
            for z in range(*params):
                q = (w, x, y, z)
                Nq = np.sqrt(np.dot(q, q))
                if not Nq == 0:
                    q = tuple([e / Nq for e in q])
                    unit_quats.add(q)


# Example rotations '''
eg_rots = []
params = (-np.pi,np.pi,np.pi/2)
zs = np.arange(*params)
ys = np.arange(*params)
xs = np.arange(*params)
for z in zs:
    for y in ys:
        for x in xs:
            eg_rots.append((x, y, z))


def our_quat(tquat):
    # converts from transformations quaternion order to ours
    x, y, z, w = tquat
    return [w, x, y, z]
    

def trans_quat(oquat):
    # converts from our quaternion to transformations order
    w, x, y, z = oquat
    return [x, y, z, w]


def test_quaternion_imps():
    for x, y, z in eg_rots:
        M = ttb.euler2mat(z, y, x)
        quat = tq.mat2quat(M)
        # Against transformations code
        tM = quaternion_matrix(trans_quat(quat))
        yield assert_array_almost_equal, M, tM[:3,:3]
        M44 = np.eye(4)
        M44[:3,:3] = M
        tQ = quaternion_from_matrix(M44)
        yield assert_true, tq.nearly_equivalent(quat, our_quat(tQ))


def test_euler_imps():
    for x, y, z in eg_rots:
        M1 = euler_matrix(z, y, x,'szyx')[:3,:3]
        M2 = ttb.euler2mat(z, y, x)
        yield assert_array_almost_equal, M1, M2
        q1 = quaternion_from_euler(z, y, x, 'szyx')
        q2 = ttb.euler2quat(z, y, x)
        yield assert_true, tq.nearly_equivalent(our_quat(q1), q2)


def test_angle_axis_imps():
    for x, y, z in eg_rots:
        M = ttb.euler2mat(z, y, x)
        q = tq.mat2quat(M)
        theta, vec = tq.quat2angle_axis(q)
        M1 = tq.angle_axis2mat(theta, vec)
        M2 = rotation_matrix(theta, vec)[:3,:3]
        yield assert_array_almost_equal, M1, M2
        M3 = np.eye(4)
        M3[:3,:3] = M
        t2, v2, point = rotation_from_matrix(M3)
        M4 = tq.angle_axis2mat(t2, v2)
        yield assert_array_almost_equal, M, M4


