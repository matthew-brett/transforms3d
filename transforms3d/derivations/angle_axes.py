''' Derivations for rotations of angle around axis '''
import numpy as np

from sympy import Symbol, symbols, sin, cos, acos, sqrt
from sympy.matrices import Matrix, eye

import os, sys
_my_path, _ = os.path.split(__file__)
sys.path.append(_my_path)
from quaternions import quat_around_axis, quat2mat, qmult
sys.path.remove(_my_path)

R = Matrix(3, 3, lambda i, j : Symbol('R%d%d' % (i, j)))
aR = eye(4)
aR[:3,:3] = R
T = eye(4)
T[:3,3] = symbols('T0', 'T1', 'T2')

# applying a rotation matrix after a translation
print T.inv() * aR * T


def angle_axis2quat(theta, vector):
    ''' Quaternion for rotation of angle `theta` around `vector`
    Notes
    -----
    Formula from http://mathworld.wolfram.com/EulerParameters.html
    '''
    t2 = theta / 2.0
    st2 = sin(t2)
    return cos(t2), vector[0]*st2, vector[1]*st2, vector[2]*st2


def quat2angle_axis(quat):
    ''' Convert quaternion to rotation of angle around axis
    '''
    w, x, y, z = quat
    vec = [x, y, z]
    n = sqrt(x*x + y*y + z*z)
    return  np.array([2 * acos(w), (np.array(vec) / n)[:]])


def angle_axis2mat(theta, vector):
    ''' Rotation matrix of angle `theta` around `vector`

    Parameters
    ----------
    theta : scalar
       angle of rotation
    vector : 3 element sequence
       vector specifying axis for rotation.
    is_normalized : bool, optional
       True if vector is already normalized (has norm of 1).  Default
       False

    Returns
    -------
    mat : array shape (3,3)
       rotation matrix specified rotation

    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    x, y, z = vector
    c = cos(theta); s = sin(theta); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    return np.array([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]])


theta, v0, v1, v2 = symbols('theta', 'v0', 'v1', 'v2')
vec = (v0, v1, v2)

q2m1 = quat2mat(quat_around_axis(theta, vec))
q2m2 = angle_axis2mat(theta, vec)
