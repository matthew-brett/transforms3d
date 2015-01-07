''' Derivations for rotations of angle around axis '''
import numpy as np

from sympy import Symbol, symbols, sin, cos, acos, sqrt, solve
from sympy.matrices import Matrix, eye

from transforms3d.derivations.utils import matrices_equal

from transforms3d.derivations.quaternions import quat_around_axis, \
    quat2mat, qmult


def orig_aa2mat(angle, direction):
    # original transformations.py implementation of angle_axis2mat
    direction = np.array(direction)
    sina = sin(angle)
    cosa = cos(angle)
    # rotation matrix around unit vector
    R = Matrix(((cosa, 0.0,  0.0),
                (0.0,  cosa, 0.0),
                (0.0,  0.0,  cosa)))
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += Matrix((( 0.0,         -direction[2],  direction[1]),
                 ( direction[2], 0.0,          -direction[0]),
                 (-direction[1], direction[0],  0.0)))
    return R


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
    return Matrix([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]])


# Formulae for axis_angle to matrix
theta, v0, v1, v2 = symbols('theta, v0, v1, v2')
vec = (v0, v1, v2)

# These give the same formula
M1 = angle_axis2mat(theta, vec)
M2 = orig_aa2mat(theta, vec)
assert matrices_equal(M1, M2)
# This does not, but leads to the same numerical result (see tests)
M3 = quat2mat(quat_around_axis(theta, vec))
assert not matrices_equal(M1, M3)

# Applying a rotation about a point
R = Matrix(3, 3, lambda i, j : Symbol('R%d%d' % (i, j)))
aR = eye(4)
aR[:3,:3] = R
T = eye(4)
point = Matrix(3, 1, symbols('P0, P1, P2'))
T[:3,3] = point

# Move to new origin (inverse point), rotate, move back to original origin
T_R_iT =  T * aR * T.inv()

