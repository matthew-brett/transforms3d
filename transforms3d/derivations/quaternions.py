''' Symbolic formulae for quaternions '''

from sympy import Symbol, cos, sin
from sympy.matrices import Matrix


def qmult(q1, q2):
    ''' Multiply two quaternions

    Parameters
    ----------
    q1 : 4 element sequence
    q2 : 4 element sequence

    Returns
    -------
    q12 : shape (4,) array

    Notes
    -----
    See : http://en.wikipedia.org/wiki/Quaternions#Hamilton_product
    '''
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return w, x, y, z


def quat_around_axis(theta, axis):
    ''' Quaternion for rotation of angle `theta` around axis `axis`

    Parameters
    ----------
    theta : symbol
       angle of rotation
    axis : 3 element sequence
       vector (assumed normalized) specifying axis for rotation

    Returns
    -------
    quat : 4 element sequence of symbols
       quaternion giving specified rotation

    Notes
    -----
    Formula from http://mathworld.wolfram.com/EulerParameters.html
    '''
    # axis vector assumed normalized
    t2 = theta / 2.0
    st2 = sin(t2)
    return (cos(t2),
            st2 * axis[0],
            st2 * axis[1],
            st2 * axis[2])


def quat2mat(quat):
    ''' Symbolic conversion from quaternion to rotation matrix

    For a unit quaternion
    
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    '''
    w, x, y, z = quat
    return Matrix([
            [1 - 2*y*y-2*z*z, 2*x*y - 2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])
