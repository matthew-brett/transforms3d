r''' Euler angle rotations and their conversions for Tait-Bryan zyx convention

See :mod:`euler` for general discussion of Euler angles and conventions.

This module has specialized implementations of the extrinsic Z axis, Y axis, X
axis rotation convention.

The conventions in this module are therefore:

* axes $i, j, k$ are the $z, y, x$ axes respectively.  Thus
  an Euler angle vector $[ \alpha, \beta, \gamma ]$ in our convention
  implies a $\alpha$ radian rotation around the $z$ axis, followed by a $\beta$
  rotation around the $y$ axis, followed by a $\gamma$ rotation around the $x$
  axis.
* the rotation matrix applies on the left, to column vectors on the
  right, so if ``R`` is the rotation matrix, and ``v`` is a 3 x N matrix with N
  column vectors, the transformed vector set ``vdash`` is given by ``vdash =
  np.dot(R, v)``.
* extrinsic rotations - the axes are fixed, and do not move with the
  rotations.
* a right-handed coordinate system

The convention of rotation around ``z``, followed by rotation around
``y``, followed by rotation around ``x``, is known (confusingly) as
"xyz", pitch-roll-yaw, Cardan angles, or Tait-Bryan angles.

Terms used in function names:

* *mat* : array shape (3, 3) (3D non-homogenous coordinates)
* *euler* : (sequence of) rotation angles about the z, y, x axes (in that
  order)
* *axangle* : rotations encoded by axis vector and angle scalar
* *quat* : quaternion shape (4,)
'''

import math

from functools import reduce

import numpy as np

from .axangles import axangle2mat

_FLOAT_EPS_4 = np.finfo(float).eps * 4.0


def euler2mat(z, y, x):
    ''' Return matrix for rotations around z, y and x axes

    Uses the convention of static-frame rotation around the z, then y, then x
    axis.

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles

    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True

    The output rotation matrix is equal to the composition of the
    individual rotations

    >>> M1 = euler2mat(zrot, 0, 0)
    >>> M2 = euler2mat(0, yrot, 0)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True

    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.

    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)

    Rotations are counter-clockwise.

    >>> zred = np.dot(euler2mat(np.pi/2, 0, 0), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(0, np.pi/2, 0), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(0, 0, np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True

    Notes
    -----
    The direction of rotation is given by the right-hand rule.  Orient the
    thumb of the right hand along the axis around which the rotation occurs,
    with the end of the thumb at the positive end of the axis; curl your
    fingers; the direction your fingers curl is the direction of rotation.
    Therefore, the rotations are counterclockwise if looking along the axis of
    rotation from positive to negative.
    '''
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, (see
    ``eulerangles.py`` in ``derivations`` subdirectory)::

      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

    This gives the following solutions for ``[z, y, x]``::

       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)

    Problems arise when ``cos(y)`` is close to zero, because both of::

       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))

    will be close to ``atan2(0, 0)``, and highly unstable.

    The ``cy`` fix for numerical instability in this code is from: *Euler Angle
    Conversion* by Ken Shoemake, p222-9 ; in: *Graphics Gems IV*, Paul Heckbert
    (editor), Academic Press, 1994, ISBN: 0123361559.  Specifically it comes
    from ``EulerAngles.c`` and deals with the case where cos(y) is close to
    zero:

    * http://www.graphicsgems.org/
    * https://github.com/erich666/GraphicsGems/blob/master/gemsiv/euler_angle/EulerAngles.c#L68

    The code appears to be licensed (from the website) as "can be used without
    restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # (-cos(y)*sin(x))**2 + (cos(x)*cos(y))**2) =
    # (cos(y)**2)(sin(x)**2 + cos(x)**2) ==> (Pythagoras)
    # cos(y) = sqrt((-cos(y)*sin(x))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r23 * r23 + r33 * r33)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x


def euler2quat(z, y, x):
    ''' Return quaternion corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Notes
    -----
    Formula from Sympy - see ``eulerangles.py`` in ``derivations``
    subdirectory
    '''
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])


def quat2euler(q):
    ''' Return Euler angles corresponding to quaternion `q`

    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion

    Returns
    -------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``quat2mat`` and ``mat2euler`` functions, but
    the reduction in computation is small, and the code repetition is
    large.
    '''
    # delayed import to avoid cyclic dependencies
    from . import quaternions as nq
    return mat2euler(nq.quat2mat(q))


def euler2axangle(z, y, x):
    ''' Return angle, axis corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    vector : array shape (3,)
       axis around which rotation occurs
    theta : scalar
       angle of rotation

    Examples
    --------
    >>> vec, theta = euler2axangle(0, 1.5, 0)
    >>> np.allclose(vec, [0, 1, 0])
    True
    >>> theta
    1.5
    '''
    # delayed import to avoid cyclic dependencies
    from . import quaternions as nq
    return nq.quat2axangle(euler2quat(z, y, x))


def axangle2euler(vector, theta):
    ''' Convert axis, angle pair to Euler angles

    Parameters
    ----------
    vector : 3 element sequence
       vector specifying axis for rotation.
    theta : scalar
       angle of rotation

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Examples
    --------
    >>> z, y, x = axangle2euler([1, 0, 0], 0)
    >>> np.allclose((z, y, x), 0)
    True

    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``angle_axis2mat`` and ``mat2euler``
    functions, but the reduction in computation is small, and the code
    repetition is large.
    '''
    return mat2euler(axangle2mat(vector, theta))
