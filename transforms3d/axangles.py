""" Functions for working with axis, angle rotations

See :mod:`quaternions` for conversions between axis, angle pairs and
quaternions.

Terms used in function names:

* *mat* : array shape (3, 3) (3D non-homogenous coordinates)
* *aff* : affine array shape (4, 4) (3D homogenous coordinates)
* *axangle* : rotations encoded by axis vector and angle scalar
"""

import math
import numpy as np


def axangle2mat(axis, angle, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`

    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.

    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation

    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    x, y, z = axis
    if not is_normalized:
        n = math.sqrt(x*x + y*y + z*z)
        x = x/n
        y = y/n
        z = z/n
    c = math.cos(angle); s = math.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    return np.array([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]])


def axangle2aff(axis, angle, point=None):
    """Return affine encoding rotation by `angle` about `axis`.

    Parameters
    ----------
    axis : array shape (3,)
       vector giving axis of rotation
    angle : scalar
       angle of rotation, in radians.

    Returns
    -------
    A : array shape (4, 4)
        Affine array to be multiplied on left of coordinate column vector to
        apply given rotation.

    Examples
    --------
    >>> angle = (np.random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = axangle2aff(direc, angle, point)
    >>> R1 = axangle2aff(direc, angle-2*math.pi, point)
    >>> np.allclose(R0, R1)
    True
    >>> R0 = axangle2aff(direc, angle, point)
    >>> R1 = axangle2aff(-direc, -angle, point)
    >>> np.allclose(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, axangle2aff(direc, math.pi*2))
    True
    >>> np.allclose(2., np.trace(axangle2aff(direc,
    ...                                      math.pi/2,
    ...                                      point)))
    True

    Notes
    -----
    Applying a rotation around a point is the same as applying a
    translation of ``-point`` to move ``point`` to the origin, rotating,
    then applying a translation of ``point``.  If ``R`` is the rotation
    matrix, than the affine for the rotation about point P is::

       [R00, R01, R02, P0 - P0*R00 - P1*R01 - P2*R02]
       [R10, R11, R12, P1 - P0*R10 - P1*R11 - P2*R12]
       [R20, R21, R22, P2 - P0*R20 - P1*R21 - P2*R22]
       [  0,   0,   0,                             1]

    (see derivations)
    """
    M = np.eye(4)
    R = axangle2mat(axis, angle)
    M[:3,:3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64)
        M[:3, 3] = point - np.dot(R, point)
    return M


def mat2axangle(mat, unit_thresh=1e-5):
    """Return axis, angle and point from (3, 3) matrix `mat`

    Parameters
    ----------
    mat : array-like shape (3, 3)
        Rotation matrix
    unit_thresh : float, optional
        Tolerable difference from 1 when testing for unit eigenvalues to
        confirm `mat` is a rotation matrix.

    Returns
    -------
    axis : array shape (3,)
       vector giving axis of rotation
    angle : scalar
       angle of rotation in radians.

    Examples
    --------
    >>> direc = np.random.random(3) - 0.5
    >>> angle = (np.random.random() - 0.5) * (2*math.pi)
    >>> R0 = axangle2mat(direc, angle)
    >>> direc, angle = mat2axangle(R0)
    >>> R1 = axangle2mat(direc, angle)
    >>> np.allclose(R0, R1)
    True

    Notes
    -----
    http://en.wikipedia.org/wiki/Rotation_matrix#Axis_of_a_rotation
    """
    M = np.asarray(mat, dtype=np.float64)
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    L, W = np.linalg.eig(M.T)
    i = np.where(np.abs(L - 1.0) < unit_thresh)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # rotation angle depending on direction
    cosa = (np.trace(M) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (M[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (M[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (M[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return direction, angle


def aff2axangle(aff):
    """Return axis, angle and point from affine

    Parameters
    ----------
    aff : array-like shape (4,4)

    Returns
    -------
    axis : array shape (3,)
       vector giving axis of rotation
    angle : scalar
       angle of rotation in radians.
    point : array shape (3,)
       point around which rotation is performed

    Examples
    --------
    >>> direc = np.random.random(3) - 0.5
    >>> angle = (np.random.random() - 0.5) * (2*math.pi)
    >>> point = np.random.random(3) - 0.5
    >>> R0 = axangle2aff(direc, angle, point)
    >>> direc, angle, point = aff2axangle(R0)
    >>> R1 = axangle2aff(direc, angle, point)
    >>> np.allclose(R0, R1)
    True

    Notes
    -----
    http://en.wikipedia.org/wiki/Rotation_matrix#Axis_of_a_rotation
    """
    R = np.asarray(aff, dtype=np.float64)
    direction, angle = mat2axangle(R[:3, :3])
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    L, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(L) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    return direction, angle, point
