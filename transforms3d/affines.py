''' Routines working on homogenous affine - usually 4x4 - matrices '''

import math

import numpy as np

from .quaternions import axangle2rmat
from .shears import sutri2mat


def decompose44(A44):
    ''' Decompose 4x4 homogenous affine matrix into parts.

    The parts are translations, rotations, zooms, shears.

    This is the same as :func:`decompose` but specialized for 4x4 affines.

    Decomposes `A44` into ``T, R, Z, S``, such that::

       Smat = np.array([[1, S[0], S[1]],
                        [0,    1, S[2]],
                        [0,    0,    1]])
       RZS = np.dot(R, np.dot(np.diag(Z), Smat))
       A44 = np.eye(4)
       A44[:3,:3] = RZS
       A44[:-1,-1] = T

    The order of transformations is therefore shears, followed by
    zooms, followed by rotations, followed by translations.

    This routine only works for shape (4,4) matrices

    Parameters
    ----------
    A44 : array shape (4,4)

    Returns
    -------
    T : array, shape (3,)
       Translation vector
    R : array shape (3,3)
        rotation matrix
    Z : array, shape (3,)
       Zoom vector.  May have one negative zoom to prevent need for negative
       determinant R matrix above
    S : array, shape (3,)
       Shear vector, such that shears fill upper triangle above
       diagonal to form shear matrix. 

    Examples
    --------
    >>> T = [20, 30, 40] # translations
    >>> R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] # rotation matrix
    >>> Z = [2.0, 3.0, 4.0] # zooms
    >>> S = [0.2, 0.1, 0.3] # shears
    >>> # Now we make an affine matrix
    >>> A = np.eye(4)
    >>> Smat = np.array([[1, S[0], S[1]],
    ...                  [0,    1, S[2]],
    ...                  [0,    0,    1]])
    >>> RZS = np.dot(R, np.dot(np.diag(Z), Smat))
    >>> A[:3,:3] = RZS
    >>> A[:-1,-1] = T # set translations
    >>> Tdash, Rdash, Zdash, Sdash = decompose44(A)
    >>> np.allclose(T, Tdash)
    True
    >>> np.allclose(R, Rdash)
    True
    >>> np.allclose(Z, Zdash)
    True
    >>> np.allclose(S, Sdash)
    True

    Notes
    -----
    The implementation inspired by:

    *Decomposing a matrix into simple transformations* by Spencer
    W. Thomas, pp 320-323 in *Graphics Gems II*, James Arvo (editor),
    Academic Press, 1991, ISBN: 0120644819.

    The upper left 3x3 of the affine consists of a matrix we'll call
    RZS::

       RZS = R * Z *S

    where R is a rotation matrix, Z is a diagonal matrix of scalings::

       Z = diag([sx, sy, sz])

    and S is a shear matrix of form::

       S = [[1, sxy, sxz],
            [0,   1, syz],
            [0,   0,   1]])

    Running all this through sympy (see 'derivations' folder) gives
    ``RZS`` as ::

       [R00*sx, R01*sy + R00*sx*sxy, R02*sz + R00*sx*sxz + R01*sy*syz]
       [R10*sx, R11*sy + R10*sx*sxy, R12*sz + R10*sx*sxz + R11*sy*syz]
       [R20*sx, R21*sy + R20*sx*sxy, R22*sz + R20*sx*sxz + R21*sy*syz]

    ``R`` is defined as being a rotation matrix, so the dot products between
    the columns of ``R`` are zero, and the norm of each column is 1.  Thus
    the dot product::

       R[:,0].T * RZS[:,1]

    that results in::

       [R00*R01*sy + R10*R11*sy + R20*R21*sy + sx*sxy*R00**2 + sx*sxy*R10**2 + sx*sxy*R20**2]

    simplifies to ``sy*0 + sx*sxy*1`` == ``sx*sxy``.  Therefore::

       R[:,1] * sy = RZS[:,1] - R[:,0] * (R[:,0].T * RZS[:,1])

    allowing us to get ``sy`` with the norm, and sxy with ``R[:,0].T *
    RZS[:,1] / sx``.

    Similarly ``R[:,0].T * RZS[:,2]`` simplifies to ``sx*sxz``, and
    ``R[:,1].T * RZS[:,2]`` to ``sy*syz`` giving us the remaining
    unknowns. 
    '''
    A44 = np.asarray(A44)
    T = A44[:-1,-1]
    RZS = A44[:-1,:-1]
    # compute scales and shears
    M0, M1, M2 = np.array(RZS).T
    # extract x scale and normalize
    sx = math.sqrt(np.sum(M0**2))
    M0 /= sx
    # orthogonalize M1 with respect to M0
    sx_sxy = np.dot(M0, M1)
    M1 -= sx_sxy * M0
    # extract y scale and normalize    
    sy = math.sqrt(np.sum(M1**2))
    M1 /= sy
    sxy = sx_sxy / sx
    # orthogonalize M2 with respect to M0 and M1
    sx_sxz = np.dot(M0, M2)
    sy_syz = np.dot(M1, M2)
    M2 -= (sx_sxz * M0 + sy_syz * M1)
    # extract z scale and normalize
    sz = math.sqrt(np.sum(M2**2))
    M2 /= sz
    sxz = sx_sxz / sx
    syz = sy_syz / sy
    # Reconstruct rotation matrix, ensure positive determinant
    Rmat = np.array([M0, M1, M2]).T
    if np.linalg.det(Rmat) < 0:
        sx *= -1
        Rmat[:,0] *= -1
    return T, Rmat, np.array([sx, sy, sz]), np.array([sxy, sxz, syz])


def decompose(A):
    ''' Decompose homogenous affine transformation matrix `A` into parts.

    The parts are translations, rotations, zooms, shears.

    `A` can be any square matrix, but is typically shape (4,4).

    Decomposes A into ``T, R, Z, S``, such that, if A is shape (4,4)::

       Smat = np.array([[1, S[0], S[1]],
                        [0,    1, S[2]],
                        [0,    0,    1]])
       RZS = np.dot(R, np.dot(np.diag(Z), Smat))
       A = np.eye(4)
       A[:3,:3] = RZS
       A[:-1,-1] = T

    The order of transformations is therefore shears, followed by
    zooms, followed by rotations, followed by translations.

    The case above (A.shape == (4,4)) is the most common, and
    corresponds to a 3D affine, but in fact A need only be square.

    Parameters
    ----------
    A : array shape (N,N)

    Returns
    -------
    T : array, shape (N-1,)
       Translation vector
    R : array shape (N-1, N-1)
        rotation matrix
    Z : array, shape (N-1,)
       Zoom vector.  May have one negative zoom to prevent need for negative
       determinant R matrix above
    S : array, shape (P,)
       Shear vector, such that shears fill upper triangle above
       diagonal to form shear matrix.  P is the (N-2)th Triangular
       number, which happens to be 3 for a 4x4 affine.

    Examples
    --------
    >>> T = [20, 30, 40] # translations
    >>> R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] # rotation matrix
    >>> Z = [2.0, 3.0, 4.0] # zooms
    >>> S = [0.2, 0.1, 0.3] # shears
    >>> # Now we make an affine matrix
    >>> A = np.eye(4)
    >>> Smat = np.array([[1, S[0], S[1]],
    ...                  [0,    1, S[2]],
    ...                  [0,    0,    1]])
    >>> RZS = np.dot(R, np.dot(np.diag(Z), Smat))
    >>> A[:3,:3] = RZS
    >>> A[:-1,-1] = T # set translations
    >>> Tdash, Rdash, Zdash, Sdash = decompose(A)
    >>> np.allclose(T, Tdash)
    True
    >>> np.allclose(R, Rdash)
    True
    >>> np.allclose(Z, Zdash)
    True
    >>> np.allclose(S, Sdash)
    True

    Notes
    -----
    We have used a nice trick from SPM to get the shears.  Let us call the
    starting N-1 by N-1 matrix ``RZS``, because it is the composition of the
    rotations on the zooms on the shears.  The rotation matrix ``R`` must have
    the property ``np.dot(R.T, R) == np.eye(N-1)``.  Thus ``np.dot(RZS.T,
    RZS)`` will, by the transpose rules, be equal to ``np.dot((ZS).T, (ZS))``.
    Because we are doing shears with the upper right part of the matrix, that
    means that the Cholesky decomposition of ``np.dot(RZS.T, RZS)`` will give
    us our ``ZS`` matrix, from which we take the zooms from the diagonal, and
    the shear values from the off-diagonal elements.
    '''
    A = np.asarray(A)
    T = A[:-1,-1]
    RZS = A[:-1,:-1]
    ZS = np.linalg.cholesky(np.dot(RZS.T,RZS)).T
    Z = np.diag(ZS).copy()
    shears = ZS / Z[:,np.newaxis]
    n = len(Z)
    S = shears[np.triu(np.ones((n,n)), 1).astype(bool)]
    R = np.dot(RZS, np.linalg.inv(ZS))
    if np.linalg.det(R) < 0:
        Z[0] *= -1
        ZS[0] *= -1
        R = np.dot(RZS, np.linalg.inv(ZS))
    return T, R, Z, S


def compose(T, R, Z, S=None):
    ''' Compose translations, rotations, zooms, [shears]  to affine

    Parameters
    ----------
    T : array-like shape (N,)
        Translations, where N is usually 3 (3D case)
    R : array-like shape (N,N)
        Rotation matrix where N is usually 3 (3D case)
    Z : array-like shape (N,)
        Zooms, where N is usually 3 (3D case)
    S : array-like, shape (P,), optional
       Shear vector, such that shears fill upper triangle above
       diagonal to form shear matrix.  P is the (N-2)th Triangular
       number, which happens to be 3 for a 4x4 affine (3D case)

    Returns
    -------
    A : array, shape (N+1, N+1)
        Affine transformation matrix where N usually == 3
        (3D case)

    Examples
    --------
    >>> T = [20, 30, 40] # translations
    >>> R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] # rotation matrix
    >>> Z = [2.0, 3.0, 4.0] # zooms
    >>> A = compose(T, R, Z)
    >>> A
    array([[  0.,  -3.,   0.,  20.],
           [  2.,   0.,   0.,  30.],
           [  0.,   0.,   4.,  40.],
           [  0.,   0.,   0.,   1.]])
    >>> S = np.zeros(3)
    >>> B = compose(T, R, Z, S)
    >>> np.all(A == B)
    True

    A null set

    >>> compose(np.zeros(3), np.eye(3), np.ones(3), np.zeros(3))
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])
    '''
    n = len(T)
    R = np.asarray(R)
    if R.shape != (n,n):
        raise ValueError('Expecting shape (%d,%d) for rotations' % (n,n))
    A = np.eye(n+1)
    if not S is None:
        Smat = sutri2mat(S)
        ZS = np.dot(np.diag(Z), Smat)
    else:
        ZS = np.diag(Z)
    A[:n,:n] = np.dot(R, ZS)
    A[:n,n] = T[:]
    return A


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
    R = axangle2rmat(axis, angle)
    M[:3,:3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


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
    R = np.asarray(aff, dtype=np.float)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return direction, angle, point
