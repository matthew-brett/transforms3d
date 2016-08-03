''' Functions for working with shears

Terms used in function names:

* *mat* : array shape (3, 3) (3D non-homogenous coordinates)
* *aff* : affine array shape (4, 4) (3D homogenous coordinates)
* *striu* : shears encoded by vector giving triangular portion above diagonal
  of NxN array (for ND transformation)
* *sadn* : shears encoded by angle scalar, direction vector, normal vector
  (with optional point vector)
'''

import math

import warnings

import numpy as np

from .utils import normalized_vector, vector_norm


# Caching dictionary for common shear Ns, indices
_shearers = {}
for n in range(1,11):
    x = (n**2 + n)/2.0
    i = n+1
    _shearers[x] = (i, np.triu(np.ones((i,i)), 1).astype(bool))


def striu2mat(striu):
    ''' Construct shear matrix from upper triangular vector

    Parameters
    ----------
    striu : array, shape (N,)
       vector giving triangle above diagonal of shear matrix.

    Returns
    -------
    SM : array, shape (N, N)
       shear matrix

    Examples
    --------
    >>> S = [0.1, 0.2, 0.3]
    >>> striu2mat(S)
    array([[ 1. ,  0.1,  0.2],
           [ 0. ,  1. ,  0.3],
           [ 0. ,  0. ,  1. ]])
    >>> striu2mat([1])
    array([[ 1.,  1.],
           [ 0.,  1.]])
    >>> striu2mat([1, 2])
    Traceback (most recent call last):
       ...
    ValueError: 2 is a strange number of shear elements

    Notes
    -----
    Shear lengths are triangular numbers.

    See http://en.wikipedia.org/wiki/Triangular_number
    '''
    n = len(striu)
    # cached case
    if n in _shearers:
        N, inds = _shearers[n]
    else: # General case
        N = ((-1+math.sqrt(8*n+1))/2.0)+1 # n+1 th root
        if N != math.floor(N):
            raise ValueError('%d is a strange number of shear elements' %
                             n)
        N = int(N)
        inds = np.triu(np.ones((N,N)), 1).astype(bool)
    M = np.eye(N)
    M[inds] = striu
    return M


def sadn2mat(angle, direction, normal):
    """Matrix for shear by `angle` along `direction` vector on shear plane.

    The shear plane is defined by normal vector `normal`, and passes through
    the origin. The direction vector must be orthogonal to the plane's normal
    vector.

    A point P is transformed by the shear matrix into P" such that
    the vector P-P" is parallel to the direction vector and its extent is
    given by the angle of P-P'-P", where P' is the orthogonal projection
    of P onto the shear plane.

    Parameters
    ----------
    angle : scalar
       angle to shear, in radians
    direction : array-like, shape (3,)
       direction along which to shear
    normal : array-like, shape (3,)
       vector defining shear plane, where shear plane passes through
       origin

    Returns
    -------
    mat : array shape (3,3)
       shear matrix

    Examples
    --------
    >>> angle = (np.random.random() - 0.5) * 4*math.pi
    >>> direct = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> S = sadn2aff(angle, direct, normal)
    >>> np.allclose(1.0, np.linalg.det(S))
    True
    """
    if abs(np.dot(normal, direction)) > 1e-5:
        raise ValueError("direction, normal vectors not orthogonal")
    normal = normalized_vector(normal)
    direction = normalized_vector(direction)
    angle = math.tan(angle)
    M = np.eye(3)
    M += angle * np.outer(direction, normal)
    return M


def sadn2aff(angle, direction, normal, point=None):
    """Affine for shear by `angle` along vector `direction` on shear plane.

    The shear plane is defined by a point and normal vector. The direction
    vector must be orthogonal to the plane's normal vector.

    A point P is transformed by the shear matrix into P" such that
    the vector P-P" is parallel to the direction vector and its extent is
    given by the angle of P-P'-P", where P' is the orthogonal projection
    of P onto the shear plane.

    Parameters
    ----------
    angle : scalar
       angle to shear, in radians
    direction : array-like, shape (3,)
       direction along which to shear
    normal : array-like, shape (3,)
       vector normal to shear-plane
    point : None or array-like, shape (3,), optional
       point, that, with `normal` defines shear plane.  Defaults to
       None, equivalent to shear-plane through origin.

    Returns
    -------
    aff : array shape (4,4)
       affine shearing matrix

    Examples
    --------
    >>> angle = (np.random.random() - 0.5) * 4*math.pi
    >>> direct = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> S = sadn2mat(angle, direct, normal)
    >>> np.allclose(1.0, np.linalg.det(S))
    True
    """
    M = np.eye(4)
    normal = normalized_vector(normal)
    direction = normalized_vector(direction)
    angle = math.tan(angle)
    M[:3, :3] = np.eye(3) + angle * np.outer(direction, normal)
    if point is not None:
        M[:3, 3] = -angle * np.dot(point, normal) * direction
    return M


def mat2sadn(mat):
    """Return shear angle, direction and plane normal from shear matrix.

    Parameters
    ----------
    mat : array-like, shape (3,3)
       shear matrix

    Returns
    -------
    angle : scalar
       angle to shear, in radians
    direction : array, shape (3,)
       direction along which to shear
    normal : array, shape (3,)
       vector defining shear plane, where shear plane passes through
       origin

    Examples
    --------
    >>> M = sadn2mat(0.5, [1, 0, 0], [0, 1, 0])
    >>> angle, direction, normal = mat2sadn(M)
    >>> angle, direction, normal
    (0.5, array([-1.,  0.,  0.]), array([ 0., -1.,  0.]))
    >>> M_again = sadn2mat(angle, direction, normal)
    >>> np.allclose(M, M_again)
    True
    """
    mat = np.asarray(mat)
    # normal: cross independent eigenvectors corresponding to the eigenvalue 1
    l, V = np.linalg.eig(mat)
    near_1, = np.nonzero(abs(np.real(l.squeeze()) - 1.0) < 1e-4)
    if near_1.size < 2:
        raise ValueError("no two linear independent eigenvectors found %s" % l)
    V = np.real(V[:, near_1]).squeeze().T
    lenorm = -1.0
    for i0, i1 in ((0, 1), (0, 2), (1, 2)):
        n = np.cross(V[i0], V[i1])
        l = vector_norm(n)
        if l > lenorm:
            lenorm = l
            normal = n
    normal /= lenorm
    # direction and angle
    direction = np.dot(mat - np.eye(3), normal)
    angle = vector_norm(direction)
    direction /= angle
    angle = math.atan(angle)
    return angle, direction, normal


def aff2sadn(aff):
    """Return shear angle, direction and plane normal from shear matrix.

    Parameters
    ----------
    mat : array-like, shape (3,3)
       shear matrix.

    Returns
    -------
    angle : scalar
       angle to shear, in radians
    direction : array, shape (3,)
       direction along which to shear
    normal : array, shape (3,)
       vector normal to shear plane
    point : array, shape (3,)
       point that, with `normal`, defines shear plane.

    Examples
    --------
    >>> A = sadn2aff(0.5, [1, 0, 0], [0, 1, 0])
    >>> angle, direction, normal, point = aff2sadn(A)
    >>> angle, direction, normal, point
    (0.5, array([-1.,  0.,  0.]), array([ 0., -1.,  0.]), array([ 0.,  0.,  0.]))
    >>> A_again = sadn2aff(angle, direction, normal, point)
    >>> np.allclose(A, A_again)
    True
    """
    warnings.warn('This function can be numerically unstable; use with care')
    aff = np.asarray(aff)
    angle, direction, normal = mat2sadn(aff[:3,:3])
    # point: eigenvector corresponding to eigenvalue 1
    l, V = np.linalg.eig(aff)
    near_1, = np.nonzero(abs(np.real(l.squeeze()) - 1.0) < 1e-8)
    if near_1.size == 0:
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    point = np.real(V[:, near_1[-1]]).squeeze()
    point = point[:3] / point[3]
    return angle, direction, normal, point
