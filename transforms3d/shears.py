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

import numpy as np

from .utils import normalized_vector


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
    array([[1. , 0.1, 0.2],
           [0. , 1. , 0.3],
           [0. , 0. , 1. ]])
    >>> striu2mat([1])
    array([[1., 1.],
           [0., 1.]])
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


def inverse_outer(mat):
    """ Return scalar t, unit vectors `a`, `b` so `mat = t * np.outer(a, b)`

    Parameters
    ----------
    mat : array-like, shape (3,3)
       shear matrix

    Returns
    -------
    t : float
        Scalar such that `mat = t * np.outer(a, b)`
    a : array, shape (3,)
        Unit vector such that `mat = t * np.outer(a, b)`
    b : array, shape (3,)
        Unit vector such that `mat = t * np.outer(a, b)`
    """
    u, s, vh = np.linalg.svd(mat)
    return s[0], u.T[0], vh[0]


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
    (0.5, array([1., 0., 0.]), array([0., 1., 0.]))
    >>> M_again = sadn2mat(angle, direction, normal)
    >>> np.allclose(M, M_again)
    True

    Notes
    -----
    The shear matrix we are decomposing was calculated using:

    .. code: python

        mat = np.eye(3) + angle * np.outer(direction, normal)

    So the idea is to use an "inverse outer product" to recover the shears.
    See :func:`inverse_outer` for the implementation.
    """
    mat = np.asarray(mat)
    # normal: cross independent eigenvectors corresponding to the eigenvalue 1
    tan, direction, normal = inverse_outer(mat - np.eye(3))
    return math.atan(tan), direction, normal


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
    >>> angle, direction, normal
    (0.5, array([1., 0., 0.]), array([0., 1., 0.]))
    >>> assert np.all(point == [0, 0, 0])
    >>> A_again = sadn2aff(angle, direction, normal, point)
    >>> np.allclose(A, A_again)
    True

    Notes
    -----
    The translation part of the affine shear matrix is calculated using:

    .. code: python

        M[:3, 3] = -angle * np.dot(point, normal) * direction

    This holds for the ``i``th coordinate:

    .. code: python

        M[i, 3] = -angle * np.dot(point, normal) * direction[i]

    Then:

    .. code: python

        np.dot(point, normal) + M[i, 3] / (angle * direction[i]) == 0

    This can be compared with the equation of the plane:

    .. code: python

        np.dot(point, normal) + d == 0

    where ``d`` is the distance from the plane to the origin.
    """
    tan, direction, normal = inverse_outer(aff[:3,:3] - np.eye(3))
    angle = math.atan(tan)
    i = np.argmax(np.abs(direction))  # Avoid division by small values
    d = aff[i, 3] / (tan * direction[i])
    point = -d * normal
    return angle, direction, normal, point
