''' zoom (scale) and shears '''

import math

import numpy as np

from .utils import normalized_vector, vector_norm


def zdir2zmat(factor, direction=None):
    """Return matrix to scale by factor around origin in direction.

    Use factor == -1 for point symmetry.

    Parameters
    ----------
    factor : scalar
       factor to zoom by (see `direction`)
    direction : None or array-like shape (3,), optional
       If None, simply apply uniform scaling by `factor`.  Otherwise,
       apply scaling along direction given by vector `direction`.  We
       convert direction to a :term:`unit vector` before application.

    Returns
    -------
    zmat : array shape (3,3)
       3x3 transformation matrix implementing zooms

    Examples
    --------
    >>> v = (np.random.rand(3, 5) - 0.5) * 20.0
    >>> S = zdir2zmat(-1.234)
    >>> np.allclose(np.dot(S, v), -1.234*v)
    True
    >>> factor = np.random.random() * 10 - 5
    >>> direct = np.random.random(3) - 0.5
    >>> S = zdir2zmat(factor, direct)
    """
    if direction is None:
        # uniform scaling
        return np.diag([factor] * 3)
    # nonuniform scaling
    direction = normalized_vector(direction)
    factor = 1.0 - factor
    M = np.eye(3)
    M -= factor * np.outer(direction, direction)
    return M


def zdir2aff(factor, direction=None, origin=None):
    """Return matrix to scale by factor around origin in direction.

    Use factor -1 for point symmetry.

    Parameters
    ----------
    factor : scalar
       factor to zoom by (see direction)
    direction : None or array-like shape (3,)
       If None, simply apply uniform scaling by `factor`.  Otherwise,
       apply scaling along direction given by vector `direction`.  We
       convert direction to a :term:`unit vector` before application.
    origin : None or array-like shape (3,)
       point at which to apply implied zooms

    Returns
    -------
    aff : array shape (4,4)
       4x4 transformation matrix implementing zooms

    Examples
    --------
    >>> v = (np.random.rand(3, 5) - 0.5) * 20.0
    >>> S = zdir2aff(-1.234)[:3,:3]
    >>> np.allclose(np.dot(S, v), -1.234*v)
    True
    >>> factor = np.random.random() * 10 - 5
    >>> direct = np.random.random(3) - 0.5
    >>> origin = np.random.random(3) - 0.5
    >>> S = zdir2aff(factor, None, origin)
    >>> S = zdir2aff(factor, direct, origin)
    """
    M = np.eye(4)
    M[:3,:3] = zdir2zmat(factor, direction)
    if origin is None:
        return M
    if direction is None:
        M[:3, 3] = origin
        M[:3, 3] *= 1.0 - factor
        return M
    # nonuniform scaling
    direction = normalized_vector(direction)
    M[:3, 3] = ((1-factor) * np.dot(origin, direction)) * direction
    return M


def zmat2zdir(zmat):
    """Return scaling factor and direction from zoom (scaling) matrix

    Parameters
    ----------
    zmat : array-like shape (3,3)
       3x3 zoom matrix

    Returns
    -------
    factor : scalar
       zoom (scale) factor as for ``zdir2zmat``
    direction : None or array, shape (3,)
       direction of zoom as for ``zdir2zmat``.  None if scaling is
       uniform.

    Examples
    --------
    Roundtrip may not generate same factor, direction, but the
    generated transformation matrices will be the same

    >>> factor = np.random.random() * 10 - 5
    >>> S0 = zdir2zmat(factor, None)
    >>> f2, d2 = zmat2zdir(S0)
    >>> S1 = zdir2zmat(f2, d2)
    >>> np.allclose(S0, S1)
    True
    >>> direct = np.random.random(3) - 0.5
    >>> S0 = zdir2zmat(factor, direct)
    >>> f2, d2 = zmat2zdir(S0)
    >>> S1 = zdir2zmat(f2, d2)
    >>> np.allclose(S0, S1)
    True
    """
    zmat = np.asarray(zmat, dtype=np.float)
    factor = np.trace(zmat) - 2.0
    # direction: unit eigenvector corresponding to eigenvalue factor
    l, V = np.linalg.eig(zmat)
    near_factors, = np.nonzero(abs(np.real(l.squeeze()) - factor) < 1e-8)
    if near_factors.size == 0:
        # uniform scaling
        factor = (factor + 2.0) / 3.0
        return factor, None
    direction = np.real(V[:, near_factors[0]])
    return factor, normalized_vector(direction)


def aff2zdir(aff):
    """Return scaling factor, direction and origin from scaling matrix.

    Parameters
    ----------
    aff : array-like shape (4,4)
       4x4 :term:`affine transformation` matrix.

    Returns
    -------
    factor : scalar
       zoom (scale) factor as for ``zdir2zmat``
    direction : None or array, shape (3,)
       direction of zoom as for ``zdir2zmat``.  None if scaling is
       uniform.
    origin : array, shape (3,)
       origin of zooms

    Examples
    --------
    >>> factor = np.random.random() * 10 - 5
    >>> direct = np.random.random(3) - 0.5
    >>> origin = np.random.random(3) - 0.5
    >>> S0 = zdir2aff(factor)
    >>> f2, d2, o2 = aff2zdir(S0)
    >>> np.allclose(S0, zdir2aff(f2, d2, o2))
    True
    >>> S0 = zdir2aff(factor, direct)
    >>> f2, d2, o2 = aff2zdir(S0)
    >>> np.allclose(S0, zdir2aff(f2, d2, o2))
    True
    >>> S0 = zdir2aff(factor, direct, origin)
    """
    M = np.asarray(aff, dtype=np.float)
    factor, direction = zmat2zdir(M[:3,:3])
    # origin: any eigenvector corresponding to eigenvalue 1
    l, V = np.linalg.eig(M)
    near_1, = np.nonzero(abs(np.real(l.squeeze()) - 1.0) < 1e-8)
    if near_1.size == 0:
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    origin = np.real(V[:, near_1[-1]]).squeeze()
    origin = origin[:3] / origin[3]
    return factor, direction, origin


def shear_adn2smat(angle, direction, normal):
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
    smat : array shape (3,3)
       shear matrix

    Examples
    --------
    >>> angle = (np.random.random() - 0.5) * 4*math.pi
    >>> direct = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> S = shear_adn2aff(angle, direct, normal)
    >>> np.allclose(1.0, np.linalg.det(S))
    True

    """
    normal = normalized_vector(normal)
    direction = normalized_vector(direction)
    if abs(np.dot(normal, direction)) > 1e-6:
        raise ValueError("direction and normal vectors are not orthogonal")
    angle = math.tan(angle)
    M = np.eye(3)
    M += angle * np.outer(direction, normal)
    return M


def shear_adn2aff(angle, direction, normal, point=None):
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
    >>> S = shear_adn2smat(angle, direct, normal)
    >>> np.allclose(1.0, np.linalg.det(S))
    True
    """
    M = np.eye(4)
    M[:3,:3] = shear_adn2smat(angle, direction, normal)
    if not point is None:
        normal = normalized_vector(normal)
        direction = normalized_vector(direction)
        angle = math.tan(angle)
        M[:3, 3] = -angle * np.dot(point, normal) * direction
    return M


def smat2shear_adn(smat):
    """Return shear angle, direction and plane normal from shear matrix.

    Parameters
    ----------
    smat : array-like, shape (3,3)
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
    >>> angle = (np.random.random() - 0.5) * 4*math.pi
    >>> direct = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> S0 = shear_adn2smat(angle, direct, normal)
    >>> angle, direct, normal = smat2shear_adn(S0)
    >>> S1 = shear_adn2smat(angle, direct, normal)
    >>> np.allclose(S0, S1)
    True
    """
    smat = np.asarray(smat)
    # normal: cross independent eigenvectors corresponding to the eigenvalue 1
    l, V = np.linalg.eig(smat)
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
    direction = np.dot(smat - np.eye(3), normal)
    angle = vector_norm(direction)
    direction /= angle
    angle = math.atan(angle)
    return angle, direction, normal


def aff2shear_adn(aff):
    """Return shear angle, direction and plane normal from shear matrix.

    Parameters
    ----------
    smat : array-like, shape (3,3)
       shear matrix

    Returns
    -------
    angle : scalar
       angle to shear, in radians
    direction : array, shape (3,)
       direction along which to shear
    normal : array, shape (3,)
       vector normal to shear plane
    point : array, shape (3,)
       point, that, with `normal` defines shear plane.

    Examples
    --------
    >>> angle = (np.random.random() - 0.5) * 4 * math.pi
    >>> direct = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> point = np.random.random(3) - 0.5
    >>> S0 = shear_adn2aff(angle, direct, normal, point)
    >>> angle, direct, normal, point = aff2shear_adn(S0)
    >>> S1 = shear_adn2aff(angle, direct, normal, point)
    >>> np.allclose(S0, S1)
    True
    """
    aff = np.asarray(aff)
    angle, direction, normal = smat2shear_adn(aff[:3,:3])
    # point: eigenvector corresponding to eigenvalue 1
    l, V = np.linalg.eig(aff)
    near_1, = np.nonzero(abs(np.real(l.squeeze()) - 1.0) < 1e-8)
    if near_1.size == 0:
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    point = np.real(V[:, near_1[-1]]).squeeze()
    point = point[:3] / point[3]
    return angle, direction, normal, point
