''' zoom (scale) and shears '''

import numpy as np

from .utils import normalized_vector


def zdir2zmat(factor, direction=None):
    """Return matrix to scale by factor around origin in direction.

    Use factor -1 for point symmetry.

    Parameters
    ----------
    factor : scalar
       factor to zoom by (see direction)
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
    >>> import transforms3d.gohlketransforms as tg
    >>> S0 = tg.scale_matrix(factor, origin, direct)
    >>> f2, o2, d2 = tg.scale_from_matrix(S0)
    >>> np.allclose(S0, tg.scale_matrix(f2, o2, d2))
    True
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

