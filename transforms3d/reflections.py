''' Reflections '''

import math

import numpy as np

from .utils import normalized_vector


def rfvec2rfmat(normal):
    """ Matrix to reflect in plane through origin, orthogonal to `normal`

    Parameters
    ----------
    normal : array-like, shape (3,)
       vector normal to plane of reflection

    Returns
    -------
    rfmat : array shape (3,3)

    Notes
    -----
    http://en.wikipedia.org/wiki/Reflection_(mathematics)

    The reflection of a vector `v` in a plane normal to vector `a` is:

    .. math::

       \\mathrm{Ref}_a(v) = v - 2\\frac{v\\cdot a}{a\\cdot a}a

    The entries of the corresponding orthogonal transformation matrix
    `R` are given by:

    .. math::

       R_{ij} = I_{ij} - 2\\frac{a_i a_j}{\|a\|^2}

    where $I$ is the identity matrix.
    """
    normal = np.asarray(normal, dtype=np.float)
    norm2 = (normal**2).sum()
    M = np.eye(3)
    return M - 2.0 * np.outer(normal, normal) / norm2


def rfvec2aff(normal, point=None):
    """Affine to mirror at plane defined by point and normal vector.

    Parameters
    ----------
    normal : 3 element sequence
       vector normal to point (and therefore mirror plane)
    point : 3 element sequence
       x, y, x coordinates of point

    Returns
    -------
    aff : array shape (4,4)

    Examples
    --------
    >>> normal = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R = rfvec2aff(normal, point)
    >>> np.allclose(2., np.trace(R))
    True
    >>> np.allclose(point, np.dot(R[:3,:3], point) + R[:3,3])
    True

    Notes
    -----
    See :func:`rfvec2rfmat`
    """
    M = np.eye(4)
    M[:3,:3] = rfvec2rfmat(normal)
    if not point is None:
        normal = normalized_vector(normal)
        M[:3, 3] = (2.0 * np.dot(point, normal)) * normal
    return M


def rfmat2rfvec(rfmat):
    """Mirror plane normal vector from `rfmat` matrix.

    Parameters
    ----------
    rfmat : array-like, shape (3,3)

    Returns
    -------
    normal : array shape (3,)
       vector normal to point (and therefore mirror plane)

    Raises
    ------
    ValueError
       If there is no eigenvector with eigenvalue -1
       
    Examples
    --------
    >>> normal = np.random.random(3) - 0.5
    >>> M0 = rfvec2rfmat(normal)
    >>> normal = rfmat2rfvec(M0)
    >>> M1 = rfvec2rfmat(normal)
    >>> np.allclose(M0, M1)
    True
    """
    M = np.asarray(rfmat)
    # normal: unit eigenvector corresponding to eigenvalue -1
    l, V = np.linalg.eig(rfmat)
    m1_factors, = np.nonzero(abs(np.real(l.squeeze()) + 1.0) < 1e-8)
    if m1_factors.size == 0:
        raise ValueError("no unit eigenvector corresponding to eigenvalue -1")
    return np.real(V[:, m1_factors[0]]).squeeze()


def aff2rfvec(aff):
    """Mirror plane normal vector and point from affine.

    Parameters
    ----------
    aff : array-like, shape (4,4)

    Returns
    -------
    normal : array shape (3,)
       vector normal to point (and therefore mirror plane)
    point : array shape (3,)
       x, y, x coordinates of point

    Raises
    ------
    ValueError
       If there is no eigenvector for aff[:3,:3] with eigenvalue -1 or if
       there is no eigenvector for `aff` with eigenvalue 1.
       
    Examples
    --------
    >>> v0 = np.random.random(3) - 0.5
    >>> v1 = np.random.random(3) - 0.5
    >>> M0 = rfvec2aff(v0, v1)
    >>> normal, point = aff2rfvec(M0)
    >>> M1 = rfvec2aff(normal, point)
    >>> np.allclose(M0, M1)
    True
    """
    aff = np.asarray(aff)
    normal = rfmat2rfvec(aff[:3,:3])
    # point: any unit eigenvector corresponding to eigenvalue 1
    l, V = np.linalg.eig(aff)
    near_1, = np.nonzero(abs(np.real(l.squeeze()) - 1.0) < 1e-8)
    if near_1.size == 0:
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    point = np.real(V[:, near_1[-1]]).squeeze()
    point = point[:3] / point[3]
    return normal, point


