""" Generic Euler rotations

See:

* http://en.wikipedia.org/wiki/Rotation_matrix
* http://en.wikipedia.org/wiki/Euler_angles
* http://mathworld.wolfram.com/EulerAngles.html

See also: *Representing Attitude with Euler Angles and Quaternions: A
Reference* (2006) by James Diebel. A cached PDF link last found here:

http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.5134

******************
Defining rotations
******************

Euler's rotation theorem tells us that any rotation in 3D can be described by 3
angles.  Let's call the 3 angles the *Euler angle vector* and call the angles
in the vector :math:`alpha`, :math:`beta` and :math:`gamma`.  The vector is [
:math:`alpha`, :math:`beta`. :math:`gamma` ] and, in this description, the
order of the parameters specifies the order in which the rotations occur (so
the rotation corresponding to :math:`alpha` is applied first).

In order to specify the meaning of an *Euler angle vector* we need to specify
the axes around which each of the rotations corresponding to :math:`alpha`,
:math:`beta` and :math:`gamma` will occur.

There are therefore three axes for the rotations :math:`alpha`, :math:`beta`
and :math:`gamma`; let's call them :math:`i` :math:`j`, :math:`k`.

Let us express the rotation :math:`alpha` around axis `i` as a 3 by 3 rotation
matrix `A`.  Similarly :math:`beta` around `j` becomes 3 x 3 matrix `B` and
:math:`gamma` around `k` becomes matrix `G`.  Then the whole rotation expressed
by the Euler angle vector [ :math:`alpha`, :math:`beta`. :math:`gamma` ], `R`
is given by::

   R = np.dot(G, np.dot(B, A))

See http://mathworld.wolfram.com/EulerAngles.html

The order :math:`G B A` expresses the fact that the rotations are
performed in the order of the vector (:math:`alpha` around axis `i` =
`A` first).

To convert a given Euler angle vector to a meaningful rotation, and a
rotation matrix, we need to define:

* the axes `i`, `j`, `k`;
* whether the rotations move the axes as they are applied (intrinsic
  rotations) - compared the situation where the axes stay fixed and the
  vectors move within the axis frame (extrinsic);
* whether a rotation matrix should be applied on the left of a vector to
  be transformed (vectors are column vectors) or on the right (vectors
  are row vectors);
* the handedness of the coordinate system.

See: http://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities

This module implements intrinsic and extrinsic axes, with standard conventions
for axes `i`, `j`, `k`.  We assume that the matrix should be applied on the
left of the vector, and right-handed coordinate systems.  To get the matrix to
apply on the right of the vector, you need the transpose of the matrix we
supply here, by the matrix transpose rule: $(M . V)^T = V^T M^T$.

*************
Rotation axes
*************

Rotations given as a set of three angles can refer to any of 24 different ways
of applying these rotations, or equivalently, 24 conventions for rotation
angles.  See http://en.wikipedia.org/wiki/Euler_angles.

The different conventions break down into two groups of 12.  In the first
group, the rotation axes are fixed (also, global, static), and do not move with
rotations.  These are called *extrinsic* axes.  The axes can also move with the
rotations.  These are called *intrinsic*, local or rotating axes.

Each of the two groups (*intrinsic* and *extrinsic*) can further be divided
into so-called Euler rotations (rotation about one axis, then a second and then
the first again), and Tait-Bryan angles (rotations about all three axes).  The
two groups (Euler rotations and Tait-Bryan rotations) each have 6 possible
choices.  There are therefore 2 * 2 * 6 = 24 possible conventions that could
apply to rotations about a sequence of three given angles.

This module gives an implementation of conversion between angles and rotation
matrices for which you can specify any of the 24 different conventions.

****************************
Specifying angle conventions
****************************

You specify conventions for interpreting the sequence of angles with a four
character string.

The first character is 'r' (rotating == intrinsic), or 's' (static ==
extrinsic).

The next three characters give the axis ('x', 'y' or 'z') about which to
perform the rotation, in the order in which the rotations will be performed.

For example the string 'szyx' specifies that the angles should be interpreted
relative to extrinsic (static) coordinate axes, and be performed in the order:
rotation about z axis; rotation about y axis; rotation about x axis. This
is a relatively common convention, with customized implementations in
:mod:`taitbryan` in this package.

The string 'rzxz' specifies that the angles should be interpreted
relative to intrinsic (rotating) coordinate axes, and be performed in the
order: rotation about z axis; rotation about the rotated x axis; rotation
about the rotated z axis. Wolfram Mathworld claim this is the most common
convention : http://mathworld.wolfram.com/EulerAngles.html.

*********************
Direction of rotation
*********************

The direction of rotation is given by the right-hand rule (orient the thumb of
the right hand along the axis around which the rotation occurs, with the end of
the thumb at the positive end of the axis; curl your fingers; the direction
your fingers curl is the direction of rotation).  Therefore, the rotations are
counterclockwise if looking along the axis of rotation from positive to
negative.

****************************
Terms used in function names
****************************

* *mat* : array shape (3, 3) (3D non-homogenous coordinates)
* *euler* : (sequence of) rotation angles about the z, y, x axes (in that
  order)
* *axangle* : rotations encoded by axis vector and angle scalar
* *quat* : quaternion shape (4,)
"""

import math

import numpy as np

from .quaternions import quat2mat, quat2axangle
from .axangles import axangle2mat
from . import taitbryan as tb

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# For testing whether a number is close to zero
_EPS4 = np.finfo(float).eps * 4.0


def euler2mat(ai, aj, ak, axes='sxyz'):
    """Return rotation matrix from Euler angles and axis sequence.

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.

    Examples
    --------
    >>> R = euler2mat(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = euler2mat(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.eye(3)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


def mat2euler(mat, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    Note that many Euler angle triplets can describe one matrix.

    Parameters
    ----------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> R0 = euler2mat(1, 2, 3, 'syxz')
    >>> al, be, ga = mat2euler(R0, 'syxz')
    >>> R1 = euler2mat(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(mat, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS4:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS4:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def euler2quat(ai, aj, ak, axes='sxyz'):
    """Return `quaternion` from Euler angles and axis sequence `axes`

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Examples
    --------
    >>> q = euler2quat(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0

    return q


def quat2euler(quaternion, axes='sxyz'):
    """Euler angles from `quaternion` for specified axis sequence `axes`

    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> angles = quat2euler([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(angles, [0.123, 0, 0])
    True
    """
    return mat2euler(quat2mat(quaternion), axes)


def euler2axangle(ai, aj, ak, axes='sxyz'):
    ''' Return angle, axis corresponding to Euler angles, axis specification

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    vector : array shape (3,)
       axis around which rotation occurs
    theta : scalar
       angle of rotation

    Examples
    --------
    >>> vec, theta = euler2axangle(0, 1.5, 0, 'szyx')
    >>> np.allclose(vec, [0, 1, 0])
    True
    >>> theta
    1.5
    '''
    return quat2axangle(euler2quat(ai, aj, ak, axes))


def axangle2euler(vector, theta, axes='sxyz'):
    ''' Convert axis, angle pair to Euler angles

    Parameters
    ----------
    vector : 3 element sequence
       vector specifying axis for rotation.
    theta : scalar
       angle of rotation
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> ai, aj, ak = axangle2euler([1, 0, 0], 0)
    >>> np.allclose((ai, aj, ak), 0)
    True
    '''
    return mat2euler(axangle2mat(vector, theta), axes)


class EulerFuncs(object):
    """ Namespace for Euler angles functions with given axes specification
    """

    def __init__(self, axes):
        """ Initialize namespace for Euler angles functions

        Parameters
        ----------
        axes : str
            Axis specification; one of 24 axis sequences as string or encoded
            tuple - e.g. ``sxyz`` (the default).
        """
        self.axes = axes

    def euler2mat(self, ai, aj, ak):
        """Return rotation matrix from Euler angles

        See :func:`euler2mat` for details.
        """
        return euler2mat(ai, aj, ak, self.axes)

    def mat2euler(self, mat):
        """Return Euler angles from rotation matrix `mat`

        See :func:`mat2euler` for details.
        """
        return mat2euler(mat, self.axes)

    def euler2quat(self, ai, aj, ak):
        """ Return `quaternion` from Euler angles

        See :func:`euler2quat` for details.
        """
        return euler2quat(ai, aj, ak, self.axes)

    def quat2euler(self, quat):
        """Euler angles from `quaternion`

        See :func:`quat2euler` for details.
        """
        return quat2euler(quat, self.axes)

    def euler2axangle(self, ai, aj, ak):
        """ Angle, axis corresponding to Euler angles

        See :func:`euler2axangle` for details.
        """
        return euler2axangle(ai, aj, ak, self.axes)

    def axangle2euler(self, vector, theta):
        """ Convert axis, angle pair to Euler angles

        See :func:`axangle2euler` for details.
        """
        return axangle2euler(vector, theta, self.axes)


# Namespaces for some common conventions
sxyz = EulerFuncs('sxyz') # Tait-Bryan XYZ
rzxz = EulerFuncs('rzxz')
physics = rzxz


class TBZYX(EulerFuncs):
    """ Namespace for Tait-Bryan ZYX Euler angle convention functions
    """

    def __init__(self):
        """ Initialize Tait-Bryan ZYX namespace
        """
        self.axes = 'szyx'

    def euler2mat(self, ai, aj, ak):
        """Return rotation matrix from Euler angles

        See :func:`transforms3d.taitbryan.euler2mat` for details.
        """
        return tb.euler2mat(ai, aj, ak)

    def mat2euler(self, mat):
        """Return Euler angles from rotation matrix `mat`

        See :func:`transforms3d.taitbryan.mat2euler` for details.
        """
        return tb.mat2euler(mat)

    def euler2quat(self, ai, aj, ak):
        """ Return `quaternion` from Euler angles

        See :func:`transforms3d.taitbryan.euler2quat` for details.
        """
        return tb.euler2quat(ai, aj, ak)


szyx = TBZYX() # Tait-Bryan ZYX
