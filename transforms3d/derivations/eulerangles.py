''' These give the derivations for Euler angles to rotation matrix and
Euler angles to quaternion.  We use the rotation matrix derivation only
in the tests.  The quaternion derivation is in the tests, and,
in more compact form, in the ``euler2quat`` code.

The rotation matrices operate on column vectors, thus, if ``R`` is the
3x3 rotation matrix, ``v`` is the 3 x N set of N vectors to be rotated,
and ``vdash`` is the matrix of rotated vectors::

   vdash = np.dot(R, v)


'''

from sympy import Symbol, cos, sin
from sympy.matrices import Matrix

from transforms3d.derivations.quaternions import quat_around_axis, \
    quat2mat, qmult


def x_rotation(theta):
    ''' Rotation angle `theta` around x-axis
    http://en.wikipedia.org/wiki/Rotation_matrix#Dimension_three
    '''
    return Matrix([[1, 0, 0],
                   [0, cos(theta), -sin(theta)],
                   [0, sin(theta), cos(theta)]])


def y_rotation(theta):
    ''' Rotation angle `theta` around y-axis
    http://en.wikipedia.org/wiki/Rotation_matrix#Dimension_three
    '''
    return Matrix([[cos(theta), 0, sin(theta)],
                  [0, 1, 0],
                  [-sin(theta), 0, cos(theta)]])


def z_rotation(theta):
    ''' Rotation angle `theta` around z-axis
    http://en.wikipedia.org/wiki/Rotation_matrix#Dimension_three
    '''
    return Matrix([[cos(theta), -sin(theta), 0],
                  [sin(theta), cos(theta), 0],
                  [0, 0, 1]])


# Formula for rotation matrix given Euler angles and z, y, x ordering
M_zyx = (x_rotation(Symbol('x')) *
         y_rotation(Symbol('y')) *
         z_rotation(Symbol('z')))

# Formula for quaternion given Euler angles, z, y, x ordering
q_zrot = quat_around_axis(Symbol('z'), [0, 0, 1])
q_yrot = quat_around_axis(Symbol('y'), [0, 1, 0])
q_xrot = quat_around_axis(Symbol('x'), [1, 0, 0])

# quaternion from composition of x on y on z rotations
q_zyx = qmult(q_xrot, qmult(q_yrot, q_zrot))
