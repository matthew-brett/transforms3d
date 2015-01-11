''' These give the derivations for Euler angles to rotation matrix and
Euler angles to quaternion.  We use the rotation matrix derivation only
in the tests.  The quaternion derivation is in the tests, and,
in more compact form, in the ``euler2quat`` code.

The rotation matrices operate on column vectors, thus, if ``R`` is the
3x3 rotation matrix, ``v`` is the 3 x N set of N vectors to be rotated,
and ``vdash`` is the matrix of rotated vectors::

   vdash = np.dot(R, v)


'''

from sympy import Symbol, cos, sin, symbols, latex
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

# Formula for gimbal lock example
alpha, beta, gamma = symbols('\\alpha, \\beta, \\gamma')
M_xyz = (z_rotation(gamma) *
         y_rotation(beta) *
         x_rotation(alpha))

# Substitute for cos(beta) = 0, sin(beta) = +-1
pm1 = Symbol('\\pm{1}')
subs = {cos(beta): 0,
        sin(beta): pm1}
M_xyz_gimbal_full = M_xyz.subs(subs)

# Substitute for cos(beta) = 0, sin(beta) = 1
subs = {cos(beta): 0,
        sin(beta): 1}
M_xyz_gimbal_sb1 = M_xyz.subs(subs)

# And combination symbols
V1, V2 = symbols('V1, V2')
v1t = cos(gamma)*sin(alpha) - cos(alpha)*sin(gamma)
v2t = cos(alpha)*cos(gamma) + sin(alpha)*sin(gamma)
subs2 = {v1t: V1,
         v2t: V2,
         v1t*-1: -V1
         }
M_xyz_gimbal_sb1_reduced = M_xyz_gimbal_sb1.subs(subs2)

# Substitute for cos(beta) = 0, sin(beta) = -1
subs = {cos(beta): 0,
        sin(beta): -1}
M_xyz_gimbal_sbm1 = M_xyz.subs(subs)

# And combination symbols
W1, W2 = symbols('W1, W2')
w1t = cos(gamma)*sin(alpha) + cos(alpha)*sin(gamma)
w2t = cos(alpha)*cos(gamma) - sin(alpha)*sin(gamma)
subs2 = {w1t: W1,
         w2t: W2,
         w1t*-1: -W1
         }
M_xyz_gimbal_sbm1_reduced = M_xyz_gimbal_sbm1.subs(subs2)

