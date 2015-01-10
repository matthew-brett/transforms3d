""" Generic Euler rotations

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
is a relatively common convention, with a custom API in :mod:`taitbryan` in
this package.

The string 'rzxz' specifies that the angles should be interpreted
relative to intrinsic (rotating) coordinate axes, and be performed in the
order: rotation about z axis; rotation about the rotated x axis; rotation
about the rotated z axis. Wolfram Mathworld claim this is the most common
convention : http://mathworld.wolfram.com/EulerAngles.html.
"""

from ._gohlketransorms import (euler_matrix as geneuler2mat,
                               euler_from_matrix as mat2geneuler)
