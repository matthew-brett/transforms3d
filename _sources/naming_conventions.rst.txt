.. _naming-conventions:

====================
 Naming conventions
====================

In the code, we try to abbreviate common concepts in a standard way.

* *aff*  - 4x4 :term:`affine matrix` for operating on homogenous coordinates
  of shape (4,) or (4, N);
* *mat* - 3x3 transformation matrix for operating on non-homogenous
  coordinate vectors of shape (3,) or (3, N). A :term:`rotation matrix` is an
  example of a transformation matrix;
* *euler* - :term:`euler angles` - sequence of three scalars giving rotations
  about defined axes;
* *axangle* - :term:`axis angle` - axis (vector) and angle (scalar) giving
  axis around which to rotate and angle of rotation;
* *quat* - :term:`quaternion` - shape (4,);
* *rfnorm* : reflection in plane defined by normal (vector) and optional point
  (vector);
* *zfdir* : zooms encoded by factor (scalar) and direction (vector)
* *zdir* - factor (scalar), direction (vector) pair to specify 3D zoom matrix;
* *striu* : shears encoded by vector giving triangular portion above diagonal
  of NxN array (for ND transformation)
* *sadn* : shears encoded by angle scalar, direction vector, normal vector
  (with optional point vector)
