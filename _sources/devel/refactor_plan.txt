==================
 Refactoring plan
==================

Note the :ref:`naming-conventions`.

Base the module distinctions on the naming conventions.

In general, routines that create or break down affines, rather than, say 3x3
matrices, go with the - say - 3x3 routines, and are named ``aff2something``,
``something2aff``.

Routines that only apply to affines, like ``compose``, ``decompose``, go into
the ``affines`` module.

Different decompositions might have different names - such as
``decompose_szrt``, returning transformations in order that they are performed
- here shears, zooms, rotations, translations.

Move the SPM (ND affine) decompose to ``decompose_szrt_nd``.

Maybe move the comments from Euler code out into ``doc`` tree.

To integrate
============

* projection_matrix, projection_from_matrix: move to own module, rename to
  ``projpn2aff``. ``aff2projpn``.
* clip_matrix: move to own module, rename to ``clip2aff``
* decompose_matrix, compose_matrix: move to ``affines`` module, rename
  to ``decompose_zsrtp``, ``compose_zsrtp``
* orthogonalization_matrix: move to ``misc`` module, rename to ``orth_aff``
* superimposition_matrix: move to ``misc`` module, rename to ``vecs2aff``
* quaterion_about_axis, quaternion_matrix, quaternion_from_matrix,
  quaternion_multiply, quaternion_conjugate, quaternion_inverse: check
  against my code
* quaternion_real, quaternion_imag: needed?
* quaternion_slerp, random_quaternion: move to ``quaternions`` module,
  rename ``qslerp``, ``rand_quat``.
* random_rotation_matrix: move to some module as ``rand_rmat``
* Arcball class, arcball_map_to_sphere, arcball_constrain_to_axis,
  arcball_nearest_axis: to own module
* vector_norm: move to ``utils`` module, review use after refactoring
* unit_vector: replace with ``utils.normalized_vector``
* random_vector, inverse_matrix, concatenate_matrices: not obviously
  used, very simple code, remove?
* _import_module: for preferring C functions.  Move to ``utils`` for
  now, consider C / python cooexistence strategy.

C / python integration
======================

See scipy.spatial - parallel C and Python routines, but there in a very
simple case.

How about:

#. Import all the Python names via the __init__.py
#. Import all the C names (after renaming with agreed scheme as above or
   differently) into a ``c.py`` __init__ -like module. Thus something
   like::

   >>> import transforms3d as t3d
   >>> M = t3d.euler3rmat(0.1, 0.3, 0.2)

for the python code, and::

   >>> import transforms3d.c as t3dc
   >>> M = t3dc.euler2mat(0.1, 0.3, 0.2)

for the C code?  Or the other way round of course::

   >>> import transforms3d as t3d
   >>> import transforms3d.python as t3dpy

or both, and switch the __init__ to import from the C or Python
namespace with a one-liner.

Questions for Christoph
=======================

Does this refactoring plan make sense?

Does the naming scheme make sense (drawing distinction for example
between 3x3 rotation matrix and affine)?

Now about the naming scheme for ``compose`` - e.g ``compose_szrtp``?

OK to return a 3x3 rotation matrix from ``decompose`` rather then Euler
angles?

How to classify ``clip_matrix``, ``orthogonalization_matrix``,
``superimposition_matrix``?

OK to remove: ``quaternion_real``, ``quaternion_imag``,
``random_vector``, ``inverse_matrix``, ``concatenate_matrices`` ?

What do you think of the C / Python scheme (it's about the same as your
current one)?

How do you see the relationship between your current code and this code?
