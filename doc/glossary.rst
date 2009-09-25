==========
 Glossary
==========

.. glossary::

   Affine matrix
      A matrix implementing an :term:`affine transformation` in
      :term:`homogenous coordinates`.  For a 3 dimensional transform, the
      matrix is shape 4 by 4.

   Affine transformation
      See `wikipedia affine`_ definition.  A affine transformation is a
      :term:`linear transformation` followed by a translation.

   Homogenous coordinates
      See `wikipedia homogenous coordinates`_

   Linear transformation
      A linear transformation is one that preserves lines - that is, if
      any three points are on a line before transformation, they are
      also on a line after transformation.  See `wikipedia linear
      transform`_.  Rotation, scaling and shear are linear
      transformations.

   Rotation matrix
      See `wikipedia rotation matrix`_.  A rotation matrix is a matrix
      implementing a rotation.  Rotation matrices are square and
      orthogonal.  That means, that the rotation matrix $R$ has columns
      that are unit vectors, and where $R^T R = I$ (where $R^T$ is the
      transpose and $I$ is the identity matrix) and therefore $R^T =
      R^{-1}$ ($R^{-1} is the inverse).  Rotation matrices also have a
      determinant of $1$.


.. include:: links_names.txt
       
       
