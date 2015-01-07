============================
 Conventions for transforms
============================

For the transforms we have coded here, we have followed some conventions
that are not universal, but seem common enough.

Transformation matrices are matrices for application on the left,
applied to coodinates on the right, stored as column vectors.

Let's say you are transforming a set of points. The points are vectors
$v^1, v^2 \ldots v^N$, and the vectors are composed of $x$, $y$ and $z$
coordinates, so that $v^1 = \left(x^1, y^1, z^1\right)$, then your points
matrix $P$ would look like:

.. math::

   P = \left(
     \begin{matrix} 
     x^1, x^2, \ldots, x^N \\
     y^1, y^2, \ldots, y^N \\
     z^1, z^2, \ldots, z^N \\
     \end{matrix}
     \right)
   
If we are applying a 3x3 transformation matrix $M$, to transform points
$P$, then the transformed points $v^{\prime}$ are given by:

.. math::

   v^\prime = M \cdot P

