############
Transforms3d
############

Code to convert between various geometric transformations.

* Composing rotations / zooms / shears / translations into affine matrix;
* Decomposing affine matrix into rotations / zooms / shears / translations;
* Conversions between different representations of rotations, including:

  * 3x3 Rotation matrices;
  * Euler angles;
  * quaternions.

We have tried to document the algorithms carefully and write clear code in the
hope that this code can be a teaching reference.  We document the math behind
some of the algorithms using `sympy <http://www.sympy.org>`_ in
``transforms3d/derivations``.  We would be very pleased if y'all would like to
add your own algorithms and derivations - please get a copy of the code from
https://github.com/matthew-brett/transforms3d and get on down,
algorithmically.  Feel free to use the github issue tracker and pull request
system to ask for advice and support.

*************
Documentation
*************

Documentation for latest released version at
http://matthew-brett.github.io/transforms3d

****
Code
****

See https://github.com/matthew-brett/transforms3d

Released under the BSD two-clause license - see the file ``LICENSE`` in the
source distribution.

Much of the code comes from `transformations.py
<http://www.lfd.uci.edu/~gohlke/code/transformations.py.html>`_ by Christoph
Gohlke, also released under the BSD license.

We use Github actions to test the code automatically under Pythons 3.7 through
3.10.

We depend on numpy >= 1.15.  You may be able to make it work on an earlier
numpy if you really needed that.

The latest released version is at https://pypi.python.org/pypi/transforms3d

*******
Support
*******

Please put up issues on the `transforms3d issue tracker
<https://github.com/matthew-brett/transforms3d/issues>`_.
