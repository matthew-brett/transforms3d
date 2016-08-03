.. image:: https://travis-ci.org/matthew-brett/transforms3d.svg?branch=master
    :target: https://travis-ci.org/matthew-brett/transforms3d

.. image:: https://coveralls.io/repos/matthew-brett/transforms3d/badge.png?branch=master
    :target: https://coveralls.io/r/matthew-brett/transforms3d?branch=master

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

`travis-ci <https://travis-ci.org/matthew-brett/transforms3d>`_ kindly tests
the code automatically under Python 2.6, 2.7, 3.2, 3.3 and 3.4.

We depend on numpy >= 1.5.  You could probably make it work on an earlier
numpy if you really needed that.

The latest released version is at https://pypi.python.org/pypi/transforms3d

*******
Support
*******

Please put up issues on the `transforms3d issue tracker
<https://github.com/matthew-brett/transforms3d/issues>`_.
