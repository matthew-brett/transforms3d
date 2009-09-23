.. _gimbal-lock:

=============
 Gimbal lock
=============

Euler angles have a major deficiency, and that is, that some rotations
are not possible, because of a phenomenon called *gimbal lock*.

Imagine that we are using the Euler angle convention of starting with a
rotation around the $x$ axis, followed by the $y$ axis, followed by the
$z$ axis.

Here we see a Spitfire aircraft, flying across the screen.  The $x$ axis
is left to right (tail to nose), the $y$ axis is from the left wing tip
to the right wing tip (going away from the screen), and the $z$ axis is
from bottom to top:

.. image:: images/spitfire_0.png

Imagine we wanted to do a slight roll with the left wing tilting down
(rotation about $x$) like this:

.. image:: images/spitfire_x.png

followed by a violent pitch so we are pointing straight up (rotation
around $y$ axis):

.. image:: images/spitfire_y.png

followed by a yaw-like turn of the nose towards the viewer (and the tail
away from the viewer):

.. image:: images/spitfire_hoped.png

But, wait.  Let's go back over that again.  Look at the result of the
rotation around the $y$ axis.  Notice that the $x$ axis, as was,
is now aligned with the $z$ axis, as it is now.  Rotating around the $z$
axis will have exactly the same effect as adding an extra rotation
around the $x$ axis at the beginning.  That means that, when there is a
$y$ axis rotation onto the $z$ axis (a rotation of $\pm\pi/2$ around the
$y$ axis) - we are *locked* from doing standard $z$ axis type rotations.

In mathematical terms, we see gimbal lock for this type of Euler axis
convention, when $\cos(\beta)$ = 0$, where $\beta$ is the angle of
rotation around the $y$ axis.  By "this type of convention" we mean
using rotation around all 3 of the $x$, $y$ and $z$ axes, rather than
using the same axis twice - e.g. the physics convention of $z$ followed
by $x$ followed by $z$ axis rotation.

See http://www.gregslabaugh.name/publications/euler.pdf for a more
detailed explanation.




