To patch against Christoph's source:

(probably should make a patch for Christophe's sources...)

Get source from: http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

Replace 'numpy.' with 'np.'

Replace occurrences of ' random.random' with 'np.random.random' (bearing
in mind there are many occurences of 'np.random.random' already.

Consider replacing 'rotation_matrix' with 'from_angle_axis_point'
(bearing in mid there are many occurences of 'random_rotation_matrix'
already.

Consider replacing 'identity_matrix' with 'compose_matrix'

Create diff against previous edited source (with same changes):

diff previous_transformations.py new_transformations.py > trans_patch

apply patch:

patch -p0 gohlketransforms.py < trans_patch

Edit...

Commit new transformations to transforms3d.original directory
