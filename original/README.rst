################################
Christoph Gohlke's original code
################################

File ``transformations.py`` downloaded from
https://www.lfd.uci.edu/~gohlke/code/transformations.py

Path ``transformations.py.patch`` is the patch to go from
``transformations.py`` in this directory (as downloaded) to modified version
in ``transforms3d/_gohlketransforms.py``.

To update the file, go to the URL above and copy / paste the text into
`transformations.py` in this directory.

To apply patch to ``transformations.py`` file in this directory and update to
``transforms3d/_gohlketransforms.py``::

    make apply-patch

To take edits from ``transforms3d/_gohlketranforms.py`` and make patch against
current ``original/transformations.py``::

    make make-patch
