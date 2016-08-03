.. _release-guide:

**************************************
Guide to making a transforms3d release
**************************************

A guide for developers who are doing a transforms3d release

.. _release-checklist:

Release checklist
=================

* Review the open list of `transforms3d issues`_.  Check whether there are
  outstanding issues that can be closed, and whether there are any issues that
  should delay the release.  Label them !

* Review and update the release notes.  Review and update the :file:`Changelog`
  file.  Get a partial list of contributors with something like::

      git shortlog -ns 0.6.0..

  where ``0.6.0`` was the last release tag name.

  Then manually go over ``git shortlog 0.6.0..`` to make sure the release notes
  are as complete as possible and that every contributor was recognized.

* Use the opportunity to update the ``.mailmap`` file if there are any
  duplicate authors listed from ``git shortlog -ns``.

* Add any new authors to the ``AUTHORS`` file.  Add any new entries to the
  ``THANKS`` file.

* Check the copyright years in ``doc/conf.py`` and ``LICENSE``

* If you have travis-ci_ building set up you might want to push the code in its
  current state to a branch that will build, e.g::

    git branch -D pre-release-test # in case branch already exists
    git co -b pre-release-test

* Clean::

    git clean -fxd

* Make sure all tests pass on your local machine (from the transforms3d root
  directory)::

    nosetests --with-doctest transforms3d

  Do this on a Python 2 and Python 3 setup.

* Run the same tests after installing into a virtualenv, to test that
  installing works correctly::

    mkvirtualenv transforms3d-test
    pip install nose wheel
    git clean -fxd
    python setup.py install
    mkdir for_test
    cd for_test
    nosetests --with-doctest transforms3d

* Check the documentation doctests::

    cd doc
    make doctest
    cd ..

* The release should now be ready.

Doing the release
=================

You might want to make tag the release commit on your local machine, push to
pypi_, review, fix, rebase, until all is good.  Then and only then do you push
to upstream on github.

* Make a signed annotated tag for the release with tag of form ``0.6.0``::

    git tag -sm 'Fifth public release' 0.6.0

  Because we're using `versioneer`_ it is the tag which sets the package
  version.

* Once everything looks good, upload the source release to PyPi.  See
  `setuptools intro`_::

    python setup.py sdist --formats=gztar,zip
    twine upload -s dist/*

* Upload wheels by building in virtualenvs, something like::

   workon py27
   rm -rf build
   python setup.py bdist_wheel upload
   workon py33
   rm -rf build
   python setup.py bdist_wheel upload
   workon py34
   rm -rf build
   python setup.py bdist_wheel upload

* Remember you'll need your ``~/.pypirc`` file set up right for this to work.
  See `setuptools intro`_.  The file should look something like this::

    [distutils]
    index-servers =
        pypi

    [pypi]
    username:your.pypi.username
    password:your-password

    [server-login]
    username:your.pypi.username
    password:your-password

* Check how everything looks on pypi - the description, the packages.  If
  necessary delete the release and try again if it doesn't look right.

* Push the tag with ``git push origin 0.6.0``

* Upload the docs with::

    pip install -e .  # if you haven't done this already
    pip install -r doc-requirements.txt
    cd doc
    make github

* Announce to the mailing lists.  With fear and trembling.

.. _setuptools intro: http://packages.python.org/an_example_pypi_project/setuptools.html

.. include:: ../links_names.inc
