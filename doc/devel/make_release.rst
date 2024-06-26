.. _release-guide:

**************************************
Guide to making a Transforms3d release
**************************************

A guide for developers who are doing a Transforms3d release

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

* Clean::

    git clean -fxd

* Run tests after installing into a virtualenv, to test that
  installing works correctly::

    mkvirtualenv transforms3d-test
    pip install pytest wheel
    git clean -fxd
    python setup.py install
    mkdir for_test
    cd for_test
    pytest --doctest-modules transforms3d

* Make sure all tests pass on your local machine (from the Transforms3d root
  directory)::

    pytest --doctest-modules transforms3d

* Check the documentation Doctests::

    pip install -r doc-requirements.txt
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

* Once everything looks good, upload the source release to PyPi::

    pip install build twine
    git clean -fxd
    python -m build . --sdist
    twine upload -s dist/transforms*gz

* Remember you'll need your ``~/.pypirc`` file set up right for this to work.
  See `setuptools intro`_.  If you have 2-factor authentication, the file may
  look something like this::

    [pypi]
    username = __token__

* Check how everything looks on Pypi - the description, the packages.  If
  necessary delete the release and try again if it doesn't look right.

* Push the tag with ``git push --tags``

* Upload the docs with::

    pip install -e .  # if you haven't done this already
    pip install -r doc-requirements.txt
    cd doc
    make github

* Announce to the mailing lists.  With fear and trembling.

.. _setuptools intro: http://packages.python.org/an_example_pypi_project/setuptools.html

.. include:: ../links_names.inc
