#!/usr/bin/env python
''' Installation script for transforms3d package '''
from os.path import join as pjoin

# Always use setuptools.
from setuptools import setup

import versioneer


setup(name='transforms3d',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      packages=['transforms3d',
                'transforms3d.derivations',
                'transforms3d.tests'],
      package_data = {'transforms3d':
                      [pjoin('tests', 'data', '*')]},
      zip_safe=False,
      )
