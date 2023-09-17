#!/usr/bin/env python
''' Installation script for transforms3d package '''
from os.path import join as pjoin

# Always use setuptools.
from setuptools import setup

import versioneer


setup(name='transforms3d',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False,
      )
