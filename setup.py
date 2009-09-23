#!/usr/bin/env python
''' Installation script for transforms3d package '''
from glob import glob
from distutils.core import setup

setup(name='transforms3d',
      version='0.1a',
      description='3D transforms - possible merge',
      author='Christoph Gohlke, Matthew Brett',
      author_email='Christoph Gohlke, matthew.brett@gmail.com',
      url='http://github.com/matthew-brett/transforms3d',
      packages=['transforms3d', 'transforms3d.derivations'],
      scripts=glob('scripts/*.py')
      )

