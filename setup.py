#!/usr/bin/env python
''' Installation script for transforms3d package '''
import sys

# For some commands, use setuptools.
if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
            'install_egg_info', 'egg_info', 'easy_install', 'bdist_wheel',
            'bdist_mpkg')).intersection(sys.argv)) > 0:
    import setuptools

from distutils.core import setup

setup(name='transforms3d',
      version='0.1a',
      description='3D transforms - possible merge',
      author='Christoph Gohlke, Matthew Brett',
      author_email='Christoph Gohlke, matthew.brett@gmail.com',
      url='http://github.com/matthew-brett/transforms3d',
      packages=['transforms3d',
                'transforms3d.derivations',
                'transforms3d.tests'],
      )
