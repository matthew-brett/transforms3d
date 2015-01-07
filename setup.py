#!/usr/bin/env python
''' Installation script for transforms3d package '''
import sys
from os.path import join as pjoin

# For some commands, use setuptools.
if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
            'install_egg_info', 'egg_info', 'easy_install', 'bdist_wheel',
            'bdist_mpkg')).intersection(sys.argv)) > 0:
    import setuptools

from distutils.core import setup

import versioneer

versioneer.VCS = 'git'
versioneer.versionfile_source = pjoin('transforms3d', '_version.py')
versioneer.versionfile_build = pjoin('transforms3d', '_version.py')
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'transforms3d-'

setup(name='transforms3d',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='3D transforms - possible merge',
      author='Christoph Gohlke, Matthew Brett',
      author_email='Christoph Gohlke, matthew.brett@gmail.com',
      url='http://github.com/matthew-brett/transforms3d',
      packages=['transforms3d',
                'transforms3d.derivations',
                'transforms3d.tests'],
      )
