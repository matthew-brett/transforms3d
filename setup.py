#!/usr/bin/env python
''' Installation script for transforms3d package '''
from os.path import join as pjoin

# Always use setuptools.
from setuptools import setup

import versioneer


setup(name='transforms3d',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Functions for 3D coordinate transformations',
      author='Matthew Brett',
      author_email='matthew.brett@gmail.com',
      maintainer='Matthew Brett',
      maintainer_email='matthew.brett@gmail.com',
      url='http://github.com/matthew-brett/transforms3d',
      packages=['transforms3d',
                'transforms3d.derivations',
                'transforms3d.tests'],
      package_data = {'transforms3d':
                      [pjoin('tests', 'data', '*')]},
      license='BSD license',
      classifiers = [
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],
      long_description = open('README.rst', 'rt').read(),
      zip_safe=False,
      # Check dependencies also in .github/workflows/testing.yml file
      requires=['numpy (>=1.15)']
      )
