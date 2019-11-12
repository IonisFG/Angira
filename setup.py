#!/usr/bin/env python

from distutils.core import setup
from distutils.command.build_py import build_py as _build_py


from setuptools import setup, find_packages

import subprocess


install_requires = [
    'unidecode', 'python-Levenshtein', 'fuzzywuzzy'
    ]

test_requires = [
    'nose',
    ]


setup(name='angira',
      cmdclass={},
      version='0.1',
      description='',
      author='Swagatam Mukhopadhyay',
      author_email='smukhopa@ionisph.com',
      url='http://www.ionisph.com',
      packages=['angira', 'angira/validators', 'angira/searchers'],
      package_dir={'angira': 'angira'},
      package_data={},
      scripts=[],
      install_requires=install_requires,
      tests_require=test_requires,
      test_suite="nose.collector",
     )


