"""Dynamical evolution of complex systems.

This package provides an interface to a set of ODE solvers for solving
dynamical time-evolution problems. The original application was quantum
dynamics via Gross-Pitaevski equations (GPE) and superfluid density
functional theory (TDDFT), but the code is quite general and should be
able to be easily used for a variety of problems.

Currently the codes only use fixed time-step method (not adaptive).

**Source:**
  https://bitbucket.org/mforbes/pytimeode
**Issues:**
  https://bitbucket.org/mforbes/pytimeode/issues
"""
import sys

from setuptools import setup, find_packages
from setuptools.command.test import test as original_test

NAME = "pytimeode"

install_requires = [
    'mmfutils>=0.4.3',
    'persist>=0.9b2',
    'zope.interface>=3.8.0'
]

test_requires = [
    'nose>=1.3.6',
    'coverage',
    'flake8',
    'pep8==1.5.7',     # Needed by flake8: dependency resolution issue if not pinned
]

# Remove NAME from sys.modules so that it gets covered in tests. See
# http://stackoverflow.com/questions/11279096
for mod in sys.modules.keys():
    if mod.startswith(NAME):
        del sys.modules[mod]
del mod


class test(original_test):
    description = "Run all tests and checks (customized for this project)"

    def finalize_options(self):
        # Don't actually run any "test" tests (we will use nosetest)
        self.test_suit = None

    def run(self):
        # Call this to do complicated distribute stuff.
        original_test.run(self)

        for cmd in ['nosetests', 'flake8', 'check']:
            try:
                self.run_command(cmd)
            except SystemExit, e:
                if e.code:
                    raise

setup(name=NAME,
      version='0.4',
      packages=find_packages(exclude=['tests']),
      cmdclass=dict(test=test),

      install_requires=install_requires,
      tests_require=test_requires,
      extras_require={},
      setup_requires=[],

      dependency_links=[
          'hg+https://bitbucket.org/mforbes/mmfutils@0.2#egg=mmfutils-0.2',
      ],

      # Metadata
      author='Michael McNeil Forbes',
      author_email='michael.forbes+bitbucket@gmail.com',
      url='https://bitbucket.org/mforbes/pytimeode',
      description="Dynamical evolution of complex systems.",
      long_description=__doc__,
      license='BSD',

      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Utilities',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: BSD License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
      ],
      )
