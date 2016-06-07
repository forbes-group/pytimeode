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

NAME = "pytimeode"

setup_requires = [
    'pytest-runner'
]

install_requires = [
    'mmfutils>=0.4.8',
    'zope.interface>=3.8.0'
]

test_requires = [
    'pytest>=2.8.1',
    'pytest-cov>=2.2.0',
    'pytest-flake8',
    'coverage',
    'flake8',
    'pep8==1.5.7',     # Needed by flake8: dependency resolution issue if not pinned
]

extras_require = dict(
    doc=['mmf_setup'],
)

# Remove NAME from sys.modules so that it gets covered in tests. See
# http://stackoverflow.com/questions/11279096
for mod in sys.modules.keys():
    if mod.startswith(NAME):
        del sys.modules[mod]
del mod


setup(name=NAME,
      version='0.8.0.dev0',
      packages=find_packages(exclude=['tests']),

      setup_requires=setup_requires,
      install_requires=install_requires,
      tests_require=test_requires,
      extras_require=extras_require,

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
