
PyTimeODE
=========

Dynamical evolution of complex systems.

This package provides an interface to a set of ODE solvers for solving
dynamical time-evolution problems. The original application was quantum
dynamics via Gross-Pitaevski equations (GPE) and superfluid density
functional theory (TDDFT), but the code is quite general and should be
able to be easily used for a variety of problems.

Currently the codes only use fixed time-step method (not adaptive).

Usage
=====

Presently you must manually install the ``pytimeode`` package or include
it with your project. I currently recommend including it as a
subrepository managed by `myrepos <http://myrepos.branchable.com>`__.
For example, I typically use the following line in a top-level
``.mrconfig`` file (use the second checkout if you access bitbucket with
ssh keys):

::

    [_ext/pytimeode]
    checkout = hg clone 'https://bitbucket.org/mforbes/pytimeode' 'pytimeode'
    #checkout = hg clone 'ssh://hg@bitbucket.org/mforbes/pytimeode' 'pytimeode'

Then running ``mr checkout`` from the toplevel will pull and update the
latest version of ``pytimeode`` and put it in ``_ext/pytimeode``. I then
create and add a symlink to this in the top level (and add ``_ext`` to
my ``.hgignore`` file) so that I can use the ``pytimeode`` module
directly:

.. code:: bash

    ln -s _ext/pytimeode/pytimeode pytimeode
    hg add pytimeode .mrconfig .hgignore

At a later date, when the package is release, it will be able to be
installed with ``pip install pytimeode``.

Development Instructions
========================

If you are a developer of this package, there are a few things to be
aware of.

1. If you modify the notebooks in ``docs/notebooks`` then you may need
   to regenerate some of the ``.rst`` files and commit them so they
   appear on bitbucket. This is done automatically by the ``pre-commit``
   hook in ``.hgrc`` if you include this in your ``.hg/hgrc`` file with
   a line like:

   ::

       %include ../.hgrc

**Security Warning:** if you do this, be sure to inspect the ``.hgrc``
file carefully to make sure that no one inserts malicious code.

