
Table of Contents
=================

-  `1. PyTimeODE <#1.-PyTimeODE>`__
-  `2. Installing <#2.-Installing>`__
-  `3. Usage <#3.-Usage>`__

   -  `3.1 Adams-Bashforth-Milne
      (ABM) <#3.1-Adams-Bashforth-Milne-%28ABM%29>`__
   -  `3.2 Split-Operator Evolution <#3.2-Split-Operator-Evolution>`__

-  `4. Development Instructions <#4.-Development-Instructions>`__

1. PyTimeODE
============

Dynamical evolution of complex systems.

This package provides an interface to a set of ODE solvers for solving
dynamical time-evolution problems. The original application was quantum
dynamics via Gross-Pitaevski equations (GPE) and superfluid density
functional theory (TDDFT), but the code is quite general and should be
able to be easily used for a variety of problems.

Currently the codes only use fixed time-step method (not adaptive).

2. Installing
=============

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

3. Usage
========

To use the solvers, you must define a class that imeplements one of the
interfaces defined in ``pytimeode.interfaces``. At a minimum, you must
provide the methods required by ``IStateMinimal``. This gives the
evolvers the ability to manipulate your state, making copies, scaling
the state, etc. Addional require functionality by the ``IState``
interface can be obtained from these by inheriting from ``StateMixin``
(though you might eventually like to provide custom implementations of
the ``IState`` interface for performance.) A variety of other mixins are
provided for implementing states from a numpy arrays
(``ArrayStateMixin``), a Mapping or Sequence of data (``StatesMixin``),
a Mapping or Sequence of arrays (``ArraysStateMixin``), or a Mapping or
Sequence of other states (``MultiStateMixin``).

.. code:: python

    from nbutils import describe_interface
    import pytimeode.interfaces
    describe_interface(pytimeode.interfaces.IStateMinimal)



.. raw:: html

    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
    <title></title>
    
    <div class="document">
    
    
    <p><tt class="docutils literal">IStateMinimal</tt></p>
    <blockquote>
    <p>Minimal interface required for state objects.  This will not satisfy all
    uses of a state.</p>
    <p>Attributes:</p>
    <blockquote>
    <p><tt class="docutils literal">dtype</tt> -- Return the dtype of the underlying state.  If this is
    real, then it is assumed that the states will always be
    real and certain optimizations may take place.</p>
    <p><tt class="docutils literal">t</tt> -- Time at which state is valid.</p>
    <p><tt class="docutils literal">writeable</tt> -- Set to <cite>True</cite> if the state is writeable, or
    <cite>False</cite> if the state should only be read.</p>
    </blockquote>
    <p>Methods:</p>
    <blockquote>
    <p><tt class="docutils literal">axpy(x, a=1)</tt> -- Perform <cite>self += a*x</cite> as efficiently as possible.</p>
    <p><tt class="docutils literal">copy()</tt> -- Return a writeable copy of the state.</p>
    <p><tt class="docutils literal">copy_from(y)</tt> -- Set this state to be a copy of the state <cite>y</cite></p>
    <p><tt class="docutils literal">scale(f)</tt> -- Perform <cite>self *= f</cite> as efficiently as possible.</p>
    </blockquote>
    </blockquote>
    </div>



Then you must satisfy the requirements of your particular solver.
Currently we support the following solvers.

3.1 Adams-Bashforth-Milne (ABM)
-------------------------------

.. code:: python

    describe_interface(pytimeode.interfaces.IStateForABMEvolvers)



.. raw:: html

    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
    <title></title>
    
    <div class="document">
    
    
    <p><tt class="docutils literal">IStateForABMEvolvers</tt></p>
    <blockquote>
    <p>Interface required by ABM and similar integration based evolvers.</p>
    <blockquote>
    These evolvers are very general, requiring only the ability for the problem
    to compute $dy/dt$.</blockquote>
    <p>This interface extends:</p>
    <blockquote>
    o <tt class="docutils literal">IState</tt></blockquote>
    <p>Methods:</p>
    <blockquote>
    <p><tt class="docutils literal">compute_dy(t, dy=None)</tt> -- Return <cite>dy/dt</cite> at time <cite>t</cite>.</p>
    <blockquote>
    If <cite>dy</cite> is provided, then use it for the result, otherwise return a new
    state.</blockquote>
    </blockquote>
    </blockquote>
    </div>



3.2 Split-Operator Evolution
----------------------------

.. code:: python

    describe_interface(pytimeode.interfaces.IStateForSplitEvolvers)



.. raw:: html

    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
    <title></title>
    
    <div class="document">
    
    
    <p><tt class="docutils literal">IStateForSplitEvolvers</tt></p>
    <blockquote>
    <p>Interface required by Split Operator evolvers.</p>
    <blockquote>
    These evolvers assume the problem can be split into two operators - $K$
    (kinetic energy) and $V$ (potential energy) so that $i dy/dt = (K+V)y$.
    The method requires that each of these operators be exponentiated.  The
    approach uses a Trotter decomposition that provides higher order accuracy,
    but requires evaluation of the potentials at an intermediate time.  The
    <tt class="docutils literal">get_potentials()</tt> method must therefore be able to compute the
    potentials at a specified time which might lie at a half-step.</blockquote>
    <p>This interface extends:</p>
    <blockquote>
    o <tt class="docutils literal">IState</tt></blockquote>
    <p>Methods:</p>
    <blockquote>
    <p><tt class="docutils literal">apply_exp_K(dt, t=None)</tt> -- Apply $e^{i K dt}$ in place</p>
    <p><tt class="docutils literal">apply_exp_V(dt, t=None, potentials=None)</tt> -- Apply $e^{i V dt}$ in place</p>
    <p><tt class="docutils literal">get_potentials(t)</tt> -- Return <cite>potentials</cite> at time <cite>t</cite>.</p>
    </blockquote>
    </blockquote>
    </div>



4. Development Instructions
===========================

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
