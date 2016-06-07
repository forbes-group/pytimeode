
Table of Contents
=================

-  `1. PyTimeODE <#1.-PyTimeODE>`__
-  `2. Installing <#2.-Installing>`__
-  `3. Usage <#3.-Usage>`__

   -  `3.1 Example (TL;DR) <#3.1-Example-%28TL;DR%29>`__
   -  `3.2 Overview <#3.2-Overview>`__

-  `4. Interfaces <#4.-Interfaces>`__

   -  `4.1 Adams-Bashforth-Milne
      (ABM) <#4.1-Adams-Bashforth-Milne-%28ABM%29>`__
   -  `4.2 Split-Operator Evolution <#4.2-Split-Operator-Evolution>`__
   -  `4.3 Others <#4.3-Others>`__

-  `5. Development Instructions <#5.-Development-Instructions>`__

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

You can install it with ``pip`` using my package index as follows:

::

    pip install --find-links https://bitbucket.org/mforbes/mypi/ --user pytimeode

At a later date, when the package is release, it will be able to be
installed directly with ``pip install pytimeode``.

3. Usage
========

3.1 Example (TL;DR)
-------------------

As a user you need to define an appropriate ``State`` class which
defines your problem, then pass an instance of this to an evolver. If
you can express your state in terms of arrays, then the following
skeleton should suffice. It solves the problem

.. math::


     \frac{\mathrm{d}y(t)}{\mathrm{d}t} = f\bigl(y(t), t\bigr) = -y^2, 
     \qquad
     y(0) = y_0 = \begin{pmatrix}
     1\\
     2
     \end{pmatrix}

.. code:: python

    from pytimeode.interfaces import implements, IStateForABMEvolvers, ArrayStateMixin
    from pytimeode.evolvers import EvolverABM
    
    import numpy as np
    
    class State(ArrayStateMixin):
        implements(IStateForABMEvolvers)
        
        def __init__(self, shape):
            # ArrayStateMixin requires you to define self.data
            self.data = np.zeros(shape)
            
        def compute_dy(self, dy):
            """Set `dy[...]` to contain f(y, t) where dy/dt = f(y,t)"""
            dy[...] = -self[...]**2
            return dy
            
    state = State([2])
    state[...] = [1, 2]  # Set the initial state
    
    evolver = EvolverABM(state, dt=0.01, t=0.0)
    ts = [evolver.y.t]
    states = [evolver.get_y()]  # Use get_y() to get a copy of the state
    
    for n in xrange(10):
        evolver.evolve(100)    
        states.append(evolver.get_y())
    
    states
    #%pylab notebook --no-import-all
    #plt.plot(ts, res)




.. parsed-literal::

    [State(t= 0., data=array([ 1.,  2.])),
     State(t= 1., data=array([ 0.5       ,  0.66666667])),
     State(t= 2., data=array([ 0.33333333,  0.4       ])),
     State(t= 3., data=array([ 0.25      ,  0.28571429])),
     State(t= 4., data=array([ 0.2       ,  0.22222222])),
     State(t= 5., data=array([ 0.16666667,  0.18181818])),
     State(t= 6., data=array([ 0.14285714,  0.15384615])),
     State(t= 7., data=array([ 0.125     ,  0.13333333])),
     State(t= 8., data=array([ 0.11111111,  0.11764706])),
     State(t= 9., data=array([ 0.1       ,  0.10526316])),
     State(t=10., data=array([ 0.09090909,  0.0952381 ]))]



3.2 Overview
------------

The design requires two components – states which define a time-depenent
problem of the form :math:`\dot{y} = f(y, t)` and evolvers which solve
this problem given an initial state. To define the interaction between
the evolvers and the states we provide a set of interfaces in the
``pytimeode.interfaces`` module.

The user must define a class ``State`` which supports the
``pytimeode.interfaces.IState`` and at least one of
``pytimeode.interfaces.IStateForABMEvolvers`` or
``pytimeode.interfaces.IStateForSplitEvolvers``. Here is a description
of the interfaces

-  ``pytimeode.interfaces.IState``: This is the general state interface.
   States act somewhat like arrays and you can use them as such. The
   preferred method for accessing the data in a state is by indexing
   ``data = state[...]`` or ``state[...] = 0.0`` for example. States can
   be copied etc. Note: you generally do not need to implement
   everything specified in this interface – most of it is specified in
   one of the various mixin classes (see below) so even custom
   applications need only define the methods specified in
   ``pytimeode.interfaces.IStateMinimal``. Using one of the array mixins
   such as ``pytimeode.interfaces.ArrayStateMixin`` allows you to forgo
   all of this and simply implement the problem in terms of arrays.

-  ``pytimeode.interfaces.IStateForABMEvolvers``: This is the interface
   used by ``pytimeode.evolvers.EvolverABM`` that uses a 5th-order
   Adams-Bashforth-Milne predictor-corrector method to solve an equation
   :math:`\dot{y} = f(y, t)`. The user only needs to define the function
   :math:`f(y, t)` by defining the ``compute_dy()`` method. This is a
   highly accurate method - if it works, then it is probably correct
   (but it requires a fairly small step size to work. Unfortunately, it
   requires about 10 copies of the state in memory (for the initial
   Runge-Kutta steps) and 8 copies in general. Energy and particle
   number are not explicitly conserved and so can be used as a check of
   the accuracy of the simulation.

-  ``pytimeode.interfaces.IStateForSplitEvolvers``: This is the
   interface used by ``pytimeode.evolvers.EvolverSplit`` that assumes
   that the problem can be expressed as
   :math:`\dot{y} = -\mathrm{i} (K + V[y])y` where one can compute
   :math:`\exp(-\mathrm{i}K t)` and :math:`\exp(-\mathrm{i}V[y] t)`
   explicitly (where :math:`V[y]` may depend nonlinearly on the state
   :math:`y(t)`). If the problem can be broken up this way – which is
   common with quantum mechanical problems – then a split-operator
   method can be applied. This is only second order, but manifestly
   conserves particle number, thus one can often get away with large
   step sizes to gain qualitative insight into the evolution of a
   system. It also only requires a couple of copies of the state. Be
   aware that the simulation might look correct, but be quantitatively
   inaccurate.

-  ``pytimeode.interfaces.IStateWithNormalize``: If the state can be
   meaningfully scaled to satisfy a normalization constraint, then
   providing this interface will allow the evolvers to explicitly
   correct the normalization of the state during evolution. The main
   application of this is for imaginary- or complex-time evolution to
   find the ground state. In general such evolution will explicitly
   violate the norm of the state. One can avoid this by projecting the
   evolution vector in such a way as to preserve the norm (this idea is
   implemented in the current code through the "constraints" mechanism),
   but then one must use a small enough timestep that the evolution is
   accurate at each step. Providing a normalization method allows one to
   use a large time step (with the split operator evolution in
   particular) to quickly descend to the minimum energy state. Here one
   does not care about the accuracy of the evolution – only that the
   ground state is ultimately achieved, thus one can evolve with some
   large steps and normalize the state, then polish off the solution
   with small time steps once one is close.

-  ``pytimeode.interfaces.INumexpr``: Providing an ``apply`` method
   allows the evolvers to use a more efficient strategy for evaluating
   expressions that improves performance. The default states used here
   are based on NumPy arrays which allow this functionality.

To use the solvers, you must define a class that implements one of the
interfaces defined in ``pytimeode.interfaces``. At a minimum, you must
provide the methods required by ``IStateMinimal``. This gives the
evolvers the ability to manipulate your state, making copies, scaling
the state, etc. Additional required functionality by the ``IState``
interface can be obtained from these by inheriting from ``StateMixin``
(though you might eventually like to provide custom implementations of
the ``IState`` interface for performance.) A variety of other mixins are
provided for implementing states from a numpy arrays
(``ArrayStateMixin``), a Mapping or Sequence of data (``StatesMixin``),
a Mapping or Sequence of arrays (``ArraysStateMixin``), or a Mapping or
Sequence of other states (``MultiStateMixin``).

4. Interfaces
=============

Here are the relevant interfaces:

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
    <p><tt class="docutils literal">dtype</tt> -- Return the dtype of the underlying state.  If this is real, then it
    is assumed that the states will always be real and certain
    optimizations may take place.</p>
    <p><tt class="docutils literal">t</tt> -- Time at which state is valid.  This is the time at which potentials
    should be evaluated etc.  (It will be set by the evolvers before
    calling the various functions like compute_dy().)</p>
    <p><tt class="docutils literal">writeable</tt> -- Set to <cite>True</cite> if the state is writeable, or <cite>False</cite> if the state
    should only be read.</p>
    </blockquote>
    <p>Methods:</p>
    <blockquote>
    <p><tt class="docutils literal">axpy(x, a=1)</tt> -- Perform <cite>self += a*x</cite> as efficiently as possible.</p>
    <p><tt class="docutils literal">braket(x)</tt> -- Perform <cite>braket(self, x)</cite> as efficiently as possible.</p>
    <blockquote>
    Note: this should conjugate self (if complex) and include any
    factors of a metric if needed.</blockquote>
    <p><tt class="docutils literal">copy()</tt> -- Return a writeable copy of the state.</p>
    <p><tt class="docutils literal">copy_from(y)</tt> -- Set this state to be a copy of the state <cite>y</cite></p>
    <p><tt class="docutils literal">scale(f)</tt> -- Perform <cite>self *= f</cite> as efficiently as possible.</p>
    </blockquote>
    </blockquote>
    </div>



Then you must satisfy the requirements of your particular solver.
Currently we support the following solvers.

4.1 Adams-Bashforth-Milne (ABM)
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
    <tt class="docutils literal">compute_dy(dy)</tt> -- Return <cite>dy/dt</cite> at time <cite>self.t</cite> using the memory in state <cite>dy</cite>.</blockquote>
    </blockquote>
    </div>



4.2 Split-Operator Evolution
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
    <p>These evolvers assume the problem can be split into two operators - $K$
    (kinetic energy) and $V$ (potential energy) so that $i dy/dt = (K+V)y$.
    The method requires that each of these operators be exponentiated.  The
    approach uses a Trotter decomposition that provides higher order accuracy,
    but requires evaluation of the potentials at an intermediate time.</p>
    <p>This interface requires that the <cite>apply_exp_V()</cite> method accept
    another state object which should be used for calculating any
    non-linear terms in $V$ which are state dependent.</p>
    <p>If your problem is linear (i.e. $V$ depends only on time, not on
    the state as in the case of the usual linear Schrodinger
    equation), then you should set the linear attribute which will
    improve performance (but do not use this for non-linear problems
    or the order of convergence will be reduced).</p>
    </blockquote>
    <p>This interface extends:</p>
    <blockquote>
    o <tt class="docutils literal">IState</tt></blockquote>
    <p>Attributes:</p>
    <blockquote>
    <tt class="docutils literal">linear</tt> -- Is the problem linear?</blockquote>
    <p>Methods:</p>
    <blockquote>
    <p><tt class="docutils literal">apply_exp_K(dt)</tt> -- Apply $e^{-i K dt}$ in place</p>
    <p><tt class="docutils literal">apply_exp_V(dt, state)</tt> -- Apply $e^{-i V dt}$ in place using <cite>state</cite> for any
    nonlinear dependence in V. (Linear problems should ignore
    <cite>state</cite>.)</p>
    </blockquote>
    </blockquote>
    </div>



4.3 Others
----------

Here are the remaining interfaces:

.. code:: python

    describe_interface(pytimeode.interfaces.IState)



.. raw:: html

    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
    <title></title>
    
    <div class="document">
    
    
    <p><tt class="docutils literal">IState</tt></p>
    <blockquote>
    <p>Interface required by the evolvers.</p>
    <blockquote>
    Many of these functions are for convenience, and can be implemented from
    those defined in <tt class="docutils literal">IState</tt> by including the <tt class="docutils literal">StateMixin</tt> mixin.</blockquote>
    <p>This interface extends:</p>
    <blockquote>
    o <tt class="docutils literal">IStateMinimal</tt></blockquote>
    <p>Methods:</p>
    <blockquote>
    <p><tt class="docutils literal">__add__(y)</tt> -- Return <cite>self + y</cite></p>
    <p><tt class="docutils literal">__div__(f)</tt> -- Return <cite>self / y</cite></p>
    <p><tt class="docutils literal">__iadd__(y)</tt> -- <cite>self += y</cite></p>
    <p><tt class="docutils literal">__idiv__(f)</tt> -- <cite>self /= f</cite></p>
    <p><tt class="docutils literal">__imul__(f)</tt> -- <cite>self *= f</cite></p>
    <p><tt class="docutils literal">__isub__(y)</tt> -- <cite>self -= y</cite></p>
    <p><tt class="docutils literal">__itruediv__(f)</tt> -- <cite>self /= f</cite></p>
    <p><tt class="docutils literal">__mul__(f)</tt> -- Return <cite>self * y</cite></p>
    <p><tt class="docutils literal">__neg__()</tt> -- <cite>-self</cite></p>
    <p><tt class="docutils literal">__pos__()</tt> -- <cite>+self</cite></p>
    <p><tt class="docutils literal">__rmul__(f)</tt> -- Return <cite>self * y</cite></p>
    <p><tt class="docutils literal">__sub__(y)</tt> -- Return <cite>self - y</cite></p>
    <p><tt class="docutils literal">__truediv__(f)</tt> -- Return <cite>self / y</cite></p>
    <p><tt class="docutils literal">empty()</tt> -- Return a writeable but uninitialized copy of the state.</p>
    <blockquote>
    Can be implemented with <cite>self.copy()</cite> but some states might be
    able to make a faster version if the data does not need to be copied.</blockquote>
    <p><tt class="docutils literal">zeros()</tt> -- Return a writeable but zeroed out copy of the state.</p>
    <blockquote>
    Can be implemented with <cite>self.copy()</cite> but some states might be
    able to make a faster version if the data does not need to be copied.</blockquote>
    </blockquote>
    </blockquote>
    </div>



.. code:: python

    describe_interface(pytimeode.interfaces.IStateWithNormalize)



.. raw:: html

    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
    <title></title>
    
    <div class="document">
    
    
    <p><tt class="docutils literal">IStateWithNormalize</tt></p>
    <blockquote>
    <p>Interface for states with a normalize function.  Solvers can then
    provide some extra features natively like allowing imaginary time evolution
    for initial state preparation.</p>
    <p>This interface extends:</p>
    <blockquote>
    o <tt class="docutils literal">IState</tt></blockquote>
    <p>Methods:</p>
    <blockquote>
    <p><tt class="docutils literal">normalize()</tt> -- Normalize (and orthogonalize) the state.</p>
    <blockquote>
    This method may be called by the evolvers if they implement non-unitary
    evolution (imaginary time cooling for example) after each step.  For
    Fermionic DFTs, the single-particle wavefunctions would then also need
    to be orthogonalized.</blockquote>
    </blockquote>
    </blockquote>
    </div>



.. code:: python

    describe_interface(pytimeode.interfaces.INumexpr)



.. raw:: html

    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
    <title></title>
    
    <div class="document">
    
    
    <p><tt class="docutils literal">INumexpr</tt></p>
    <blockquote>
    <p>Allows for numexpr optimizations</p>
    <p>Attributes:</p>
    <blockquote>
    <tt class="docutils literal">dtype</tt> -- Return the dtype of the underlying state.  If this is real, then it
    is assumed that the states will always be real and certain optimizations
    may take place.</blockquote>
    <p>Methods:</p>
    <blockquote>
    <tt class="docutils literal">apply(expr, **kwargs)</tt> -- Evaluate the expression using the arguments in <tt class="docutils literal">kwargs</tt> and store
    the result in <tt class="docutils literal">self</tt>.  For those instance of the class in <tt class="docutils literal">kwargs</tt>,
    the expression must be applied over all components.  This is used by
    the <tt class="docutils literal">utils.expr.Expression</tt> class to allow numexpr expressions to be
    applied to custom state objects.</blockquote>
    </blockquote>
    </div>



.. code:: python

    describe_interface(pytimeode.interfaces.IStatePotentialsForSplitEvolvers)



.. raw:: html

    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
    <title></title>
    
    <div class="document">
    
    
    <p><tt class="docutils literal">IStatePotentialsForSplitEvolvers</tt></p>
    <blockquote>
    <p>Interface required by Split Operator evolvers.</p>
    <blockquote>
    This is a specialization of <cite>IStateForSplitEvolvers</cite> that uses an
    alternative method <cite>get_potentials()</cite> to compute the non-linear
    portion of the potential. It is intended for use when the state is
    much more complicated than the non-linear portion of the
    potential, hence only a separate copy of the potentials is maintained.</blockquote>
    <p>This interface extends:</p>
    <blockquote>
    o <tt class="docutils literal">IStateForSplitEvolvers</tt></blockquote>
    <p>Methods:</p>
    <blockquote>
    <p><tt class="docutils literal">apply_exp_V(dt, potentials)</tt> -- Apply $e^{-i V dt}$ in place using <cite>potentials</cite></p>
    <p><tt class="docutils literal">get_potentials()</tt> -- Return <cite>potentials</cite> at time <cite>self.t</cite>.</p>
    </blockquote>
    </blockquote>
    </div>



5. Development Instructions
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
