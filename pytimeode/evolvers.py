r"""Superfluid Hydrodynamic Evolution.

This package provides classes for performing the time evolution of superfluid
hydrodynamic systems.  These evolvers use only the methods provided by the
IState interface so as to work with a wide variety of problems.

Compared with the previous ``superfluid.evolvers`` module, we change a few of
things:

  * The ``State`` (i.e. ``y0``) replaces the role of the basis, providing
    functionality of computing norms, brakets, etc. that may need to utilize
    parallel or GPU computing.
"""
from __future__ import division

__all__ = ['EvolverABM', 'EvolverSplit']

import numpy as np

from mmfutils import interface

from . import interfaces
from .utils import numexpr, Object
from .utils import expr


######################################################################
# Evolvers: These are the core evolvers.
class EvolverBase(Object):
    r"""Base for all evolution objects.  Provides :meth:`evolve` which performs
    the time evolution.

    Attributes:
    t :
       Current time
    step :
       Current step number
    """
    interface.implements(interfaces.IEvolver)

    def __init__(self, y, dt, t=0.0, normalize=False, copy=True):
        r"""
        Parameter
        ---------
        y : State
           Initial state
        dt : float
           Timestep
        t : float
           Initial time.
        normalize : bool
           If `True` and the state `y` implements the `IStateWithNormalization`
           interface, then the state will be normalized after each step.  This
           allows for imaginary time evolution.  Note, however, that this can
           also be implemented in the states `compute_dy` method as a
           constraint on fixed particle number by making `dy` and `y`
           orthogonal, so this is an optional feature.
        copy : bool
           If `True`, then first make a copy of the state.
        """
        self.y = y.copy() if copy else y
        self.t = t
        self.dt = dt
        self.normalize = normalize
        if self.normalize:
            if not interface.verifyObject(interfaces.IStateWithNormalize, y):
                raise ValueError(
                    "Can only set normalize=True if state y implements " +
                    "IStateWithNormalize")

        if interfaces.INumexpr.providedBy(y):
            interface.verifyObject(interfaces.INumexpr, y)
            self.numexpr = numexpr
        else:
            self.numexpr = None

        Object.__init__(self)

    def init(self):
        self.y.t = self.t

    ######################################################################
    # Defaults:  Subclasses may want to overload these for performance.
    def get_dy(self, y=None, t=None, dy=None):
        r"""Return the `y'=dy/dt`.  This is used by ABM evolvers."""
        if t is None:
            t = self.t
        if y is None:
            y = self.get_y()
        if dy is None:
            dy = y.empty()
        with y.lock:
            dy = y.compute_dy(t=t, dy=dy)
        return dy

    def evolve(self, steps=None, omega=None):
        r"""Evolve the system by `steps`."""
        t0 = self.t
        assert steps > 1

        getattr(self.y, 'pre_evolve_hook', lambda: None)()

        self.do_step(first=True)

        for kt in xrange(1, steps-1):
            self.do_step()

        self.do_step(final=True)
        assert np.allclose(self.t, t0 + steps * self.dt)
        self.y.t = self.t

        getattr(self.y, 'post_evolve_hook', lambda: None)()


class EvolverSplit(EvolverBase):
    r"""Split operator evolution.

    This class implements the purely unitary evolution obtained by splitting
    the Hamiltonian into Kinetic and Potential pieces:

    .. math::
       U_{\delta t} = e^{-i\hbar \delta t (K + V)}
         \approx e^{-i\hbar \delta t K/2}
                 e^{-i\hbar \delta t V}
                 e^{-i\hbar \delta t K/2}

    This requires that :math:`V` is diagonal in position space and that the
    kinetic operator is diagonal in momentum space.

    Note that we need to include a factor of `degen` for the potential piece
    (this is already included in the kinetic pieces).
    """
    def __init__(self, y, dt, t=0.0, copy=True, **kw):
        interface.verifyObject(interfaces.IStateForSplitEvolvers, y)
        EvolverBase.__init__(self, y=y, dt=dt, t=t, copy=copy, **kw)
        if y.linear:
            pass
        elif interfaces.IStatePotentialsForSplitEvolvers.providedBy(y):
            interface.verifyObject(interfaces.IStatePotentialsForSplitEvolvers, y)
            self.use_nonlinear_potentials = True
        else:
            self.use_nonlinear_potentials = False
            self._nonlinear_tmp_state = y.copy()

    def init(self):
        EvolverBase.init(self)

    def do_step(self, first=False, final=False):
        r"""Perform one step of the Split method.

        The formal steps are grouped as::

          K_2 V K_2   K_2 V K_2   ...  K_2 V K_2

        where ``K_2`` is half the energy evolution.  We group these as::

          K_2 V K         V K     ...      V K_2

        The chemical potential is updated before V is applied, and the
        potentials are applied at the midpoint times.  (During evolution, the
        times will be staggered.  This is corrected at the `final` step.)
        """
        t = self.t
        dt = self.dt
        y = self.y

        if first:
            # First step with half of the kinetic energy
            y.apply_exp_K(dt=dt/2, t=t)
            t += 0.5*dt

        # Here is the application of the potential.  We first take a full step
        # with the self-consistent potentials at the starting time, then we
        # correct.

        # Compute and store V(t)
        if y.linear:
            # For linear problems, we can just evolve one full step.
            y.apply_exp_V(dt=dt, state=None, t=t)  # full step with V(t)
        elif self.use_nonlinear_potentials:
            # Nonlinear problems with a get_potentials function
            V = y.get_potentials(t=t)
            y.apply_exp_V(dt=dt, t=t, potentials=V)    # full step with V(t)

            # Compute and store V(t+dt)
            V -= y.get_potentials(t=t+dt)   # V0 - V1
            V *= -0.5                       # (V1 - V0)/2

            # Correct step
            y.apply_exp_V(dt=dt, t=t, potentials=V)
        else:
            # General non-linear problems require the temporary state.
            y1 = self._nonlinear_tmp_state
            y1.copy_from(y)
            y1.apply_exp_V(dt=dt, t=t, state=y1)
            y1.axpy(y)
            y1.scale(0.5)
            # Correct step
            y.apply_exp_V(dt=dt, t=t, state=y1)

        if final:
            # Only half of K at the end
            y.apply_exp_K(dt=dt/2, t=t)
            t += 0.5*dt
        else:
            y.apply_exp_K(dt=dt, t=t)
            t += dt

        if self.normalize:
            y.normalize()

        # This should be a float, not a view of an array otherwise one might
        # accumulate a bunch of times that are all the same since they refer to
        # the same array.
        y.t = t
        self.t = float(t)

    def get_y(self):
        r"""Return a copy of the current `y`."""
        return self.y.copy()


class EvolverABM(EvolverBase):
    r"""Simple class to manage the ABM steps and associated memory.

    This version uses storage for 2 previous states, 2
    predictor/corrector differences, and 4 previous derivatives `dy = -iH(y)`
    for a total of 8 arrays.  One can reduce this to 7 with some convoluted
    manipulations to reuse previous memory.

    Note that a copy of the original array will be made by default,
    but this copy can be suppressed by setting `copy=False` in the
    constructor.  (Note that the original state will then be mutated
    to an unspecified value.)

    If using `numexpr`, then we need one extra array _tmp until Issue
    92 is merged.

    Notes
    -----
    * We store `161/170*(c - p)` values in :attr:`dcps`.  These are scaled
      predictor-corrector differences.
    * We store the previous y's and dy's in the lists :attr:`ys` and
      :attr:`dys`.  All of these are sorted so that the most recent elemnt is
      first (i.e. ``ys[0]``).
    """

    def __init__(self, y, dt, t=0.0,
                 mu=None, no_runge_kutta=False,
                 **kw):
        r"""
        Parameters
        ----------
        no_runge_kutta : bool
           The initial four steps are taken using a fourth order Runge Kutta
           integrator in order to fill out the required predictor corrector
           arrays :attr:`ys`, `:attr:`dys`, :attr:`dcps` for the ABM method.
           If this is `True`, then we assume that the previous four steps were
           stationary and populate the arrays accordingly.
        """
        interface.verifyObject(interfaces.IStateForABMEvolvers, y)

        self.mu = mu
        self.no_runge_kutta = no_runge_kutta

        # Will need to store these for pickling.
        self.ys = None
        self.dcps = None
        self.dys = None
        EvolverBase.__init__(self, y=y, dt=dt, t=t, **kw)

    def init(self):
        EvolverBase.init(self)

        y0 = self.y
        dt = self.dt

        # 2 copies for the ys, 2 (predictor - corrector) differences,
        # and 4 copies for dy = -1j*H*y
        if self.no_runge_kutta:
            self.ys = [y0, y0.copy()]
            self.dcps = [_y*(161/170*0) for _y in self.ys]
            self.dys = [_y*0 for _y in [y0]*4]
        else:
            self.ys = [y0]
            self.dcps = [161/170*0*y0]
            self.dys = []

        # Coefficients for the ABM method
        h = dt
        self._ap = h/48 * np.array([119, -99, 69, -17], dtype=float)
        _tmp = h * 161./48./170.
        self._am = _tmp * (17)
        self._ac = _tmp * np.array([-68, 102, -68, 17], dtype=float)

        if self.numexpr:
            # Use sympy to simplify the expressions
            constants = dict(h=h)
            m = '((y0+y1)/2 + h/48*(119*dy0-99*dy1+69*dy2-17*dy3) + dcp0)'
            dcp = 'h*161/48/170*(17*dm-68*dy0+102*dy1-68*dy2+17*dy3)'
            self._expr_m = expr.Expression(
                m, constants=constants, state=y0,
                args=['y0', 'y1', 'dy0', 'dy1', 'dy2', 'dy3', 'dcp0'],
                ex_uses_vml=False)
            self._expr_dcp = expr.Expression(
                dcp, constants=constants, state=y0,
                args=['dm', 'dy0', 'dy1', 'dy2', 'dy3'],
                ex_uses_vml=False)

            self._expr_y = expr.Expression(
                'm + dcp - dcp0', constants=constants, state=y0,
                args=['m', 'dcp', 'dcp0'],
                ex_uses_vml=False)

            if numexpr.__version__ < "2.3":
                # Until issue 93 makes it into a release, we will not use the same
                # variable in and out, so we need one extra array
                # http://code.google.com/p/numexpr/issues/detail?id=93
                pass
            self._tmp = self.y.copy()

    def do_step(self, first=None, final=None):
        if len(self.dys) < 4:
            self.do_step_runge_kutta()
            self.ys = self.ys[:2]            # Only keep two previous steps
            if len(self.dys) == 4:
                # Only allocate these here.  Not exactly sure what
                # values to use.
                self.dcps = [0*_y for _y in self.ys]
        else:
            #self.do_step_ABM()
            self.do_step_ABM_numexpr()

        # This should be atomic, not an array otherwise one might accumulate a
        # bunch of times that are all the same.
        self.t = float(self.t)

    def do_step_runge_kutta(self):
        r"""4th order Runge Kutta for the first four steps to populate the
        predictor/corrector arrays."""
        t = self.t
        h = self.dt
        ys = self.ys
        dys = self.dys

        y = self.ys[0].copy()
        if len(self.dys) < len(self.ys):
            # Need to compute dy
            dy = self.get_dy(y=y, t=self.t)
            dys.insert(0, dy)
        else:
            dy = self.dys[0]

        # h might be an array so multiply this on the right so dy does not get
        # converted to an array (a problem for dy which support
        # __array_interface__)

        if False:           # pragma: no cover
            # A simple but memory-inefficient approach
            # Uses a total of 15 arrays!
            k = [None, None, None, None]
            k[0] = dy * h
            k[1] = (self.get_dy(y + k[0]/2., t=t + h/2.)) * h
            k[2] = (self.get_dy(y + k[1]/2., t=t + h/2.)) * h
            k[3] = (self.get_dy(y + k[2],    t=t + h)) * h
            y += (k[0] + 2*k[1] + 2*k[2] + k[3])/6.0
            del k
        elif False:             # pragma: no cover
            # Better: uses a total of 11 arrays
            f0 = dy
            y.axpy(dy, h/2.)
            f1 = self.get_dy(y, t=t + h/2.)
            y.axpy(dy, -h/2.)
            y.axpy(f1, h/2.)
            f2 = self.get_dy(y, t=t + h/2.)
            y.axpy(f1, -h/2.)
            y.axpy(f2, h)
            f3 = self.get_dy(y, t=t + h)
            y.axpy(f2, -h)
            y.axpy(f0, h/6.)
            y.axpy(f1, h/3.)
            y.axpy(f2, h/3.)
            y.axpy(f3, h/6.)
            del f0, f1, f2, f3
        else:
            # I think this is the best we can do memory wise: (10 arrays)
            f0 = dy
            y.axpy(dy, h/2.)
            f1 = self.get_dy(y, t=t + h/2.)
            y.axpy(dy, -h/2.)
            y.axpy(f1, h/2.)
            f2 = self.get_dy(y, t=t + h/2.)
            y.axpy(f1, -h/2.)
            y.axpy(f2, h)
            f1.axpy(f2, -2.)
            f3 = self.get_dy(y, dy=f2, t=t + h)
            del f2
            y.axpy(f1, h/3.)
            y.axpy(f0, h/6.)
            y.axpy(f3, h/6.)
            del f0, f1, f3

        if self.normalize:
            y.normalize()

        self.t += h
        dy = self.get_dy(y=y, t=self.t)

        ys.insert(0, y)
        dys.insert(0, dy)

    def do_step_ABM(self):
        r"""Perform one step of the ABM method."""
        t = self.t
        dt = self.dt
        ys = self.ys            # Slightly faster to make these local
        dcps = self.dcps
        dys = self.dys

        # Remove array from the end.  We will use this for the modifier, and
        # then finally for the new y
        y = ys.pop()

        y *= 0.5
        y.axpy(x=ys[0], a=0.5)
        for _i in xrange(4):
            y.axpy(x=dys[_i], a=self._ap[_i])
        y.axpy(x=dcps[0], a=1)

        dcp = dcps.pop()

        # Compute m' in next dcp array, then update
        dcp = self.get_dy(y=y, t=t+dt, dy=dcp)
        dcp *= self._am
        for _i in xrange(4):
            dcp.axpy(x=dys[_i], a=self._ac[_i])

        y.axpy(x=dcp, a=1)
        y.axpy(x=dcps[0], a=-1)

        t += dt

        dy = dys.pop()
        dy = self.get_dy(y=y, t=t, dy=dy)

        if self.normalize:
            y.normalize()

        ys.insert(0, y)
        dys.insert(0, dy)
        dcps.insert(0, dcp)
        self.t = t

    def do_step_ABM_numexpr(self):
        r"""Perform one step of the ABM method.  This version uses numexpr."""
        if not self.numexpr:
            return self.do_step_ABM()

        t = self.t
        dt = self.dt
        ys = self.ys            # Slightly faster to make these local
        dcps = self.dcps
        dys = self.dys

        # Remove array from the end.  We will use this for the modifier, and
        # the finally for the new y
        y0 = ys.pop()
        m = dcps.pop()
        m.apply(self._expr_m,
                y0=y0, y1=ys[0], dy0=dys[0], dy1=dys[1], dy2=dys[2],
                dy3=dys[3], dcp0=dcps[0])
        dcp = y0
        del y0

        # Compute dm = m' in the _tmp array
        dm = self._tmp
        dm = self.get_dy(y=m, t=t+dt, dy=dm)

        # Computed dcp
        dcp.apply(self._expr_dcp,
                  dm=dm, dy0=dys[0], dy1=dys[1], dy2=dys[2], dy3=dys[3])

        y = dm
        del dm

        y.apply(self._expr_y, m=m, dcp=dcp, dcp0=dcps[0])
        if self.normalize:
            y.normalize()

        self._tmp = m
        del m

        t += dt

        dy = dys.pop()
        dy = self.get_dy(y=y, t=t, dy=dy)

        ys.insert(0, y)
        dys.insert(0, dy)
        dcps.insert(0, dcp)
        self.t = t

    @property
    def y(self):
        return self.ys[0]

    @y.setter
    def y(self, y0):
        self.ys = [y0]
        self.dcps = None
        self.dys = None

    def get_y(self):
        r"""Return a copy of the current y"""
        return self.y.copy()


interface.verifyClass(interfaces.IEvolver, EvolverABM)
interface.verifyClass(interfaces.IEvolver, EvolverSplit)
