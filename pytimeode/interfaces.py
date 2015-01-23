"""Interfaces.

At the top level, one probably wants to use the tested evolvers in
``evolver.py``.  These require only an object implementing the appropriate
``IState`` interface.

If you want to reuse other components like bases, then you will need to
implement the additional interfaces define here.  Here is the dependency graph.
"""

import numpy as np
from .utils.interface import (Interface, Attribute, implements,
                              verifyObject, verifyClass)


class IEvolver(Interface):
    """General interface for evolvers"""
    """Requires:

    y.update(t)

    # Split Operator uses
    y.apply_exp_K(dt)
    y.apply_exp_V(dt, potentials)
    y.copy()
    y.copy_from(y)
    y.compute_dy_inplace(potentials)
    y.axpy(x, a)
    y.scale(self, f)

    Also uses +=, +, -, * etc. but all these have default implementations in
    terms of y.axpy and y.scale


    V = y.potentials
    V - V
    0.5*V
    y.potentials.copy()


    # ABM uses

    """
    y = Attribute("y", "Current state")
    t = Attribute("t", "Current time")

    def __init__(y, dt, t=0.0):
        """Return an evolver starting with state `y` at time `t` and evolve
        with step `dt`."""

    def evolve(steps):
        """Evolve the initial state by `steps` of length `dt` in time"""


class IStateMinimal(Interface):
    """Minimal interface required for state objects.  This will not satisfy all
    uses of a state."""

    # For self-consistent problems such as a DFT, the state must keep track of
    # the current set of self-consistent potentials.  These are computed by
    # `update()`.  The potentials must act like arrays or numbers (i.e. they
    # must support addition, scaling, and copying).
    potentials = Attribute("potentials",
                           """Array-like object for representing the current
                           potentials.""")

    writeable = Attribute("writeable",
                          """Set to `True` if the state is writeable, or
                          `False` if the state should only be read.""")

    dtype = Attribute("dtype",
                      """Return the dtype of the underlying state.  If this is
                      real, then it is assumed that the states will always be
                      real and certain optimizations may take place.""")

    def copy():
        """Return a copy of the state."""

    def copy_from(y):
        """Set this state to be a copy of the state `y`"""

    def update(t=0):
        """Set the time to be "t" and compute `self.potentials`."""

    def axpy(x, a=1):
        """Perform `self += a*x` as efficiently as possible."""

    def scale(f):
        """Perform `self *= f` as efficiently as possible."""

    # Note: we could get away with __imul__ but it requires one return self
    # which can be a little confusing, so we allow the user to simply define
    # `axpy` and `scale` instead.

    def apply(expr, **kwargs):
        """Evaluate the expression using the arguments in ``kwargs`` and store
        the result in ``self``.  For those instance of the class in ``kwargs``,
        the expression must be applied over all components.  This is used by
        the ``utils.expr.Expression`` class to allow numexpr expressions to be
        applied to custom state objects.
        """


class IState(IStateMinimal):
    """Interface required by the evolvers.

    Many of these functions are for convenience, and can be implemented from
    those defined in ``IState`` by including the ``StateMixin`` mixin.
    """
    def __imul__(f):
        """`self *= f`"""

    def __iadd__(y):
        """`self += y`"""

    def __isub__(y):
        """`self -= y`"""

    def __idiv__(f):
        """`self /= f`"""

    def __itruediv__(f):
        """`self /= f`"""

    def __add__(y):
        """Return `self + y`"""

    def __sub__(y):
        """Return `self - y`"""

    def __mul__(f):
        """Return `self * y`"""

    def __rmul__(f):
        """Return `self * y`"""

    def __truediv__(f):
        """Return `self / y`"""

    def __div__(f):
        """Return `self / y`"""


class IStateForABMEvolvers(IState):
    """Interface required by ABM and similar integration based evolvers."""
    def compute_dy_inplace(potentials=None):
        """Compute `dy` in place using the specified potentials (or
        `self.potentials`."""


class IStateForSplitEvolvers(IState):
    """Interface required by Split Operator evolvers."""
    def apply_exp_K(dt):
        r"""Apply $e^{i K dt}$"""

    def apply_exp_V(dt, potentials=None):
        r"""Apply $e^{i V dt}$`"""


class IStateWithNormalize(IState):
    """Interface for states with a normalize function.  Solvers can then
    provide some extra features natively like allowing imaginary time evolution
    for initial state preparation."""
    def normalize():
        """Normalize (and orthogonalize) the state.

        This method may be called by the evolvers if they implement non-unitary
        evolution (imaginary time cooling for example) after each step.  For
        Fermionic DFTs, the single-particle wavefunctions would then also need
        to be orthogonalized."""


######################################################################
# Defaults
#
# These mixins implement many of the required operations using only the
# methods required by the Minimal interfaces

class StateMixin(object):
    # Note: we could get away with __imul__ but it requires one return self
    # which can be a little confusing, so we allow the user to simply define
    # `axpy` and `scale` instead.
    def __imul__(self, f):
        """`self *= f`"""
        self.scale(f)
        return self

    def __iadd__(self, y):
        """`self += y`"""
        assert isinstance(y, self.__class__)
        self.axpy(y)
        return self

    def __isub__(self, y):
        """`self -= y`"""
        assert isinstance(y, self.__class__)
        self.axpy(y, a=-1)
        return self

    def __itruediv__(self, f):
        """`self /= f`"""
        self *= (1./f)
        return self

    __idiv__ = __itruediv__

    def __add__(self, y):
        """Return `self + y`"""
        assert isinstance(y, self.__class__)
        res = self.copy()
        res.axpy(y)
        return res

    def __sub__(self, y):
        """Return `self - y`"""
        assert isinstance(y, self.__class__)
        res = self.copy()
        res.axpy(y, -1)
        return res

    def __mul__(self, f):
        """Return `self * y`"""
        res = self.copy()
        res *= f
        return res

    __rmul__ = __mul__

    def __truediv__(self, f):
        """Return `self / y`"""
        res = self.copy()
        res *= (1./f)
        return res

    __div__ = __truediv__

    def __iter__(self):
        """Return the list of quantum numbers.

        This default version assumes that there is only once quantum number and
        that the corresponding array is self.y.
        """
        return [0].__iter__()

    def __getitem__(self, key):
        """Return the array associated with the specified quantum number.

        This default version assumes that there is only one array `self.y`.
        """
        assert key == 0
        return self.y

    @property
    def dtype(self):
        for _l in self:
            return self[_l].dtype

    def apply(self, expr, **kwargs):
        for l in self:
            kw = {}
            for _k in kwargs:
                kw[_k] = kwargs[_k]
                if isinstance(kw[_k], self.__class__):
                    kw[_k] = kw[_k][l]

            expr(out=self[l], **kw)
