"""Interfaces.

At the top level, one probably wants to use the tested evolvers in
``evolver.py``.  These require only an object implementing the appropriate
``IState`` interface.

If you want to reuse other components like bases, then you will need to
implement the additional interfaces define here.  Here is the dependency graph.
"""
import contextlib

from mmfutils.interface import (Interface, Attribute)


class IEvolver(Interface):
    """General interface for evolvers"""
    """Requires:

    # Split Operator uses
    y.apply_exp_K(dt)
    y.apply_exp_V(dt, potentials)
    y.copy()
    y.copy_from(y)
    y.compute_dy()
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

    def __init__(y, dt, t=0.0, copy=True):
        """Return an evolver starting with state `y` at time `t` and evolve
        with step `dt`."""

    def evolve(steps):
        """Evolve the initial state by `steps` of length `dt` in time"""


class IStateMinimal(Interface):
    """Minimal interface required for state objects.  This will not satisfy all
    uses of a state."""
    writeable = Attribute("writeable",
                          """Set to `True` if the state is writeable, or
                          `False` if the state should only be read.""")

    dtype = Attribute("dtype",
                      """Return the dtype of the underlying state.  If this is
                      real, then it is assumed that the states will always be
                      real and certain optimizations may take place.""")

    t = Attribute("t", """Time at which state is valid.""")

    def copy():
        """Return a copy of the state."""

    def copy_from(y):
        """Set this state to be a copy of the state `y`"""

    def axpy(x, a=1):
        """Perform `self += a*x` as efficiently as possible."""

    def scale(f):
        """Perform `self *= f` as efficiently as possible."""

    # Note: we could get away with __imul__ but it requires one return self
    # which can be a little confusing, so we allow the user to simply define
    # `axpy` and `scale` instead.


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

    def apply(expr, **kwargs):
        """Evaluate the expression using the arguments in ``kwargs`` and store
        the result in ``self``.  For those instance of the class in ``kwargs``,
        the expression must be applied over all components.  This is used by
        the ``utils.expr.Expression`` class to allow numexpr expressions to be
        applied to custom state objects.
        """


class IStateForABMEvolvers(IState):
    """Interface required by ABM and similar integration based evolvers.

    These evolvers are very general, requiring only the ability for the problem
    to compute $dy/dt$.
    """
    def compute_dy(y, t, dy=None):
        """Return `dy/dt` at time `t`.

        If `dy` is provided, then use it for the result, otherwise return a new
        state.
        """


class IStateForSplitEvolvers(IState):
    r"""Interface required by Split Operator evolvers.

    These evolvers assume the problem can be split into two operators - $K$
    (kinetic energy) and $V$ (potential energy) so that $i dy/dt = (K+V)y$.
    The method requires that each of these operators be exponentiated.  The
    approach uses a Trotter decomposition that provides higher order accuracy,
    but requires evaluation of the potentials at an intermediate time.  The
    ``get_potentials()`` method must therefore be able to compute the
    potentials at a specified time which might lie at a half-step.
    """
    def get_potentials(t):
        """Return `potentials` at time `t`."""

    def apply_exp_K(dt, t=None):
        r"""Apply $e^{i K dt}$ in place"""

    def apply_exp_V(dt, t=None, potentials=None):
        r"""Apply $e^{i V dt}$ in place"""


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
        that the corresponding array is self.data.
        """
        return [0].__iter__()

    def __getitem__(self, key):
        """Return the array associated with the specified quantum number.

        This default version assumes that there is only one array `self.data`.
        """
        assert key == 0
        return self.data

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

    @property
    @contextlib.contextmanager
    def lock(self):
        writeable = self.writeable
        self.writeable = False
        try:
            yield
        finally:
            self.writeable = writeable


class ArrayStateMixin(StateMixin):
    """Mixin providing support for states with a single data array.

    Assumes that the data is an array-like object called `self.data`.
    """
    t = 0.0
    data = None
    potentials = None

    @property
    def writeable(self):
        """Set to `True` if the state is writeable, or `False` if the state
        should only be read.
        """
        return self.data.flags.writeable

    @writeable.setter
    def writeable(self, value):
        self.data.flags.writeable = value

    @property
    def dtype(self):
        """Return the dtype of the underlying state.  If this is real, then it
        is assumed that the states will always be real and certain
        optimizations may take place.
        """
        return self.data.dtype

    def copy(self, y=None):
        """Return a copy of the state.

        Subclasses can override this to initialize a different `y` object.
        """
        if y is None:
            y = self.__class__()
        y.data = self.data.copy()
        y.t = self.t
        if self.potentials is not None:
            y.potentials = self.potentials.copy()
        return y

    def copy_from(self, y):
        """Set this state to be a copy of the state `y`"""
        assert self.writeable
        self.t = y.t
        self.data[...] = y.data
        if y.potentials is not None:
            self.potentials[...] = y.potentials

    def axpy(self, x, a=1):
        """Perform `self += a*x` as efficiently as possible."""
        assert self.writeable
        self.data += a*x.data

    def scale(self, f):
        """Perform `self *= f` as efficiently as possible."""
        assert self.writeable
        self.data *= f

    # Note: we could get away with __imul__ but it requires one return self
    # which can be a little confusing, so we allow the user to simply define
    # `axpy` and `scale` instead.

    def __repr__(self):
        return repr(self.data)

    @property
    def __array_interface__(self):
        """Allows states to act as arrays with ``np.asarray(state)``."""
        return self.data.__array_interface__
