"""Interfaces.

At the top level, one probably wants to use the tested evolvers in
``evolver.py``.  These require only an object implementing the appropriate
``IState`` interface.

If you want to reuse other components like bases, then you will need to
implement the additional interfaces define here.  Here is the dependency graph.
"""
import contextlib
import copy

import numpy as np

from mmfutils.interface import (implements, Interface, Attribute)

__all__ = ['IEvolver', 'IStateMinimal', 'IState', 'INumexpr',
           'IStateForABMEvolvers', 'IStateForSplitEvolvers',
           'IStateWithNormalize',
           'StateMixin', 'ArrayStateMixin', 'MultiArrayStateMixin',
           'ArrayListStateMixin', 'ArrayDictStateMixin']


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
        """Return a writeable copy of the state."""

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


class INumexpr(Interface):
    """Allows for numexpr optimizations"""
    dtype = Attribute("dtype",
                      """Return the dtype of the underlying state.  If this is
                      real, then it is assumed that the states will always be
                      real and certain optimizations may take place.""")

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
    def compute_dy(t, dy=None):
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

    @property
    @contextlib.contextmanager
    def lock(self):
        writeable = self.writeable
        self.writeable = False
        try:
            yield
        finally:
            self.writeable = writeable


class StatesMixin(object):
    """Mixin for states with a set of "quantum numbers".

    These states contain a collection of data indexed by a set of keys that we
    call "quantum numbers".
    """
    implements([INumexpr])

    def __len__(self):
        return len(list(self))

    ######################################################################
    # Requires these methods
    def __iter__(self):
        """Return the list of quantum numbers."""

    def __getitem__(self, key):
        """Return the data associated with `key`"""

    def apply(self, expr, **kwargs):
        for _l in self:
            kw = {}
            for _k in kwargs:
                kw[_k] = kwargs[_k]
                if isinstance(kw[_k], self.__class__):
                    kw[_k] = kw[_k][_l]

            expr(out=self[_l], **kw)


class ArrayStateMixin(StateMixin):
    """Mixin providing support for states with a single data array.

    Assumes that the data is an array-like object called `self.data`.  This
    provides all the functionality required by IState.  All the user needs to
    provide are the methods for the required `IStateFor...Evolvers`.
    """
    implements([INumexpr])
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

    def copy(self):
        """Return a copy of the state.

        Uses `copy.copy()` to shallow-copy attributes, and copy.deepcopy()` to
        copy the data and potentials.
        """
        y = copy.copy(self)
        y.writeable = True      # Copies should be writeable
        y.data = copy.deepcopy(self.data)
        y.potentials = copy.deepcopy(self.potentials)
        return y

    def copy_from(self, y):
        """Set this state to be a copy of the state `y`"""
        assert self.writeable
        args = {}
        for key in ['data', 'potentials']:
            y_array = getattr(y, key)
            if key in self.__dict__:
                args[key] = getattr(self, key)
                args[key][...] = y_array
            else:
                args[key] = copy.deepcopy(y_array)

        self.__dict__.update(y.__dict__, **args)

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
        """We can't really do this since we don't know the constructor.  We
        just show the data here."""
        return "{}({})".format(self.__class__.__name__, repr(self.data))

    @property
    def __array_interface__(self):
        """Allows states to act as arrays with ``np.asarray(state)``."""
        return self.data.__array_interface__

    def apply(self, expr, **kwargs):
        kw = {}
        for _k in kwargs:
            kw[_k] = kwargs[_k]
            if isinstance(kw[_k], self.__class__):
                kw[_k] = kw[_k].data

        expr(out=self.data, **kw)


class MultiArrayStateMixin(StatesMixin, ArrayStateMixin):
    """Mixin providing support for states with a list of data arrays.

    Requires `__iter__()` provide keys `key` so that `self.data[key]` is an
    array representing all the data. This provides all the functionality
    required by IState.  All the user needs to provide are the methods for the
    required `IStateFor...Evolvers`.
    """
    def __getitem__(self, key):
        """Return the array associated with the specified quantum number."""
        return self.data[key]

    @property
    def dtype(self):
        # For now assume all arrays have same type
        dtype = self[self.__iter__().next()].dtype
        assert np.all([dtype is self[_k].dtype for _k in self])
        return dtype

    @property
    def writeable(self):
        """Set to `True` if the state is writeable, or `False` if the state
        should only be read.
        """
        return np.all(self.data[key].flags.writeable for key in self)

    @writeable.setter
    def writeable(self, value):
        for key in self:
            self[key].flags.writeable = value

    def copy(self):
        """Return a copy of the state.

        Uses `copy.copy()` to shallow-copy attributes, and copy.deepcopy()` to
        copy the data and potentials.
        """
        y = copy.copy(self)
        y.data = copy.deepcopy(self.data)
        y.potentials = copy.deepcopy(self.potentials)
        for key in self:
            y[key][...] = self[key]
        return y

    def copy_from(self, y):
        """Set this state to be a copy of the state `y`"""
        assert self.writeable
        data = self.data
        for key in self:
            self[key][...] = y[key]
        args = dict(data=data, potentials=copy.deepcopy(y.potentials))
        self.__dict__.update(y.__dict__, **args)

    def axpy(self, x, a=1):
        """Perform `self += a*x` as efficiently as possible."""
        assert self.writeable
        for key in self:
            # Can't use += here because python translates that to __setitem__
            # which we do not support
            self[key].__iadd__(a*x[key])

    def scale(self, f):
        """Perform `self *= f` as efficiently as possible."""
        assert self.writeable
        for key in self:
            self[key].__imul__(f)

    @property
    def __array_interface__(self):
        """Allows states to act as arrays with ``np.asarray(state)``."""
        raise NotImplementedError


class ArrayListStateMixin(MultiArrayStateMixin):
    """Mixin providing support for states with a list of data arrays.

    Assumes that `self.data` is a list of arrays.
    """
    def __iter__(self):
        """Return the list of quantum numbers."""
        return xrange(len(self.data)).__iter__()


class ArrayDictStateMixin(MultiArrayStateMixin):
    """Mixin providing support for states with a dict of data arrays.

    Assumes that `self.data` is a dict of arrays.
    """
    def __iter__(self):
        """Return the list of quantum numbers."""
        return self.data.__iter__()
