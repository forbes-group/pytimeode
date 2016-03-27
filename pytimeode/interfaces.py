"""Interfaces.

At the top level, one probably wants to use the tested evolvers in
``evolver.py``.  These require only an object implementing the appropriate
``IState`` interface.

If you want to reuse other components like bases, then you will need to
implement the additional interfaces define here.  Here is the dependency graph.
"""
import collections
import contextlib
import copy

import numpy as np

from mmfutils.interface import (implements, Interface, Attribute)

__all__ = ['IEvolver', 'IStateMinimal', 'IState', 'INumexpr',
           'IStateForABMEvolvers',
           'IStateForSplitEvolvers',
           'IStatePotentialsForSplitEvolvers',
           'IStateWithNormalize',
           'StateMixin', 'ArrayStateMixin', 'ArraysStateMixin',
           'MultiStateMixin',
           'implements'
           ]


class IEvolver(Interface):
    """General interface for evolvers"""
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
    def __pos__():
        """`+self`"""

    def __neg__():
        """`-self`"""

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

    def empty():
        """Return a writeable but uninitialized copy of the state.

        Can be implemented with `self.copy()` but some states might be
        able to make a faster version if the data does not need to be copied.
        """


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
    def compute_dy(t, dy):
        """Return `dy/dt` at time `t` using the memory in state `dy`."""


class IStateForSplitEvolvers(IState):
    r"""Interface required by Split Operator evolvers.

    These evolvers assume the problem can be split into two operators - $K$
    (kinetic energy) and $V$ (potential energy) so that $i dy/dt = (K+V)y$.
    The method requires that each of these operators be exponentiated.  The
    approach uses a Trotter decomposition that provides higher order accuracy,
    but requires evaluation of the potentials at an intermediate time.

    This interface requires that the `apply_exp_V()` method accept
    another state object which should be used for calculating any
    non-linear terms in $V$ which are state dependent.

    If your problem is linear (i.e. $V$ depends only on time, not on
    the state as in the case of the usual linear Schrodinger
    equation), then you should set the linear attribute which will
    improve performance (but do not use this for non-linear problems
    or the order of convergence will be reduced).
    """

    linear = Attribute("linear", "Is the problem linear?")

    def apply_exp_K(dt, t=None):
        r"""Apply $e^{-i K dt}$ in place"""

    def apply_exp_V(dt, state, t=None):
        r"""Apply $e^{-i V dt}$ in place using `state` for any
        nonlinear dependence in V. (Linear problems should ignore
        `state`.)"""


class IStatePotentialsForSplitEvolvers(IStateForSplitEvolvers):
    r"""Interface required by Split Operator evolvers.

    This is a specialization of `IStateForSplitEvolvers` that uses an
    alternative method `get_potentials()` to compute the non-linear
    portion of the potential. It is intended for use when the state is
    much more complicated than the non-linear portion of the
    potential, hence only a separate copy of the potentials is maintained.
    """
    def get_potentials(t):
        """Return `potentials` at time `t`."""

    def apply_exp_V(dt, potentials, t=None):
        r"""Apply $e^{-i V dt}$ in place using `potentials`"""


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
# Default Mixins
#
# These mixins implement many of the required operations using only the
# methods required by the Minimal interfaces

class StateMixin(object):
    linear = False          # By default assume problems are nonlinear

    # Note: we could get away with __imul__ but it requires one return self
    # which can be a little confusing, so we allow the user to simply define
    # `axpy` and `scale` instead.
    def __pos__(self):
        """`+self`"""
        return self

    def __neg__(self):
        """`-self`"""
        return -1*self

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
        self.scale(1./f)
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
        res.scale(f)
        return res

    __rmul__ = __mul__

    def __truediv__(self, f):
        """Return `self / y`"""
        res = self.copy()
        res.scale(1./f)
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

    def empty(self):
        return self.copy()

    @property
    def writable(self):
        """Common spelling of `writeable`.  Disable with useful error message."""
        raise AttributeError(
            "Cannot get attribute `writable`.  Did you mean `writeable`?")

    @writable.setter
    def writable(self, value):
        """Common spelling of `writeable`.  Disable with useful error message."""
        raise AttributeError(
            "Cannot set attribute `writable`.  Did you mean `writeable`?")


class StatesMixin(object):
    """Mixin for states with a collection (Sequence or Mapping) of data.

    The general interface is provided through the ``__iter__()`` and
    ``__getitem__()`` methods which are assumed to give complete access to the
    data through objects with behave like arrays (i.e. support arithmetic,
    assignment with ``x[...] = y``, and a ``flags.writeable`` attribute.)
    """
    implements([INumexpr])

    linear = False          # By default assume problems are nonlinear
    data = None

    def __len__(self):
        return len(list(self))

    def apply(self, expr, **kwargs):
        for _l in self:
            kw = {}
            for _k in kwargs:
                kw[_k] = kwargs[_k]
                if isinstance(kw[_k], self.__class__):
                    kw[_k] = kw[_k][_l]

            expr(out=self[_l], **kw)

    ######################################################################
    # Requires these methods
    #
    # These default implementations assume self.data is a Sequence of Mapping,
    # but can be overridden to support custom objects.
    def __iter__(self):
        """Return the list of quantum numbers.

        This version assumes `self.data` is either a Sequence or a Mapping.
        """
        if isinstance(self.data, collections.Sequence):
            return xrange(len(self.data)).__iter__()
        else:
            return self.data.__iter__()

    def __getitem__(self, key):
        """Return the data associated with `key`.

        This version assumes `self.data` is either a Sequence or a Mapping.
        """
        return self.data[key]

    def __setitem__(self, key, value):
        """Set the data associated with `key`.

        This version assumes `self.data` is either a Sequence or a Mapping.
        """
        self.data[key] = value

    ######################################################################
    # Default methods using the __iter__() and __getitem__()
    @property
    def dtype(self):
        # For now assume all arrays have same type
        if 'dtype' in self.__dict__:
            dtype = self.__dict__['dtype']
        else:
            dtype = self[self.__iter__().next()].dtype
        assert all(dtype == self[_k].dtype for _k in self)
        return dtype

    @property
    def writeable(self):
        """Set to `True` if the state is writeable, or `False` if the state
        should only be read.
        """
        # We we carefully use short-circuiting so that
        # self[key].flags.writeable is only evaluated if 'writeable'
        # is not found.
        return all(
            (self[key] if hasattr(self[key], 'writeable') else self[key].flags)
            .writeable
            for key in self)

    @writeable.setter
    def writeable(self, value):
        for key in self:
            data = self[key]
            if hasattr(data, 'writeable'):
                data.writeable = value
            else:
                data.flags.writeable = value


class ArrayStateMixin(StateMixin):
    """Mixin providing support for states with a single data array.

    Assumes that the data is an array-like object called `self.data`.  This
    provides all the functionality required by IState.  All the user needs to
    provide are the methods for the required `IStateFor...Evolvers`.
    """
    implements([INumexpr])
    t = 0.0
    data = None

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
        if 'dtype' in self.__dict__:
            dtype = self.__dict__['dtype']
        else:
            dtype = self.data.dtype
        return dtype

    def copy(self):
        """Return a copy of the state.

        Uses `copy.copy()` to shallow-copy attributes, and copy.deepcopy()` to
        copy the data.
        """
        y = copy.copy(self)
        y.data = copy.deepcopy(self.data)
        y.writeable = True      # Copies should be writeable
        return y

    def copy_from(self, y):
        """Set this state to be a copy of the state `y`"""
        assert self.writeable
        self[...] = y[...]
        self.__dict__.update(y.__dict__, data=self.data)

    def empty(self):
        """Return an uninitialized copy of the state."""
        y = copy.copy(self)
        y.data = np.empty_like(self.data)
        y.writeable = True      # Copies should be writeable
        return y

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

    ######################################################################
    # Convenience methods
    def __getitem__(self, key):
        """Provides direct access to the array."""
        return self.data[key]

    def __setitem__(self, key, value):
        """Provides direct access to the array."""
        self.data[key] = value


class ArraysStateMixin(StatesMixin, ArrayStateMixin):
    """Mixin providing support for states with a list of data arrays.

    Requires `__iter__()` provide keys `key` so that `self.data[key]` is an
    array representing all the data. This provides all the functionality
    required by IState.  All the user needs to provide are the methods for the
    required `IStateFor...Evolvers`.
    """
    def copy(self):
        """Return a copy of the state.

        Uses `copy.copy()` to shallow-copy attributes, and copy.deepcopy()` to
        copy the data.
        """
        y = copy.copy(self)
        y.data = copy.deepcopy(self.data)
        for key in self:
            y[key][...] = self[key]
        return y

    def empty(self):
        """Return an uninitialized copy of the state."""
        y = copy.copy(self)
        y.data = copy.copy(self.data)
        for key in self:
            y[key] = np.empty_like(self[key])
        return y

    def copy_from(self, y):
        """Set this state to be a copy of the state `y`"""
        assert self.writeable
        for key in self:
            self[key][...] = y[key]
        self.__dict__.update(y.__dict__, data=self.data)

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


class MultiStateMixin(ArraysStateMixin):
    """Mixin providing support for states comprising multiple states.

    Requires `__iter__()` provide keys `key` so that `self.data[key]` is an
    IState representing all the data.
    """
    def apply(self, expr, **kwargs):
        for key in self:
            kw = {}
            for _k in kwargs:
                kw[_k] = kwargs[_k]
                if isinstance(kw[_k], self.__class__):
                    kw[_k] = kw[_k][key]

            self[key].apply(expr, **kw)

    def empty(self):
        """Return an uninitialized copy of the state."""
        y = copy.copy(self)
        y.data = copy.copy(self.data)
        for key in self:
            y[key] = self[key].empty()
        return y
