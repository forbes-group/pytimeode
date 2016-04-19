import nose.tools as nt
import numpy as np
from zope.interface.exceptions import (
    BrokenImplementation, BrokenMethodImplementation)

from mmfutils.interface import implements, verifyClass, verifyObject


from ..interfaces import (IStateForABMEvolvers, ArrayStateMixin)


class State0(ArrayStateMixin):
    """Incomplete State to test interface checking"""
    implements([IStateForABMEvolvers])

    def __init__(self, N=4, dim=2):
        self.N = N
        self.dim = dim
        self.data = np.zeros((self.N,)*self.dim, dtype=complex)

    # Missing method compute_dy


class State1(State0):
    """Broken State to test interface checking"""
    implements([IStateForABMEvolvers])

    def compute_dy(self):
        # Missing argument dy
        self.data[...] = -self.data


class State(State1):
    """Correct State to test interface checking"""
    implements([IStateForABMEvolvers])

    def compute_dy(self, dy):
        dy.data[...] = -self.data


class TestInterfaces(object):
    @nt.raises(BrokenImplementation)
    def test_missing_method(self):
        verifyClass(IStateForABMEvolvers, State0)

    @nt.raises(BrokenMethodImplementation)
    def test_broken_method(self):
        verifyClass(IStateForABMEvolvers, State1)

    def test_class_interface(self):
        verifyClass(IStateForABMEvolvers, State)

    def test_object_interface(self):
        verifyObject(IStateForABMEvolvers, State())


class TestInterfacesDoctests(object):
    """Doctests to ensure reasonable error messages are given.

    >>> verifyClass(IStateForABMEvolvers, State0)
    Traceback (most recent call last):
       ...
    BrokenImplementation: An object has failed to implement interface \
             <InterfaceClass pytimeode.interfaces.IStateForABMEvolvers>
    <BLANKLINE>
            The compute_dy attribute was not provided.
    <BLANKLINE>

    >>> verifyClass(IStateForABMEvolvers, State1)
    Traceback (most recent call last):
       ... State1 ...
    BrokenMethodImplementation: The implementation of compute_dy violates\
     its contract because implementation doesn't allow enough arguments.

    >>> verifyClass(IStateForABMEvolvers, State)
    True

    >>> s = State()
    >>> verifyObject(IStateForABMEvolvers, s)
    True

    >>> s.writable
    Traceback (most recent call last):
       ...
    AttributeError: Cannot get attribute `writable`.  Did you mean `writeable`?

    >>> s.writable = False
    Traceback (most recent call last):
       ...
    AttributeError: Cannot set attribute `writable`.  Did you mean `writeable`?
    """
