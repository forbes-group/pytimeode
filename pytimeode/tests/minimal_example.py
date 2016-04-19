"""Minimal Example for testing and demonstrating how to us the interfaces.

Here we solve the following zero-dimensional problem:

.. math::
   y = e^{i (t-1)^2}
   y' = 2i(t-1)y
   y'' = 2iy + 2i(t-1)y'
   y''' = 4iy' + 2i(t-1)y''
   y'''' = 6iy'' + 2i(t-1)y'''

This corresponds to evolution with potential $V = -2(t-1)$
"""
from __future__ import division

import copy

import numpy as np
from mmfutils import interface

from .. import interfaces


class State(interfaces.StateMixin):
    interface.implements(interfaces.IStateForABMEvolvers)

    writeable = True

    def __init__(self, t=1.0, data=[1.0, 0.0]):
        """Not part of the interface"""
        self.t = float(t)
        self.data = np.array(data, dtype=float).reshape((2,))
        self.dtype = float

    def copy(self):
        y = copy.copy(self)
        y.data = self.data.copy()
        y.writeable = True      # Copies should be writeable
        return y

    def copy_from(self, y):
        assert self.writeable
        self.data[...] = y.data
        self.t = float(y.t)

    def axpy(self, x, a=1):
        assert self.writeable
        self.data += a*x.data

    def scale(self, f):
        assert self.writeable
        self.data.dtype = complex
        self.data *= f
        self.data.dtype = float

    def apply_V(self):
        assert self.writeable
        v_ext = -2.0*(self.t - 1.0)
        self.data *= v_ext

    def compute_dy(self, dy):
        if dy is not self:
            dy.copy_from(self)

        dy.apply_V()
        dy *= -1j
        return dy

    def __repr__(self):
        return "State(t=%g, data=%s)" % (self.t, repr(self.data))


class StateNumexpr(State):
    """Provides apply so we can use numexpr"""
    interface.implements(interfaces.INumexpr)

    def apply(self, expr, **kw):
        # Need to pass arrays to expr, not the states.
        for k in kw:
            if isinstance(kw[k], State):
                kw[k] = kw[k].data
        expr(out=self.data, **kw)

interface.verifyClass(interfaces.IStateForABMEvolvers, State)
interface.verifyClass(interfaces.IStateForABMEvolvers, StateNumexpr)
interface.verifyClass(interfaces.INumexpr, StateNumexpr)

interface.verifyObject(interfaces.IStateForABMEvolvers, State())
interface.verifyObject(interfaces.IStateForABMEvolvers, StateNumexpr())
interface.verifyObject(interfaces.INumexpr, StateNumexpr())
