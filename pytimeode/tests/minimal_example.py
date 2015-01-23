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

import numpy as np

from .. import interfaces
from ..utils import interface


class State(interfaces.StateMixin):
    interface.implements(interfaces.IStateForABMEvolvers)

    writeable = True
    potentials = None

    def __init__(self, t=1.0, data=[1.0, 0.0]):
        """Not part of the interface"""
        self.t = float(t)
        self.data = np.array(data, dtype=float).reshape((2,))

    def __getitem__(self, key=None):
        return self.data

    def copy(self):
        return State(t=self.t, data=self.data)

    def copy_from(self, y):
        assert self.writeable
        self.data = np.array(y.data)
        self.t = float(y.t)

    def axpy(self, x, a=1):
        assert self.writeable
        self.data += a*x.data

    def scale(self, f):
        assert self.writeable
        self.data.dtype = complex
        self.data *= f
        self.data.dtype = float

    def update(self, t=0, mu=None):
        assert self.writeable
        self.t = t

    def apply_V(self, v=None):
        assert self.writeable
        v_ext = -2.0*(self.t - 1.0)
        self.data *= v_ext

    def compute_dy_inplace(self, potentials=None):
        assert self.writeable
        self.apply_V()
        self *= -1j

    def __repr__(self):
        return "State(t=%g, data=%s)" % (self.t, repr(self.data))


interface.verifyClass(interfaces.IStateForABMEvolvers, State)
