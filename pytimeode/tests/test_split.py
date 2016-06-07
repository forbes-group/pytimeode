import nose.tools as nt

import numpy as np
from scipy.linalg import expm

from ..evolvers import EvolverABM
from ..utils.testing import TestState
from ..interfaces import (implements, IStateForABMEvolvers,
                          IStateForSplitEvolvers, ArrayStateMixin)

import minimal_example


class State(ArrayStateMixin):
    implements(IStateForABMEvolvers, IStateForSplitEvolvers)

    def __init__(self, N=2):
        np.random.seed(1)
        self.K = (np.random.random((N, N))
                  + 1j*np.random.random((N, N)) - 0.5 - 0.5j)
        self.V = (np.random.random((N, N))
                  + 1j*np.random.random((N, N)) - 0.5 - 0.5j)
        self.y0 = (np.random.random(N)
                   + 1j*np.random.random(N) - 0.5 - 0.5j)
        self.data = self.y0.copy()

    def compute_dy(self, dy):
        """Return `dy/dt` at time `self.t`.

        If `dy` is provided, then use it for the result, otherwise return a new
        state.
        """
        dy[...] = (self.K + self.V).dot(self[...])/1j
        return dy

    def get_potentials(self):
        """Return `potentials` at time `self.t`."""

    def apply_exp_K(self, dt):
        r"""Apply $e^{-i K dt}$ in place"""
        self[...] = expm(self.K/1j*dt).dot(self[...])

    def apply_exp_V(self, dt, state):
        r"""Apply $e^{-i V dt}$ in place"""
        self[...] = expm(self.V/1j*dt).dot(self[...])

    def get_y_exact(self, t):
        H = self.K + self.V
        return expm(-1j*t*H).dot(self.y0)


class Test(object):
    def y(self, t):
        y = np.array([np.exp(1j*(t - 1)**2)])
        y.dtype = float
        return y

    def test_no_numexpr(self):
        y0 = minimal_example.State()
        e = EvolverABM(y=y0, dt=0.01)
        e.evolve(steps=100)
        # y = (e.y.data, self.y(t=e.y.t))
        nt.ok_(np.allclose(e.y.data, self.y(t=e.y.t)))

    def test_numexpr(self):
        y0 = minimal_example.StateNumexpr()
        e = EvolverABM(y=y0, dt=0.01)
        e.evolve(steps=100)
        # y = (e.y.data, self.y(t=e.y.t))
        nt.ok_(np.allclose(e.y.data, self.y(t=e.y.t)))

    def test_testing(self):
        y = State()
        t = TestState(y)
        assert all(t.check_split_operator())
