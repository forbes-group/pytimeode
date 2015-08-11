"""Test memory usage"""
import nose.tools as nt
import numpy as np

from zope.interface import classImplementsOnly

from mmfutils.interface import implements

from ..interfaces import (IStateForABMEvolvers, IStateForSplitEvolvers,
                          ArrayStateMixin)
from ..evolvers import EvolverABM, EvolverSplit


class State(ArrayStateMixin):
    """This class keeps track of all copies."""
    implements([IStateForABMEvolvers, IStateForSplitEvolvers])

    copies = 0
    max_copies = 0

    @classmethod
    def _reset(cls):
        cls.copies = 0
        cls.max_copies = 0

    @classmethod
    def _copy(cls):
        cls.copies += 1
        cls.max_copies = max(cls.max_copies, cls.copies)

    @classmethod
    def _del(cls):
        cls.copies -= 1

    def __init__(self):
        self.data = np.zeros(2, dtype=complex)
        self._copy()

    def compute_dy(self, t=0, dy=None):
        if dy is None:
            dy = self.copy()
        dy[...] = -self[...]
        return dy

    def get_potentials(self, t):
        """Return `potentials` at time `t`."""
        return -1.0

    def apply_exp_K(self, dt, t=None):
        r"""Apply $e^{i K dt}$ in place"""
        pass

    def apply_exp_V(self, dt, t=None, potentials=None):
        r"""Apply $e^{i V dt}$ in place"""
        if potentials is None:
            potentials = self.get_potentials(t=t)
        self *= np.exp(1j*potentials*dt)

    def copy(self):
        self._copy()
        return ArrayStateMixin.copy(self)

    def __del__(self):
        self._del()


class StateNoNumexpr(State):
    copies = 0
    max_copies = 0

classImplementsOnly(StateNoNumexpr, [IStateForABMEvolvers,
                                     IStateForSplitEvolvers])


class TestMemory(object):
    def setUp(self):
        StateNoNumexpr._reset()
        self.state = StateNoNumexpr()

    def test_abm(self):
        nt.assert_equal(StateNoNumexpr.max_copies, 1)
        e = EvolverABM(y=self.state, dt=0.01, copy=False, no_runge_kutta=True)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 8)
        e.evolve(10)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 8)
        e.evolve(10)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 8)

    def test_abm_runge_kutta(self):
        nt.assert_equal(StateNoNumexpr.max_copies, 1)
        e = EvolverABM(y=self.state, dt=0.01, copy=False)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 2)
        e.evolve(10)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 10)
        e.evolve(10)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 10)

    def test_split(self):
        """The Split evolver should not require any new states"""
        nt.assert_equal(StateNoNumexpr.max_copies, 1)
        e = EvolverSplit(y=self.state, dt=0.01, copy=False)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 1)
        e.evolve(10)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 1)
        e.evolve(10)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 1)


class TestMemoryNumexpr(object):
    def setUp(self):
        State._reset()
        self.state = State()

    def test_abm(self):
        nt.assert_equal(State.max_copies, 1)
        e = EvolverABM(y=self.state, dt=0.01, copy=False, no_runge_kutta=True)
        nt.assert_less_equal(State.max_copies, 9)
        e.evolve(10)
        nt.assert_less(State.max_copies, 10)
        e.evolve(10)
        nt.assert_less(State.max_copies, 10)

    def test_abm_runge_kutta(self):
        nt.assert_equal(State.max_copies, 1)
        e = EvolverABM(y=self.state, dt=0.01, copy=False)
        nt.assert_less_equal(State.max_copies, 3)
        e.evolve(10)
        nt.assert_less_equal(State.max_copies, 11)
        e.evolve(10)
        nt.assert_less_equal(State.max_copies, 11)

    def test_split(self):
        """The Split evolver should not require any new states"""
        nt.assert_equal(State.max_copies, 1)
        e = EvolverSplit(y=self.state, dt=0.01, copy=False)
        nt.assert_less_equal(State.max_copies, 1)
        e.evolve(10)
        nt.assert_less_equal(State.max_copies, 1)
        e.evolve(10)
        nt.assert_less_equal(State.max_copies, 1)
