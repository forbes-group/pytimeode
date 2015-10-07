"""Test memory usage"""
import nose.tools as nt
import numpy as np

from zope.interface import classImplementsOnly, classImplements

from mmfutils.interface import implements

from ..interfaces import (IStateForABMEvolvers,
                          IStateForSplitEvolvers,
                          IStatePotentialsForSplitEvolvers,
                          ArrayStateMixin)
from ..evolvers import EvolverABM, EvolverSplit


class State(ArrayStateMixin):
    """This class keeps track of all copies."""
    implements([IStateForABMEvolvers, IStateForSplitEvolvers])
    linear = True
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

    def apply_exp_K(self, dt, t=None):
        r"""Apply $e^{-i K dt}$ in place"""
        pass

    def apply_exp_V(self, dt, t=None, state=None):
        r"""Apply $e^{-i V dt}$ in place"""
        if self.linear:
            # Linear problems should never be called with state
            nt.ok_(state is None)

        V = -1
        self *= np.exp(-1j*V*dt)

    def copy(self):
        self._copy()
        return ArrayStateMixin.copy(self)

    def __del__(self):
        self._del()


class StatePotentials(State):
    implements(IStatePotentialsForSplitEvolvers)

    linear = False
    copies = 0
    max_copies = 0

    def get_potentials(self, t):
        return -1

    def apply_exp_V(self, dt, t=None, potentials=None):
        V = potentials
        self *= np.exp(-1j*V*dt)


class StateNoNumexpr(State):
    copies = 0
    max_copies = 0


classImplementsOnly(StateNoNumexpr, [IStateForABMEvolvers,
                                     IStateForSplitEvolvers])
classImplements(StatePotentials, [IStateForABMEvolvers,
                                  IStateForSplitEvolvers,
                                  IStatePotentialsForSplitEvolvers])


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

    def test_split_nonlinear(self):
        """The Split evolver should require only 1 new states"""
        nt.assert_equal(StateNoNumexpr.max_copies, 1)
        self.state.linear = False
        e = EvolverSplit(y=self.state, dt=0.01, copy=False)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 2)
        e.evolve(10)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 2)
        e.evolve(10)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 2)

    def test_split_linear(self):
        """The Split evolver should not require any new states"""
        nt.assert_equal(StateNoNumexpr.max_copies, 1)
        self.state.linear = True
        e = EvolverSplit(y=self.state, dt=0.01, copy=False)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 1)
        e.evolve(10)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 1)
        e.evolve(10)
        nt.assert_less_equal(StateNoNumexpr.max_copies, 1)

    def test_split_potential(self):
        """The Split evolver should require no new states"""
        StatePotentials._reset()
        state = StatePotentials()
        state.linear = False
        nt.assert_equal(StatePotentials.max_copies, 1)
        e = EvolverSplit(y=state, dt=0.01, copy=False)
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
