"""Test memory usage"""
import numpy as np
import pytest

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

    def compute_dy(self, dy):
        dy[...] = -self[...]
        return dy

    def apply_exp_K(self, dt):
        r"""Apply $e^{-i K dt}$ in place"""
        pass

    def apply_exp_V(self, dt, state=None):
        r"""Apply $e^{-i V dt}$ in place"""
        if self.linear:
            # Linear problems should never be called with state
            assert state is None

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

    def get_potentials(self):
        return -1

    def apply_exp_V(self, dt, potentials=None):
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
    @pytest.fixture
    def state(self):
        StateNoNumexpr._reset()
        return StateNoNumexpr()

    def test_abm(self, state):
        assert StateNoNumexpr.max_copies == 1
        e = EvolverABM(y=state, dt=0.01, copy=False, no_runge_kutta=True)
        assert StateNoNumexpr.max_copies <= 8
        e.evolve(10)
        assert StateNoNumexpr.max_copies <= 8
        e.evolve(10)
        assert StateNoNumexpr.max_copies <= 8

    def test_abm_runge_kutta(self, state):
        assert StateNoNumexpr.max_copies == 1
        e = EvolverABM(y=state, dt=0.01, copy=False)
        assert StateNoNumexpr.max_copies <= 2
        e.evolve(10)
        assert StateNoNumexpr.max_copies <= 10
        e.evolve(10)
        assert StateNoNumexpr.max_copies <= 10

    def test_split_nonlinear(self, state):
        """The Split evolver should require only 1 new states"""
        assert StateNoNumexpr.max_copies == 1
        state.linear = False
        e = EvolverSplit(y=state, dt=0.01, copy=False)
        assert StateNoNumexpr.max_copies <= 2
        e.evolve(10)
        assert StateNoNumexpr.max_copies <= 2
        e.evolve(10)
        assert StateNoNumexpr.max_copies <= 2

    def test_split_linear(self, state):
        """The Split evolver should not require any new states"""
        assert StateNoNumexpr.max_copies == 1
        state.linear = True
        e = EvolverSplit(y=state, dt=0.01, copy=False)
        assert StateNoNumexpr.max_copies <= 1
        e.evolve(10)
        assert StateNoNumexpr.max_copies <= 1
        e.evolve(10)
        assert StateNoNumexpr.max_copies <= 1

    def test_split_potential(self, state):
        """The Split evolver should require no new states"""
        StatePotentials._reset()
        state = StatePotentials()
        state.linear = False
        assert StatePotentials.max_copies == 1
        e = EvolverSplit(y=state, dt=0.01, copy=False)
        assert StateNoNumexpr.max_copies <= 1
        e.evolve(10)
        assert StateNoNumexpr.max_copies <= 1
        e.evolve(10)
        assert StateNoNumexpr.max_copies <= 1


class TestMemoryNumexpr(object):
    @pytest.fixture
    def state(self):
        State._reset()
        return State()

    def test_abm(self, state):
        assert State.max_copies == 1
        e = EvolverABM(y=state, dt=0.01, copy=False, no_runge_kutta=True)
        assert State.max_copies <= 9
        e.evolve(10)
        assert State.max_copies < 10
        e.evolve(10)
        assert State.max_copies < 10

    def test_abm_runge_kutta(self, state):
        assert State.max_copies == 1
        e = EvolverABM(y=state, dt=0.01, copy=False)
        assert State.max_copies <= 3
        e.evolve(10)
        assert State.max_copies <= 11
        e.evolve(10)
        assert State.max_copies <= 11

    def test_split(self, state):
        """The Split evolver should not require any new states"""
        assert State.max_copies == 1
        e = EvolverSplit(y=state, dt=0.01, copy=False)
        assert State.max_copies <= 1
        e.evolve(10)
        assert State.max_copies <= 1
        e.evolve(10)
        assert State.max_copies <= 1
