import numpy as np

import minimal_example

from ..evolvers import EvolverABM


class Mixin(object):
    def pre_evolve_hook(self):
        self._hook_called = True

    def compute_dy(self, *v, **kw):
        if not self._hook_called:
            raise Exception
        return self._State.compute_dy(self, *v, **kw)


class State(Mixin, minimal_example.State):
    _State = minimal_example.State


class StateNumexpr(Mixin, minimal_example.StateNumexpr):
    _State = minimal_example.StateNumexpr


class Test(object):
    def y(self, t):
        y = np.array([np.exp(1j*(t - 1)**2)])
        y.dtype = float
        return y

    def test_issue_9(self):
        y0 = State()
        e = EvolverABM(y=y0, dt=0.01, t=y0.t)
        e.evolve(steps=100)
        print e.y._hook_called

        # y = (e.y.data, self.y(t=e.t))
        assert np.allclose(e.y.data, self.y(t=e.t))

    def test_issue_9_numexpr(self):
        y0 = StateNumexpr()
        e = EvolverABM(y=y0, dt=0.01, t=y0.t)
        e.evolve(steps=100)
        # y = (e.y.data, self.y(t=e.t))
        assert np.allclose(e.y.data, self.y(t=e.t))
