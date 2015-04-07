import numpy as np

import minimal_example

from ..evolvers import EvolverABM


class Test(object):
    def y(self, t):
        y = np.array([np.exp(1j*(t - 1)**2)])
        y.dtype = float
        return y

    def test1(self):
        y0 = minimal_example.State()
        e = EvolverABM(y=y0, dt=0.01, t=y0.t)
        e.evolve(steps=100)
        # y = (e.y.data, self.y(t=e.t))
        assert np.allclose(e.y.data, self.y(t=e.t))
