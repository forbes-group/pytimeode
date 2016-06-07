import numpy as np

import pytest

import minimal_example

from ..evolvers import EvolverABM


class Test(object):
    def y(self, t):
        y = np.array([np.exp(1j*(t - 1)**2)])
        y.dtype = float
        return y

    def test_no_numexpr(self):
        y0 = minimal_example.State()
        e = EvolverABM(y=y0, dt=0.01)
        e.evolve(steps=100)
        # y = (e.y.data, self.y(t=e.t))
        assert np.allclose(e.y.data, self.y(t=e.t))

    def test_numexpr(self):
        y0 = minimal_example.StateNumexpr()
        e = EvolverABM(y=y0, dt=0.01)
        e.evolve(steps=100)
        # y = (e.y.data, self.y(t=e.t))
        assert np.allclose(e.y.data, self.y(t=e.t))

    def test_zeros(self):
        y0 = minimal_example.State()
        y0.t = 1.2
        y0.data[...] = 1.0
        y1 = y0.zeros()
        assert np.allclose(y1.data[...], 0.0)
        assert np.allclose(y1.t, 1.2)
        y1.data[...] = 2.0
        assert np.allclose(y0.data[...], 1.0)

    def test_evolver_t(self):
        y0 = minimal_example.State()
        y0.t = 0.0
        e = EvolverABM(y=y0, dt=0.01, t=1.2)
        assert np.allclose(e.y.t, 1.2)


class TestCoverage(object):
    """Some tests to help with coverage."""
    def test_no_normalize(self):
        y0 = minimal_example.State()
        with pytest.raises(ValueError):
            e = EvolverABM(y=y0, dt=0.01, normalize=True)
            assert np.allclose(e.y.t, y0.t)
