import numpy as np
import pytest

from mmfutils.interface import implements

from ..interfaces import (IStateForABMEvolvers,
                          # IStateForSplitEvolvers, IStateWithNormalize,
                          ArrayStateMixin, ArraysStateMixin)

from ..evolvers import EvolverABM


class State(ArrayStateMixin):
    """
    >>> State(N=2, dim=1)
    State(array([ 1.+0.j,  1.+0.j]))
    """
    implements([IStateForABMEvolvers])

    def __init__(self, N=4, dim=2):
        self.N = N
        self.dim = dim
        self.data = np.ones((self.N,)*self.dim, dtype=complex)

    def compute_dy(self, t=0.0, dy=None):
        if dy is None:
            dy = self.copy()
        dy[...] = -self[...]
        return dy


class States(ArraysStateMixin):
    """
    >>> States(N=2)
    States([array([ 1.+0.j,  1.+0.j]),
            array([ 1.+0.j,  1.+0.j,  1.+0.j, 1.+0.j])])
    """
    implements([IStateForABMEvolvers])

    def __init__(self, N=4):
        self.N = N
        self.data = [np.ones(self.N, dtype=complex),
                     np.ones(2*self.N, dtype=complex)]

    def compute_dy(self, t=0.0, dy=None):
        if dy is None:
            dy = self.copy()
        dy[0][...] = -self[0]
        dy[1][...] = self[1]
        return dy


class StatesDict(ArraysStateMixin):
    """
    >>> StatesDict(N=2)
    StatesDict({'a': array([ 1.+0.j,  1.+0.j]),
                'b': array([ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j])})

    """
    implements([IStateForABMEvolvers])

    def __init__(self, N=4):
        self.N = N
        self.data = dict(a=np.ones(self.N, dtype=complex),
                         b=np.ones(2*self.N, dtype=complex))

    def compute_dy(self, t=0.0, dy=None):
        if dy is None:
            dy = self.copy()
        dy['a'][...] = -self['a']
        dy['b'][...] = self['b']
        return dy


class TestArrayStateMixin(object):
    @classmethod
    def setup_class(cls):
        cls.State = State
        s = cls.State()
        shape = (s.N,)*s.dim
        cls.n = np.arange(s.N**s.dim).reshape(shape)

    def test_lock(self):
        s = self.State()
        with pytest.raises(ValueError):
            with s.lock:
                s[...] = self.n

    def test_writeable(self):
        s = self.State()
        s.writeable = False
        with pytest.raises(ValueError):
            s[...] = self.n

    def test_copy(self):
        s = self.State()
        s[...] = self.n
        s1 = s.copy()
        s *= 2
        assert np.allclose(s.data, s1.data*2)

        # Regression test for issue 10
        for writeable in [True, False]:
            s.writeable = writeable
            assert s.writeable == writeable
            s1 = s.copy()
            assert s.writeable == writeable

    def test_empty(self):
        s = self.State()
        s[...] = self.n
        s1 = s.empty()
        s1[...] = s[...]
        s *= 2
        assert np.allclose(s.data, s1.data*2)

        # Regression test for issue 10
        for writeable in [True, False]:
            s.writeable = writeable
            assert s.writeable == writeable
            s1 = s.empty()
            assert s.writeable == writeable

    def test_copy_from(self):
        s = self.State()
        s[...] = self.n
        s1 = self.State()
        s1.copy_from(s)
        s1 += s
        assert np.allclose(s.data*2, s1.data)

    def test_evolve(self):
        y0 = self.State()
        e = EvolverABM(y=y0, dt=0.01)
        e.evolve(10)
        y = e.y
        t = e.t
        assert np.allclose(y.data, y0.data*np.exp(-t))

    def test_array_interface(self):
        s = self.State()
        assert np.allclose(s.data, np.asarray(s))

    def test_array_ops(self):
        def check(f, s):
            assert np.allclose(f, s.data)

        s1 = self.State()
        s2 = self.State()

        f1 = f2 = 1.0

        check(f2, +s2)          # Unary +
        check(-f2, -s2)         # Unary -

        s2 *= 2.0
        f2 *= 2.0

        check(f2, s2)

        s2 /= 1.5
        f2 /= 1.5
        check(f2, s2)

        s2 += s1
        f2 += f1
        check(f2, s2)

        s2 -= s1
        f2 -= f1
        check(f2, s2)

        s3 = s2 + s1
        f3 = f2 + f1
        check(f3, s3)

        s3 = s2 - s1
        f3 = f2 - f1
        check(f3, s3)

        s3 = s2 * 1.5
        f3 = f2 * 1.5
        check(f3, s3)

        s3 = s2 / 1.5
        f3 = f2 / 1.5
        check(f3, s3)

    def test_setting_dtype(self):
        """Regression test for issue 5"""
        s = self.State()
        s.__dict__['dtype'] = float
        s.data = s.data.astype(float)
        assert s.dtype is float
        s.__dict__['dtype'] = complex
        s.data = s.data.astype(complex)
        assert s.dtype is complex


class TestArrayStatesMixin(object):
    @classmethod
    def setup_class(cls):
        cls.State = States
        s = cls.State()
        cls.ns = [np.arange(len(s[_d])) for _d in s]

    def test_lock0(self):
        s = self.State()
        with s.lock:
            with pytest.raises(ValueError):
                s[0][...] = self.ns[0]

    def test_lock1(self):
        s = self.State()
        with s.lock:
            with pytest.raises(ValueError):
                s[1][...] = self.ns[1]

    def test_writeable(self):
        s = self.State()
        s.writeable = False
        with pytest.raises(ValueError):
            s[0][...] = self.ns[0]

    def test_copy(self):
        s = self.State()
        for _n, _k in enumerate(s):
            s[_k][...] = self.ns[_n]
        s1 = s.copy()
        s *= 2
        for _k in s:
            assert np.allclose(s[_k], s1[_k]*2)

    def test_empty(self):
        s = self.State()
        for _n, _k in enumerate(s):
            s[_k][...] = self.ns[_n]
        s1 = s.empty()
        for _k in s:
            s1[_k][...] = s[_k]
        s *= 2
        for _k in s:
            assert np.allclose(s[_k], s1[_k]*2)

    def test_copy_from(self):
        s = self.State()
        for _n, _k in enumerate(s):
            s[_k][...] = self.ns[_n]
        s1 = self.State()
        s1.copy_from(s)
        s1 += s
        assert all([np.allclose(s[_k]*2, s1[_k]) for _k in s])

    def test_evolve(self):
        y0 = self.State()
        e = EvolverABM(y=y0, dt=0.01)
        e.evolve(10)
        y = e.y
        t = e.t
        assert np.allclose(y[0], y0[0]*np.exp(-t))
        assert np.allclose(y[1], y0[1]*np.exp(t))

    def test_array_interface(self):
        s = self.State()

        # This just makes a singleton array of the class s.
        a = np.asarray(s)
        assert a.ravel()[0] is s

    def test_array_ops(self):
        def check(f, s):
            for _k in s:
                assert np.allclose(f, s[_k])

        s1 = self.State()
        s2 = self.State()

        f1 = f2 = 1.0

        check(f2, +s2)          # Unary +
        check(-f2, -s2)         # Unary -

        s2 *= 2.0
        f2 *= 2.0

        check(f2, s2)

        s2 /= 1.5
        f2 /= 1.5
        check(f2, s2)

        s2 += s1
        f2 += f1
        check(f2, s2)

        s2 -= s1
        f2 -= f1
        check(f2, s2)

        s3 = s2 + s1
        f3 = f2 + f1
        check(f3, s3)

        s3 = s2 - s1
        f3 = f2 - f1
        check(f3, s3)

        s3 = s2 * 1.5
        f3 = f2 * 1.5
        check(f3, s3)

        s3 = s2 / 1.5
        f3 = f2 / 1.5
        check(f3, s3)

    def test_setting_dtype(self):
        """Regression test for issue 5"""
        s = self.State()
        s.__dict__['dtype'] = float
        for _k in s:
            s.data[_k] = s.data[_k].astype(float)
        assert s.dtype is float
        s.__dict__['dtype'] = complex
        for _k in s:
            s.data[_k] = s.data[_k].astype(complex)
        assert s.dtype is complex


class TestArrayStatesDictMixin(object):
    @classmethod
    def setup_class(cls):
        cls.State = StatesDict
        s = cls.State()
        cls.ns = [np.arange(len(s[_d])) for _d in s]

    def test_copy(self):
        s = self.State()
        for _n, _k in enumerate(s):
            s[_k][...] = self.ns[_n]
        s1 = s.copy()
        s *= 2
        for _k in s:
            assert np.allclose(s[_k], s1[_k]*2)

    def test_empty(self):
        s = self.State()
        for _n, _k in enumerate(s):
            s[_k][...] = self.ns[_n]
        s1 = s.empty()
        for _k in s:
            s1[_k][...] = s[_k]
        s *= 2
        for _k in s:
            assert np.allclose(s[_k], s1[_k]*2)

    def test_copy_from(self):
        s = self.State()
        for _n, _k in enumerate(s):
            s[_k][...] = self.ns[_n]
        s1 = self.State()
        s1.copy_from(s)
        s1 += s
        assert all([np.allclose(s[_k]*2, s1[_k]) for _k in s])

    def test_evolve(self):
        y0 = self.State()
        e = EvolverABM(y=y0, dt=0.01)
        e.evolve(10)
        y = e.y
        t = e.t
        assert np.allclose(y['a'], y0['a']*np.exp(-t))
        assert np.allclose(y['b'], y0['b']*np.exp(t))

    def test_array_ops(self):
        def check(f, s):
            for _k in s:
                assert np.allclose(f, s[_k])

        s1 = self.State()
        s2 = self.State()

        f1 = f2 = 1.0

        s2 *= 2.0
        f2 *= 2.0

        check(f2, s2)

        s2 /= 1.5
        f2 /= 1.5
        check(f2, s2)

        s2 += s1
        f2 += f1
        check(f2, s2)

        s2 -= s1
        f2 -= f1
        check(f2, s2)

        s3 = s2 + s1
        f3 = f2 + f1
        check(f3, s3)

        s3 = s2 - s1
        f3 = f2 - f1
        check(f3, s3)

        s3 = s2 * 1.5
        f3 = f2 * 1.5
        check(f3, s3)

        s3 = s2 / 1.5
        f3 = f2 / 1.5
        check(f3, s3)
