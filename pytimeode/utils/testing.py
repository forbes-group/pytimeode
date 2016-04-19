"""Some utilities for testing.
"""
from __future__ import division

import numpy as np

from mmfutils import interface
from mmfutils.math.differentiate import differentiate

from .. import interfaces
from ..evolvers import EvolverSplit, EvolverABM

_EPS = np.finfo(float).eps


class TestState(object):
    """Class with methods to perform some checks on various states.
    """

    def __init__(self, state):
        self.state = state

    def check_split_operator(self, normalize=False,
                             atol=np.sqrt(_EPS), rtol=np.sqrt(_EPS),
                             dt=None, dir=0, rdy=1e-2):
        """Checks consistency between `compute_dy` and the split
        operator methods `apply_exp_K` and `apply_exp_V` using
        numerical differentiation of the latter.

        Note: the state must be able to be converted to an array for
        this to work.

        Arguments
        ---------
        normalize : bool, None
           Passed through to the evolver.  If `None`, then state is
           normalized only if it provides the IStateWithNormalize
           interface.  Note: It is tricky to implement a state where
           both methods have a consistent normalization scheme,
           especially with quantum mechanical problems where the
           different evolution schemes might differ by an overall phase.
        dt : float
           Initial step dt. The numerical differentiation scheme uses
           a Richardson extrapolation dividing dt successively by a
           constant factor and extrapolating to the dt=0 limit.
        dir : float
           If `dir < 0`, then the function is only evaluated for
           positive `t + dt` where `dt > 0`. If zero, then both
           positive and negative `dt` are used (centered difference
           formula has better convergence properties).
        rdy : float
           If `dt` is not provided, then we start from `dt=1` and
           reduce it until the maximum change normalized by the maximum
           value in the state is less than `rdy`.
        """
        y0 = self.state.copy()
        assert interface.verifyObject(interfaces.IStateForABMEvolvers, y0)
        assert interface.verifyObject(interfaces.IStateForSplitEvolvers, y0)
        if not interfaces.IStateWithNormalize.providedBy(y0):
            if normalize:
                raise ValueError(
                    "Can only set normalize=True if state y implements " +
                    "IStateWithNormalize")
        else:
            assert interface.verifyObject(interfaces.IStateWithNormalize, y0)
            if normalize is None:
                normalize = True

        def f_split(dt):
            evolver = EvolverSplit(y=y0, dt=dt/6.0, normalize=normalize)
            evolver.evolve(6)
            return np.asarray(evolver.y)

        def f_abm(dt):
            evolver = EvolverABM(y=y0, dt=dt/6.0, normalize=normalize)
            evolver.evolve(6)
            return np.asarray(evolver.y)

        if dt is None:
            dt = 1.0
            # Choose a small enough `dt` so that the relative change in the
            # state is small.
            _y0 = np.asarray(y0)
            for f in [f_split, f_abm]:
                while _EPS < dt:
                    _y1 = f(dt)
                    dt *= 0.5
                    if abs(_y0 - _y1).max()/abs(_y0.max()) < rdy:
                        break
                if dt <= _EPS:
                    raise ValueError("Could not find a reasonable step size dt.")

        dy_split = differentiate(f_split, h0=dt)
        dy_abm = differentiate(f_abm, h0=dt)
        dy_exact = np.asarray(y0.compute_dy(dy=y0.empty()))

        errs = []
        for dy in [dy_split, dy_abm]:
            aerr = abs(dy - dy_exact)
            err = np.minimum(aerr / atol,
                             np.divide(aerr, abs(dy_exact)) / rtol)
            if np.any(err > 1.0):
                print("Maximum error = {}".format(err.max()))
            errs.append(err)

        aerr = abs(dy_abm - dy_split)
        err = np.minimum(aerr / atol,
                         np.divide(aerr, abs(dy_abm)) / rtol)
        if np.any(err > 1.0):
            print("Maximum error = {}".format(err.max()))
        errs.append(err)
        return [np.all(_err <= 1.0) for _err in errs]
