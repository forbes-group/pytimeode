r"""Utilities"""
from __future__ import division

import numpy as np

_EPS = np.finfo(float).eps

__all__ = ['Object', 'numexpr']


numexpr = False
try:
    import numexpr

    # These convolutions are needed to deal with a common failure mode: If the
    # MKL libraries cannot be found, then the whole python process crashes with
    # a library error.  We test this in a separate process and if it fails, we
    # disable the MKL.
    import multiprocessing

    def check(q):
        import numexpr
        q.put(numexpr.get_vml_version())

    q = multiprocessing.Queue()
    _p = multiprocessing.Process(target=check, args=[q])
    _p.start()
    _p.join()
    if q.empty():
        # Fail
        numexpr.use_vml = False

except ImportError:
    pass


######################################################################
# General utilities
class Object(object):
    r"""General base class with a few convenience methods.

    Constructors: The `__init__` method should simply be used to set variables,
    all initialization that computes attributes etc. should be done in `init()`
    which will be called at the end of `__init__`.

    This aids pickling which will save only those variables defined when the
    base `__init__` is finished, and will call `init()` upon unpickling,
    thereby allowing unpicklable objects to be used (in particular function
    instances).

    .. note:: Do not use a variable named `_empty_state`... this is reserved
       for objects without any state.
    """
    def __init__(self):
        self.picklable_attributes = [_k for _k in self.__dict__]
        self.init()

    def init(self):
        r"""Define any computed attributes here."""

    def __getstate__(self):
        state = dict((_k, self.__dict__[_k])
                     for _k in self.picklable_attributes)
        # From the docs:
        # "For new-style classes, if __getstate__() returns a false value,
        #  the __setstate__() method will not be called."
        # Don't return an empty state!
        if not state:
            state = dict(_empty_state=True)
        return state

    def __setstate__(self, state):
        if '_empty_state' in state:
            state.pop('_empty_state')
        self.__dict__.update(state)
        self.init()

        # init() may reset an evolver state, for example, so we once again set
        # the variables from the pickle.
        self.__dict__.update(state)
