"""Wrappers for the fft.  Try to use fftw if possible.
"""
__all__ = ['fftn', 'ifftn', 'fftfreq']

import atexit
import functools
import itertools
import warnings

import numpy as np
import scipy as sp
from scipy.linalg import get_blas_funcs


def fft(Phi, axis=None):
    return np.fft.fft(Phi, axis=axis)


def ifft(Phit, axis=None):
    return np.fft.ifft(Phit, axis=axis)


def fftn(Phi, axes=None):
    return np.fft.fftn(Phi, axes=axes)


def ifftn(Phit, axes=None):
    return np.fft.ifftn(Phit, axes=axes)


fftfreq = np.fft.fftfreq

_THREADS = 8

try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fft, ifft, fftn, ifftn

    # By default, pyfftw does not cache the plans.  Here we enable the cache
    # and set the keepalive time to an hour.
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60*60)

    # Without this, nosetests hangs indefinitely.
    atexit.register(pyfftw.interfaces.cache.disable)

    # Also, the number of threads is set by default to 1.  Here we set the
    # default value to 8 and use FFT_MEASURE to actually check.
    fft, ifft, fftn, ifftn = [functools.partial(_f, threads=_THREADS,
                                                planner_effort='FFTW_MEASURE')
                              for _f in fft, ifft, fftn, ifftn]
except ImportError:
    try:
        import anfft
        def fft(Phi, axis=None):
            shape = Phi.shape
            if axis is None or axis == -1 or axis == len(shape)-1:
                return anfft.fft(Phi)
            elif 0 == axis:
                return np.rollaxis(
                    anfft.fft(np.rollaxis(Phi, 0, len(shape))),
                    -1, 0)
            else:
                raise NotImplementedError

        def ifft(Phit, axis=None):
            shape = Phit.shape
            if axis is None or axis == -1 or axis == len(shape)-1:
                return anfft.ifft(Phit)
            elif 0 == axis:
                return np.rollaxis(
                    anfft.ifft(np.rollaxis(Phit, 0, len(shape))),
                    -1, 0)
            else:
                raise NotImplementedError

        def fftn(Phi, axes=None):
            return anfft.fftn(Phi, axes=axes)

        def ifftn(Phit, axes=None):
            return anfft.ifftn(Phit, axes=axes)

    except ImportError:
        warnings.warn("Could not import anfft... performance not optimal.")


def resample(f, N):
    """Resample f to a new grid of size N.

    This uses the FFT to resample the function `f` on a new grid with `N`
    points.  Note: this assumes that the function `f` is periodic.  Resampling
    non-periodic functions to finer lattices may introduce aliasing artifacts.

    Arguments
    ---------
    f : array
       The function to be resampled.  May be n-dimensional
    N : int or array
       The number of lattice points in the new array.  If this is an integer,
       then all dimensions of the output array will have this length.

    Examples
    --------
    >>> def f(x, y):
    ...     "Function with only low frequencies"
    ...     return (np.sin(2*np.pi*x)-np.cos(4*np.pi*y))
    >>> L = 1.0
    >>> Nx, Ny = 16, 13   # Small grid
    >>> NX, NY = 31, 24   # Large grid
    >>> dx, dy = L/Nx, L/Ny
    >>> dX, dY = L/NX, L/NY
    >>> x = (np.arange(Nx)*dx - L/2)[:, None]
    >>> y = (np.arange(Ny)*dy - L/2)[None, :]
    >>> X = (np.arange(NX)*dX - L/2)[:, None]
    >>> Y = (np.arange(NY)*dY - L/2)[None, :]
    >>> f_XY = resample(f(x,y), (NX, NY))
    >>> np.allclose(f_XY, f(X,Y))                      # To larger grid
    True
    >>> np.allclose(resample(f_XY, (Nx, Ny)), f(x,y))  # Back down
    True
    """
    newshape = np.array(f.shape)
    newshape[:] = N
    fk = fftn(f)
    fk1 = np.zeros(newshape, dtype=complex)
    for _s in itertools.product(
            *((slice(0, (_N + 1) // 2), slice(-(_N - 1) // 2, None))
              for _N in np.minimum(f.shape, newshape))):
        fk1[_s] = fk[_s]

    return ifftn(fk1) * np.prod(newshape.astype(float)/f.shape)
