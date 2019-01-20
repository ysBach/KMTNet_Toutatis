import numpy as np
from astropy import constants as c
from scipy.integrate import trapz

HH = c.h.si.value
KB = c.k_B.si.value
CC = c.c.si.value
PI = np.pi


def B_nu(freq, temp):
    ''' The B_nu(nu, T) Planck black body function.
    Parameters
    ----------
    freq, temp: float, ~np.ndarray
        The frequency (nu) and temperatures to calculate B_nu function. It is
        better *not* to give astropy units.
    '''
    factor = 2 * HH * freq**3 / CC**2
    expon = np.exp(HH * freq / (KB * temp))
    return factor / (expon - 1)


def B_lambda(wave, temp):
    ''' The B_lambda(lambda, T) Planck black body function.
    Parameters
    ----------
    wave, temp: float, ~np.ndarray
        The wavelength (lambda) and temperatures to calculate B_nu function. It
        is better *not* to give astropy units.
    '''
    factor = 2 * HH * CC**2
    expon = np.exp(HH * CC / (wave * KB * temp))
    return factor / (expon - 1)


def SED_aT(x, Tbin, dTbin=None, aT_func=None, aT_value=None, mode='nu'):
    ''' Generate combined spectrum for the given nu/lambda and aT function.

    Parameters
    ----------
    x: float, ~np.ndarray
        The frequency/wavelength to calculate W_nu/W_lambda function.
    Tbin: float, ~np.ndarray
        The temperature bins to be used for the calculation in the unit of K.
    dTbin: float, ~np.ndarray
        The delta-temperature bins to be used for integration. Use ``np.ediff1d``
        such that ``len(dTbin) = len(Tbin) - 1``.
    aT_func: function object, optional
        The function which describes the area-temperature distribution.
    aT_value: float, ~np.ndarray, list of floats, optional
        If area-temperature function is not analytic, just give the a(T) value
        for corresponding ``temp`` array. ``len(aT_value)`` must be identical
        to ``len(temp)``.
    mode: str in {'lambda', 'nu'}, optional
        Whether to use B_lambda or B_nu. Default is ``'nu'``.

    Returns
    -------
    integrated: ~np.ndarray
        W_nu (nu) of length ``len(x)`` is returned.

    Example
    -------
    >>> # Test of Hung-Yi Li 2005 IEEE paper
    >>> def Li_aT(T):
    >>>     return np.exp( - (T-450)**2 / 25000)
    >>> F_BIN = np.linspace(1.e10, 2.e14, 51)
    >>> T_BIN = np.arange(100, 800, 1)
    >>> SED_Li = th.SED_aT(F_BIN, T_BIN, Li_aT, mode='nu')
    '''
    x_calc = np.atleast_1d(x)
    T_calc = np.atleast_1d(Tbin)
    if dTbin is None:
        dTbin = np.ediff1d(T_calc)
    dT_calc = np.atleast_1d(dTbin)
    xx, TT = np.meshgrid(x_calc, T_calc)

    if mode == 'nu':
        Bs = B_nu(xx, TT)
    elif mode == 'lambda':
        Bs = B_lambda(xx, TT)
    else:
        raise ValueError('mode must be either "nu" or "lambda".')

    if aT_func is not None:
        areas = aT_func(TT)
    elif aT_value is not None:
        aT_value = np.atleast_1d(aT_value)
        aT_value = np.atleast_1d(aT_value)
        na = aT_value.shape[0]
        nx = x_calc.shape[0]
        if na != T_calc.shape[0]:
            raise ValueError("length of aT_value and temp differ.")
        areas = np.tile(aT_value, nx).reshape(na, nx)
    else:
        raise ValueError("One and only one of aT_value or aT_func must be given.")

    integrated = trapz(Bs * areas, x=T_calc, dx=dT_calc, axis=0)
    summed = np.sum(Bs * areas, axis=0)

    return integrated, summed


# Deprecated since not using trapz...
def SED_aT_deprecated(x, aT_func=None, aT_value=None, mode='nu',
                      temp=np.arange(100, 801, 1)):
    ''' Generate combined spectrum for the given nu/lambda and aT function.

    Parameters
    ----------
    x: float, ~np.ndarray
        The frequency/wavelength to calculate W_nu/W_lambda function.
    aT_func: function object, optional
        The function which describes the area-temperature distribution.
    aT_value: float, ~np.ndarray, list of floats, optional
        If area-temperature function is not analytic, just give the a(T) value
        for corresponding ``temp`` array. ``len(aT_value)`` must be identical
        to ``len(temp)``.
    mode: str in {'lambda', 'nu'}, optional
        Whether to use B_lambda or B_nu. Default is ``'nu'``.
    temp: array, optional
        The temperature bins to be used for the calculation in the unit of K.

    Returns
    -------
    summed: ~np.ndarray
        W_nu (nu) of length ``len(x_calc)`` is returned.

    Example
    -------
    >>> # Test of Hung-Yi Li 2005 IEEE paper
    >>> def Li_aT(T):
    >>>     return np.exp( - (T-450)**2 / 25000)
    >>> F_BIN = np.linspace(1.e10, 2.e14, 51)
    >>> T_BIN = np.arange(100, 800, 1)
    >>> SED_Li = th.SED_aT(F_BIN, Li_aT, mode='nu', temp=T_BIN)
    '''
    x_calc = np.atleast_1d(x)
    temp_calc = np.atleast_1d(temp)
    xx, TT = np.meshgrid(x_calc, temp_calc)

    if mode == 'nu':
        Bs = B_nu(xx, TT)

    elif mode == 'lambda':
        Bs = B_lambda(xx, TT)

    else:
        raise ValueError('mode must be either "nu" or "lambda".')

    if aT_func is not None:
        areas = aT_func(TT)

    elif aT_value is not None:
        aT_value = np.atleast_1d(aT_value)
        na = aT_value.shape[0]
        nx = x_calc.shape[0]
        if na != temp_calc.shape[0]:
            raise ValueError("length of aT_value and temp differ.")
        areas = np.tile(aT_value, nx).reshape(na, nx)

    else:
        raise ValueError(
            "One and only one of aT_value or aT_func must be given.")

    summed = np.sum(Bs * areas, axis=0)

    return summed
