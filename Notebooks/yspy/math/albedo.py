import numpy as np
from astropy import constants as c
from astropy import units as u
from ..util import astropy_util


def solar_const(r_hel=1. * u.au):
    ''' Calculates the solar constant at the given heliocentric distance.
    Parameters
    ----------
    r_hel: Quantity or float, optional
        The heliocentric distance to the target. If it is given in float, it is
        interpreted as AU.
    '''
    r_hel = astropy_util.change_to_quantity(r_hel, u.au)
    sol = c.L_sun / (4 * np.pi * r_hel**2)
    return sol.to(u.W / (u.m**2))


def phase_int_HG(Gpar=0.15, option='Myhrvold'):
    ''' Calculates the phase integral from linear approximation (q value).
    Parameters
    ----------
    Gpar: float, optional
        The slope parameter defined in the IAU standard H G magnitude system
        (Bowell et al 1989).
    option: str, optional
        Which option to use for the calculation.
    '''
    if option == 'Myhrvold':
        C1, C2 = 0.286, 0.656
    elif option == 'Bowell' or option == 'IAU':
        C1, C2 = 0.290, 0.684
    return C1 + C2 * Gpar


def albedo_Bond_HG(pV, Gpar=0.15, option='Myhrvold'):
    ''' Calculates the Bond albedo (q value) in IAU standard H G mag system.
    Parameters
    ----------
    pV: float
        The geometric albedo in V band.
    Gpar: float, optional
        The slope parameter defined in the IAU standard H G magnitude system
        (Bowell et al 1989).
    option: str, optional
        Which option to use for the calculation.
    '''
    q = phase_int_HG(Gpar, option=option)
    return q * pV


def T_ref(pV, Gpar=0.15, r_hel=1. * u.au, epsilon=0.9, option='Myhrvold'):
    ''' Calculates the reference temperature (zero TI subsolar temperature).
    Parameters
    ----------
    pV: float
        The geometric albedo in V band.
    r_hel: Quantity or float, optional
        The heliocentric distance to the target. If it is given in float, it is
        interpreted as AU.
    Gpar: float, optional
        The slope parameter defined in the IAU standard H G magnitude system
        (Bowell et al 1989).
    epsilon: float, optional
        The black body emissivity (0 ~ 1).
    option: str, optional
        Which option to use for the calculation.
    '''
    eps_err = 'epsilon must lie between 0 to 1.0.'
    r_hel = astropy_util.change_to_quantity(r_hel, u.au)

    if epsilon > 1 or epsilon < 0:
        raise ValueError(eps_err)

    alb_Bond = albedo_Bond_HG(pV=pV, Gpar=Gpar, option=option)
    sol = solar_const(r_hel=r_hel)
    T4 = (1 - alb_Bond) * sol / (epsilon * c.sigma_sb)
    return (T4**(1 / 4)).to(u.K)


def d2p(D_eff, Hmag_V):
    ''' Calculates the effective diameter from geometric albedo in V band.
    Inverse of ``p2d``.
    Note
    ----
    The constant 1329 km is used since Fowler & Chillemi (1992) in the book
    "IRAS Minor Planet Survey". It is derived as 2AU * 10^(V_sun/5), and using
    modern values, this is about 1300~1330 km. Following the tradition, I will
    use 1329 km.

    Parameters
    ----------
    D_eff: Quantity or float.
        The effective diameter of the object. If float, it is interpreted as
        ``u.km``.
    Hmag_V: Quantity or float.
        The absolute magnitude of the object in V band. If float, it is
        interpreted as ``u.mag``.
    '''
    C = 1329 * u.km
    D_eff = astropy_util.change_to_quantity(D_eff, u.km)
    Hmag_V = astropy_util.change_to_quantity(Hmag_V, u.dimensionless_unscaled)
    pV = (C / D_eff * np.power(10, - Hmag_V / 5))**2

    return pV.to(u.dimensionless_unscaled)


def p2d(pV, Hmag_V):
    ''' Calculates the geometric albedo in V band from the effective diameter.
    Inverse of ``d2p``.
    Note
    ----
    The constant 1329 km is used since Fowler & Chillemi (1992) in the book
    "IRAS Minor Planet Survey". It is derived as 2AU * 10^(V_sun/5), and using
    modern values, this is about 1300~1330 km. Following the tradition, I will
    use 1329 km.

    Parameters
    ----------
    pV: float
        The geometric albedo in V band.
    Hmag_V: Quantity or float.
        The absolute magnitude of the object in V band. If float, it is
        interpreted as ``u.mag``.
    '''
    C = 1329 * u.km
    Hmag_V = astropy_util.change_to_quantity(Hmag_V, u.mag)
    D_eff = C / np.sqrt(pV) * np.power(10, -Hmag_V / 5)

    return D_eff.to(u.km)
