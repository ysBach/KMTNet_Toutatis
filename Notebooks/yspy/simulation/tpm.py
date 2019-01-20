import numpy as np
from astropy import constants as c
# from astropy.coordinates import UnitSphericalRepresentation as usph
# from astropy.coordinates import SphericalRepresentation as sph
from scipy.optimize import newton


__all__ = ["PI", "SB", "D_EFF0", "SOL_1AU"]


PI = np.pi

# The Stefan-Boltzmann constant in SI
SB = c.sigma_sb.value

AU2M = 149597870700  # unit = m

# The phase integral coefficients used for Bond albedo calculation


# The effective diameter value
D_EFF0 = 1329 * 1000  # unit = m
# which is 2AU * 10^(m_V_sun / 5).
# Classically m_V_sun = -26.762 (Campins+1985AJ, 90, 896; 26.762 +- 0.017)
# and 1AU ~ 1.496e11 m, D_EFF0 = 1329.11 km.
# Here D_EFF0 is especially sensitive to m_V_sun (see link below).
# m_V_sun =  26.70    26.74    26.762   26.77  [mag]
# D_EFF0  = 1367.61  1342.65  1329.11  1324.22 [km]
# rel_diff=    2.90     1.02     0.00    -0.37 [%]
# https://sites.google.com/site/mamajeksstarnotes/basic-astronomical-data-for-the-sun

# The solar constant at 1 AU
SOL_1AU = 1361  # unit = W/m^2.
# Basically it's a varying constant (amplitude <~ 0.2%).
# See http://lasp.colorado.edu/home/sorce/data/tsi-data/


def A_B_HG(Gpar, pV):
    A_B_C1 = 0.286
    A_B_C2 = 0.656
    return (A_B_C1 + A_B_C2 * Gpar) * pV


def H2D(Hmag, pV):
    return D_EFF0 / np.sqrt(pV) * 10**(-Hmag / 5)


def STM_T0(A_Bond, rh_AU, epsilon=0.90, eta=1.0):
    term = (1 - A_Bond) * (SOL_1AU / rh_AU**2) / (epsilon * SB * eta)
    return term**(1 / 4)


def P2omega(Prot_hour):
    ''' Converts the rotational period in hour to omega in rad/s'''
    return 2 * PI / (Prot_hour * 3600)


def Thetapar(TI, omega_rot, T0, epsilon=0.90):
    return TI * np.sqrt(omega_rot) / (epsilon * SB * T0**3)


def gammapar(dt, dz):
    return dt / dz**2


def newton_iter(newu0_first, newu1, mu_sun, Theta, dz):
    ''' Finds the root for the energy balance equation on the surface.
    Parameters
    ----------
    newu0_first: float
        The first trial to the ``newu[0]`` (surface temperature) value, i.e.,
        the ansatz of ``newu[0]`` value.

    newu1: float
        The second top layer's temperaute, which will have been calculated
        before this function will be called.

    mu_sun: float
        The mu-factor for the sun on the patch (``max(0, cos(i_sun))``).

    Theta: float
        The thermal parameter, Theta.

    dz: float
        The depth resolution.
    '''

    def _iterfunc(x, newu1, mu, Theta, dz):
        return x**4 - mu - Theta / dz * (newu1 - x)

    def _iterfuncprime(x, newu1, mu, Theta, dz):
        return 4 * x**3 + Theta / dz
    # mu is not used but included due to scipy's newton requires same number
    # of positional args for prime function.

    u0 = newton(_iterfunc,
                newu0_first,
                _iterfuncprime,
                args=(newu1, mu_sun, Theta, dz),
                maxiter=50,   # default values
                tol=1.48e-8)  # default values
    return u0


# class SmallBody:
#     def __init__(self, targetname, discreteepochs=None):
#         self.targetname = targetname
#         self.discreteepochs = discreteepochs
#         self.observatory_code = None # used only if queried
#         # self.query_table = None      # used only if queried
#         self.queried = None          # used only if queried
#         self.vecs = None
#         self.spin = None
#         self.obs_pm = None

#     def __str__(self):
#         _str = "Small body {:s}"
#         print(_str.format(self.targetname))


#     def calc_vectors(self, sc_T=None, sc_O=None, returns=True):
#         ''' Calculates the Sun, Target, Observer vectors.
#         If the query has already been done, it uses the queried table to get
#         the vectors. If not, the ``SkyCoord`` objects for the locations of
#         the target and the observer (``sc_T`` and ``sc_O``) must be given.
#         #TODO: One may want to give ``sc_T`` and ``sc_S``, for example...

#         Parameters
#         ----------
#         sc_T, sc_O: SkyCoord, optional
#             The locations of the target and observer (usually with respect to
#             the Sun, i.e., ``frame='hcrs'``).
#         '''
#         err = 'If the query has not been done, you must give sc_T and sc_O.'

#         if self.queried is not None:
#             self.vecs = self.queried.calc_vectors(returns=True)
#         else:
#             if (sc_T is None) or (sc_O is None):
#                 raise ValueError(err)



