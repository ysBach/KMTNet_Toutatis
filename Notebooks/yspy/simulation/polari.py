import numpy as np
from astropy.modeling.functional_models import Fittable1DModel, Parameter
from scipy.optimize import minimize_scalar


def shestopalovP(x, h=0.1, k1=1.e-4, k2=1.e-4, k3=1.e-4, a0=20):
    ''' The polarimetric curve function from Shestopalov 2004 JQS&RT 88 351.
    Note
    ----
    If the angle ``x`` is in degree or radian, all others should match the unit
    (e.g., ``h`` is percent per degree or radian, ``k1`` is per degree or
    radian).
    '''
    term1 = (1 - np.exp(-k1 * (x - 0.))) / (1 - np.exp(-k1 * a0))
    term2 = (1 - np.exp(-k2 * (x - a0))) / k2
    term3 = (1 - np.exp(-k3 * (x - 180))) / (1 - np.exp(-k3 * (a0 - 180)))
    return h * term1 * term2 * term3


def shestopalovExt(h=0.1, k1=0.01, k2=0.01, k3=0.01, a0=20):
    ''' Calculates the extrema from Shestopalov function
    '''
    def _Pr(x):
        return shestopalovP(x, h=h, k1=k1, k2=k2, k3=k3, a0=a0)

    resmin = minimize_scalar(_Pr, bounds=(0, 180), method='bounded')
    amin = resmin.x
    Pmin = resmin.fun
    # Use -P curve to find maximum:
    resmax = minimize_scalar(-_Pr, bounds=(0, 180), method='bounded')
    amax = resmax.x
    Pmax = -resmax.fun
    return dict(amin=amin, Pmin=Pmin, amax=amax, Pmax=Pmax)


class Shestopalov1D(Fittable1DModel):
    """ 1-D Shestopalov polarimetric curve model
    angles are in degrees, not radians.
    """
    slope = Parameter(default=0.1)
    mpar = Parameter(default=0.1)
    npar = Parameter(default=0.1)
    lpar = Parameter(default=0.1)
    ainv = Parameter(default=20)

    @staticmethod
    def evaluate(x, slope, mpar, npar, lpar, ainv):
        return shestopalovP(x, slope, mpar, npar, lpar, ainv)

    @staticmethod
    def fit_deriv(x, slope, mpar, npar, lpar, ainv):

        pass

