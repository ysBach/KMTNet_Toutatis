import numpy as np


def iauHG_full(alpha, Gpar, degree=True):
    if degree:
        alpha = np.deg2rad(alpha)

    def exp(x):
        return np.exp(x)

    sina = np.sin(alpha)
    tanp2 = np.tan(alpha / 2)  # tangent per 2 value
    fe = exp(-90.56 * tanp2**2)  # an exponetial value

    term2 = ((1.31 - 0.992 * Gpar) * sina * fe
             / (-0.158 + (-1.79 + sina) * sina))
    term3 = Gpar * exp(-1.862 * tanp2**1.218) * (1 - fe)
    term4 = (1 - Gpar) * exp(-3.332 * tanp2**0.631) * (1 - fe)
    return fe + term2 + term3 + term4


def iauHG(alpha, Gpar, degree=True):
    if degree:
        alpha = np.deg2rad(alpha)

    def exp(x):
        return np.exp(x)

    tanp2 = np.tan(alpha / 2)  # tangent per 2 value
    term1 = Gpar * exp(-1.87 * tanp2**1.22)
    term2 = (1 - Gpar) * exp(-3.33 * tanp2**0.63)

    return term1 + term2


def iauHa2H0(Halpha, alpha, Gpar, degree=True):
    ''' Converts H(alpha) = V(1, 1, alpha) to H(0) using IAU HG model.
    '''
    H = Halpha + 2.5 * np.log10(iauHG(alpha=alpha, Gpar=Gpar, degree=degree))
    return H