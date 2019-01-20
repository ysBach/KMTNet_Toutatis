import warnings
import numpy as np
from astropy import coordinates
from astropy.coordinates import SkyCoord


def normalize(num, lower=0, upper=360, b=False):
    """Normalize number to range [lower, upper) or [lower, upper].
    From phn: https://github.com/phn/angles
    Parameters
    ----------
    num : float
        The number to be normalized.
    lower : int
        Lower limit of range. Default is 0.
    upper : int
        Upper limit of range. Default is 360.
    b : bool
        Type of normalization. Default is False. See notes.
        When b=True, the range must be symmetric about 0.
        When b=False, the range must be symmetric about 0 or ``lower`` must
        be equal to 0.
    Returns
    -------
    n : float
        A number in the range [lower, upper) or [lower, upper].
    Raises
    ------
    ValueError
      If lower >= upper.
    Notes
    -----
    If the keyword `b == False`, then the normalization is done in the
    following way. Consider the numbers to be arranged in a circle,
    with the lower and upper ends sitting on top of each other. Moving
    past one limit, takes the number into the beginning of the other
    end. For example, if range is [0 - 360), then 361 becomes 1 and 360
    becomes 0. Negative numbers move from higher to lower numbers. So,
    -1 normalized to [0 - 360) becomes 359.
    When b=False range must be symmetric about 0 or lower=0.
    If the keyword `b == True`, then the given number is considered to
    "bounce" between the two limits. So, -91 normalized to [-90, 90],
    becomes -89, instead of 89. In this case the range is [lower,
    upper]. This code is based on the function `fmt_delta` of `TPM`.
    When b=True range must be symmetric about 0.
    Examples
    --------
    >>> normalize(-270,-180,180)
    90.0
    >>> import math
    >>> math.degrees(normalize(-2*math.pi,-math.pi,math.pi))
    0.0
    >>> normalize(-180, -180, 180)
    -180.0
    >>> normalize(180, -180, 180)
    -180.0
    >>> normalize(180, -180, 180, b=True)
    180.0
    >>> normalize(181,-180,180)
    -179.0
    >>> normalize(181, -180, 180, b=True)
    179.0
    >>> normalize(-180,0,360)
    180.0
    >>> normalize(36,0,24)
    12.0
    >>> normalize(368.5,-180,180)
    8.5
    >>> normalize(-100, -90, 90)
    80.0
    >>> normalize(-100, -90, 90, b=True)
    -80.0
    >>> normalize(100, -90, 90, b=True)
    80.0
    >>> normalize(181, -90, 90, b=True)
    -1.0
    >>> normalize(270, -90, 90, b=True)
    -90.0
    >>> normalize(271, -90, 90, b=True)
    -89.0
    """
    if lower >= upper:
        ValueError("lower must be lesser than upper")
    if not b:
        if not ((lower + upper == 0) or (lower == 0)):
            raise ValueError(
                'When b=False lower=0 or range must be symmetric about 0.')
    else:
        if not (lower + upper == 0):
            raise ValueError('When b=True range must be symmetric about 0.')

    from math import floor, ceil
    # abs(num + upper) and abs(num - lower) are needed, instead of
    # abs(num), since the lower and upper limits need not be 0. We need
    # to add half size of the range, so that the final result is lower +
    # <value> or upper - <value>, respectively.
    res = num
    if not b:
        res = num
        if num > upper or num == lower:
            num = lower + abs(num + upper) % (abs(lower) + abs(upper))
        if num < lower or num == upper:
            num = upper - abs(num - lower) % (abs(lower) + abs(upper))

        res = lower if num == upper else num
    else:
        total_length = abs(lower) + abs(upper)
        if num < -total_length:
            num += ceil(num / (-2 * total_length)) * 2 * total_length
        if num > total_length:
            num -= floor(num / (2 * total_length)) * 2 * total_length
        if num > upper:
            num = total_length - num
        if num < lower:
            num = -total_length - num

        res = num

    res *= 1.0  # Make all numbers float, to be consistent

    return res


# TODO: Remove these two functions in favor of SkyCoord?
def sphere2cart(r, theta, phi):
    ''' Transform spherical coordinate to Cartesian vector.
    '''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def dot_prod(theta1, phi1, theta2, phi2):
    vec1 = sphere2cart(1., theta1, phi1)
    vec2 = sphere2cart(1., theta2, phi2)
    return np.dot(vec1, vec2)


# TODO: Solving Spherical Triangle
class SphericalTriangle():
    ''' Solves the spherical triangle for given values
    '''
    def __init__(self, A=None, B=None, C=None, a=None, b=None, c=None):
        self.A = A
        self.B = B
        self.C = C
        self.a = a
        self.b = b
        self.c = c

    def __str__(self):
        print('Angles: ', self.A, self.B, self.C)
        print('Sides : ', self.a, self.b, self.c)

    def cos_rule(self):
        pass


def cart2sph(sc, frame='icrs'):
    ''' Transforms the cartesian ``SkyCoord`` to spherical
    '''
    if not isinstance(sc, coordinates.representation.CartesianRepresentation):
        raise TypeError('The input sc should be CartesianRepresentation'
                        + 'but now it is {:s}'.format(type(sc)))

    sc_cart = SkyCoord(*sc.xyz, representation='cartesian', frame=frame)
    sc_cart.representation = 'spherical'
    return sc_cart


def add_sc(sc1, sc2):
    ''' Subtract two ``SkyCoord`` objects, return the distance and angles.
    Note
    ----
    To subtract, use ``sc1`` and ``revert_sc(sc2)``:
    >>> test = add_sc(SkyCoord(0, 0.1), revert_sc(SkyCoord(0.1, 0.3))
    '''

    w_diff_frame = 'Two coordinates have different frames! Using the first one.'

    if not sc1.is_equivalent_frame(sc2.frame):
        warnings.warn(w_diff_frame)

    subtracted = sc1.cartesian + sc2.cartesian
    sphericalsc = cart2sph(subtracted, frame=sc1.frame)
    sphericalsc.representation = 'spherical'
    return sphericalsc


def revert_sc(sc):
    ''' Gives the reverse vector of a given ``SkyCoord`` object.
    '''
    sc_c = -1. * sc.cartesian
    sphericalsc = cart2sph(sc_c, frame=sc.frame)
    return sphericalsc


def crossdot_sc(sc1, sc2, sc3):
    ''' Calculates (sc1 cross sc2) dot sc3.
    '''
    uvec1 = sc1.represent_as('unitspherical')
    uvec2 = sc2.represent_as('unitspherical')
    uvec3 = sc3.represent_as('unitspherical')
    calculated = (uvec1.cross(uvec2)).dot(uvec3)
    return calculated
