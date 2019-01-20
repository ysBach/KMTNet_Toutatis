import numpy as np

from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.models import (Gaussian1D, Gaussian2D, Const1D,
                                     Const2D, CONSTRAINTS_DOC)

from astropy.modeling.fitting import LevMarLSQFitter
from ..math import angle


__all__ = ['_GaussianConst1D', 'GaussianConst2D']


def simple_fit(model, x, y, z=None, fitter=LevMarLSQFitter(), init_params=None,
               **kwargs):
    f_init = model(init_params)
    f_fit = fitter(f_init, x, y, z, **kwargs)
    return f_fit


class _GaussianConst1D(Const1D + Gaussian1D):
    """A model for a 1D Gaussian plus a constant.
    From ``photutils.centroid`` v 0.4.
    """


class GaussianConst2D(Fittable2DModel):
    """ A model for a 2D Gaussian plus a constant.
    From ``photutils.centroid`` v 0.4.
    Parameters
    ----------
    constant : float
        Value of the constant.
    amplitude : float
        Amplitude of the Gaussian.
    x_mean : float
        Mean of the Gaussian in x.
    y_mean : float
        Mean of the Gaussian in y.
    x_stddev : float
        Standard deviation of the Gaussian in x.
        ``x_stddev`` and ``y_stddev`` must be specified unless a covariance
        matrix (``cov_matrix``) is input.
    y_stddev : float
        Standard deviation of the Gaussian in y.
        ``x_stddev`` and ``y_stddev`` must be specified unless a covariance
        matrix (``cov_matrix``) is input.
    theta : float, optional
        Rotation angle in radians. The rotation angle increases
        counterclockwise.
    """

    constant = Parameter(default=1)
    amplitude = Parameter(default=1)
    x_mean = Parameter(default=0)
    y_mean = Parameter(default=0)
    x_stddev = Parameter(default=1)
    y_stddev = Parameter(default=1)
    theta = Parameter(default=0)

    @staticmethod
    def evaluate(x, y, constant, amplitude, x_mean, y_mean, x_stddev,
                 y_stddev, theta):
        """Two dimensional Gaussian plus constant function."""

        model = Const2D(constant)(x, y) + Gaussian2D(amplitude, x_mean,
                                                     y_mean, x_stddev,
                                                     y_stddev, theta)(x, y)
        return model


GaussianConst2D.__doc__ += CONSTRAINTS_DOC


def Gaussian2D_correct(model, theta_lower=-np.pi / 2, theta_upper=np.pi / 2):
    ''' Sets x = semimajor axis and theta to be in [-pi/2, pi/2] range.
    Example
    -------
    >>> from astropy.modeling.functional_models import Gaussian2D
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from yspy.util import astropy_util as au
    >>> gridsize = np.zeros((40, 60))
    >>> common = dict(x_mean=20, y_mean=20, x_stddev=5)
    >>> y, x = np.mgrid[:gridsize.shape[0], :gridsize.shape[1]]
    >>> for sig_y in [-1, -0.1, 0.1, 1, 10]:
    >>>     for theta in [-12345.678, -100, -1, -0.1, 0, 0.1, 1, 100, 12345.678]:
    >>>         g = Gaussian2D(**common, theta=theta)
    >>>         f, ax = plt.subplots(3)
    >>>         ax[0].imshow(g(x, y), vmax=1, vmin=1.e-12)
    >>>         ax[1].imshow(au.Gaussian2D_correct(g)(x, y), vmax=1, vmin=1.e-12)
    >>>         ax[2].imshow(g(x, y) - au.Gaussian2D_correct(g)(x, y), vmin=1.e-20, vmax=1.e-12)
    >>>         np.testing.assert_almost_equal(g(x, y) - au.Gaussian2D_correct(g)(x, y), gridsize)
    >>>         plt.pause(0.1)
    You may see some patterns in the residual image, they are < 10**(-13).
    '''
    new_model = Gaussian2D(*model.parameters)
    sig_x = model.x_stddev.value
    sig_y = model.y_stddev.value

    if sig_x > sig_y:
        theta_norm = angle.normalize(
            model.theta.value, theta_lower, theta_upper)
        new_model.x_stddev.value = sig_x
        new_model.y_stddev.value = sig_y
        new_model.theta.value = theta_norm

    else:
        theta_norm = angle.normalize(
            model.theta.value + np.pi / 2, theta_lower, theta_upper)
        new_model.x_stddev.value = sig_y
        new_model.y_stddev.value = sig_x
        new_model.theta.value = theta_norm

    return new_model

