import warnings

import numpy as np
from astropy.stats import sigma_clip
from astropy.table import Table

__all__ = ['sky_fit']


def sky_fit(ccd, annulus, method='mode', sky_nsigma=3,
            sky_iters=5, mode_option='sex'):
    """ Estimate the sky value from image and annulus.
    Parameters
    ----------
    ccd: CCDData
        The image data to extract sky at given annulus.
    annulus: annulus object
        The annulus which will be used to estimate sky values.
    # fill_value: float or nan
    #     The pixels which are masked by ``ccd.mask`` will be replaced with
    #     this value.
    method : {"mean", "median", "mode"}, optional
        The method to estimate sky value. You can give options to "mode"
        case; see mode_option.
        "mode" is analogous to Mode Estimator Background of photutils.
    sky_nsigma : float, optinal
        The input parameter for sky sigma clipping.
    sky_iters : float, optinal
        The input parameter for sky sigma clipping.
    mode_option : {"sex", "IRAF", "MMM"}, optional.
        sex  == (med_factor, mean_factor) = (2.5, 1.5)
        IRAF == (med_factor, mean_factor) = (3, 2)
        MMM  == (med_factor, mean_factor) = (3, 2)
        where ``msky = (med_factor * med) - (mean_factor * mean)``.

    Returns
    -------
    skytable: astropy.table.Table
        The table of the followings.
    msky : float
        The estimated sky value within the all_sky data, after sigma clipping.
    ssky : float
        The sample standard deviation of sky value within the all_sky data,
        after sigma clipping.
    nsky : int
        The number of pixels which were used for sky estimation after the
        sigma clipping.
    nrej : int
        The number of pixels which are rejected after sigma clipping.
    """

    skydicts = []
    skys = annul2values(ccd, annulus)

    for i, sky in enumerate(skys):
        skydict = {}

        if method == 'mean':
            skydict["msky"] = np.ma.mean(sky)
            skydict["ssky"] = np.ma.std(sky, ddof=1)
            skydict["nsky"] = sky.shape[0]
            skydict["nrej"] = 0

        elif method == 'median':
            skydict["msky"] = np.ma.median(sky)
            skydict["ssky"] = np.ma.std(sky, ddof=1)
            skydict["nsky"] = sky.shape[0]
            skydict["nrej"] = 0

        elif method == 'mode':
            sky_clip = sigma_clip(sky, sigma=sky_nsigma, iters=sky_iters)

            sky_clipped = sky[~sky_clip.mask]
            nsky = np.count_nonzero(~sky_clip.mask)
            mean = np.ma.mean(sky_clipped)
            med = np.ma.median(sky_clipped)
            std = np.ma.std(sky_clipped, ddof=1)
            nrej = np.count_nonzero(sky) - nsky

            skydict["nrej"] = nrej
            skydict["nsky"] = nsky
            skydict["ssky"] = std

            if nrej < 0:
                raise ValueError('nrej < 0: check the code')

            if nrej > nsky:  # rejected > survived
                warnings.warn('More than half of the pixels rejected.')

            if mode_option == 'IRAF':
                if (mean < med):
                    msky = mean
                else:
                    msky = 3 * med - 2 * mean
                skydict["msky"] = msky

            elif mode_option == 'MMM':
                msky = 3 * med - 2 * mean
                skydict["msky"] = msky

            elif mode_option == 'sex':
                if (mean - med) / std > 0.3:
                    msky = med
                else:
                    msky = (2.5 * med) - (1.5 * mean)
                skydict["msky"] = msky

            else:
                raise ValueError('mode_option not understood')

        skydicts.append(skydict)
    skytable = Table(skydicts)
    # skytable["msky"].unit = u.adu / u.pix
    # skytable["ssky"].unit = u.adu / u.pix
    # skytable["nrej"].unit = u.pix
    return skytable


def annul2values(ccd, annulus):
    ''' Extracts the pixel values from the image with annuli

    Parameters
    ----------
    ccd: CCDData
        The image which the annuli in ``annulus`` are to be applied.
    annulus: ~photutils aperture object
        The annuli to be used to extract the pixel values.
    # fill_value: float or nan
    #     The pixels which are masked by ``ccd.mask`` will be replaced with
    #     this value.
    Returns
    -------
    values: list
        The list of pixel values. Length is the same as the number of annuli in
        ``annulus``.
    '''
    values = []
    _ccd = ccd.copy()
    if _ccd.mask is None:
        _ccd.mask = np.zeros_like(_ccd.data).astype(bool)

    mask_an = annulus.to_mask(method='center')
    for i, an in enumerate(mask_an):
        an_idx = np.nonzero(an.data)
        skys_1d = an.multiply(_ccd.data)[an_idx]
        mask_1d = an.multiply(_ccd.mask)[an_idx]
        # skys_i = an.multiply(_ccd, fill_value=np.nan)
        # sky_xy = np.nonzero(an.data)
        # sky_all = mask_im[sky_xy]
        # sky_values = sky_all[~np.isnan(sky_all)]
        # values.append(sky_values)
        skys_1d = np.ma.array(skys_1d, dtype=_ccd.dtype, mask=mask_1d)
        values.append(skys_1d)

    return values


# def sky_fit(all_sky, method='mode', sky_nsigma=3, sky_iters=5,
#             mode_option='sex'):
#     '''
#     Estimate sky from given sky values.

#     TODO: will it be necessary to include med_factor=2.5, mean_factor=1.5?

#     Parameters
#     ----------
#     all_sky : ~numpy.ndarray
#         The sky values as numpy ndarray format. It MUST be 1-d for proper use.
#     method : {"mean", "median", "mode"}, optional
#         The method to estimate sky value. You can give options to "mode"
#         case; see mode_option.
#         "mode" is analogous to Mode Estimator Background of photutils.
#     sky_nsigma : float, optinal
#         The input parameter for sky sigma clipping.
#     sky_iters : float, optinal
#         The input parameter for sky sigma clipping.
#     mode_option : {"sex", "IRAF", "MMM"}, optional.
#         sex  == (med_factor, mean_factor) = (2.5, 1.5)
#         IRAF == (med_factor, mean_factor) = (3, 2)
#         MMM  == (med_factor, mean_factor) = (3, 2)
#         where ``msky = (med_factor * med) - (mean_factor * mean)``.
#     Returns
#     -------
#     skytable: astropy.table.Table
#         The table of the followings.
#     msky : float
#         The estimated sky value within the all_sky data, after sigma clipping.
#     ssky : float
#         The sample standard deviation of sky value within the all_sky data,
#         after sigma clipping.
#     nsky : int
#         The number of pixels which were used for sky estimation after the
#         sigma clipping.
#     nrej : int
#         The number of pixels which are rejected after sigma clipping.
#     -------

#     '''
#     skys = np.atleast_2d(all_sky)
#     skydicts = []

#     for sky in skys:
#         skydict = {}
#         if method == 'mean':
#             skydict["msky"] = np.mean(sky)
#             skydict["ssky"] = np.std(sky, ddof=1)
#             skydict["nsky"] = sky.shape[0]
#             skydict["nrej"] = 0

#         elif method == 'median':
#             skydict["msky"] = np.median(sky)
#             skydict["ssky"] = np.std(sky, ddof=1)
#             skydict["nsky"] = sky.shape[0]
#             skydict["nrej"] = 0

#         elif method == 'mode':
#             # median centered sigma clipping:
#             sky_clip = sigma_clip(sky, sigma=sky_nsigma, iters=sky_iters)

#             sky_clipped = sky[~sky_clip.mask]
#             nsky = np.count_nonzero(~sky_clip.mask)
#             mean = np.mean(sky_clipped)
#             med = np.median(sky_clipped)
#             std = np.std(sky_clipped, ddof=1)
#             nrej = sky.shape[0] - nsky

#             skydict["nrej"] = nrej
#             skydict["nsky"] = nsky
#             skydict["ssky"] = std

#             if nrej < 0:
#                 raise ValueError('nrej < 0: check the code')

#             if nrej > nsky:  # rejected > survived
#                 warnings.warn('More than half of the pixels rejected.')

#             if mode_option == 'IRAF':
#                 if (mean < med):
#                     msky = mean
#                 else:
#                     msky = 3 * med - 2 * mean
#                 skydict["msky"] = msky

#             elif mode_option == 'MMM':
#                 msky = 3 * med - 2 * mean
#                 skydict["msky"] = msky

#             elif mode_option == 'sex':
#                 if (mean - med) / std > 0.3:
#                     msky = med
#                 else:
#                     msky = (2.5 * med) - (1.5 * mean)
#                 skydict["msky"] = msky

#             else:
#                 raise ValueError('mode_option not understood')

#         skydicts.append(skydict)
#     skytable = Table(skydicts)
#     return skytable

