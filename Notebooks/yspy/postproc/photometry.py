import warnings
import numpy as np
# from matplotlib import pyplot as plt

from photutils import aperture_photometry as apphot
# from photutils.detection import find_peaks

from astropy.table import hstack
from astropy.nddata import CCDData, Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy import units as u
# from astropy.convolution import convolve
# from photutils.centroids import centroid_com


from . import background
from ..query import astroquery_util
# from ..util.graphics import zimshow

# def make_pixelwise_error(imagedata, effective_gain, dark_map,
# ronoise_electron):
#     ''' Generate pixel-wise error map

#     Parameters
#     ----------
#     imagedata: array-like
#         The image in ADU
#     effective_gain: float or Quantity
#         The gain in e/ADU
#     dark_map: array-like
#         The dark current (bias subtracted) in ADU
#     ronoise_electron: float or Quantity
#         The readout noise in electrons, not ADU

#     Note
#     ----
#     Poisson noise from image (``sqrt(imagedata/effective_gain)``)
#     Poisson noise from dark (``sqrt(dark_map/effective_gain)``)
#     Gaussian noise from ronoise (``(ronoise_electron/effective_gain)**2``)
#     '''
#     e_img = np.sqrt(imagedata / effective_gain)
#     e_dark = np.sqrt(dark_map / effective_gain)
#     e_ronoise = np.ones_like(e_img) * (ronoise_electron / effective_gain)**2
#     err =

# TODO: photutils v0.4 had problem with centroid_com. So I made this.
# Delete it after it is updated properly.


def centroid_com(data, mask=None):
    """
    Calculate the centroid of an n-dimensional array as its "center of
    mass" determined from moments.
    Invalid values (e.g. NaNs or infs) in the ``data`` array are
    automatically masked.
    Parameters
    ----------
    data : array_like
        The input n-dimensional array.
    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
    Returns
    -------
    centroid : `~numpy.ndarray`
        The coordinates of the centroid in pixel order (e.g. ``(x, y)``
        or ``(x, y, z)``), not numpy axis order.
    """
    from astropy.utils.exceptions import AstropyUserWarning

    data = data.astype(np.float)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asarray(mask, dtype=bool)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data[mask] = 0.

    badidx = ~np.isfinite(data)
    if np.any(badidx):
        warnings.warn('Input data contains input values (e.g. NaNs or infs), '
                      'which were automatically masked.', AstropyUserWarning)
        data[badidx] = 0.

    total = np.sum(data)
    indices = np.ogrid[[slice(0, i) for i in data.shape]]

    # note the output array is reversed to give (x, y) order
    return np.array([np.sum(indices[axis] * data) / total
                     for axis in range(data.ndim)])[::-1]


def apphot_annulus(ccd, aperture, annulus, t_exposure, error=None, mask=None,
                   sky_keys={}, t_exposure_unit=u.s, **kwargs):
    ''' Do aperture photometry using annulus.
    Parameters
    ----------
    ccd: CCDData
        The data to be photometried. Preferably in ADU.
    aperture, annulus: photutils aperture and annulus object
        The aperture and annulus to be used for aperture photometry.
    error: array-like or Quantity, optional
        See ``photutils.aperture_photometry`` documentation.
        The pixel-wise error map to be propagated to magnitued error.
        One common example is the Poisson error propagated with readout noise,
        i.e., if ``e_Poisson = np.sqrt(gain * imagedata) / gain`` and
        ``e_RONoise = R/gain`` in ADU, then
        ``error = np.sqrt(e_Poisson**2 + e_RONoise**2)`` in ADU.
        See ``photutils.utils.calc_total_error`` documentation.
    sky_keys: dict
        kwargs of ``sky_fit``. Mostly one doesn't change the default setting,
        so I intentionally made it to be dict rather than usual kwargs, etc.
    **kwargs:
        kwargs for ``photutils.aperture_photometry``.

    Returns
    -------
    phot_f: astropy.table.Table
        The photometry result.
    '''
    _ccd = ccd.copy()

    if error is not None:
        err = error.copy()
        if isinstance(err, CCDData):
            err = err.data
    else:
        err = np.zeros_like(_ccd.data)

    if mask is not None:
        if _ccd.mask is not None:
            warnings.warn("ccd contains mask, so given mask will be added to it.")
            _ccd.mask += mask
        else:
            _ccd.mask = mask

    skys = background.sky_fit(_ccd, annulus, **sky_keys)
    n_ap = aperture.area()
    phot = apphot(_ccd.data, aperture, mask=_ccd.mask, error=err, **kwargs)
    # If we use ``_ccd``, photutils deal with the unit, and the lines below
    # will give a lot of headache for units. It's not easy since aperture
    # can be pixel units or angular units (Sky apertures).
    # ysBach 2018-07-26

    phot_f = hstack([phot, skys])

    phot_f["source_sum"] = phot_f["aperture_sum"] - n_ap * phot_f["msky"]
    phot_f["source_sum_err"] = (np.sqrt(phot_f["aperture_sum_err"]**2
                                        + (n_ap * phot_f['ssky'])**2 / phot_f['nsky']))

    phot_f["mag"] = -2.5 * np.log10(phot_f['source_sum'] / t_exposure)
    phot_f["merr"] = (2.5 / np.log(10)
                      * phot_f["source_sum_err"] / phot_f['source_sum'])

    return phot_f


def centroiding_iteration(ccd, position_xy, cbox_size=5., csigma=3.):
    ''' Find the intensity-weighted centroid of the image iteratively

    Returns
    -------
    xc_img, yc_img : float
        The centroided location in the original image coordinate in image XY.

    shift : float
        The total distance between the initial guess and the fitted centroid,
        i.e., the distance between `(xc_img, yc_img)` and `position_xy`.
    '''

    imgX, imgY = position_xy
    cutccd = Cutout2D(ccd.data, position=position_xy, size=cbox_size)
    avg, med, std = sigma_clipped_stats(cutccd.data, sigma=3, iters=5)
    cthresh = med + csigma * std
    # using pixels only above med + 3*std for centroiding is recommended.
    # See Ma+2009, Optics Express, 17, 8525
    mask = (cutccd.data < cthresh)
    if ccd.mask is not None:
        mask += ccd.mask
    xc_cut, yc_cut = centroid_com(data=cutccd.data, mask=mask)
    # Find the centroid with pixels have values > 3sigma, by center of mass
    # method. The position is in the cutout image coordinate, e.g., (3, 3).

    xc_img, yc_img = cutccd.to_original_position((xc_cut, yc_cut))
    # convert the cutout image coordinate to original coordinate.
    # e.g., (3, 3) becomes something like (137, 189)

    dx = xc_img - imgX
    dy = yc_img - imgY
    shift = np.sqrt(dx**2 + dy**2)
    return xc_img, yc_img, shift


def find_centroid_com(ccd, position_xy, iters=5, cbox_size=5., csigma=3.,
                      tol_shift=1.e-4, max_shift=1, verbose=False, full=False):
    ''' Find the intensity-weighted centroid iteratively.
    Simply run `centroiding_iteration` function iteratively for `iters` times.
    Given the initial guess of centroid position in image xy coordinate, it
    finds the intensity-weighted centroid (center of mass) after rejecting
    pixels by sigma-clipping.

    Parameters
    ----------
    ccd : CCDData or ndarray
        The whole image which the `position_xy` is calculated.

    position_xy : array-like
        The position of the initial guess in image XY coordinate.

    cbox_size : float or int, optional
        The size of the box to find the centroid. Recommended to use 2.5 to
        4.0 `FWHM`. See: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?centerpars
        Minimally about 5 pixel is recommended. If extended source (e.g.,
        comet), recommend larger cbox.

    csigma : float or int, optional
        The parameter to use in sigma-clipping. Using pixels only above 3-simga
        level for centroiding is recommended. See Ma+2009, Optics Express, 17,
        8525.

    tol_shift : float
        The tolerance for the shift. If the shift in centroid after iteration
        is smaller than this, iteration stops.

    max_shift: float
        The maximum acceptable shift. If shift is larger than this, raises
        warning.

    verbose : bool
        Whether to print how many iterations were needed for the centroiding.

    full : bool
        Whether to return the original and final cutout images.
    Returns
    -------
    com_xy : list
        The iteratively found centroid position.
    '''
    if not isinstance(ccd, CCDData):
        ccd = CCDData(ccd, unit='adu')  # Just a dummy

    i_iter = 0
    xc_iter = [position_xy[0]]
    yc_iter = [position_xy[1]]
    shift = []
    d = 0
    if verbose:
        print(f"Initial xy: ({xc_iter[0]}, {yc_iter[0]}) [0-index]")
        print(f"With max iteration {iters:d}, shift tolerance {tol_shift}")

    while (i_iter < iters) and (d < tol_shift):
        xy_old = [xc_iter[-1], yc_iter[-1]]

        x, y, d = centroiding_iteration(ccd=ccd,
                                        position_xy=xy_old,
                                        cbox_size=cbox_size,
                                        csigma=csigma)
        xc_iter.append(x)
        yc_iter.append(y)
        shift.append(d)
        i_iter += 1
        if verbose:
            print(f"Iteration {i_iter:d} / {iters:d}: "
                  + f"({x:.2f}, {y:.2f}), shifted {d:.2f}")

    newpos = [xc_iter[-1], yc_iter[-1]]
    dx = x - position_xy[0]
    dy = y - position_xy[1]
    total = np.sqrt(dx**2 + dy**2)

    if verbose:
        print(f"Final shift: dx={dx:+.2f}, dy={dy:+.2f}, total={total:.2f}")

    if total > max_shift:
        warnings.warn(f"Shift is larger than {max_shift} ({total:.2f}).")

    # if verbose:
    #     print('Found centroid after {} iterations'.format(i_iter))
    #     print('Initially {}'.format(position_xy))
    #     print('Converged ({}, {})'.format(xc_iter[i_iter], yc_iter[i_iter]))
    #     shift = position_xy - np.array([xc_iter[i_iter], yc_iter[i_iter]])
    #     print('(Python/C-like indexing, not IRAF/FITS/Fortran)')
    #     print()
    #     print('Shifted to {}'.format(shift))
    #     print('\tShift tolerance was {}'.format(tol_shift))

    if full:
        original_cut = Cutout2D(data=ccd.data,
                                position=position_xy,
                                size=cbox_size)
        final_cut = Cutout2D(data=ccd.data,
                             position=newpos,
                             size=cbox_size)
        return newpos, original_cut, final_cut

    return newpos


def check_nostar_query(coord, minsep, catalog="USNO-B", keywords=None,
                       column_filters={}):
    ''' Checks whether there's any sidereal object near the target.
    Parameters
    ----------
    coord: SkyCoord
        The position of the object
    minsep: astropy.Quantity
        Minimum separation to tolerate.
    catalog: str or list of str
        Catalog(s) to use.

    '''
    viz = astroquery_util.QueryVizier(coordinates=coord,
                                      radius=minsep,
                                      catalog=catalog,
                                      keywords=keywords,
                                      column_filters=column_filters)
    res = viz.query()
    if len(res) > 0:
        warnings.warn(f"Objects from {len(res)} catalog(s) found within "
                      + f"{minsep}:\n\t{res}")
        return False

    return True


# def check_nostar_convolve(ccd, kernel, minsep_I0, position=None, size=None,
#                           tolerance_I0=0.005, sky=None,
#                           threshold=None, mask_size=5, box_size=5, border_width=5,
#                           ksigma=3., csigma=3., show_plot=False):

#     def _show_plot_macro(ax, sources, stars, cent_x, cent_y):
#         from matplotlib.patches import Circle
#         ax.plot(sources["x_peak"][~stars], sources["y_peak"][~stars], 'kx', ms=10)
#         ax.plot(sources["x_peak"][stars], sources["y_peak"][stars], 'rx', ms=10)
#         ax.plot(cent_x, cent_y, 'k+', ms=10)
#         ax.add_patch(Circle((cent_x, cent_y), minsep_I0, color='r', fill=False))

#     def minsep_calc(sources, tolerance_I0, sky):
#         Is = sources["peak"] - sky


#     if position is not None and size is not None:
#         ccd = Cutout2D(ccd, position, size, copy=True)
#         cent_x, cent_y = ccd.input_position_cutout
#     else:
#         cent_x, cent_y = ccd.shape
#         cent_x -= 0.5
#         cent_y -= 0.5

#     if sky is None or threshold is None:
#         avg, med, std = sigma_clipped_stats(ccd.data, sigma=csigma, iters=1)

#         if sky is None:
#             if (avg - med) / std > 0.3:
#                 sky = med
#             else:
#                 sky = (2.5 * med) - (1.5 * avg)

#         if threshold is None:
#             threshold = med + ksigma * std

#     mask = np.zeros_like(ccd.data).astype(bool)

#     xmin = np.around(cent_x - mask_size / 2).astype(int)
#     xmax = np.around(cent_x + mask_size / 2).astype(int)
#     ymin = np.around(cent_y - mask_size / 2).astype(int)
#     ymax = np.around(cent_y + mask_size / 2).astype(int)
#     mask[ymin:ymax, xmin:xmax] = True

#     convolved = convolve(ccd.data, kernel, fill_value=np.nan)
#     sources = find_peaks(convolved, threshold=threshold, mask=mask,
#                          box_size=box_size, border_width=border_width)

#     if show_plot:
#         fig, axs = plt.subplots(1, 2)
#         zimshow(axs[0], ccd.data)
#         zimshow(axs[1], convolved)
#         axs[0].set_title("Original data")
#         axs[1].set_title("Convolved")

#     if len(sources) == 0:
#         return True

#     sources["distance"] = np.sqrt((sources["x_peak"] - cent_x)**2
#                                   + (sources["y_peak"] - cent_y)**2)
#     stars = sources["distance"] < minsep_pix
#     nstars = np.count_nonzero(stars)

#     if show_plot:
#         _show_plot_macro(axs[0], sources, stars, cent_x, cent_y)
#         _show_plot_macro(axs[1], sources, stars, cent_x, cent_y)

#     if nstars > 0:
#         warnings.warn(f"Maybe {nstars} stars near the target:\n"
#                       + f"{sources[stars]}")
#         if show_plot:
#             plt.suptitle(f"REJECTED: {nstars} stars")
#         return False

#     return True
