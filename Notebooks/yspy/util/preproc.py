import warnings
from pathlib import Path

import numpy as np

from scipy.stats import itemfreq

from ccdproc import (combine, subtract_bias, subtract_dark, flat_correct,
                     trim_image, cosmicray_lacosmic)
from ccdproc import sigma_func as ccdproc_mad2sigma_func

from astropy import units as u
from astropy.wcs import WCS
from astropy.nddata import CCDData, StdDevUncertainty, Cutout2D
from astropy.modeling.functional_models import Gaussian1D
from astropy.modeling.fitting import LevMarLSQFitter

from . import fits_util as fu

__all__ = ["Gfit2hist", "bias2rdnoise", 'combine_ccd', 'bdf_process',
           "cutout2CCDData", 'crrej_LA']


# def check_exptime(table, colname_file, colname_nx, colname_ny, colname_exptime):
#    exptimes = table[colname_exptime].data.data
#
#    if len(np.unique(exptimes)) != 1:
#        print('There are more than one exposure times:')
#        print('\texptimes = ', end=' ')
#        print(np.unique(exptimes), end=' ')
#        print('seconds')
#        table[colname_file, colname_nx, colname_ny,
#              colname_exptime].pprint(max_width=150)
#
#    return exptimes

def Gfit2hist(data):
    ''' Gaussian fit to the frequency distribution of the nddata.
    '''
    freq = itemfreq(data.flatten())
    fitter = LevMarLSQFitter()
    mode = freq[freq[:, 1] == freq[:, 1].max(), 0][0]
    init = Gaussian1D(mean=mode)
    fitG = fitter(init, freq[:, 0], freq[:, 1])
    return fitG


def bias2rdnoise(data):
    ''' Infer readout noise from bias image.
    '''
    fitG = Gfit2hist(data)
    return fitG.stddev.value


def combine_ccd(fitslist, trim_fits_section=None, output=None, unit='adu',
                subtract_frame=None, combine_method='median', reject_method=None,
                normalize=False, exposure_key='EXPTIME',
                combine_uncertainty_function=ccdproc_mad2sigma_func,
                extension=0, type_key=None, type_val=None,
                dtype=np.float32, output_verify='fix', overwrite=False,
                **kwargs):
    ''' Combining images
    Slight variant from ccdproc.
    # TODO: accept the input like ``sigma_clip_func='median'``, etc.
    # TODO: normalize maybe useless..?
    Parameters
    ----------
    fitslist: list of str, path-like
        list of FITS files.

    combine: str
        The ``method`` for ``ccdproc.combine``, i.e., {'average', 'median', 'sum'}

    reject: str
        Made for simple use of ``ccdproc.combine``,
        {None, 'minmax', 'sigclip' == 'sigma_clip', 'extrema'}. Automatically turns
        on the option, e.g., ``clip_extrema = True`` or ``sigma_clip = True``.
        Leave it blank for no rejection.

    type_key, type_val: str, list of str
        The header keyword for the ccd type, and the value you want to match.
        For an open HDU named ``hdu``, e.g., only the files which satisfies
        ``hdu[extension].header[type_key] == type_val`` among all the ``fitslist``
        will be used.

    **kwarg:
        kwargs for the ``ccdproc.combine``. See its documentation.
        This includes (RHS are the default values)
        ```
        weights=None,
        scale=None,
        mem_limit=16000000000.0,
        clip_extrema=False,
        nlow=1,
        nhigh=1,
        minmax_clip=False,
        minmax_clip_min=None,
        minmax_clip_max=None,
        sigma_clip=False,
        sigma_clip_low_thresh=3,
        sigma_clip_high_thresh=3,
        sigma_clip_func=<numpy.ma.core._frommethod instance>,
        sigma_clip_dev_func=<numpy.ma.core._frommethod instance>,
        dtype=None,
        combine_uncertainty_function=None, **ccdkwargs
        ```

    Returns
    -------
    master: astropy.nddata.CCDData
        Resulting combined ccd.

    '''

    def _set_reject_method(reject_method):
        ''' Convenience function for ccdproc.combine reject switches
        '''
        clip_extrema, minmax_clip, sigma_clip = False, False, False

        if reject_method == 'extrema':
            clip_extrema = True
        elif reject_method == 'minmax':
            minmax_clip = True
        elif ((reject_method == 'sigma_clip') or (reject_method == 'sigclip')):
            sigma_clip = True
        else:
            if reject_method is not None:
                raise KeyError("reject must be one of "
                               "{None, 'minmax', 'sigclip' == 'sigma_clip', 'extrema'}")

        return clip_extrema, minmax_clip, sigma_clip

    def _print_info(combine_method, Nccd, reject_method, **kwargs):
        if reject_method is None:
            reject_method = 'no'

        info_str = ('"{:s}" combine {:d} images by "{:s}" rejection')

        print(info_str.format(combine_method, Nccd, reject_method))
        print(dict(**kwargs))
        return

    # def _ccdproc_combine(ccdlist, combine_method, min_value=0,
    #                     combine_uncertainty_function=ccdproc_mad2sigma_func,
    #                     **kwargs):
    #     ''' Combine after minimum value correction and then rejection/trimming.
    #     ccdlist:
    #         list of CCDData

    #     combine_method: str
    #         The ``method`` for ``ccdproc.combine``, i.e., {'average', 'median',
    #         'sum'}

    #     **kwargs:
    #         kwargs for the ``ccdproc.combine``. See its documentation.
    #     '''
    #     if not isinstance(ccdlist, list):
    #         ccdlist = [ccdlist]

    #     # copy for safety
    #     use_ccds = ccdlist.copy()

    #     # minimum value correction and trim
    #     for ccd in use_ccds:
    #         ccd.data[ccd.data < min_value] = min_value

    #     #combine
    #     ccd_combined = combine(img_list=use_ccds,
    #                         method=combine_method,
    #                         combine_uncertainty_function=combine_uncertainty_function,
    #                         **kwargs)

    #     return ccd_combined

    def _normalize_exptime(ccdlist, exposure_key):
        _ccdlist = ccdlist.copy()
        exptimes = []

        for i in range(len(_ccdlist)):
            exptime = _ccdlist[i].header[exposure_key]
            exptimes.append(exptime)
            _ccdlist[i] = _ccdlist[i].divide(exptime)

        if len(np.unique(exptimes)) != 1:
            print('There are more than one exposure times:')
            print('\texptimes =', end=' ')
            print(np.unique(exptimes), end=' ')
            print('seconds')
        print('Normalized images by exposure time ("{:s}").'.format(
            exposure_key))

        return _ccdlist

    fitslist = list(fitslist)

    if (output is not None) and (Path(output).exists()):
        if overwrite:
            print(f"{output} already exists:\n\t", end='')
            print("But will be overridden.")
        else:
            print(f"{output} already exists:\n\t", end='')
            return fu.load_if_exists(output, loader=CCDData.read, if_not=None)

    ccdlist = fu.stack_FITS(filelist=fitslist,
                            extension=extension,
                            unit=unit,
                            trim_fits_section=trim_fits_section,
                            type_key=type_key,
                            type_val=type_val)
    header = ccdlist[0].header

    _print_info(combine_method=combine_method,
                Nccd=len(ccdlist),
                reject_method=reject_method,
                dtype=dtype,
                **kwargs)

    # Normalize by exposure
    if normalize:
        ccdlist = _normalize_exptime(ccdlist, exposure_key)

    # Set rejection switches
    clip_extrema, minmax_clip, sigma_clip = _set_reject_method(reject_method)

    master = combine(img_list=ccdlist,
                     combine_method=combine_method,
                     clip_extrema=clip_extrema,
                     minmax_clip=minmax_clip,
                     sigma_clip=sigma_clip,
                     combine_uncertainty_function=combine_uncertainty_function,
                     **kwargs)

    str_history = '{:d} images with {:s} = {:s} are {:s} combined '
    ncombine = len(ccdlist)
    header["NCOMBINE"] = ncombine
    header.add_history(str_history.format(ncombine,
                                          str(type_key),
                                          str(type_val),
                                          str(combine_method)))

    if subtract_frame is not None:
        subtract = CCDData(subtract_frame.copy())
        master.data = master.subtract(subtract).data
        header.add_history("Subtracted a user-provided frame")

    master.header = header
    master = fu.CCDData_astype(master, dtype=dtype)

    if output is not None:
        master.write(output, output_verify=output_verify, overwrite=overwrite)

    return master


def bdf_process(ccd, output=None, mbiaspath=None, mdarkpath=None, mflatpath=None,
                fits_section=None, calc_err=False, unit='adu', gain=None,
                rdnoise=None, gain_key="GAIN", rdnoise_key="RDNOISE",
                gain_unit=u.electron / u.adu, rdnoise_unit=u.electron,
                dark_exposure=None, data_exposure=None, exposure_key="EXPTIME",
                exposure_unit=u.s, dark_scale=False,
                min_value=None, norm_value=None,
                verbose=True, output_verify='fix', overwrite=True,
                dtype="float32"):
    ''' Do bias, dark and flat process.
    Parameters
    ----------
    ccd: array-like
        The ccd to be processed.
    output: path-like
        Saving directory
    '''

    proc = CCDData(ccd)
    hdr_new = proc.header

    if mbiaspath is None:
        do_bias = False
        # mbias = CCDData(np.zeros_like(ccd), unit=unit)
    else:
        do_bias = True
        mbias = CCDData.read(mbiaspath, unit=unit)
        hdr_new.add_history(f"Bias subtracted using {mbiaspath}")

    if mdarkpath is None:
        do_dark = False
        mdark = None
    else:
        do_dark = True
        mdark = CCDData.read(mdarkpath, unit=unit)
        hdr_new.add_history(f"Dark subtracted using {mdarkpath}")
        if dark_scale:
            hdr_new.add_history(
                f"Dark scaling {dark_scale} using {exposure_key}")

    if mflatpath is None:
        do_flat = False
        # mflat = CCDData(np.ones_like(ccd), unit=unit)
    else:
        do_flat = True
        mflat = CCDData.read(mflatpath)
        hdr_new.add_history(f"Flat corrected using {mflatpath}")

    if fits_section is not None:
        proc = trim_image(proc, fits_section)
        mbias = trim_image(mbias, fits_section)
        mdark = trim_image(mdark, fits_section)
        mflat = trim_image(mflat, fits_section)
        hdr_new.add_history(f"Trim by FITS section {fits_section}")

    if do_bias:
        proc = subtract_bias(proc, mbias)

    if do_dark:
        proc = subtract_dark(proc,
                             mdark,
                             dark_exposure=dark_exposure,
                             data_exposure=data_exposure,
                             exposure_time=exposure_key,
                             exposure_unit=exposure_unit,
                             scale=dark_scale)
        # if calc_err and verbose:
        #     if mdark.uncertainty is not None:
        #         print("Dark has uncertainty frame: Propagate in arithmetics.")
        #     else:
        #         print("Dark does NOT have uncertainty frame")

    if calc_err:
        if gain is None:
            gain = fu.get_from_header(hdr_new, gain_key,
                                      unit=gain_unit,
                                      verbose=verbose,
                                      default=1.).value

        if rdnoise is None:
            rdnoise = fu.get_from_header(hdr_new, rdnoise_key,
                                         unit=rdnoise_unit,
                                         verbose=verbose,
                                         default=0.).value

        err = fu.make_errmap(proc,
                             gain_epadu=gain,
                             subtracted_dark=mdark)

        proc.uncertainty = StdDevUncertainty(err)
        errstr = (f"Error calculated using gain = {gain:.3f} [e/ADU] and "
                  + f"rdnoise = {rdnoise:.3f} [e].")
        hdr_new.add_history(errstr)

    if do_flat:
        if calc_err:
            if (mflat.uncertainty is not None) and verbose:
                print("Flat has uncertainty frame: Propagate in arithmetics.")
                hdr_new.add_history(
                    "Flat had uncertainty and is also propagated.")

        proc = flat_correct(proc,
                            mflat,
                            min_value=min_value,
                            norm_value=norm_value)

    proc = fu.CCDData_astype(proc, dtype=dtype)
    proc.header = hdr_new

    if output is not None:
        proc.write(output, output_verify=output_verify, overwrite=overwrite)

    return proc


def cutout2CCDData(ccd, position, size, mode='trim', fill_value=np.nan, full=True):
    ''' Converts the Cutout2D object to proper CCDData.
    Parameters
    ----------
    ccd: CCDData
        The ccd to be trimmed.
    position, size, mode, fill_value:
        See the ``ccdproc.trim_image`` doc.
    full: bool
        If ``True`` (default), returns the ``Cutout2D`` object.
        If ``False``, only the trimmed ccd is returned.
    '''
    w = WCS(ccd.header)
    cut = Cutout2D(data=ccd.data, position=position, size=size, wcs=w,
                   mode=mode, fill_value=fill_value, copy=True)
    # Copy True just to avoid any contamination to the original ccd.
    y1 = cut.ymin_original
    y2 = cut.ymax_original
    x1 = cut.xmin_original
    x2 = cut.xmax_original
    trimmed_ccd = trim_image(ccd[y1:y2, x1:x2])
    if full:
        return trimmed_ccd, cut
    return trimmed_ccd


def crrej_LA(ccd, dtype='float32', output=None, nomask=False,
             output_verify='fix', overwrite=False, **kwargs):
    ''' Does the cosmic-ray rejection using default L.A.Cosmic algorithm.

    Parameters
    ----------
    ccd: CCDData, ndarray
        The ccd to be cosmic-ray removed. If ndarray, changed to CCDData by
        ``ccd = CCDData(ccd)``.

    dtype: dtype-like
        The dtype of the output ccd (CCDData's data)

    output: path-like
        The path to save the rejected ccd.

    nomask: bool
        If ``False`` (default), the returned and saved ``CCDData`` will contain
        the mask extension (extension 1 with name MASK). If ``True``, the
        mask will be set as ``None`` after the cosmic-ray rejection. Can be
        turned on when the mask is unimportant and the disk storage is running
        out.

    kwargs:
        The kwargs for the cosmic-ray rejection. By default,
        ``sigclip=4.5``, ``sigfrac=0.3``, ``objlim=5.0``, ``gain=1.0``,
        ``readnoise=6.5``, ``satlevel=65535.0``, ``pssl=0.0``, ``niter=4``,
        ``sepmed=True``, ``cleantype='meanmask'``, ``fsmode='median'``,
        ``psfmodel='gauss'``, ``psffwhm=2.5``, ``psfsize=7``, ``psfk=None``,
        ``psfbeta=4.765``, ``verbose=False``
    '''

    if not isinstance(ccd, CCDData):
        warnings.warn("ccd is not CCDData. Convert using ccd = CCDData(ccd)")
        ccd = CCDData(ccd)

    nccd = cosmicray_lacosmic(ccd, **kwargs)
    nccd = fu.CCDData_astype(nccd, dtype=dtype)

    if nomask:
        nccd.mask = None

    if output is not None:
        nccd.write(output, output_verify=output_verify, overwrite=overwrite)

    return nccd


# DEPRECATED
# def response_correct(data, normdata1d, dispaxis=0, output='', threshold=0.,
#                      low_reject=3., high_reject=3.,
#                      iters=3, function='legendre', order=3):
#     ''' Response correction for a 2-D spectrum
#     Parameters
#     ----------
#     data : numpy.ndarray
#         The data to be corrected. Usually a (combined) flat field image.
#     normdata1d: numpy.ndarray
#         1-D numpy array which contains the suitable normalization image.
#     dispaxis : {0, 1}
#         The dispersion axis. 0 and 1 mean column and line, respectively.
#     threshold : float
#         The final 2-D response map pixels smaller than this value will be
#         replaced by 1.0.

#     Usage
#     -----
#     nsmooth = 7
#     normdata1d = np.sum(mflat[700:900, :] , axis=0)
#     normdata1d = convolve(normdata1d, Box1DKernel(nsmooth), boundary='extend')
#     response = preproc.response_correct(data = mflat.data,
#                                         normdata1d=normdata1d,
#                                         dispaxis=1,
#                                         order=10)

#     '''

#     nlambda = len(normdata1d)
#     nrepeat = data.shape[dispaxis - 1]

#     if data.shape[dispaxis] != nlambda:
#         wstr = "data shape ({:d}, {:d}) with dispaxis {:d} \
#         does not match with normdata1d ({:d})"
#         wstr = wstr.format(data.shape[0], data.shape[1],
#                            dispaxis, normdata1d.shape[0])
#         warnings.warn(wstr)

#     x = np.arange(0, nlambda)

#     if function == 'legendre':
#         fitted = legval(x, legfit(x, normdata1d, deg=order))
#         # TODO: The iteration here should be the iteration over the
#         # fitting, not the sigma clip itself.
#         residual = normdata1d - fitted
#         clip = sigma_clip(residual, iters=iters,
#                           sigma_lower=low_reject, sigma_upper=high_reject)
#     else:
#         warnings.warn("{:s} is not implemented yet".format(function))

#     mask = clip.mask
#     weight = (~mask).astype(float)  # masked pixel has weight = 0.
#     coeff = legfit(x, normdata1d, deg=order, w=weight)

#     if function == 'legendre':
#         response = legval(x, coeff)

#     response /= np.average(response)
#     response[response < threshold] = 1.
#     response_map = np.repeat(response, nrepeat)

#     if dispaxis == 0:
#         response_map = response_map.reshape(nlambda, nrepeat)
#     elif dispaxis == 1:
#         response_map = response_map.reshape(nrepeat, nlambda)

#     response2d = data / response_map

#     return response2d


# DEPRECATED
# def bdfgt_process(fname, mbiaspath=None, mflatpath=None, mdarkpath=None,
#                   extension=0,
#                   trim=[0, -1, 0, -1], mdark_seconds=1., gain=1.,
#                   exposure_key='EXPTIME', dtype=np.float32,
#                   output=''):
#     ''' Bias, Dark, Flat, Gain, and Trimming process.
#     Parameters
#     ----------
#     fname, mbiaspath, mdarkpath, mflatpath : Path
#         The path to the fits files.

#     trim : list, tuple, optional
#         The section for trimming as [y_lower, y_upper, x_lower, x_upper] order.

#     exposure_key : str, optional
#         The keyword for the exposure time in the header of the file ``fname``.
#         This is NOT used for the dark frame, since the dark frame is supposed
#         to be normalized to 1-second. If it is not, then tune the
#         ``mdark_seconds``.

#     mdark_seconds : int or float, optional
#         The exposure time for the master dark frame. Defaults to 1.

#     gain : float, optional
#         The gain value in classical e/ADU.

#     '''
#     hdu = fits.open(fname)
#     mbiasname = str(mbiaspath)
#     mdarkname = str(mdarkpath)
#     mflatname = str(mflatpath)
#     header = hdu[extension].header
#     data = hdu[extension].data.astype(dtype)
#     trim_naxis1 = trim[3] - trim[2]
#     trim_naxis2 = trim[1] - trim[0]

#     if mbiaspath is None:
#         mbias = np.zeros_like(data, dtype=dtype)
#     else:
#         mbias = fits.getdata(mbiaspath).astype(dtype)
#         header.add_history("Bias subtracted by {:s}".format(mbiasname))

#     if mdarkpath is None:
#         mdark = np.zeros_like(data, dtype=dtype)
#     else:
#         print('Dark frame exposure = {:.1f} sec'.format(mdark_seconds))
#         if mdark_seconds == 1.:
#             mdark = fits.getdata(mdarkpath).astype(dtype)
#         else:
#             mdark_seconds = fits.getheader(mdarkpath)[exposure_key]
#             mdark = fits.getdata(mdarkpath).astype(dtype) / mdark_seconds
#         header.add_history("Dark subtracted by {:s}".format(mdarkname))

#     if mflatpath is None:
#         mflat = np.ones_like(data, dtype=dtype)
#     else:
#         mflat = fits.getdata(mflatpath).astype(dtype)
#         header.add_history("Flat divided by {:s}".format(mflatname))

#     data -= (mbias + mdark * header[exposure_key])
#     data /= (mflat / np.average(mflat))
#     data *= gain
#     data.astype(dtype)
#     header['NAXIS1'], header['NAXIS2'] = trim_naxis1, trim_naxis2
#     hdu = fits.PrimaryHDU(data=data[trim[0]:trim[1], trim[2]:trim[3]],
#                           header=header)
#     hdu.writeto(output,
#                 output_verify='fix',
#                 overwrite=True)
#     return hdu


#
# def preproc(fnames, mbias, mflat, min_value=0, crrej=False):
#
#    if not isinstance(mbias, CCDData):
#        master_bias = CCDData(data=mbias, unit='adu')
#    else:
#        master_bias = mbias.copy()
#
#    if not isinstance(mflat, CCDData):
#        master_flat = CCDData(data=mflat, unit='adu')
#    else:
#        master_flat = mflat.copy()
#
#    processed_ccds = []
#
#    for fname in fnames:
#        print('Preprocessing started for {:s}'.format(fname))
#        obj_p = CCDData(fits.getdata(fname),
#                        meta=fits.getheader(fname),
#                        unit='adu')
#        gain = obj_p.header['gain']
#        xbin = obj_p.header['bin-fct1']
#        chip = obj_p.header['det-id']
#        # TODO: change ccd.data to just ccd (after ccdproc ver>1.3 release)
#        # TODO: gain value differs from ch to ch.
#
#        if crrej:
#            rdnoise = 4.0
#            import astroscrappy
#            # TODO: implement spec crrej
#            m, obj_p.data = astroscrappy.detect_cosmics(obj_p.data,
#                                                        satlevel=np.inf,
#                                                        sepmed=False,
#                                                        cleantype='medmask',
#                                                        fsmode='median',
#                                                        gain=gain,
#                                                        readnoise=rdnoise)
#
#
# TODO: Use ccdproc when ccdproc 1.3 is released
#        obj_p = subtract_bias(obj_p, master_bias)
#        obj_p = flat_correct(obj_p, master_flat, min_value=min_value)
# #        obj_p = gain_correct(obj_p, gain=gain, gain_unit=u.electron/u.adu)
#
#        obj_p = (obj_p.data - master_bias)
#        obj_p = suboverscan(data=obj_p, xbin=xbin, chip=chip, ybox=5)
#        obj_p = obj_p / master_flat * np.mean(master_flat)
#        obj_p = obj_p * gain
#        obj_p.astype(np.float32)
#        processed_ccds.append(obj_p)
#
#        print('\tDone')
#
#    return processed_ccds
