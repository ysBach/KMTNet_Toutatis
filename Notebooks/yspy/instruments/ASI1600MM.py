from pathlib import Path
from astropy.io import fits
from ..util import graphics, fits_util


useful_header_keys = ["NAXIS1", "NAXIS2", "XBINNING", "YBINNING",
                      "MAXDATA", "BUNIT", "DATE-OBS", "EXPTIME", "JD",
                      "IMAGETYP", "OBJECT", "EGAIN"]


def rdnoise(gain_epadu):
    ''' Gives the readout noise from ASI 1600MM manual.
    Rough estimate from the manual's plot...
    >>> gain = fits_util.dB2epadu(np.arange(0, 31, 5))
    >>> rdnoise = [3.6, 2.5, 1.9, 1.7, 1.4, 1.3, 1.25]
    >>> np.polyfit(gain, rdnoise, deg=1)
    # array([0.48090526, 1.17909034])
    # similar to 0.5 (intercept) and 1.2 (slope).
    '''
    return 1.2 + 0.5 * gain_epadu


def keys(frame, exptime=None):
    ''' Gives ``type_key`` and ``type_val`` for ``yspy.preproc.combine``.
    '''
    if frame.lower() == "bias":
        return dict(type_key=["IMAGETYP"],
                    type_val=["Bias Frame"])

    elif frame.lower() == "dark":
        return dict(type_key=["IMAGETYP", "EXPTIME"],
                    type_val=["Dark Frame", exptime])

    elif frame.lower() == "flat":
        if exptime is not None:
            return dict(type_key=["IMAGETYP", "EXPTIME"],
                        type_val=["Flat Field", exptime])
        else:
            return dict(type_key=["IMAGETYP"],
                        type_val=["Flat Field"])

    else:
        if exptime is not None:
            return dict(type_key=["OBJECT", "EXPTIME"],
                        type_val=[frame, exptime])
        else:
            return dict(type_key=["OBJECT"],
                        type_val=[frame])


def fit2fits(fitlist, toppath='.', convert_bit=False, original_bit=12, target_bit=16,
             remove_fit=False, save_thumbnail=False, overwrite_thumbnail=False):
    ''' Convert FIT to FITS with appropriate bit conversion.

    Parameters
    ----------
    fitlist: list of str or list of Path
        The list of files' locations in str or Path format.

    toppath: str or Path-like, optional
        The relative path where you want to work at with respect toppath with
        respect to the present working directory.

    original_bit, target_bit: int
        The original and target conversion bit numbers.

    remove_fit: bool, optional
        Whether to remove the original ``.fit`` file.

    save_thumbnail, overwrite_thumbnail: bool, optional
        Whether to save the thumbnail of converted FITS file and whether to
        overwrite thumbnail if it exists.

    Returns
    -------
    fitslist: list of Path
        The list of converted FITS files' Path.
    '''
    toppath = Path(toppath)
    for f in fitlist:
        print(f"Doing {f}...")
        # Set the Path for the converted FITS file
        fitspath = Path(toppath) / (f.stem + '.fits')

        # If the FITS already exists, just open it for thumbnail.
        # If not, convert and save with updating header.
        if fitspath.exists():
            hdul = fits.open(fitspath)

        else:
            if convert_bit:
                hdul = fits_util.convert_bit(f, original_bit=original_bit,
                                             target_bit=target_bit)
            else:
                hdul = fits.open(f)
                hdul[0].header["BUNIT"] = "ADU"
                hdul[0].header["MAXDATA"] = (65504,
                                             "maximum valid physical value in raw data")
                #  Hardcode for ASI 1600MM

            # If no 'OBJECT' is given, use 'IMAGETYPE' to guess it:
            try:
                obj = hdul[0].header['OBJECT']
                assert obj != ''
                assert obj is not None
            except (KeyError, AssertionError):
                hdul[0].header['OBJECT'] = (hdul[0].header["IMAGETYP"].split(' ')[0],
                                            "The name of Object Imaged")
            hdul.writeto(fitspath)

        # Save thumbnail if ordered
        if save_thumbnail:
            plotpath = Path(toppath) / (f.stem + '.png')

            if overwrite_thumbnail or (not plotpath.exists()):
                graphics.save_thumbnail(hdul[0].data, plotpath)

        hdul.close()

        # Remove the file if ordered
        if remove_fit:
            f.unlink()

    fitslist = []
    for f in fitlist:
        fitslist.append(Path(str(f) + 's'))

    return fitslist

# def master_bias(fitslist, **kwargs):
#     ''' Makes master bias frame.
#     Convenience function to make master bias using ASI1600MM(Pro).
#     the only thing added to ``preproc.combine_ccd`` here is the
#     ``type_key`` and ``type_val``.

#     Parameters
#     ----------
#     fitslist : list of Path-like
#         The list of paths to the fits files to be used for searching bias
#         frames.

#     Returns
#     -------
#     mbias: CCDData
#         The master bias frame.
#     '''
#     mbias = preproc.combine_ccd(fitslist,
#                                 type_key=["OBJECT"],
#                                 type_val=["Bias"],
#                                 **kwargs)
#     return mbias

# def master_dark(fitslist, exptime, master_bias = None, **kwargs):
#     ''' Makes master dark frame.

#     Convenience function to make master dark using ASI1600MM(Pro).
#     the only thing added to ``preproc.combine_ccd`` here is the
#     ``type_key`` and ``type_val``.

#     Parameters
#     ----------
#     fitslist: list of Path-like
#         The list of paths to the fits files to be used for searching dark
#         frames.

#     exptime: float
#         The exposure time of dark frames to be combined in seconds.

#     master_bias: CCDData, optional
#         If given, dark subtraction will be made. Sometimes we want to

#     Returns
#     -------
#     mdark: CCDData
#         The master dark frame.
#     '''
#     mdark = preproc.combine_ccd(fitslist,
#                                 type_key=["OBJECT", "EXPTIME"],
#                                 type_val=["Dark", exptime],
#                                 **kwargs)

#     return mdark

