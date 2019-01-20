import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


def astrometry_extract(data, fwhm=3., ksigma=5., csigma=3., indexing=1, bintable=True):
    ''' Extracts the star positions (0-indexing), but not extended ones.
    Note
    ----
    This is just a convenience function for DAOStarFinder. First used for the
    astrometry client. This is why the xy positions are sorted by flux.

    Parameters
    ----------
    data: ndarray
        The array containing the pixel values
    fwhm: float
        The estimated FWHM of stellar objects in the image.
    ksigma, csigma: float
        The threshold for the detection will be calculated by median plus
        ``ksigma`` times standard deviation, where the median and standard
        deviation is calculated from the ``csigma``-sigma clipping on to the
        original image (``data``).
    indexing: int, float
        Whether to use 0 or 1 indexing. The user may use any floating number
        for their own indexing, although 0 or 1 is the most usual case.
    bintable: bool
        Whether to convert to FITS BINTABLE format. This is required for astrometry.net

    Example
    -------
    >>> xy = extracter(orig.data, fwhm=4, ksigma=5, csigma=3) + 1  # 1-indexing
    >>> np.savetxt(srcpath, xy, fmt='%d')
    >>> plt.plot(*xy.T, 'rx', ms=10)
    '''
    avg, med, std = sigma_clipped_stats(data, sigma=csigma, iters=1)
    finder = DAOStarFinder(fwhm=fwhm, threshold=med + ksigma * std,
                           exclude_border=True)
    sources = finder(data)
    sources.sort(["flux"])
    xy = np.vstack((sources["xcentroid"].round().astype(int).tolist(),
                    sources["ycentroid"].round().astype(int).tolist())).T
    xy += indexing

    if bintable:
        x = fits.Column(name='x', format='Iw', array=xy[:, 0])
        y = fits.Column(name='y', format='Iw', array=xy[:, 1])
        xy = fits.BinTableHDU.from_columns([x, y])
        return xy

    return xy