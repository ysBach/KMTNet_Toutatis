import numpy as np
from astropy.wcs import WCS


def xyinFOV(header, table, ra_key='ra', dec_key='dec', bezel=0):
    ''' Convert RA/DEC to pixel with rejection at bezels
    Parameters
    ----------
    header: astropy.io.fits.Header
        The header to extract WCS information.
    table: astropy.table.Table
        The queried result table.
    ra_key, dec_key: str
        The column names containing RA/DEC.
    bezel: int or float
        The bezel size to exclude stars at the image edges. If you want to
        keep some stars outside the edges, put negative values (e.g., ``-5``).
    '''
    w = WCS(header)
    _tab = table.copy()
    x, y = w.wcs_world2pix(_tab[ra_key], _tab[dec_key], 0)

    if bezel != 0:
        nx, ny = header['naxis1'], header['naxis2']
        mask = ((x < (0 + bezel))
                | (x > (nx - bezel))
                | (y < (0 + bezel))
                | (y > (ny - bezel)))
        x = x[~mask]
        y = y[~mask]
        _tab.remove_rows(mask)

    _tab["x"] = x
    _tab["y"] = y

    return _tab


def sdss2BV(g, r, gerr=None, rerr=None):
    '''
    Pan-STARRS DR1 (PS1) uses AB mag.
    https://www.sdss.org/dr12/algorithms/fluxcal/#SDSStoAB
    Jester et al. (2005) and Lupton (2005):
    https://www.sdss.org/dr12/algorithms/sdssubvritransform/
    Here I used Lupton. Application to PS1, it seems like Jester - Lupton VS
    Lupton V mag is scattered around -0.013 +- 0.003 (minmax = -0.025, -0.005)
    --> Lupton conversion is fainter.
    V = g - 0.5784*(g - r) - 0.0038;  sigma = 0.0054
    '''
    if gerr is None:
        gerr = np.zeros_like(g)

    if rerr is None:
        rerr = np.zeros_like(r)

    V = g - 0.5784 * (g - r) - 0.0038
    dV = np.sqrt((1.5784 * gerr)**2 + (0.5784 * rerr)**2 + 0.0052**2)
    return V, dV