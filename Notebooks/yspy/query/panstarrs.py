import warnings
import tempfile
import requests
from astropy.io.votable import parse_single_table


SERVER = 'https://archive.stsci.edu/panstarrs/search.php'


def panstarrs_query(ra_deg, dec_deg, rad_deg, mindet=1,
                    maxsources=10000):
    """ Query Pan-STARRS DR1 @ MAST.
    The code obtained from [1].

    Note
    ----
    Excerpt from [1].

    The Pan-STARRS DR1 catalog resides at MAST, which unfortunately does not
    yet have an ``astroquery`` interface. Hence, we have to use a different
    approach: we download the data as an xml file and read that in using
    ``astroquery``, again providing an ``astropy.table`` object.

    The file download makes this query significantly slower than comparable
    ``astroquery`` routines. Please note that STScI currently limits the
    Pan-STARRS queries on their servers to field radii smaller than 0.5 degrees.

    Example query can be done by
    >>> print(panstarrs_query(12.345, 67.89, 0.1))


    Parameters
    ----------
    ra_deg, dec_deg, rad_deg: float
        RA, Dec, field radius in degrees
    mindet: int, optional
        The minimum number of detection.
    maxsources: int, optional
        The maximum number of sources
    server: str
        The servername

    Returns
    -------
    datatab: astropy.table object

    References
    ----------
    [1] https://michaelmommert.wordpress.com/2017/02/13/accessing-the-gaia-and-pan-starrs-catalogs-using-python/
    """
    if rad_deg > 0.5:
        warnings.warn('STScI currently limits the Pan-STARRS queries on their '
                      + 'servers to field radii smaller than 0.5 degrees. '
                      + 'Retrieved 2018 Feburary.')

    r = requests.get(SERVER,
                     params={'RA': ra_deg, 'DEC': dec_deg,
                             'SR': rad_deg, 'max_records': maxsources,
                             'outputformat': 'VOTable',
                             'ndetections': ('>%d' % mindet)})

    with tempfile.TemporaryFile() as tmp:
        tmp.write(r.content)
        # parse local file into astropy.table object
        data = parse_single_table(tmp)
        datatab = data.to_table(use_names_over_ids=True)
        # tmp file automatically deleted.

    return datatab
