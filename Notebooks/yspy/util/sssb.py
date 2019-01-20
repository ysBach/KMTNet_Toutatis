"""
Convenience functions related to SSSB (small solar system bodies).
"""
from astroquery.jplhorizons import Horizons


def permanent_designation(designation, **kwargs):
    ''' Gives the permanent designation.
    Note
    ----
    Query to JPL/HORIZONS and reads the byproduct ``targetname``.
    '''

    dq = Horizons(id=designation, **kwargs)
    fullname = str(dq.elements()['targetname'][0])

    try:
        perm_desig = int(fullname.split(' ')[0])
        return perm_desig

    except ValueError:
        print(f"No permanent designation found for {designation}.")
        # first and last characters are '(' and ')', so crop them:
        return fullname[1:-1]


"""
import requests
from bs4 import BeautifulSoup

NEODYS_SERVER = "http://newton.dm.unipi.it/neodys/index.php?"
SBDB_SERVER = "https://ssd.jpl.nasa.gov/sbdb.cgi?"

def permanent_designation(designation):
    '''
    This function queries to NEODyS (http://newton.dm.unipi.it/neodys/index.php?pc=0)
    or JPL SBDB (https://ssd.jpl.nasa.gov/sbdb.cgi) which eventually queries
    the permanent designation based on the given designation.
        1984 QY1 = 2007 VB1 = 2011 NQ  ---->  331471
    NEODyS only has NEOs, but it has advantage that it contains the spin axis
    information. JPL SBDB contains any small bodies, but it does not contain
    the spin axis information.
    '''

    #TODO: Make it parallelizable.

    NEODYS_params = dict(pc = '1.1.9', n = designation)
    SBDB_params = dict(sstr = designation)

    is_inNEODYS = True
    obj_type = 'neo'
    # TODO: These are not used currently. Maybe we can return these objects?

    NEODYS_page = requests.get(NEODYS_SERVER,
                               timeout=timeout,
                               params=NEODYS_params)
    soup = BeautifulSoup(NEODYS_page.content, 'html.parser')

    try:
        url = soup.find_all('iframe')[0]['src']
        perm_desig = int(url.split('/')[-1].split('.')[0])
        return perm_desig

    except IndexError: # if no page exists
        is_inNEODYS = False

    SBDB_page = requests.get(SBDB_SERVER,
                             timeout=timeout,
                             params=SBDB_params)
    soup = BeautifulSoup(SBDB_page.content, 'html.parser')
    spkid = int((soup.find_all("font")[8]).get_text())
    obj_type_int = spkid // 1.e6

    if obj_type_int == 1:
        obj_type = 'comet'
    elif obj_type_int == 2:
        obj_type = 'asteroid'
    else:
        obj_type = 'unknown'
        print(f"No permanent designation found for {designation}")
        return 0

    perm_desig = int(spkid % 1.e6)
    return perm_desig
"""




