import numpy as np
import ephem

# Many of here are obtained from ``callhorizons.export2pyephem`` in callhorizons
# v 1.0.13 by M. Mommert.

__all__ = ['jd2djd', 'djd2jd', 'get_readdb_epoch',
           'mpcorb2pyephem', 'make_pyephem']

# TODO: Make mpcorb class??


def mpcorb_num(mpcorbdata):
    ''' MPCORB contains parentheses in 'Number', so remove it.
    '''
    data = mpcorbdata.copy()
    raw_num = data['Number'].astype(str)
    num = np.char.replace(raw_num, '(', '')
    num = np.char.replace(num, ')', '')
    data['Number'] = num.astype(int)
    return data


def jd2djd(jd):
    ''' Converts the JD to Dublin JD
    Obtained from ``callhorizons.export2pyephem`` in callhorizons
    v 1.0.13 by M. Mommert.
    '''
    djd = jd - 2415020.0  # Dublin Julian date
    return djd


def djd2jd(djd):
    ''' Converts the ublin JD to JD
    Obtained from ``callhorizons.export2pyephem`` in callhorizons
    v 1.0.13 by M. Mommert.
    '''
    jd = djd + 2415020.0  # Dublin Julian date
    return jd


def get_readdb_epoch(jd):
    ''' Converts jd to month/date/year format for PyEphem.
    Obtained from ``callhorizons.export2pyephem`` in callhorizons
    v 1.0.13 by M. Mommert.
    '''
    epoch_djd = jd2djd(jd)  # Dublin Julian date
    epoch = ephem.date(epoch_djd)
    y, m, d = epoch.triple()
    epoch_str = f"{m}/{d}/{y}"
    return epoch_str


def mpcorb2pyephem(mpcorb_line, equinox):
    ''' Converts the IAUMPCORB line to PyEphem line.
    Obtained from ``callhorizons.export2pyephem`` in callhorizons
    v 1.0.13 by M. Mommert.
    '''
    name = mpcorb_line['Number']
    a, e, i = mpcorb_line['a'], mpcorb_line['e'], mpcorb_line['i']
    o, Om = mpcorb_line['Peri'], mpcorb_line['Node']
    M, H, G = mpcorb_line['M'], mpcorb_line['H'], mpcorb_line['G']
    E = get_readdb_epoch(mpcorb_line['Epoch'])
    n = 0.9856076686 / np.sqrt(a**3)  # mean daily motion
    line = f'{name},e,{i},{Om},{o},{a},{n},{e},{M},{E},{equinox},{H},{G}'
    return line


def make_pyephem_object(obj, equinox):
    ''' Makes a pyephem EllipticalBody object.
    Obtained from ``callhorizons.export2pyephem`` in callhorizons
    v 1.0.13 by M. Mommert.
    '''
    pyephem_line = mpcorb2pyephem(obj, equinox=equinox)
    ephobj = ephem.readdb(pyephem_line)
    return ephobj


def make_pyephem(mpcorbdata, equinox):
    ''' Make a list of ephem.EllipticalBody objects.
    Obtained from ``callhorizons.export2pyephem`` in callhorizons
    v 1.0.13 by M. Mommert.
    '''
    objects = []
    for obj in mpcorbdata:
        # export to PyEphem
        ephobj = make_pyephem_object(obj, equinox)
        objects.append(ephobj)

    return objects


def check_near_opp(mpcorbobj, times, equinox, elong_min, mag_max):
    from astropy.time import Time
    if isinstance(times, Time):
        times = times.jd

    is_visible = False
    ephobj = make_pyephem_object(mpcorbobj, equinox)
    for date in times:
        date_djd = jd2djd(date)
        if is_visible:
            print(mpcorbobj['Number'], end=' ')
            break
        else:
            ephobj.compute(date_djd)
            if ((ephobj.elong > np.deg2rad(elong_min))
                    and (ephobj.mag < mag_max)):
                is_visible = True
    return is_visible
