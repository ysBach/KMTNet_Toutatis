import warnings
import numpy as np
import callhorizons
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
from astropy import units as u

__all__ = ["calc_airmass", "airmass_obs", "airmass_hdr",
           "ecliptic_RADEC_rough", "ecliptic_RADEC_horizons"]


def calc_airmass(zd_deg=None, cos_zd=None, scale=750.):
    ''' Calculate airmass by nonrefracting radially symmetric atmosphere model.
    Note
    ----
    Wiki:
        https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Nonrefracting_radially_symmetrical_atmosphere
    Identical to the airmass calculation at a given ZD of IRAF's
    asutil.setairmass:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?setairmass

    Parameters
    ----------
    zd_deg: float, optional
        The zenithal distance in degrees
    cos_zd: float, optional
        The cosine of zenithal distance. If given, ``zd_deg`` is not used.
    scale: float, optional
        Earth radius divided by the atmospheric height (usually scale height)
        of the atmosphere. In IRAF documentation, it is mistakenly written that
        this ``scale`` is the "scale height".
    '''
    if zd_deg is None and cos_zd is None:
        raise ValueError("Either zd_deg or cos_zd should not be None.")

    if cos_zd is None:
        cos_zd = np.cos(np.deg2rad(zd_deg))

    am = np.sqrt((scale * cos_zd)**2 + 2 * scale + 1) - scale * cos_zd

    return am


def airmass_obs(targetcoord, obscoord, ut, exptime, scale=750., full=False):
    ''' Calculate airmass by nonrefracting radially symmetric atmosphere model.
    Note
    ----
    Wiki:
        https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Nonrefracting_radially_symmetrical_atmosphere
    Identical to the airmass calculation for a given observational run of
    IRAF's asutil.setairmass:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?setairmass
    Partly contributed by Kunwoo Kang (Seoul National University) in Apr 2018.

    '''
    if not isinstance(ut, Time):
        warnings.warn("ut is not Time object. "
                      + "Assume format='isot', scale='utc'.")
        ut = Time(ut, format='isot', scale='utc')
    if not isinstance(exptime, u.Quantity):
        warnings.warn("exptime is not astropy Quantity. "
                      + "Assume it is in seconds.")
        exptime = exptime * u.s

    t_start = ut
    t_mid = ut + exptime / 2
    t_final = ut + exptime

    altaz = {"alt": [], "az": [], "zd": [], "airmass": []}
    for t in [t_start, t_mid, t_final]:
        C_altaz = AltAz(obstime=t, location=obscoord)
        target = targetcoord.transform_to(C_altaz)
        alt = target.alt.to_string(unit=u.deg, sep=':')
        az = target.az.to_string(unit=u.deg, sep=':')
        zd = target.zen.to(u.deg).value
        am = calc_airmass(zd_deg=zd, scale=scale)
        altaz["alt"].append(alt)
        altaz["az"].append(az)
        altaz["zd"].append(zd)
        altaz["airmass"].append(am)

    am_simpson = (altaz["airmass"][0]
                  + 4 * altaz["airmass"][1]
                  + altaz["airmass"][2]) / 6

    if full:
        return am_simpson, altaz

    return am_simpson


def airmass_hdr(header, ra=None, dec=None, ut=None, exptime=None,
                lon=None, lat=None, height=None, equinox=None, frame=None,
                scale=750.,
                ra_key="RA", dec_key="DEC", ut_key="DATE-OBS",
                exptime_key="EXPTIME", lon_key="LONGITUD", lat_key="LATITUDE",
                height_key="HEIGHT", equinox_key="EPOCH", frame_key="RADECSYS",
                ra_unit=u.hourangle, dec_unit=u.deg,
                exptime_unit=u.s, lon_unit=u.deg, lat_unit=u.deg,
                height_unit=u.m,
                ut_format='isot', ut_scale='utc',
                full=False
                ):
    ''' Calculate airmass using the header.
    Parameters
    ----------
    ra, dec: float or Quantity, optional
        The RA and DEC of the target. If not specified, it tries to find them
        in the header using ``ra_key`` and ``dec_key``.

    ut: str or Time, optional
        The *starting* time of the observation in UT.

    exptime: float or Time, optional
        The exposure time.

    lon, lat, height: str, float, or Quantity
        The longitude, latitude, and height of the observatory. See
        astropy.coordinates.EarthLocation.

    equinox, frame: str, optional
        The ``equinox`` and ``frame`` for SkyCoord.

    scale: float, optional
        Earth radius divided by the atmospheric height (usually scale height)
        of the atmosphere.

    XX_key: str, optional
        The header key to find XX if ``XX`` is ``None``.

    XX_unit: Quantity, optional
        The unit of ``XX``

    ut_format, ut_scale: str, optional
        The ``format`` and ``scale`` for Time.

    full: bool, optional
        Whether to return the full calculated results. If ``False``, it returns
        the averaged (Simpson's 1/3-rule calculated) airmass only.
    '''
    # TODO: compare this with fits_util.get_from_header
    def _conversion(header, val, key, unit=None, instance=None):
        if val is None:
            val = header[key]
        elif (instance is not None) and (unit is not None):
            if isinstance(val, instance):
                val = val.to(unit).value

        return val

    ra = _conversion(header, ra, ra_key, ra_unit, u.Quantity)
    dec = _conversion(header, dec, dec_key, dec_unit, u.Quantity)
    exptime = _conversion(header, exptime, exptime_key,
                          exptime_unit, u.Quantity)
    lon = _conversion(header, lon, lon_key, lon_unit, u.Quantity)
    lat = _conversion(header, lat, lat_key, lat_unit, u.Quantity)
    height = _conversion(header, height, height_key, height_unit, u.Quantity)
    equinox = _conversion(header, equinox, equinox_key)
    frame = _conversion(header, frame, frame_key)

    if ut is None:
        ut = header[ut_key]
    elif isinstance(ut, Time):
        ut = ut.isot
        # ut_format = 'isot'
        # ut_scale = 'utc'

    targetcoord = SkyCoord(ra=ra,
                           dec=dec,
                           unit=(ra_unit, dec_unit),
                           frame=frame,
                           equinox=equinox)

    try:
        observcoord = EarthLocation(lon=lon * lon_unit,
                                    lat=lat * lat_unit,
                                    height=height * height_unit)

    except ValueError:
        observcoord = EarthLocation(lon=lon,
                                    lat=lat,
                                    height=height * height_unit)

    result = airmass_obs(targetcoord=targetcoord,
                         obscoord=observcoord,
                         ut=ut,
                         exptime=exptime,
                         scale=scale,
                         full=full)

    return result


def ecliptic_RADEC_rough(l_min=0, l_max=360, num=50):
    ''' Calculates the approximation of RA and DEC of the ecliptic plane.
    https://community.dur.ac.uk/john.lucey/users/solar_year.html

    Parameters
    ----------
    l_min, l_max : float, optional
        The minimum and the maximum ecliptic longitude in degrees you want to
        calculate.
    num : int, optional
        The number of ecliptic longitude bins (i.e., resolution).
    '''
    lambda_sun = np.linspace(l_min, l_max, num)
    # Now alpha and delta are in degrees:
    alpha_sun = lambda_sun + 2.45 * np.sin(2 * np.deg2rad(lambda_sun))
    delta_sun = 23.5 * np.sin(np.deg2rad(lambda_sun))
    return alpha_sun, delta_sun


def ecliptic_RADEC_horizons(discrete_epoch=None, start_epoch=None,
                            stop_epoch=None, step_size=None, alpha_sort=True):
    ''' Queries the exact RA DEC of the ecliptic plane for given time.
    Parameters
    ----------
    alpha_sort : bool, optional
        Wether to sort the output RA and DEC based on the RA.
    '''
    is_discrete = True
    if discrete_epoch is None:
        if (start_epoch is None) ^ (stop_epoch is None) ^ (step_size is None):
            raise ValueError(
                'If discrete_epoch is not given, all others must be given.')

    dq = callhorizons.query('sun', smallbody=False)
    if is_discrete:
        dq.set_discreteepochs(discrete_epoch)
    else:
        dq.set_epochrange(start_epoch, stop_epoch, step_size)
    dq.get_ephemerides(500)

    # TODO: There must be a better way to achieve this...
    ra, dec = dq['RA'], dq['DEC']
    if alpha_sort:
        sort_idx = ra.argsort()
        ra, dec = ra[sort_idx], dec[sort_idx]
    return ra, dec
