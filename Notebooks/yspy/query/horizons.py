"""
FIXME: Use astroquery.jplhorizons!!
"""

import callhorizons
import numpy as np
from astropy import units as u
from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from ..math import angle


class DiscreteEpochQuery:
    '''
    Example
    -------
    >>> OBSTIME = Time('2016-08-06T00:00:00', 'utc')
    >>> queried = query.DiscreteEpochQuery('1984 QY1', '511', [OBSTIME.jd])
    >>> queried.query()
    >>> vec=queried.vectors()
    >>> vec_spin = SkyCoord(20, 10, unit='deg', frame='hcrs')
    >>> lat_sun = vec_spin.separation(vec['TS'])[0]
    >>> lon_sun = -PI * u.rad
    >>> lat_obs = vec_spin.separation(vec['TO'])[0]
    >>> print((vec['TO'].separation(vec['TS'])).value)
    >>> print(queried.query_table['alpha'])
    # 30.4699...
    # 30.6902...
    NOTE: The two values differ slightly, and that is because the position
        of the target seen by the sun is calculated in J2000.0, for example,
        and that seen by the observer is calculated in Jxxxx.x, the observation
        time. So they can differ by at most ~ +- 0.4 degree.
    '''
    def __init__(self, targetname, observatory_code, discreteepochs):
        '''
        Note
        ----
        Parameters have the same name as that of ``callhorizons`` v 1.0.11.
        '''
        self.targetname = str(targetname)
        self.observatory_code = str(observatory_code)
        self.discreteepochs = np.asarray(discreteepochs)
        self.query_table = None
        self.vecs = None
        self.spin = None
        self.obs_pm = None
        # if positive (``obs_pm > 0``), the subobserver point is at afternoon
        # in the targetocentric coord. The subsolar point is at the midday,
        # which must be by definition.

    def __str__(self):
        _str = "Query {:s} at {:s} observatory for given discrete epochs."
        return _str.format(self.targetname, self.obsevatory_code)

    def query(self, depoch=100, smallbody=True, cap=True, comet=False,
              asteroid=False, airmass_lessthan=99, solar_elongation=(0, 180),
              skip_daylight=False):
        '''
        Parameters
        ----------
        depoch: int, optional
            The number of discrete epochs to be chopped. This is needed because
            the HORIZONS query does not accept infinitely many epochs at one
            time. I guess ~ 600 epochs are maximum we can query at a time.

        Note
        ----
            Other parameters are explained in ``callhorizons`` manual v 1.0.11.
        '''
        # TODO: add ``comet`` and ``asteroid`` to ``callhorizons.query``

        if depoch > 100:
            Warning('If query for more than about 100 epochs, '
                    'HORIZONS sometimes does not work correctly.')

        Nepoch = np.shape(self.discreteepochs)[0]
        Nquery = (Nepoch - 1) // depoch + 1
        tabs = []

        print(f'Query: {self.targetname} at {self.observatory_code} for {Nepoch} epochs')
        for i in range(Nquery):
            if Nquery != 1:
                print(f"Querying {i+1} / {Nquery} batch.")
            dq = callhorizons.query(self.targetname, smallbody=smallbody,
                                    cap=cap)
            # TODO: add comet and asteroids to ``callhorizons.query``
            dq.set_discreteepochs(self.discreteepochs[i * depoch:(i + 1) * depoch])
            dq.get_ephemerides(self.observatory_code,
                               airmass_lessthan=airmass_lessthan,
                               solar_elongation=solar_elongation,
                               skip_daylight=skip_daylight)
            tabs.append(Table(dq.data))

        if len(tabs) == 1:
            self.query_table = tabs[0]

        elif len(tabs) > 1:
            self.query_table = vstack(tabs)

        print("Query done.")

    def calc_vectors(self, returns=True):
        ''' Calculates the Sun, Target, Observer vectors.
        Returns
        -------
        vecs: dict of SkyCoord
            A dict object which contains all 6 combinations of vectors made
            by the Sun, Target, and Observer.
        '''
        if self.query_table is None:
            raise ValueError('Please do query first!')

        def _makevec(name_r, name_lon, name_lat):
            r = self.query_table[name_r] * u.au
            lon = self.query_table[name_lon] * u.deg
            lat = self.query_table[name_lat] * u.deg
            return SkyCoord(lon, lat, distance=r, frame='hcrs')

        vec_ST = _makevec('r', 'EclLon', 'EclLat')
        vec_OT = _makevec('delta', 'ObsEclLon', 'ObsEclLat')

        vec_SO = angle.add_sc(vec_ST, angle.revert_sc(vec_OT))

        self.vecs = {'SO': vec_SO,
                     'OS': angle.revert_sc(vec_SO),
                     'ST': vec_ST,
                     'TS': angle.revert_sc(vec_ST),
                     'OT': vec_OT,
                     'TO': angle.revert_sc(vec_OT)}
        if returns:
            return self.vecs

    def subpoints(self, spin, midnight_rad=-np.pi):
        ''' Calculates the subsolar and sub-observer point for given spin.
        Parameters
        ----------
        spin: SkyCoord
            The spin vector. Usually with ``frame='hcrs'``.

        midnight_rad: float
            The angle (longitude) at the midnight. Defaults to ``-PI``.

        Returns
        -------
        : dict of astropy.coordinates.angles.Angle
        '''
        self.spin = spin
        if self.vecs is None:
            self.calc_vectors(returns=False)

        alpha = self.query_table['alpha']
        # NOTE: Change the above 'alpha' if callhorizons changes.

        def _compare_sun_obs():
            comp = angle.crossdot_sc(self.vecs['TS'], self.vecs['TO'], self.spin)
            self.obs_pm = comp

        def _get_lon_obs(lon_sun, lat_sun, lat_obs):
            numer = np.cos(alpha) - np.sin(lat_sun) * np.sin(lat_obs)
            denom = np.cos(lat_sun) * np.cos(lat_obs)
            cosine = numer / denom
            _compare_sun_obs()

            sign = (self.obs_pm > 0) * 2 - 1
            # sign = +1 if self.obs_pm > 0 and -1 otherwise.

            lon = coordinates.Angle(lon_sun + sign * np.arccos(cosine))
            lon.wrap_at('180d', inplace=True)
            return lon

        lat_sun = spin.separation(self.vecs['TS'])
        lon_sun = coordinates.Angle(np.ones(len(lat_sun)) * midnight_rad * u.rad)
        lat_obs = spin.separation(self.vecs['TO'])
        lon_obs = _get_lon_obs(lon_sun, lat_sun, lat_obs)

        # TODO: Return two vectors, not one dict
        return {'lon_sun': lon_sun, 'lat_sun': lat_sun,
                'lon_obs': lon_obs, 'lat_obs': lat_obs}

    def writeto(self, output, format='ascii.csv', overwrite=True):
        ''' Writes the queried table.
        '''
        if output is not None:
            tosave = self.query_table
            tosave.write(output, format=format, overwrite=overwrite)
