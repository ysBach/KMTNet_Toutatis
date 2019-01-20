"""
IRSA Moving Object
==================

API from

 https://irsa.ipac.caltech.edu/applications/Gator/GatorAid/irsa/catsearch.html

The URL of the IRSA catalog query service, CatQuery, is

 https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query

The service accepts the following keywords, which are analogous to the search
fields on the Gator search form:


catalog     Required    Catalog name in the IRSA database management system.
                        NOTE: Catalog needs to be mocing-object search enabled.

                        Examples: 'allsky_4band_p1bs_psd'
                                  'allsky_3band_p1bs_psd'
                                  'allsky_2band_p1bs_psd'
                                  'neowiser_p1bs_psd'

moradius    Optional    Cone search radius (arcsec)
                        (default = 10 arcsec)

mobj        Required    Type of input
                        smo - by name or number
                        mpc - MPC format
                        obt - orbital elements


mobjstr                 Name or numeric designation of object
                        Required if mobj = smo.

                        Examples: "324"
                                  "Bamberga" (asteroid)
                                  "29P" (comet)

mobjtype                Solar-system object type
                        Required if mobj = mpc or obt.

                        Asteroid - for asteroids
                        Comet - for comets

mpc                     MPC one-line format string.
                        Required if mobj = mpc.

mobjmaj/perih_dist      Semi-major axis (asteroid) or perihelion distance
                        (comet) in AU. Must be larger than 0.
                        Required when mobj = obt.

mobjecc                 Eccentricity of orbit (0.0-1.0)
                        Required when mobj = obt.

mobjinc                 Inclination of orbit (deg) (0.0-180.0)
                        Required when mobj = obt.
mobjper                 Argument of perihelion (deg) (0.0-360.0)
                        Required when mobj = obt.

mobjasc                 Longitude of ascending node (deg) (0.0-360.0)
                        Required when mobj = obt.

mobjanom/perih_time     Mean anomaly (asteroid) in deg, or perihelion time
                        (comet) in yyyy+mm+dd+hh:mm:ss
                        (0.0-360.0 for mean anomaly)
                        Required when mobj = obt.

mobjdsg                 Designation for returned ephemeris
                        Required when mobj = obt.

                        Example: Bamberga

mobjepo                 Epoch of coordinates in MJD (Modified Julian Date)
                        Required when mobj = obt.

                        Example: 55203.0 for start of WISE mission
btime       Optional    Earliest observation date (UT) to include in
                        yyyy+mm+dd+hh:mm:ss

etime       Optional    Latest observation date (UT) to include in
                        yyyy+mm+dd+hh:mm:ss

outfmt      Optional    Defines query's output format.
                        6 - returns a program interface in XML
                        3 - returns a VO Table (XML)
                        2 - returns SVC message
                        1 - returns an ASCII table
                        0 - returns Gator Status Page in HTML (default)

#TODO: Check below (excerpt from astroquery v 0.3.7)
onlist      Optional    1 - catalog is visible through Gator web interface
                        (default)

                        0 - catalog has been ingested into IRSA but not yet
                        visible through web interface.

                        This parameter will generally only be set to 0 when
                        users are supporting testing and evaluation of new
                        catalogs at IRSA's request.

If onlist=0, the following parameters are required:

    server              Symbolic DataBase Management Server (DBMS) name

    database            Name of Database.

    ddfile              The data dictionary file is used to get column
                        information for a specific catalog.

    selcols             Target column list with value separated by a comma(,)

                        The input list always overwrites default selections
                        defined by a data dictionary.

    outrows             Number of rows retrieved from database.

                        The retrieved row number outrows is always less than or
                        equal to available to be retrieved rows under the same
                        constraints.
"""


class IrsaMOQuery:
    server = 'https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?'

    def __init__(self, catalog):
        self.catalog = catalog
        self.moradius = None
        self.mobjstr = None
        self.mobjtype = None
        self.mpc = None
        self.mobjmaj = None
        self.perih_dist = None
        self.mobjecc = None
        self.mobjinc = None
        self.mobjper = None
        self.mobjasc = None
        self.mobjanom = None
        self.perih_time = None
        self.mobjdsg = None
        self.mobjepo = None
        self.btime = None
        self.etime = None
        self.outfmt = None
        self.selcols = None
        self.outrows = None

    def query_smo(self, mobjstr, moradius=None, btime=None, etime=None,
                  selcols=None, outrows=None, outfmt=6):
        ''' Query with the mobj = smo option.
        mobjstr: str
            The name or the number of the target.

        moradius: int, float
            The cone search radius in arcsec. Default is None and will be
            regarded as 10 arcsec by IRSA MO search.
        '''


