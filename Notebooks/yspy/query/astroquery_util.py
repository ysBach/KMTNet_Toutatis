from astroquery.vizier import Vizier


# # TODO: Remove AstroqueryColumns....
# class AstroqueryColumns:
#     def __init__(self, columns=[], column_filters={}):
#         self.columns = columns
#         self.column_filters = column_filters
#         self.query_dict = dict(columns=self.columns,
#                                column_filters=self.column_filters)

#     def __str__(self):
#         return self.columns, self.column_filters


class QueryVizier:
    def __init__(self, coordinates, columns=None, radius=None, keywords=None,
                 inner_radius=None, width=None, height=None, catalog=None,
                 column_filters={}):
        '''
        Parameters
        ----------
        columns : list
            List of strings

        column_filters : dict

        keywords : str or None

        coordinates : str, `astropy.coordinates` object, or `~astropy.table.Table`
            The target around which to search. It may be specified as a
            string in which case it is resolved using online services or as
            the appropriate `astropy.coordinates` object. ICRS coordinates
            may also be entered as a string.  If a table is used, each of
            its rows will be queried, as long as it contains two columns
            named ``_RAJ2000`` and ``_DEJ2000`` with proper angular units.

        radius : convertible to `~astropy.coordinates.Angle`
            The radius of the circular region to query.

        inner_radius : convertible to `~astropy.coordinates.Angle`
            When set in addition to ``radius``, the queried region becomes
            annular, with outer radius ``radius`` and inner radius
            ``inner_radius``.

        width : convertible to `~astropy.coordinates.Angle`
            The width of the square region to query.

        height : convertible to `~astropy.coordinates.Angle`
            When set in addition to ``width``, the queried region becomes
            rectangular, with the specified ``width`` and ``height``.

        catalog : str or list, optional
            The catalog(s) which must be searched for this identifier.
            If not specified, all matching catalogs will be searched.
        '''
        self.coordinates = coordinates
        self.radius = radius
        self.inner_radius = inner_radius
        self.width = width
        self.height = height
        self.catalog = catalog
        self.column_filters = column_filters
        self.columns = columns

        if self.columns is None:
            if catalog == 'UCAC4':
                self.columns = ['+_r', 'RAJ2000', 'DEJ2000', 'mfa', 'No',
                                'Bmag', 'e_Bmag', 'f_Bmag',
                                'Vmag', 'e_Vmag', 'f_Vmag',
                                'gmag', 'e_gmag', 'f_gmag',
                                'rmag', 'e_rmag', 'f_rmag',
                                'imag', 'e_imag', 'f_imag']
            else:
                self.columns = ['+_r', '*']

    def __str__(self):
        return self.coordinates, self.catalog

    def query(self):
        '''

        Example
        -------
        >>> coords = SkyCoord([100, 90],[30, 20], unit=(u.deg, u.deg))
        >>> r = 10 * u.arcmin
        >>> test = QueryVizier(coords, catalog='UCAC4', radius=r)
        >>> tq = test.query()
        '''
        viz = Vizier(columns=self.columns,
                     column_filters=self.column_filters)

        viz.ROW_LIMIT = -1
        # query up to infinitely many rows. By default, this is 50.

        result = viz.query_region(self.coordinates, radius=self.radius,
                                  inner_radius=self.inner_radius,
                                  width=self.width, height=self.height,
                                  catalog=self.catalog)

        self.queried = result

        return self.queried
