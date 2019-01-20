import numpy as np
from astropy.coordinates.matrix_utilities import rotation_matrix as RM

# TODO: Make a function to parse the SPIN file from DAMIT(until line2)


def read_obj(objfile):
    objstr = np.loadtxt(objfile, dtype=bytes).astype(str)
    vertices = objstr[objstr[:, 0] == 'v'][:, 1:].astype(float)
    facets = objstr[objstr[:, 0] == 'f'][:, 1:].astype(int)

    # Normals include direction + area information
    facet_normals_ast = []
    facet_areas = []

    # I don't think we need to speed this for loop  up too much since it takes
    # only ~ 1 s even for 20000 facet case.
    for facet in facets:
        verts = vertices[facet - 1]  # Python is 0-indexing!!!
        vec10 = verts[1] - verts[0]
        vec20 = verts[2] - verts[0]

        area = np.linalg.norm(np.cross(vec10, vec20)) / 2  # Triangular
        facet_com_ast = np.sum(verts, axis=0) / 3

        facet_normals_ast.append(facet_com_ast)
        facet_areas.append(area)

    facet_normals_ast = np.array(facet_normals_ast)
    facet_areas = np.array(facet_areas)

    return dict(objstr=objstr, vertices=vertices, facets=facets,
                normals=facet_normals_ast, areas=facet_areas)


def read_damit_spin(spinfile, spin=None):
    ''' Simple reader for the obj file and spin file.
    If there is no proper spin file from DAMIT, manually give spin as a dict
    object. The dict should have phi0, longitude and latitude in the unit of
    deg, rotational period of hours, and t0 of days (JD).
    '''
    # Get spin-related informations
    spin = dict(lon=None, lat=None, P=None, t0=None, phi0=None)
    with open(spinfile, 'r') as f:
        l1 = np.array(f.readline().replace(
            '\n', '').split(' ')).astype(float)
        l2 = np.array(f.readline().replace(
            '\n', '').split(' ')).astype(float)
        spin["lon"], spin["lat"], spin["P"] = l1
        spin["t0"], spin["phi0"] = l2

    return spin


class ObjShape:
    '''
    Example
    -------
    PI = np.pi
    d2r = PI / 180

    objfile = "3200.obj"
    spinfile = "A1011.M1730.spin.txt"

    objstr, spin = read_obj_spin(objfile, spinfile=spinfile)
    obj = ObjShape(objstr, spin)

    obj.set_rotmat("dtP", 0.1)
    obj.solve_facets()
    print(obj)

    plt.close('all')
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    x = obj.vectors[:, 0]
    y = obj.vectors[:, 1]
    z = obj.vectors[:, 2]
    ax.scatter(x, y, z)
    cax.contourf(x, y, z, zdir='z', offset=0)
    '''

    def __init__(self, objstr=None, spin=None):
        self.objstr = objstr
        self.spin = spin
        self.vertices = self.objstr[self.objstr[:, 0]
                                    == 'v'][:, 1:].astype(float)
        self.facets = self.objstr[self.objstr[:, 0] == 'f'][:, 1:].astype(int)
        self.M_ast2ecl = None
        self.facet_normals_ast = None
        self.facet_normals_ecl = None
        self.facet_areas = None
        self.mus = None

    def __str__(self):
        nvertices = self.objstr[self.objstr[:, 0] == 'v'].shape[0]
        nfacets = self.objstr[self.objstr[:, 0] == 'f'].shape[0]
        return f"{nvertices} vertices and {nfacets} facets model."

    def set_rotmat(self, mode, value, testing=True):
        ''' Set rotational matrix by one of the methods.
        Parameters
        ----------
        mode: str in {"t", "dt", "dtP", "wt"}
            See Note.

        value: float
            The corresponding value. See Note.

        testing: bool
            Whether to test the rotation matrix is unitary.

        Note
        ----
        Depending on ``mode``, the calculation for the `2 * pi * (t - t0) / P`
        term will be different.

        * ``mode = "t"``: ``value = t``. ``value`` in JD.
        * ``mode = "dt"``: ``value = t - t0``. ``value`` in JD.
        * ``mode = "dtP"``: ``value = (t - t0) / P``. ``value`` has no unit.
        * ``mode = "wt"``: ``value = 2 * pi * (t - t0) / P``. ``value`` in rad.
        '''
        P_day = self.spin["P"] / 24.

        if mode == "t":
            term = 2 * np.pi * (value - self.spin["t0"] / P_day)
        elif mode == "dt":
            term = 2 * np.pi * (value / P_day)
        elif mode == "dtP":
            term = 2 * np.pi * value
        elif mode == "wt":
            term = value
        else:
            raise ValueError("mode not understood.")

        # Transformation matrix from obj file's xyz to ecliptic coordinate
        # Angle must be in degree and it rotates the axes, not point. Example:
        # RM(a, axis='z') = [[ cos a, sin a, 0],
        #                    [-sin a, cos a, 0],
        #                    [     0,     0, 1]]
        # RM(30, axis='z') @ [1, 0, 0] = [0.866, -0.5, 0]

        self.M_ast2ecl = (RM(-1 * (self.spin["lon"]), axis='z')
                          @ RM(-1 * (90 - self.spin["lat"]), axis='y')
                          @ RM(-1 * (term), axis='z'))

        if testing:
            # Rotation matrix should be unitary
            np.testing.assert_almost_equal([1, 1, 1],
                                           np.sum(self.M_ast2ecl**2, axis=0))
            np.testing.assert_almost_equal([1, 1, 1],
                                           np.sum(self.M_ast2ecl**2, axis=1))

    def solve_facets(self):
        # Normals include direction + area information
        facet_normals_ast = []
        facet_normals_ecl = []
        facet_areas = []

        for facet in self.facets:
            verts = self.vertices[facet - 1]  # Python is 0-indexing!!!
            vec10 = verts[1] - verts[0]
            vec20 = verts[2] - verts[0]

            area = np.linalg.norm(np.cross(vec10, vec20))
            facet_com_ast = np.sum(verts, axis=0) / 3
            facet_com_ecl = self.M_ast2ecl @ facet_com_ast

            facet_normals_ast.append(facet_com_ast)
            facet_normals_ecl.append(facet_com_ecl)
            facet_areas.append(area)

        facet_normals_ast = np.array(facet_normals_ast)
        facet_normals_ecl = np.array(facet_normals_ecl)
        facet_areas = np.array(facet_areas)

        self.facet_normals_ast = facet_normals_ast
        self.facet_normals_ecl = facet_normals_ecl
        self.facet_areas = facet_areas
