import math
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import warnings

from htvlearn.grid import Grid


class Lattice():
    """ Implements a Bravais lattice

    The lattice points are defined as the set:
    A = {a1*x1 + ... + an*xn: 0 <= ai < lsize, x1,...,xn in R^n}
      = {X_mat @ a : 0 <= ai < lsize, X_mat in R^{nxn}}.

    X_mat changes coordinates in the basis x1, ..., xn of R^n to the euclidian
    standard basis e1, ..., en; its columns are formed by the lattice vectors
    x1, x2, ..., xn. We can define the function f: A -> R, which produces a
    set C containing the values of the box spline coefficients.
    For now, n = 2, such that the lattice is 2-dimensional.

    Tensor representation formats:
    . batch format (m,n) (default) - vectors along rows
    . mat (matrix) format (n,m) - vectors along columns
    . batch_mat format - batch of matrices (b,n,m)
    """
    ldim = 2  # lattice dimension
    centers_barycentric_coordinates = Tensor([0.333, 0.333, 0.334])
    # requires that lattice is hexagonal and lsize*h = 1 (both enforced)
    bottom_right_std = Tensor([0.2500, -0.4330])
    hexagonal_matrix = Tensor([[1., 0.5],
                               [0., 0.5 * math.sqrt(3)]])

    def __init__(self,
                 X_mat=None,
                 C_mat=None,
                 lsize=10,
                 C_init='zero',
                 **kwargs):
        """
        Args:
            Option 1 (Custom Lattice):
                X_mat: lattice matrix; size (n, n).
                C_mat: observation matrix; size ([k] * n);
                    C_mat(i1, .., in) = f(a1*x1 + ... + an*xn) = f(X_mat @ a),
                    e.g. for n=2: x direction -> rows, y -> columns.
            Option 2 (Hexagonal Lattice):
                lsize: Lattice size; number of vertices: (lsize+1)**2
                h: lattice step size;
                C_init: Lattice values initialization.
        """
        # C_mat is indexed as (i, j) = (x, y)
        # X_mat or C_mat attributes should not be changed directly
        # instead, assign to new_X_mat, new_C_mat and use update_lattice()
        if X_mat is not None and C_mat is not None:
            self.verify_input(X_mat, C_mat)
        else:
            assert X_mat == C_mat, \
                'X_mat is None and C_mat is not, or vice-versa.'
            X_mat, C_mat = self.init_hexagonal_lattice(lsize, C_init, **kwargs)

        self.init_origin_triangles()
        self.init_neighborhood()
        self.update_lattice(X_mat, C_mat)

    @classmethod
    def init_hexagonal_lattice(cls, lsize=10, C_init='zero', **kwargs):
        """ """
        assert C_init in ['zero', 'normal']

        h = 1. / lsize
        X_mat = cls.hexagonal_matrix.mul(h)
        C_mat = torch.zeros((lsize + 1, lsize + 1))
        if C_init == 'normal':
            C_mat.normal_().mul_(0.05)  # for visualization purposes.

        assert cls.is_hexagonal(X_mat), 'Lattice is not hexagonal'

        return X_mat, C_mat

    @staticmethod
    def is_hexagonal(X_mat):
        """ """
        if not np.allclose(X_mat[1, 0], 0.):
            return False
        if not np.allclose(X_mat[0, 1], X_mat[0, 0] / 2):
            return False
        if not np.allclose(X_mat[1, 1], X_mat[0, 1] * math.sqrt(3)):
            return False

        return True

    @classmethod
    def verify_input(cls, X_mat, C_mat):
        """ """
        # verify types
        if X_mat.dtype != torch.float32 or C_mat.dtype != torch.float32:
            raise ValueError('Expected float tensors.')

        # verify if lattice is hexagonal
        if not cls.is_hexagonal(X_mat):
            raise ValueError('Invalid X_mat --- lattice not hexagonal.')

        h, lsize = X_mat[0, 0], C_mat.size(0) - 1
        if not np.allclose(lsize * h, 1.):
            raise ValueError(f'lsize*h = {lsize * h}. Should be 1.')

        # verify sizes
        X_mat_size, C_mat_size = tuple(X_mat.size()), tuple(C_mat.size())
        ldim = X_mat_size[0]  # lattice dimension
        if ldim != cls.ldim:
            raise ValueError('Only 2D lattice allowed, for now')
        if X_mat_size != (ldim, ldim):
            raise ValueError(
                f'Expected size ({ldim},{ldim}). Found size {X_mat_size}.')
        if len(C_mat_size) != ldim or (not all(k == C_mat_size[0]
                                               for k in C_mat_size)):
            raise ValueError(
                f'Expected size ([k] * {ldim}). Found size {C_mat_size}.')
        if C_mat_size[0] % 2 != 1:
            raise ValueError(
                f'Expected odd dimensions. Found size {C_mat_size}.')

        # verify consistency
        if X_mat.det().allclose(Tensor([0.])):
            raise ValueError(
                'X_mat is not invertible or very ill-conditioned.')
        if not X_mat.allclose(X_mat.clamp(min=0)):
            # To facilitate computations.
            raise ValueError('Both lattice vectors should be positive')

    def init_origin_triangles(self):
        """ Initialize the first and second origin triangles, their barycentric
        matrices and respective inverses. The rows of each triangle are the
        lattice coordinates of the vertices.

        Some computations, such as computation of barycentric coordinates
        of x wrt to its triangle can be translated to computation of
        barycentric coordinates of translated x wrt one of the origin
        triangles. (see get_barycentric_mask/coordinates())
        """
        # vertices of origin square
        vA, vB, vC, vD = [0, 0], [1, 0], [1, 1], [0, 1]
        self.batch_origin_triangles = torch.tensor([[vA, vB, vD],
                                                    [vB, vC, vD]])  # (2,3,2)
        assert self.batch_origin_triangles.size() == (2, 3, 2)

        # size (2,3,3); for each of the 2 triangles (dim=0),
        # get barycentric matrices
        self.batch_origin_triangles_barycentric_mat = \
            self.append_ones(self.batch_origin_triangles,
                             dim=-1).transpose(1, 2)

        # batch inverse - size (2,3,3)
        self.batch_inv_origin_triangles_barycentric_mat = \
            torch.inverse(self.batch_origin_triangles_barycentric_mat)

        assert self.batch_inv_origin_triangles_barycentric_mat.size() == \
            (2, 3, 3)

    def init_neighborhood(self):
        """ Initialize neighborhood for HTV computations.
        """
        v_Ma, v_Mb, v_Mc = [0, 0], [1, 0], [0, 1]
        v_A, v_B, v_C = [1, 1], [-1, 1], [1, -1]

        # size: (6,2)
        self.neighborhood_diff = \
            torch.tensor([v_Ma, v_Mb, v_Mc, v_A, v_B, v_C])
        self.neighborhood_size = self.neighborhood_diff.size(0)

    def update_lattice_values(self, new_C_mat):
        """ """
        assert new_C_mat.size() == self.C_mat.size()
        self.C_mat = new_C_mat
        self.init_unique_triangles()
        self.init_affine_coefficients()

    def update_lattice(self, X_mat, C_mat):
        """ Updates which are always required when X_mat or C_mat
        are changed; e.g., after dividing lattice.
        """
        self.h = X_mat[0, 0]

        self.X_mat, self.C_mat = X_mat, C_mat
        # so that inversion does not have to be repeated
        self.X_mat_inv = torch.inverse(X_mat)

        self.lsize = C_mat.size(0) - 1
        if self.lsize % 2 != 0:
            raise ValueError(f'lsize {self.lsize} should be even...')

        # centered lattice; bottom left lattice coordinates: (lmin, lmin)
        self.lmin = -(self.lsize // 2)
        self.lmax = self.lmin + self.lsize

        # list of (x, y) locations of vertices in lattice coordinates
        self.lattice_grid = Grid(self.lmin,
                                 self.lmax + 1,
                                 h=1,
                                 square=True,
                                 to_numpy=False).x

        assert self.lattice_grid.size() == ((self.lsize + 1)**2, 2)

        # Now we get three corresponding tensors:
        # self.unique_triangles[i] <-> self.triangle_centers[i]
        # <-> self.affine_coeff[i]
        self.init_unique_triangles()
        self.init_triangle_centers()
        self.init_triangle_simplices()
        self.init_affine_coefficients()

    def get_lattice_mask(self, x_lat, pad=0):
        """ Get mask of points inside lattice to access as
        x_lat[in_lattice_mask]
        """
        x_lat_min, _ = x_lat.min(1)  # minimum along dim=1
        x_lat_max, _ = x_lat.max(1)  # maximum along dim=1

        lattice_mask = ((x_lat_min >= self.lmin + pad) *
                        (x_lat_max <= self.lmax - pad))

        return lattice_mask

    def inside_lattice(self, x_lat):
        """ """
        lattice_mask = self.get_lattice_mask(x_lat)
        return lattice_mask.all()

    def init_unique_triangles(self):
        """ """
        # x_lat does not include upper and right boundary vertices.
        # x_lat goes row-wise (i.e. has x_axis stride = 1);
        # e.g. lsize=1: x_lat = [(0, 0), (1, 0), (0, 1), (1, 1)]
        x_lat = Grid(self.lmin,
                     self.lmax,
                     h=1,
                     square=True,
                     to_numpy=False).x

        m = x_lat.size(0)
        x_lat_expand = x_lat.view(m, 1, 1, 2).expand(m, 2, 3, 2)

        # (m,2,3,2)
        unique_triangles_aux = x_lat_expand + self.batch_origin_triangles
        self.unique_triangles = unique_triangles_aux.view(m * 2, 3, 2)
        assert self.unique_triangles.size(0) == (self.lsize**2) * 2

        # index relative to bottom left vertice
        # size (m, 3, 2)
        unique_triangles_rel = self.unique_triangles.sub(self.lmin)

        # values of vertices of each unique triangle
        self.unique_triangles_values = \
            self.C_mat[unique_triangles_rel[:, :, 0],
                       unique_triangles_rel[:, :, 1]]

        assert self.unique_triangles_values.size() == \
            (self.unique_triangles.size(0), 3)

    def init_triangle_centers(self):
        """ """
        try:
            self.triangle_centers = \
                self.get_triangle_centers(self.unique_triangles)
        except AttributeError:
            print(
                'Need to initialize unique_triangles before triangle_centers.')
            raise

    @classmethod
    def get_triangle_centers(cls, triangles):
        """
        Get tensor with centers of lattice triangles
        in lattice coordinates
        """
        # unique_triangles size (k,3,2)
        k = triangles.size(0)
        batch_xcenter_barycentric_coordinates = \
            cls.centers_barycentric_coordinates.view(1, 3).expand(k, 3)
        assert batch_xcenter_barycentric_coordinates.size() == (k, 3)

        triangles_mat = triangles.transpose(1, 2).float()
        assert triangles_mat.size() == (k, 2, 3)
        xcenter = (
            triangles_mat
            @ batch_xcenter_barycentric_coordinates.unsqueeze(-1)).squeeze(-1)
        assert xcenter.size() == (k, 2)

        return xcenter

    @classmethod
    def append_ones(cls, x, dim=0):
        """ Append ones to x with the final goal of constructing a
        barycentric matrix.

        Args:
            dim: dimension along which to concatenate with ones vector.
        """
        if dim not in [0, -1]:
            raise ValueError(
                'Can only append ones to first or last dimension.')

        x_size = x.size()
        assert x_size[dim] == cls.ldim, ('concatenation dimension should be '
                                         'the same as the lattice dimension.')

        if dim == 0:
            ones_vec = torch.ones((1, *x_size[1::]))
            x_out = torch.cat((x.float(), ones_vec), dim=0)
        elif dim == -1:
            ones_vec = torch.ones((*x_size[0:-1], 1))
            x_out = torch.cat((x.float(), ones_vec), dim=-1)

        return x_out

    def init_triangle_simplices(self):
        """ """
        unique_triangles_rel = self.unique_triangles.sub(self.lmin)
        vertices_vectorized_idx = (unique_triangles_rel[:, :, 1] *
                                   (self.lsize + 1) +
                                   unique_triangles_rel[:, :, 0])
        i = vertices_vectorized_idx[:, 0]
        j = vertices_vectorized_idx[:, 1]
        k = vertices_vectorized_idx[:, 2]

        self.simplices = torch.cat(
            (i.view(-1, 1), j.view(-1, 1), k.view(-1, 1)), dim=1)

    # ==== Affine subsets coefficients ===================================== #

    def init_affine_coefficients(self):
        """ """
        try:
            x_lat = self.triangle_centers
        except AttributeError:
            print('Need to initialize triangle_centers before affine_coeff.')
            raise

        self.affine_coeff = \
            self.get_affine_coefficients(x_lat, use_saved_affine=False)

    def get_affine_coefficients(self, x_lat, use_saved_affine=True, **kwargs):
        """ Get the coefficients for the affine space where x=x_std belongs.

        The equation of the plane where x lives is
        a1*x1 + a2*x2 + z + d = 0 (coefficients: (a1, a2, 1, d));
        The corresponding affine space is
        z = f(x1,x2) = -a1.x1 -a2.x2 - d
        (coefficients: (a'1, a'2, d') = (-a1, -a2, -d));

        In this case z in d' + U, where d' + U is an affine subset,
        d' + U = {d' + u: u in U}, U = {a'1.x1, a'2.x2, a'1, a'2 in F}.

        Args:
            --> see get_plane_coefficients()
            use_saved_affine: use saved self.affine_coeff,
            resulting in a speedup.
        Returns:
            affine_coeff size: size (m, 3);
                {(a'i1, ai'i2, di')}_{i=1}^m = {(-ai1, -ai2, -di)}_{i=1}^m.
        """
        if use_saved_affine is True:
            try:
                lattice_mask = self.get_lattice_mask(x_lat)
                x_lat_in = x_lat[lattice_mask]
                idx_unique_triangles, _, _ = \
                    self.check_my_triangle(x_lat_in, **kwargs)

                affine_coeff = torch.zeros(x_lat.size(0), 3)
                affine_coeff[lattice_mask] = \
                    self.affine_coeff[idx_unique_triangles]

            except AttributeError:
                print('affine_coeff attribute still not created...')
                raise
        else:
            plane_coeff = self.get_plane_coefficients(x_lat, **kwargs)
            affine_coeff = self.get_affine_coeff_from_plane_coeff(plane_coeff)

        assert affine_coeff.size() == (x_lat.size(0), 3)

        return affine_coeff

    @staticmethod
    def get_affine_coeff_from_plane_coeff(plane_coeff):
        """ Get affine coefficients from plane coefficients

        Plane equation: a1*x1 + a2*x2 + z + d = 0
        (coefficients (a1, a2, 1, d));
        Affine space equation: z = f(x1,x2) = -a1*x1 -a2*x2 -d
        (coefficients: (a'1, a'2, d') = (-a1, -a2, -d))
        """
        return plane_coeff.index_select(1, torch.tensor([0, 1, 3])).mul(-1)

    # ==== Check My Triangle =============================================== #

    def check_my_triangle(self, x_lat, check_method='halfplane', **kwargs):
        """ Returns the lattice idx/coordinates of the vertices which form
            the triangle in which x_lat lives, using the halfplane or
            barycentric method.

        Args:
            x_lat: size(m, 2); batch of m 2-dimensional vectors in
                   lattice basis.
            check_method: 'halfplane' or 'barycentric'

        Returns:
            idx_unique_triangles: size (m,); index of triangle in
                        self.unique_triangles.
            vertices_lat: size (m,3,2); each element is an array of 3 vertices
                        which constitute the triangle in which x_lat[i] lives.
            second_mask: size (m,); mask which is one for each point belonging
                        to the second possible triangle.

        """
        assert check_method in ['halfplane', 'barycentric']
        assert x_lat.size(1) == self.ldim, f'found size {x_lat.size(1)}...'
        if not self.inside_lattice(x_lat):
            raise ValueError(
                'check_my_triangle(): x_lat should lie inside lattice.')

        x_lat_floor = x_lat.floor()
        x_lat_remainder = x_lat - x_lat_floor

        if check_method == 'halfplane':
            second_mask = self.get_halfplane_mask(x_lat_remainder)
        else:
            second_mask = self.get_barycentric_mask(x_lat_remainder)

        base_triangle_idx = \
            x_lat_floor[:, 1].long().sub(self.lmin) * (self.lsize * 2) + \
            x_lat_floor[:, 0].long().sub(self.lmin) * 2

        idx_unique_triangles = base_triangle_idx + second_mask
        vertices_lat = self.unique_triangles[idx_unique_triangles]

        return idx_unique_triangles, vertices_lat, second_mask

    @staticmethod
    def get_halfplane_mask(x_lat_remainder):
        """ Using the halfplane method, get the mask which is 1 for the
            tensors which live in the second triangle.

        Args:
            x_lat_remainder: size (m, 2); in lattice basis
        """
        # {x1} + {x2} - 1 >= 0 -> belongs to second triangle;
        # {xi} - remainder of xi
        half_plane = x_lat_remainder.sum(1).sub(1)  # (m,)
        second_mask = (half_plane > 0.).long()  # (m,)

        return second_mask

    def get_barycentric_mask(self, x_lat_remainder):
        """ Using the barycentric method, get the mask which is 1 for the
            tensors which live in the second triangle. This amounts to
            checking that all of the three barycentric coordinates wrt
            the second triangle are positive.

        Args:
            x_lat_remainder: size (m, 2); in lattice basis
        """
        # r - see wiki barycentric coordinate system
        # (m,3,1)
        r_mat = self.append_ones(x_lat_remainder, dim=-1).unsqueeze(-1)

        # x_barycentric_coordinates wrt second triangle
        inv_second_triangle_barycentric_mat = \
            self.batch_inv_origin_triangles_barycentric_mat[1, :, :]

        # (3,3) x (m,3,1) = (m,3,1) -> (m,3)
        x_lat_barycentric_coordinates = \
            (inv_second_triangle_barycentric_mat @ r_mat).squeeze(-1)

        num_positive_barycentric_coordinates = \
            (x_lat_barycentric_coordinates > 0.).sum(1)  # (m,)

        second_mask = (num_positive_barycentric_coordinates == 3).long()

        return second_mask

    # ==== Plane coefficients ============================================== #

    def get_plane_coefficients(self, x_lat, plane_method='solve', **kwargs):
        """ Get the coefficients for the plane where x lives.

        This function should only be called in get_affine_coefficients()
        to create self.affine_coeff tensor. After that, it is much faster
        to call get_affine_coefficients() and use self.affine_coeff.

        General plane equation at (x1, ..., xn, z):
        a1.x1 +...+ an.xn + c.z + d = 0; here, c is set to 1.

        Args:
            x_lat: size(m, 2); batch of m 2-dimensional vectors in
            lattice basis.
            plane_method: 'solve' or 'normal'.

        Returns:
            plane_coeff: size (m, 4); {(ai1, ai2, 1, di)}_{i=1}^m

        """
        assert plane_method in ['normal', 'solve']

        lattice_mask = self.get_lattice_mask(x_lat)
        x_lat_in = x_lat[lattice_mask]

        _, vertices_lat_in, _ = self.check_my_triangle(x_lat_in, **kwargs)
        vertices_lat_in_rel = vertices_lat_in.sub(self.lmin)

        vertices_values_in = \
            self.C_mat[vertices_lat_in_rel[:, :, 0],
                       vertices_lat_in_rel[:, :, 1]]

        vertices_size = vertices_lat_in.size()  # (m,3,2)
        # (m*3, 2) to convert all vertices to standard
        vertices_lat_in_reshape = vertices_lat_in.view(-1, 2)
        vertices_std_in = \
            self.lattice_to_standard(
                vertices_lat_in_reshape.float()).reshape(vertices_size)

        vertices_std_in_with_values = \
            torch.cat(
                (vertices_std_in,
                 vertices_values_in.unsqueeze(-1)),
                dim=-1)

        if plane_method == 'normal':
            plane_coeff_in = self.normal_method(vertices_std_in_with_values)
        else:
            plane_coeff_in = self.solve_method(vertices_std_in_with_values)

        plane_coeff = torch.tensor([0., 0., 1., 0.])\
            .unsqueeze(0).expand(x_lat.size(0), -1).clone()
        plane_coeff[lattice_mask] = plane_coeff_in

        # check that c = plane_coeff[:, 2] = 1
        assert (plane_coeff[:, 2]).allclose(torch.ones_like(plane_coeff[:, 2]))

        return plane_coeff

    @classmethod
    def normal_method(cls, vertices_std_with_values):
        """ Get plane coefficients for each datapoint from
        vertices_std_with_values, i.e., (x1,x2,z), using the normal method.

        Plane equation: a1*x1 +...+ an*xn + z + d = 0; normal: (a1,...,an,1)

        For n=2, the normal can be found using the outer product of the
        triangle indexes. If point (x1,x2) lives in triangle A,B,C, the normal
        (n1,n2,n3) can be obtained by doing the cross product ABxAC.
        After this, we normalize according to the third coordinate
        (n1/n3, n2/n3, 1), then choose a point in the plane where (x1,x2)
        lives and plug into the equation d = -(a1.x1 + a2.x2 + z).

        Args:
            vertices_std_with_values: size (m,3,3)
        """
        m = vertices_std_with_values.size(0)
        AB = vertices_std_with_values[:, 1, :] - \
            vertices_std_with_values[:, 0, :]  # (m,3)
        AC = vertices_std_with_values[:, 2, :] - \
            vertices_std_with_values[:, 0, :]  # (m,3)
        assert AB.size() == (m, 3)

        # AB x AC
        normals = torch.cross(AB, AC, dim=1)  # (m,3)
        c = normals[:, -1]
        # normalize normals by last coordinate
        # a1.x1 + a2.x2 + z + d = 0
        normalized_normals = normals.div(c.view(-1, 1))
        assert normalized_normals.size() == (m, 3)

        plane_coeff = \
            cls.get_plane_coeff_from_normals_vertices(normalized_normals,
                                                      vertices_std_with_values)

        return plane_coeff

    @classmethod
    def solve_method(cls, vertices_std_with_values):
        """ Get plane coefficients for each datapoint from
        vertices_std_with_values, i.e., (x1,x2,z), using the solve method.

        plane equation: a1.x1 +...+ an.xn + z + d = 0; normal: (a1,...,an,1)
        For n = 2, we have 3 vertices and 3 unkowns (a1, a2, d)
        for each datapoint.
        So, we can substitute these and solve the system of 3 equations
        to get (a1, a2, d).

        Args:
            vertices_std_with_values: size (m,3,3)
        """
        m = vertices_std_with_values.size(0)
        vertices_std = vertices_std_with_values[:, :, 0:2]
        vertices_values = vertices_std_with_values[:, :, 2]

        # vertices for (x1, x2, z) -> {(x1_K, x2_K, z_K)}, K in {A,B,C}
        # (x1-x1_C).a1 + (x2-x2_C).a2 + (z-z_C) = 0
        # plug (x1_A, x2_A, z_A) and (x1_B, x2_B, z_B) above to find a1, a2
        z_diff = (vertices_values[:, 0:2] -
                  vertices_values[:, 2:3]).unsqueeze(-1)
        assert z_diff.size() == (m, 2, 1)

        x_diff = vertices_std[:, 0:2, :] - vertices_std[:, 2:3, :]
        assert x_diff.size() == (m, 2, 2)

        a1_a2, _ = torch.solve(z_diff.mul(-1), x_diff)
        assert a1_a2.size() == (m, 2, 1)
        a1_a2 = a1_a2.squeeze(-1)
        assert a1_a2.size() == (m, 2)

        # normals are normalized by last coordinate
        normalized_normals = cls.append_ones(a1_a2, dim=-1)
        assert normalized_normals.size() == (m, 3)

        plane_coeff = \
            cls.get_plane_coeff_from_normals_vertices(normalized_normals,
                                                      vertices_std_with_values)

        return plane_coeff

    @staticmethod
    def get_plane_coeff_from_normals_vertices(normalized_normals,
                                              vertices_std_with_values):
        """
        Args:
            normalized_normals: size (m,3); normals for each datapoint
                                            (normalized by last coordinate);
            vertices_std_with_values: size (m,3,3).
        """
        m = vertices_std_with_values.size(0)
        # plug (x1A, x2A, zA) into plane equation to find d
        d = (normalized_normals *
             vertices_std_with_values[:, 0, :]).sum(1).mul(-1)
        assert d.size() == (m, )
        plane_coeff = torch.cat((normalized_normals, d.view(-1, 1)), dim=-1)
        assert plane_coeff.size() == (m, 4)

        return plane_coeff

    # ==== Coordinate transformations ====================================== #

    def lattice_to_standard(self, x_lat):
        """ Transform coordinates in lattice basis to standard basis

        Args:
            x_lat: size(m, n) batch of m n-dimensional vectors
            represented in the lattice basis
        Returns:
            x_standard: size(m, n) batch of m n-dimensional vectors
            represented in the standard basis

        The ith output vector corresponds to taking the linear combination
        of the columns of lattice matrix with scalars forming vector xi:
        xi[0]X_mat[:, 0] + xi[1]X_mat[:, 1] + ... + xi[n]X_mat[:,n]
        = X_mat @ xi
        """
        assert len(x_lat.size()) == 2, 'x_lat should be 2D'
        assert x_lat.size(1) == self.ldim, f'found size {x_lat.size(1)}...'

        X_mat = self.X_mat.to(x_lat.dtype)
        x_standard = (X_mat @ x_lat.t()).t()

        return x_standard

    def standard_to_lattice(self, x_standard):
        """ Transform coordinates in lattice basis to standard basis

        Args:
            x_standard: size(m, n) batch of m n-dimensional vectors
            represented in the standard basis
        Returns:
            x_lat: size(m, n) batch of m n-dimensional tensors vectors
            represented in the lattice basis

        The ith output vector corresponds to taking the linear combination
        of the columns of lattice matrix with scalars forming vector xi:
        xi[0]X_mat^(-1)[:, 0] + xi[1]X_mat^(-1)[:, 1] + ...
        + xi[n]X_mat^(-1)[:,n] = X_mat^(-1) @ xi
        """
        assert len(x_standard.size()) == 2, 'x_standard should be 2D'
        assert x_standard.size(1) == self.ldim, \
            f'found size {x_standard.size(1)}...'

        X_mat_inv = self.X_mat_inv.to(x_standard.dtype)
        x_lat = (X_mat_inv @ x_standard.t()).t()

        return x_lat

    # ==== Compute z = f(x1,x2) ============================================ #

    def get_values_from_interpolation(self, x_lat, **kwargs):
        """ Get values for x

        Args:
            x_lat: size (m, 2);
                   batch of m 2-dimensional vectors in lattice basis
            vertices_lat: size (m,3,2); vertices of triangle where each data
                        data point is located.
                        3x2 matrix of -1 if outside lattice;
        Returns:
            x_values, vertices_lat: size (m,), size (m, 3, 2)
        """
        lattice_mask = self.get_lattice_mask(x_lat)
        x_lat_in = x_lat[lattice_mask]

        m = x_lat_in.size(0)
        # sizes: ((m,3), (m,3,3))
        x_lat_in_barycentric_coordinates, vertices_lat_in = \
            self.get_barycentric_coordinates(x_lat_in, **kwargs)

        # rel = relative to bottom left corner ("origin")
        vertices_lat_in_rel = vertices_lat_in.sub(self.lmin)

        # TODO: Generalize next operation (e.g.)
        # C_mat[vertices_lat_rel.split(dim=-1)]
        vertices_values_in = \
            self.C_mat[vertices_lat_in_rel[:, :, 0],
                       vertices_lat_in_rel[:, :, 1]]

        assert vertices_values_in.size() == (m, 3)

        # batch inner product:
        # f(xi) = a1f(v1) + a2f(v2) + a3f(v3), i \in {1,...,m}
        # where v1, v2, v3 are the vertices of the triangle where xi lives
        x_values_in = (vertices_values_in *
                       x_lat_in_barycentric_coordinates).sum(1)  # (m,)
        assert x_values_in.size() == (m, )

        vertices_lat = torch.ones((x_lat.size(0), 3, 2),
                                  dtype=torch.int64).mul(-1)
        vertices_lat[lattice_mask] = vertices_lat_in

        x_values = torch.zeros(x_lat.size(0))
        x_values[lattice_mask] = x_values_in

        return x_values, vertices_lat

    def get_barycentric_coordinates(self, x_lat, in_origin=True, **kwargs):
        """ Get barycentric coordinates for x_lat

        Args:
            x_lat: size(m, 2); batch of m 2-dimensional vectors in
                   lattice basis
            in_origin: Compute barycentric coordinates in origin instead
                of in original location (should give the same result,
                but be faster, since we don't need to compute inverses)

        Returns:
            x_lat_barycentric_coordinates: size (m, 3);
            vertices_lat: size (m, 3, 2);
                          vertices of triangles where x_lat lives.
        """
        if not self.inside_lattice(x_lat):
            raise ValueError('get_barycentric_coordinates(): '
                             'x_lat should lie inside lattice.')

        m = x_lat.size(0)
        # vertices size (m,3,2), second_mask size (m,)
        _, vertices_lat, second_mask = self.check_my_triangle(x_lat, **kwargs)

        if in_origin is True:
            x_lat_remainder = x_lat - x_lat.floor()
            batch_r = self.append_ones(x_lat_remainder, dim=-1)  # (m,3)
            batch_R_mat_inv = \
                self.batch_inv_origin_triangles_barycentric_mat[second_mask]
        else:
            warnings.warn('in_origin=False is poorly conditioned. '
                          'Prefer in_origin=True.')
            batch_r = self.append_ones(x_lat, dim=-1)  # (m,3)
            batch_R_mat = self.append_ones(vertices_lat,
                                           dim=-1).transpose(1, 2)  # (m,3,3)
            batch_R_mat_inv = torch.inverse(batch_R_mat)  # (m,3,3)

        assert batch_r.size() == (m, 3)
        assert batch_R_mat_inv.size() == (m, 3, 3)

        # (m,3,3) x (m,3,1) = (m,3,1) -> squeeze -> (m,3)
        x_lat_barycentric_coordinates = \
            (batch_R_mat_inv @ batch_r.unsqueeze(-1)).squeeze(-1)
        assert x_lat_barycentric_coordinates.size() == (m, 3)

        return x_lat_barycentric_coordinates, vertices_lat

    def get_values_from_affine_coefficients(self, x_lat, **kwargs):
        """ Get values for x from affine coefficients a'1, a'2, d'
        z = f(x1,x2) = a'1.x1 + a'2.x2 + d'.

        Args:
            x_lat: size(m, 2); batch of m 2-dimensional vectors in
            lattice basis
        Returns:
            x_values: size (m,)
        """
        affine_coeff = self.get_affine_coefficients(x_lat, **kwargs)  # (m, 3)
        # get (a'1, a'2, d')
        al1, al2, dl = \
            affine_coeff[:, 0], affine_coeff[:, 1], affine_coeff[:, 2]

        x_std = self.lattice_to_standard(x_lat)
        x_values = al1 * x_std[:, 0] + al2 * x_std[:, 1] + dl

        return x_values

    # ====================================================================== #

    @property
    def flattened_C(self):
        """ Flatten C_mat, moving first along rows (x-direction)
        and then columns (y-direction).
        """
        return self.C_mat.t().flatten()

    def C_mat_to_flattened_C(self, C_mat):
        """ """
        return C_mat.t().flatten()

    def flattened_C_to_C_mat(self, flattened_C, C_mat_size=None):
        """ """
        mat_size = self.C_mat.size() if C_mat_size is None else C_mat_size
        return flattened_C.reshape(mat_size).t()

    def flat_idx_to_C_idx(self, flat_idx, interior=False, **kwargs):
        """ """
        eps = 0.01
        if interior:
            # flat_idx corresponds to interior lattice
            y_idx_aux = (flat_idx.float() + eps).div(self.lsize - 1).long()
            x_idx = (flat_idx - (y_idx_aux * (self.lsize - 1))).add(1)
            y_idx = y_idx_aux.add(1)
        else:
            y_idx = (flat_idx.float() + eps).div(self.lsize + 1).long()
            x_idx = flat_idx - (y_idx * (self.lsize + 1))

        return torch.cat((x_idx.view(-1, 1), y_idx.view(-1, 1)), dim=1)

    def flat_idx_to_x(self, flat_idx, return_basis='standard', **kwargs):
        """ """
        assert return_basis in ['lattice', 'standard']

        C_idx = self.flat_idx_to_C_idx(flat_idx, **kwargs)
        x_lat = C_idx + self.lmin

        if return_basis == 'lattice':
            return x_lat
        else:
            return self.lattice_to_standard(x_lat.float())

    def is_padded(self, pad_size=1):
        """ Check if lattice is padded with pad=pad_size zeros.
        """
        p = pad_size
        x_zeros = self.C_mat.new_zeros((p, self.C_mat.size(0)))
        y_zeros = x_zeros.t()

        if self.C_mat[0:p, :].allclose(x_zeros) and \
           self.C_mat[:, 0:p].allclose(y_zeros) and \
           self.C_mat[-p::, :].allclose(x_zeros) and \
           self.C_mat[:, -p::].allclose(y_zeros):

            return True

        return False

    def pad_lattice(self, pad_size=1):
        """ Check if lattice is padded with pad=pad_size zeros.
        """
        p = pad_size
        new_size = self.C_mat.size(0) + 2 * p
        new_C_mat = self.C_mat.new_zeros((new_size, new_size))
        new_C_mat[p:-p, p:-p] = self.C_mat
        self.C_mat = new_C_mat

    def save(self, save_str, lattice_dict=None, **kwargs):
        """ """
        if lattice_dict is None:
            lattice_dict = {}
        if save_str not in lattice_dict:
            lattice_dict[save_str] = {}

        lattice_dict[save_str]['X_mat'] = self.X_mat.clone()
        lattice_dict[save_str]['C_mat'] = self.C_mat.clone()

        return lattice_dict

    # ====================================================================== #

    def divide_lattice(self):
        """ Divide lattice """
        new_X_mat = self.X_mat.div(2)  # divide lattice vectors by 2
        new_lsize = self.lsize * 2

        new_C_mat = self.C_mat.new_zeros((new_lsize + 1, new_lsize + 1))

        # fill borders by piecewise linear interpolation
        xp = np.arange(0, self.lsize + 1) * 2
        x = np.arange(0, self.lsize * 2 + 1)
        new_C_mat[0, :] = \
            torch.from_numpy(np.interp(x, xp, self.C_mat[0, :].numpy()))
        new_C_mat[:, 0] = \
            torch.from_numpy(np.interp(x, xp, self.C_mat[:, 0].numpy()))
        new_C_mat[-1, :] = \
            torch.from_numpy(np.interp(x, xp, self.C_mat[-1, :].numpy()))
        new_C_mat[:, -1] = \
            torch.from_numpy(np.interp(x, xp, self.C_mat[:, -1].numpy()))

        # fill interior by piecewise linear interpolation
        grid = Grid(self.lmin + 0.5,
                    self.lmax - 0.1,
                    h=0.5,
                    square=True,
                    to_numpy=False,
                    to_float32=True)

        int_x_values, _ = self.get_values_from_interpolation(grid.x)
        int_new_C_mat = int_x_values.reshape(grid.meshgrid_size).t()
        new_C_mat[1:-1, 1:-1] = int_new_C_mat

        # Apply changes
        self.update_lattice(new_X_mat, new_C_mat)

    def convert_to_standard_basis(self, *datasets):
        """ Convert data to standard coordinates if it is in
        lattice coordinates.
        """
        for dataset in datasets:
            try:
                dataset['input'] = self.lattice_to_standard(dataset['input'])

            except KeyError:
                print(f'{dataset} does not contain "input" key.')
                raise

    @staticmethod
    def box_spline(x_lat, **kwargs):
        """ Returns k[i] = k(xi1,xi2) for each element in x_lat, where k
        is the box spline function.
        """
        assert x_lat.dtype == torch.float32
        assert x_lat.size(1) == 2, f'found size {x_lat.size(1)} ...'

        a1, a2 = x_lat[:, 0].view(-1, 1), -x_lat[:, 1].view(-1, 1)
        zeros_vec = torch.zeros_like(a1)

        f1 = torch.cat((zeros_vec, a1, a2), dim=-1)

        f1_min, _ = f1.min(dim=1)
        f1_max, _ = f1.max(dim=1)

        k = F.relu(1 - f1_max + f1_min)

        return k
