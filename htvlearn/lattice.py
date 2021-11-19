import math
import torch
from torch import Tensor
import numpy as np
import warnings
from django.utils.functional import classproperty

from htvlearn.grid import Grid


class Lattice():
    r"""
    Class for cpwl functions ``f:\R² \to \R`` with a uniform hexagonal lattice
    domain. This class of cpwl functions is in the span of linear hexagonal
    box splines shifted on the hexagonal lattice. A function in this space is
    uniquely specified by the values at the lattice vertices (box spline
    coefficients).

    The set of lattice vertices A for a centered hexagonal lattice of size
    (lsize x lsize) with ``(lsize + 1)**2`` vertices is given by:
    ``
    A = {a1*v1 + a2*v2: -lsize//2 <= ai < lsize//2, ai \in \Z, v1,v2 in R^2}
      = {X_mat @ a : -lsize//2 <= ai < lsize//2, ai \in \Z, X_mat in R^{2x2}}
    ``
    where ``v1 = h*(1, 0), v2 = h*(1/2, sqrt(3)/2)`` and ``h`` is the grid
    step.

    X_mat changes coordinates in the lattice basis v1,v2 \in R^2 to the
    standard basis e1,e2; X_mat's columns are formed by the lattice vectors
    v1,v2. Each lattice vertex is associated to a value corresponding to
    the coefficient of the box spline at that location.

    Tensor representation formats:
    . batch format (m,n) (default) - vectors along rows
    . mat (matrix) format (n,m) - vectors along columns
    . batch_mat format - batch of matrices (b,n,m)
    """

    ldim = 2  # lattice dimension
    # barycentric coordinates for simplex centers (approximately)
    centers_barycentric_coordinates = Tensor([0.333, 0.333, 0.334])
    # hexagonal lattice matrix with unit vectors
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
            Option 1 (direct) --- X_mat (lattice vectors) and the matrix of
            lattice coefficients C_mat are given directly. These two matrices
            need to satisfy several conditions (see self.verify_input).
                X_mat (torch.Tensor):
                    lattice matrix; size (2, 2).
                C_mat (torch.Tensor):
                    coefficients matrix; size (lsize+1, lsize+1);
                    C_mat(i1, i2) = f(a1*x1 + a2*x2) = f(X_mat @ a),
                    x direction -> rows, y -> columns.

            Option 2 (indirect) --- define X_mat via lsize and initialize
            C_mat with zeros or values from a normal distribution.
                lsize (int):
                    Lattice size; number of vertices is ``(lsize+1)**2``.
                C_init (str):
                    'zero' or 'normal' lattice coefficients initialization.
        """
        # C_mat is indexed as (i, j) = (x, y)
        if X_mat is not None and C_mat is not None:
            self.verify_input(X_mat, C_mat)
        else:
            assert X_mat == C_mat, \
                'X_mat is None and C_mat is not None, or vice-versa.'
            X_mat, C_mat = self.init_hexagonal_lattice(lsize, C_init, **kwargs)

        self.init_origin_simplices()
        self.init_neighborhood()
        self.update_lattice(X_mat, C_mat)

    @property
    def X_mat(self):
        return self._X_mat

    @X_mat.setter
    def X_mat(self, new_X_mat):
        raise AttributeError('X_mat can only be updated via'
                             'self.update_lattice().')

    @property
    def C_mat(self):
        return self._C_mat

    @C_mat.setter
    def C_mat(self, new_C_mat):
        raise AttributeError('C_mat can only be updated via self.update_'
                             'lattice() or self.update_coefficients().')

    @property
    def lsize(self):
        return self._lsize

    @lsize.setter
    def lsize(self, new_lsize):
        raise AttributeError('lsize can only be set in __init__().')

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, new_h):
        raise AttributeError('h cannot be set directly.')

    @classmethod
    def init_hexagonal_lattice(cls, lsize=10, C_init='zero', **kwargs):
        """
        Initialize hexagonal lattice.

        Args:
            lsize (int):
                Lattice size; number of vertices is ``(lsize+1)**2``.
            C_init (str):
                'zero' or 'normal' lattice coefficients initialization.
        """
        assert C_init in ['zero', 'normal']

        h = 1. / lsize
        X_mat = cls.hexagonal_matrix.mul(h)
        C_mat = torch.zeros((lsize + 1, lsize + 1))
        if C_init == 'normal':
            C_mat.normal_().mul_(0.05)  # for visualization purposes.

        assert cls.is_hexagonal(X_mat), 'Lattice is not hexagonal'

        return X_mat, C_mat

    # These two class properties require that lattice
    # is hexagonal and lsize*h = 1 (both enforced)
    @classproperty
    def bottom_right_std(cls):
        """Get the bottom right corner of lattice in standard coordinates"""
        return cls.hexagonal_matrix @ Tensor([0.5, -0.5])

    @classproperty
    def upper_right_std(cls):
        """Get the upper right corner of lattice in standard coordinates"""
        return cls.hexagonal_matrix @ Tensor([0.5, 0.5])

    @staticmethod
    def is_hexagonal(X_mat):
        """
        Check if lattice is hexagonal.

        Args:
            X_mat (torch.Tensor):
                lattice matrix; size (2, 2).
        """
        if not np.allclose(X_mat[1, 0], 0.):
            return False
        if not np.allclose(X_mat[0, 1], X_mat[0, 0] / 2):
            return False
        if not np.allclose(X_mat[1, 1], X_mat[0, 1] * math.sqrt(3)):
            return False

        return True

    @classmethod
    def verify_input(cls, X_mat, C_mat):
        """
        Verify lattice matrix and coefficients.

        Check if X_mat is hexagonal, that sizes are compatible, and that
        the resulting lsize * h = 1. Throw an exception if not.

        Args:
            X_mat (torch.Tensor):
                lattice matrix; size (2, 2).
            C_mat (torch.Tensor):
                coefficients matrix; size (lsize+1, lsize+1);
                C_mat(i1, i2) = f(a1*x1 + a2*x2) = f(X_mat @ a),
                x direction -> rows, y -> columns.
        """
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

    def init_origin_simplices(self):
        """
        Initialize the first and second origin simplices, their barycentric
        matrices and respective inverses. The rows of each simplex are the
        lattice coordinates of the vertices.

        Some computations, such as computation of barycentric coordinates
        of x wrt to its simplex can be translated to computation of
        barycentric coordinates of translated x wrt one of the origin
        simplices. (see get_barycentric_coordinates())
        """
        # vertices of origin square
        vA, vB, vC, vD = [0, 0], [1, 0], [1, 1], [0, 1]
        self.batch_origin_simplices = torch.tensor([[vA, vB, vD],
                                                    [vB, vC, vD]])  # (2,3,2)
        assert self.batch_origin_simplices.size() == (2, 3, 2)

        # size (2,3,3); for each of the 2 simplices (dim=0),
        # get barycentric matrices
        self.batch_origin_simplices_barycentric_mat = \
            self.append_ones(self.batch_origin_simplices,
                             dim=-1).transpose(1, 2)

        # batch inverse - size (2,3,3)
        self.batch_inv_origin_simplices_barycentric_mat = \
            torch.inverse(self.batch_origin_simplices_barycentric_mat)

        assert self.batch_inv_origin_simplices_barycentric_mat.size() == \
            (2, 3, 3)

    def init_neighborhood(self):
        """
        Initialize neighborhood steps for given vertex, such that
        ``v + self.neighborhood_diff`` gives the lattice coordinates of its
        neighbors for the HTV computation.
        """
        v_Ma, v_Mb, v_Mc = [0, 0], [1, 0], [0, 1]
        v_A, v_B, v_C = [1, 1], [-1, 1], [1, -1]

        # size: (6,2)
        self.neighborhood_diff = \
            torch.tensor([v_Ma, v_Mb, v_Mc, v_A, v_B, v_C])
        self.neighborhood_size = self.neighborhood_diff.size(0)

    def update_coefficients(self, new_C_mat):
        """
        Update the lattice values and the information concerning the function
        simplices.

        Args:
            new_C_mat (torch.Tensor):
                new coefficients matrix; size (lsize+1, lsize+1);
                new_C_mat(i1, i2) = f(a1*x1 + a2*x2) = f(X_mat @ a),
                x direction -> rows, y -> columns.
        """
        assert new_C_mat.size() == self._C_mat.size()
        self._C_mat = new_C_mat
        self.init_unique_simplices()
        self.init_affine_coefficients()

    def update_lattice(self, new_X_mat, new_C_mat):
        """
        Update_lattice matrix and coefficients.

        Also perform updates which are always required when X_mat or C_mat
        are changed; e.g., after dividing lattice.

        Args:
            new_X_mat (torch.Tensor):
                new lattice matrix; size (2, 2).
            new_C_mat (torch.Tensor):
                new coefficients matrix; size (lsize+1, lsize+1);
                C_mat(i1, i2) = f(a1*x1 + a2*x2) = f(X_mat @ a),
                x direction -> rows, y -> columns.
        """
        self._h = new_X_mat[0, 0]

        self._X_mat, self._C_mat = new_X_mat, new_C_mat
        # so that inversion does not have to be repeated
        self._X_mat_inv = torch.inverse(new_X_mat)

        self._lsize = new_C_mat.size(0) - 1
        if self._lsize % 2 != 0:
            raise ValueError(f'lsize {self._lsize} should be even...')

        # centered lattice; bottom left lattice coordinates: (lmin, lmin)
        self.lmin = -(self._lsize // 2)
        self.lmax = self.lmin + self._lsize

        # list of (x, y) locations of vertices in lattice coordinates
        self.lattice_grid = Grid(self.lmin,
                                 self.lmax + 1,
                                 h=1,
                                 square=True,
                                 to_numpy=False).x

        assert self.lattice_grid.size() == ((self._lsize + 1)**2, 2)

        # Now we get three corresponding tensors:
        # self.unique_simplices[i] <-> self.simplex_centers[i]
        # <-> self.affine_coeff[i]
        self.init_unique_simplices()
        self.init_simplex_centers()
        self.init_affine_coefficients()

    def get_lattice_mask(self, x_lat, pad=0):
        """
        Get mask of points inside lattice to access as
        x_lat[in_lattice_mask].

        Args:
            x_lat (torch.Tensor):
                input in lattice coordinates; size (m, 2).

        Returns:
            lattice_mask (torch.bool):
                signals which points are inside lattice; size: (m,).
        """
        x_lat_min, _ = x_lat.min(1)  # minimum along dim=1
        x_lat_max, _ = x_lat.max(1)  # maximum along dim=1

        lattice_mask = ((x_lat_min >= self.lmin + pad) *
                        (x_lat_max <= self.lmax - pad))

        return lattice_mask

    def inside_lattice(self, x_lat):
        """
        Check that all datapoints in x_lat lie inside lattice.

        Args:
            x_lat (torch.Tensor):
                input in lattice coordinates; size (m, 2).

        Returns:
            True if all datapoints in x_lat are inside lattice.
        """
        lattice_mask = self.get_lattice_mask(x_lat)
        return lattice_mask.all()

    def init_unique_simplices(self):
        """
        Initialize tensor with lattice simplices, without repetition.

        saves self.unique_simplices which is of size (k, 3). Each row i gives
        the indices of the vertices of simplex i <= k.
        """
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

        # (m, 2, 3, 2)
        unique_simplices_aux = x_lat_expand + self.batch_origin_simplices
        self.unique_simplices = unique_simplices_aux.view(m * 2, 3, 2)
        assert self.unique_simplices.size(0) == (self._lsize**2) * 2

        # index relative to bottom left vertice
        # size (m, 3, 2)
        unique_simplices_rel = self.unique_simplices.sub(self.lmin)

        # values of vertices of each unique simplex
        self.unique_simplices_values = \
            self._C_mat[unique_simplices_rel[:, :, 0],
                        unique_simplices_rel[:, :, 1]]

        assert self.unique_simplices_values.size() == \
            (self.unique_simplices.size(0), 3)

    def init_simplex_centers(self):
        """
        Initialize tensor with centers of the lattice simplices, without
        repetition, in lattice coordinates.

        saves self.simplex_centers which is of size (k, 2). Each row i gives
        the centers of simplex i <= k.
        """
        try:
            self.simplex_centers = \
                self.get_simplex_centers(self.unique_simplices)
        except AttributeError:
            print(
                'Need to initialize unique_simplices before simplex_centers.')
            raise

    @classmethod
    def get_simplex_centers(cls, simplices):
        """
        Get tensor with centers of the lattice simplices, without repetition,
        in lattice coordinates.

        Args:
            simplices (torch.int64):
                size: (k, 3).

        Returns:
            xcenter (torch.Tensor):
                Each row i gives the centers of simplex i <= k; size (k, 2).
        """
        # unique_simplices size (k,3,2)
        k = simplices.size(0)
        batch_xcenter_barycentric_coordinates = \
            cls.centers_barycentric_coordinates.view(1, 3).expand(k, 3)
        assert batch_xcenter_barycentric_coordinates.size() == (k, 3)

        simplices_mat = simplices.transpose(1, 2).float()
        assert simplices_mat.size() == (k, 2, 3)
        xcenter = (
            simplices_mat
            @ batch_xcenter_barycentric_coordinates.unsqueeze(-1)).squeeze(-1)
        assert xcenter.size() == (k, 2)

        return xcenter

    @classmethod
    def append_ones(cls, x, dim=0):
        """
        Append ones to ``x`` along dimension ``dim``.

        Useful for constructing a barycentric matrix.

        Args:
            dim (int>=0):
                dimension along which to concatenate with ones vector.
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

    # ==== Affine subsets coefficients ===================================== #

    def init_affine_coefficients(self):
        """
        Initialize affine coefficients of each simplex in the lattice.

        The affine coefficients completely specify the function in each
        simplex, i.e., f(x1,x2) = a1.x1 + a2.x2 + d, where (a1, a2, d) are the
        affine coefficients for a given simplex.
        """
        try:
            x_lat = self.simplex_centers
        except AttributeError:
            print('Need to initialize simplex_centers before affine_coeff.')
            raise

        self.affine_coeff = \
            self.get_affine_coefficients(x_lat, use_saved_affine=False)

    def get_affine_coefficients(self, x_lat, use_saved_affine=True, **kwargs):
        """
        Get the affine coefficients for the simplex to which x_lat belongs to.

        Plane equation: a1*x1 + a2*x2 + z + d = 0
        (plane coefficients: (a1, a2, 1, d));
        Affine space equation: z = f(x1,x2) = -a1*x1 -a2*x2 -d
        (affine coefficients: (a'1, a'2, d') = (-a1, -a2, -d));

        Args:
            x_lat (torch.Tensor):
                input in lattice coordinates; size (m, 2).
            use_saved_affine (bool):
                use saved self.affine_coeff, resulting in a speedup.

        Returns:
            affine_coeff (torch.Tensor):
                size: (m, 3).
        """
        if use_saved_affine is True:
            try:
                lattice_mask = self.get_lattice_mask(x_lat)
                x_lat_in = x_lat[lattice_mask]
                idx_unique_simplices, _, _ = \
                    self.check_my_simplex(x_lat_in, **kwargs)

                affine_coeff = torch.zeros(x_lat.size(0), 3)
                affine_coeff[lattice_mask] = \
                    self.affine_coeff[idx_unique_simplices]

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
        """
        Get affine coefficients from plane coefficients.

        Plane equation: a1*x1 + a2*x2 + z + d = 0
        (plane coefficients (a1, a2, 1, d));
        Affine space equation: z = f(x1,x2) = -a1*x1 -a2*x2 -d
        (affine coefficients: (a'1, a'2, d') = (-a1, -a2, -d))

        Args:
            plane_coeff (torch.Tensor):
                size: (m, 4).

        Returns:
            affine_coeff (torch.Tensor):
                size: (m, 3).
        """
        return plane_coeff.index_select(1, torch.tensor([0, 1, 3])).mul(-1)

    # ==== Check My Simplex =============================================== #

    def check_my_simplex(self, x_lat, **kwargs):
        """
        Get the lattice idx/coordinates of the vertices which form
        the simplex in which x_lat lives, using the halfplane method.

        Args:
            x_lat (torch.Tensor):
                input in lattice coordinates; size (m, 2).

        Returns:
            idx_unique_simplices (torch.int64):
                indices (in self.unique_simplices) of simplices in which
                x_lat lives; size (m,).
            vertices_lat (torch.int64):
                each element is an array of 3 vertices which constitute
                the simplex in which x_lat[i] lives; size (m, 3, 2).
            second_mask (torch.int64):
                mask which is one for each point belonging to the second
                possible simplex; size (m,).
        """
        assert x_lat.size(1) == self.ldim, f'found size {x_lat.size(1)}...'
        if not self.inside_lattice(x_lat):
            raise ValueError(
                'check_my_simplex(): x_lat should lie inside lattice.')

        # compute fractional part of x_lat
        x_lat_floor = x_lat.floor()
        x_lat_fractional = x_lat - x_lat_floor

        second_mask = self.get_halfplane_mask(x_lat_fractional)

        base_simplex_idx = \
            x_lat_floor[:, 1].long().sub(self.lmin) * (self._lsize * 2) + \
            x_lat_floor[:, 0].long().sub(self.lmin) * 2

        idx_unique_simplices = base_simplex_idx + second_mask
        vertices_lat = self.unique_simplices[idx_unique_simplices]

        return idx_unique_simplices, vertices_lat, second_mask

    @staticmethod
    def get_halfplane_mask(x_lat_fractional):
        """
        Use the halfplane method to get the mask which is 1 for the
        tensors which live in the second simplex and 0 otherwise.

        Args:
            x_lat_fractional (torch.Tensor):
                fractional part of x_lat in lattice basis; size (m, 2).

        Returns:
            second_mask (torch.int64):
                mask which is one for each point belonging to the second
                possible simplex; size (m,).
        """
        # {x1} + {x2} - 1 >= 0 -> belongs to second simplex;
        # {xi} - fractional of xi
        half_plane = x_lat_fractional.sum(1).sub(1)  # (m,)
        second_mask = (half_plane > 0.).long()  # (m,)

        return second_mask

    # ==== Plane coefficients ============================================== #

    def get_plane_coefficients(self, x_lat, **kwargs):
        """
        Get the plane coefficients for the simplex to which x_lat belongs to.

        Plane equation: a1*x1 + a2*x2 + z + d = 0
        (plane coefficients: (a1, a2, 1, d)).
        This function should only be called in get_affine_coefficients()
        to create self.affine_coeff tensor. After that, it is much faster
        to call get_affine_coefficients() with use_saved_affine=True to
        retrieve self.affine_coeff directly.

        Args:
            x_lat (torch.Tensor):
                input in lattice coordinates. size (m, 2)

        Returns:
            plane_coeff (torch.Tensor):
                size: (m, 4).
        """
        lattice_mask = self.get_lattice_mask(x_lat)
        x_lat_in = x_lat[lattice_mask]

        _, vertices_lat_in, _ = self.check_my_simplex(x_lat_in, **kwargs)
        vertices_lat_in_rel = vertices_lat_in.sub(self.lmin)

        vertices_values_in = \
            self._C_mat[vertices_lat_in_rel[:, :, 0],
                        vertices_lat_in_rel[:, :, 1]]

        vertices_size = vertices_lat_in.size()  # (m, 3, 2)
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

        plane_coeff_in = self.solve_method(vertices_std_in_with_values)
        plane_coeff = torch.tensor([0., 0., 1., 0.])\
            .unsqueeze(0).expand(x_lat.size(0), -1).clone()
        plane_coeff[lattice_mask] = plane_coeff_in

        # check that c = plane_coeff[:, 2] = 1
        assert (plane_coeff[:, 2]).allclose(torch.ones_like(plane_coeff[:, 2]))

        return plane_coeff

    @classmethod
    def solve_method(cls, vertices_std_with_values):
        """
        Get plane coefficients for each datapoint from vertices_std_with
        _values, i.e., (x1,x2,z), using the 'solve' method.

        Plane equation: a1*x1 + a2*x2 + z + d = 0.
        We have 3 vertices and 3 unkowns (a1, a2, d) for each datapoint,
        so we can substitute these and solve the system of 3 equations
        to get (a1, a2, d).

        Args:
            vertices_std_with_values (torch.Tensor):
                concatenation of datapoint locations in standard coordinates
                and values; size (m, 3, 3).

        Returns:
            plane_coeff (torch.Tensor):
                size: (m, 4).
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

        a1_a2 = torch.linalg.solve(x_diff, z_diff.mul(-1))
        assert a1_a2.size() == (m, 2, 1)
        a1_a2 = a1_a2.squeeze(-1)
        assert a1_a2.size() == (m, 2)

        # normals are normalized by last coordinate
        normalized_normals = cls.append_ones(a1_a2, dim=-1)
        assert normalized_normals.size() == (m, 3)

        plane_coeff = cls.get_plane_coeffs_from_normals_vertices(
            normalized_normals, vertices_std_with_values)

        return plane_coeff

    @staticmethod
    def get_plane_coeffs_from_normals_vertices(normalized_normals,
                                               vertices_std_with_values):
        """
        Get plane coefficients from the normalized simplex normals
        and the vertices.

        Args:
            normalized_normals (torch.Tensor):
                normals for each datapoint normalized by last coordinate,
                i.e., of the form (a1, a2, 1); size (m, 3).
            vertices_std_with_values (torch.Tensor):
                concatenation of datapoint locations in standard coordinates
                and values; size (m, 3, 3).

        Returns:
            plane_coeff (torch.Tensor):
                size: (m, 4).
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
        """
        Transform x_lat coordinates in lattice basis to coordinates in
        standard basis.

        Args:
            x_lat (torch.Tensor):
                input in lattice basis; size(m, 2).
        Returns:
            x_standard (torch.Tensor):
                input in standard basis; size(m, 2).

        The ith output vector corresponds to taking the linear combination
        of the columns of lattice matrix with scalars forming vector xi:
        xi[0]X_mat[:, 0] + xi[1]X_mat[:, 1] = X_mat @ xi
        """
        assert len(x_lat.size()) == 2, 'x_lat should be 2D'
        assert x_lat.size(1) == self.ldim, f'found size {x_lat.size(1)}...'

        X_mat = self._X_mat.to(x_lat.dtype)
        x_standard = (X_mat @ x_lat.t()).t()

        return x_standard

    def standard_to_lattice(self, x_standard):
        """
        Transform x_standard coordinates in standard basis to coordinates in
        lattice basis.

        Args:
            x_standard (torch.Tensor):
                input in standard basis; size(m, 2).
        Returns:
            x_lat (torch.Tensor):
                input in lattice basis; size(m, 2).

        The ith output vector corresponds to taking the linear combination
        of the columns of lattice matrix with scalars forming vector xi:
        xi[0]X_mat^(-1)[:, 0] + xi[1]X_mat^(-1)[:, 1] = X_mat^(-1) @ xi
        """
        assert len(x_standard.size()) == 2, 'x_standard should be 2D'
        assert x_standard.size(1) == self.ldim, \
            f'found size {x_standard.size(1)}...'

        X_mat_inv = self._X_mat_inv.to(x_standard.dtype)
        x_lat = (X_mat_inv @ x_standard.t()).t()

        return x_lat

    # ==== Compute z = f(x1,x2) ============================================ #

    def get_values_from_interpolation(self, x_lat, **kwargs):
        """
        Get values for each datapoint in x_lat from interpolation via
        barycentric coordinates in its simplex.

        Args:
            x_lat (torch.Tensor):
                input in lattice basis; size(m, 2).

        Returns:
            vertices_lat (torch.int64):
                each element is an array of 3 vertices which constitute
                the simplex in which x_lat[i] lives; size (m, 3, 2).
            x_values (torch.Tensor):
                size (m,).
        """
        lattice_mask = self.get_lattice_mask(x_lat)
        x_lat_in = x_lat[lattice_mask]

        m = x_lat_in.size(0)
        # sizes: ((m, 3), (m, 3, 3))
        x_lat_in_barycentric_coordinates, vertices_lat_in = \
            self.get_barycentric_coordinates(x_lat_in, **kwargs)

        # rel = relative to bottom left corner ("origin")
        vertices_lat_in_rel = vertices_lat_in.sub(self.lmin)

        # TODO: Generalize next operation (e.g.)
        # C_mat[vertices_lat_rel.split(dim=-1)]
        vertices_values_in = \
            self._C_mat[vertices_lat_in_rel[:, :, 0],
                        vertices_lat_in_rel[:, :, 1]]

        assert vertices_values_in.size() == (m, 3)

        # batch inner product:
        # f(xi) = a1f(v1) + a2f(v2) + a3f(v3), i \in {1,...,m}
        # where v1, v2, v3 are the vertices of the simplex where xi lives
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
        """
        Get the barycentric coordinates of each datapoint in x_lat.

        Args:
            x_lat (torch.Tensor):
                input in lattice basis; size(m, 2).
            in_origin (bool):
                If True, compute barycentric coordinates in origin instead
                of in original location (should give the same result,
                but be faster, since we don't need to compute inverses).

        Returns:
            x_lat_barycentric_coordinates (torch.Tensor):
                barycentric coordinates of each datapoint; size (m, 3).
            vertices_lat (torch.int64):
                each element is an array of 3 vertices which constitute
                the simplex in which x_lat[i] lives; size (m, 3, 2).
        """
        if not self.inside_lattice(x_lat):
            raise ValueError('get_barycentric_coordinates(): '
                             'x_lat should lie inside lattice.')

        m = x_lat.size(0)
        # vertices size (m, 3, 2), second_mask size (m,)
        _, vertices_lat, second_mask = self.check_my_simplex(x_lat, **kwargs)

        if in_origin is True:
            x_lat_fractional = x_lat - x_lat.floor()
            batch_r = self.append_ones(x_lat_fractional, dim=-1)  # (m, 3)
            batch_R_mat_inv = \
                self.batch_inv_origin_simplices_barycentric_mat[second_mask]
        else:
            warnings.warn('in_origin=False is poorly conditioned. '
                          'Prefer in_origin=True.')
            batch_r = self.append_ones(x_lat, dim=-1)  # (m, 3)
            batch_R_mat = self.append_ones(vertices_lat,
                                           dim=-1).transpose(1, 2)  # (m, 3, 3)
            batch_R_mat_inv = torch.inverse(batch_R_mat)  # (m, 3, 3)

        assert batch_r.size() == (m, 3)
        assert batch_R_mat_inv.size() == (m, 3, 3)

        # (m, 3, 3) x (m, 3, 1) = (m, 3, 1) -> squeeze -> (m, 3)
        x_lat_barycentric_coordinates = \
            (batch_R_mat_inv @ batch_r.unsqueeze(-1)).squeeze(-1)
        assert x_lat_barycentric_coordinates.size() == (m, 3)

        return x_lat_barycentric_coordinates, vertices_lat

    def get_values_from_affine_coefficients(self, x_lat, **kwargs):
        """
        Get values for each datapoint in x_lat from the affine coefficients of
        its simplex.

        affine coefficients: (a'1, a'2, d');
        z = f(x1,x2) = a'1.x1 + a'2.x2 + d'.

        Args:
            x_lat (torch.Tensor):
                input in lattice basis; size(m, 2).

        Returns:
            x_values (torch.Tensor):
                size (m,).
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
        """
        Get flattened coefficients from matrix, moving first along rows
        (x-direction) and then columns (y-direction).
        """
        return self._C_mat.t().flatten()

    def flattened_C_to_C_mat(self, flattened_C):
        """
        Get coefficients matrix from flattened coefficients.

        Args:
            flattened_C (torch.Tensor):
                flattened coefficients tensor; size (m, ).

        Returns:
            C_mat (torch.Tensor):
                coefficients matrix; size (n, n), such that n * n = m.
        """
        assert flattened_C.size() == (self._C_mat.size(0) *
                                      self._C_mat.size(1),)
        return flattened_C.reshape(self._C_mat.size()).t()

    def save(self, save_str, lattice_dict=None, **kwargs):
        """
        Save lattice in ``lattice_dict`` under ``save_str`` key.

        Args:
            save_str (str):
                key for saving lattice in dictionary.
            lattice_dict (dict or None):
                If not None, save lattice in this dictionary. Otherwise,
                initialize a new dictionary and save it there.

        Returns:
            lattice_dict (dict):
                dictionary with saved lattice.
        """
        if lattice_dict is None:
            lattice_dict = {}
        if save_str not in lattice_dict:
            lattice_dict[save_str] = {}

        lattice_dict[save_str]['X_mat'] = self._X_mat.clone()
        lattice_dict[save_str]['C_mat'] = self._C_mat.clone()

        return lattice_dict

    # ====================================================================== #

    def refine_lattice(self):
        """Refine the lattice into a finer scale with half the grid size."""
        new_X_mat = self._X_mat.div(2)  # divide lattice vectors by 2
        new_lsize = self._lsize * 2

        new_C_mat = self._C_mat.new_zeros((new_lsize + 1, new_lsize + 1))

        # fill borders by piecewise linear interpolation
        xp = np.arange(0, self._lsize + 1) * 2
        x = np.arange(0, self._lsize * 2 + 1)
        new_C_mat[0, :] = \
            torch.from_numpy(np.interp(x, xp, self._C_mat[0, :].numpy()))
        new_C_mat[:, 0] = \
            torch.from_numpy(np.interp(x, xp, self._C_mat[:, 0].numpy()))
        new_C_mat[-1, :] = \
            torch.from_numpy(np.interp(x, xp, self._C_mat[-1, :].numpy()))
        new_C_mat[:, -1] = \
            torch.from_numpy(np.interp(x, xp, self._C_mat[:, -1].numpy()))

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
