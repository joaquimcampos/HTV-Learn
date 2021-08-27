import torch
import math
from scipy import sparse


class Operators():

    # regularization vector before multiplication with (2*math.sqrt(3)/3)
    reg_mat = torch.tensor([[1, -1, -1, 1, 0, 0],
                            [-1, 1, -1, 0, 1, 0],
                            [-1, -1, 1, 0, 0, 1]])

    hexagonal_correction = 2 * math.sqrt(3) / 3

    def __init__(self, lattice_obj, input, lmbda=None, **kwargs):
        """
        Args:
            lattice_obj (Lattice):
                instance of Lattice class (lattice.py)
            input (torch.Tensor):
                training datapoints.
            lmbda (None or float):
                If not None, multiply regularization matrix by this value.
            """
        self.input = input
        self.lmbda = lmbda
        self.H_mat_sparse = self.get_forward_op(lattice_obj, self.input)
        self.L_mat_sparse = self.get_regularization_op(lattice_obj, self.lmbda)

    @staticmethod
    def get_forward_op(lat, input):
        """
        Construct sparse H in (||Hz - y||_2)^2 + ||Lz||_1 ;
        H is a sparse matrix of barycentric coordinates.

        This method does not depend on boundary conditions since all points
        are inside the interior lattice.

        Args:
            lattice_obj (Lattice):
                instance of Lattice class (lattice.py)
            input (torch.Tensor):
                training datapoints.

        Returns:
            H_mat_sparse (scipy.sparse.csr_matrix):
                Forward matrix with the 3 barycentric coordinates of each
                datapoint in each row. size: (input.size(0), num_vertices).
        """
        x_lat = lat.standard_to_lattice(input)
        # get in-lattice points
        # (f-g assumed to be in lattice where g is an affine function)
        lattice_mask = lat.get_lattice_mask(x_lat, pad=0)
        x_lat_in = x_lat[lattice_mask]
        m = x_lat_in.size(0)

        assert m == x_lat.size(0), f'inside: {m}, outside: {x_lat.size(0)}.'

        # sizes: (m, 3), (m, 3, 2)
        x_lat_in_barycentric_coordinates, vertices_lat_in = \
            lat.get_barycentric_coordinates(x_lat_in)

        # rel = relative to bottom left corner ("origin")
        vertices_lat_in_rel = vertices_lat_in.sub(lat.lmin)  # (k,3,2)

        # (m, 3)
        vertices_in_vectorized_idx = \
            vertices_lat_in_rel[:, :, 1] * (lat.lsize + 1) + \
            vertices_lat_in_rel[:, :, 0]

        assert vertices_in_vectorized_idx.size() == (x_lat_in.size(0), 3)

        col_idx_mat = vertices_in_vectorized_idx
        col_idx = col_idx_mat.view(-1)  # vectorized

        row_idx_mat = torch.arange(0, m).view(-1, 1).expand(col_idx_mat.size())
        row_idx = row_idx_mat.reshape(-1)
        assert row_idx.size() == col_idx.size()

        H_mat = x_lat_in_barycentric_coordinates
        H_vectorized_mat = H_mat.view(-1)
        assert H_vectorized_mat.size() == col_idx.size()

        num_vertices = lat.lattice_grid.size(0)
        H_mat_sparse = sparse.csr_matrix(
            (H_vectorized_mat.numpy(), (row_idx.numpy(), col_idx.numpy())),
            shape=(m, num_vertices))

        H_mat_sparse.eliminate_zeros()
        H_mat_sparse.check_format()

        return H_mat_sparse

    @classmethod
    def get_regularization_op(cls, lat, lmbda=None):
        """
        Construct sparse L (based on HTV regularization)
        in (||Hz - y||_2)^2 + lmbda*||Lz||_1.

        See calculations of HTV in the paper.
        If lmbda is None, does not multiply by it.

        Args:
            lattice_obj (Lattice):
                instance of Lattice class (lattice.py)
            lmbda (None or float):
                If not None, multiply regularization matrix by this value.

        Returns:
            L_mat_sparse (scipy.sparse.csr_matrix):
                regularization matrix. Contains 4 values in each row.
                If lmbda not None, it is multiplied by this value.
                size: (#junctions, num_vertices).
        """
        assert cls.reg_mat.size() == (3, lat.neighborhood_size)

        # For each location in C_mat get C_Ma, C_Mb, C_Mc, C_A, C_B, C_C
        # sizes: (num_vertices,1,2) + (6,2) = (num_vertices,6,2)
        neighborhood = lat.lattice_grid.unsqueeze(1) + \
            lat.neighborhood_diff
        # rel = relative to bottom left corner ("origin")
        neighborhood_rel = neighborhood.sub(lat.lmin)

        # boundary mask size: (num_vertices, 6)
        boundary_mask = (neighborhood_rel[:, :, 0] < 0) + \
                        (neighborhood_rel[:, :, 0] > lat.lsize) + \
                        (neighborhood_rel[:, :, 1] < 0) + \
                        (neighborhood_rel[:, :, 1] > lat.lsize)

        # assign to something impossible to signal that this is a boundary
        # vertice (in the edge of lattice or outside)
        neighborhood_rel[boundary_mask] = torch.tensor([-1, -1])

        # will be negative for the boundary ones. size: (num_vertices, 6)
        neighborhood_vectorized_idx = \
            neighborhood_rel[:, :, 1] * (lat.lsize + 1) + \
            neighborhood_rel[:, :, 0]

        # notation: neighb = neighbors
        # given t = num_vertices
        # row_idx_mat (ndim=2) =   col_idx_mat (ndim=2) =
        # [[0, ..., 0],            [[neighb_idx11, ..., neighb_idx16],
        #  1, ..., 1,              REPEAT vector above 2 more times,
        #     ...   ,                            ...
        #  t*3, ..., t*3]]         neighb_idxt1, ..., neighb_idxt6,
        #                          REPEAT vector above 2 more times]
        #
        # L_mat (ndim=2) =
        # [[1, -1, -1, 1, 0, 0],
        # [-1,  1, -1, 0, 1, 0,
        # [-1, -1,  1, 0, 0, 1],
        # REPEAT matrix above t times]

        # number of vertices
        num_vertices = lat.lattice_grid.size(0)

        # (num_vertices * 3, 6)
        L_mat = cls.reg_mat.repeat(num_vertices, 1)
        assert L_mat.size() == (num_vertices * 3, 6)

        # (num_vertices, 6) -> (num_vertices, 6*3) ->
        # (num_vertices*6*3,) -> (num_vertices*3, 6)
        col_idx_mat = neighborhood_vectorized_idx.repeat(1, 3)\
            .view(-1).reshape(L_mat.size())
        assert torch.equal(col_idx_mat[3], col_idx_mat[5])  # check

        # check if side corresponding row has an invalid vertice that
        # counts for HTV (L_mat != 0).
        # If L_mat is zero somewhere, it does not matter if there is
        # an outside-lattice vertice in that position. These will be
        # deleted later.
        invalid_side_mask = ((col_idx_mat < 0) * (L_mat != 0)).any(1)
        L_mat = L_mat[~invalid_side_mask]
        col_idx_mat = col_idx_mat[~invalid_side_mask]

        rows = torch.arange(0, col_idx_mat.size(0))
        row_idx_mat = rows.view(-1, 1).expand(col_idx_mat.size())

        L_vectorized_mat = L_mat.reshape(-1)
        col_idx = col_idx_mat.reshape(-1)
        row_idx = row_idx_mat.reshape(-1)
        assert col_idx.size() == L_vectorized_mat.size()
        assert row_idx.size() == L_vectorized_mat.size()

        # eliminate non-relevant invalid vertices
        invalid_vertices = (col_idx < 0)
        L_vectorized_mat = L_vectorized_mat[~invalid_vertices]
        col_idx = col_idx[~invalid_vertices]
        row_idx = row_idx[~invalid_vertices]

        zero_vertices = (L_vectorized_mat == 0)
        L_vectorized_mat = L_vectorized_mat[~zero_vertices]
        col_idx = col_idx[~zero_vertices]
        row_idx = row_idx[~zero_vertices]

        if lmbda is not None:
            # multiply by lmbda weight
            L_vectorized_mat = L_vectorized_mat.mul(lmbda)

        # multiply by hexagonal correction factor 2*sqrt(3)/3.
        L_vectorized_mat = L_vectorized_mat * cls.hexagonal_correction

        L_mat_shape = (L_mat.size(0), num_vertices)
        L_mat_sparse = sparse.csr_matrix(
            (L_vectorized_mat.numpy(), (row_idx.numpy(), col_idx.numpy())),
            shape=L_mat_shape)
        L_mat_sparse.eliminate_zeros()
        L_mat_sparse.check_format()

        return L_mat_sparse
