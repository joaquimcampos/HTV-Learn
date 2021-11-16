import torch
import numpy as np
import scipy.spatial

from htvlearn.lattice import Lattice
from htvlearn.grid import Grid
from htvlearn.hessian import (
    get_finite_second_diff_Hessian,
    get_exact_grad_Hessian
)


class Delaunay():
    """Wrapper around scipy.spatial.Delaunay"""

    centers_barycentric_coordinates = np.array([0.333, 0.333, 0.334])

    def __init__(self,
                 points=None,
                 values=None,
                 npoints=120,
                 points_range=2,
                 values_range=1,
                 add_extreme_points=False,
                 pad_factor=0.05,
                 **kwargs):
        """
        Args:
            points, values (torch.tensor/np.array):
                Specific set of points/values to generate Delaunay
                triangulation from. size: (n, 2), (n,), respectively.
                if not given, npoints are randomly generated.
            npoints (int):
                number of delaunay triangulation points to generate, if
                ``points`` and ``values`` are not given.
            points_range (float):
                range for generated points.
            values_range (float):
                range for generated values.
            add_extreme_points (bool):
                Add extreme points of the rectangle to triangulation.
            pad_factor (float):
                relative pad factor for extreme points
                (relative to data range).
        """
        super().__init__()

        if isinstance(points, torch.Tensor):
            points = points.numpy()
            values = values.numpy()

        if isinstance(points, np.ndarray):
            self._verify_input(points, values)
        else:
            interior_npoints = ((npoints -
                                 4) if add_extreme_points is True else npoints)
            # randomly generate points in [-points_range/2, points_range/2]^2
            points = ((np.random.rand(int(interior_npoints), 2) - 0.5) *
                      points_range)
            # randomly generate values in [-values_range/2, values_range/2]
            values = ((np.random.rand(int(interior_npoints), ) - 0.5) *
                      values_range)

        if add_extreme_points:
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            # pf -> absolute padding factor
            pf = min(x_max - x_min, y_max - y_min) * pad_factor
            # extreme points of domain
            extreme_points = np.array([[x_min - pf, y_min - pf],
                                       [x_max + pf, y_min - pf],
                                       [x_min - pf, y_max + pf],
                                       [x_max + pf, y_max + pf]])
            points = np.vstack((points, extreme_points))
            values = np.concatenate((values, np.zeros(4)))

        # generate delaunay triangulation
        self.tri = scipy.spatial.Delaunay(points)

        for attstr in [
                'values', 'simplices_points', 'simplices_values',
                'simplices_affine_coeff'
        ]:
            assert not hasattr(self.tri, attstr), \
                f'{self.tri} does not have attribute {attstr}...'

        # idx of convex hull points
        self.convex_hull_points_idx = np.unique(self.tri.convex_hull.flatten())
        # triangulation vertice values
        self.tri.values = values

        # sanity check
        # https://math.stackexchange.com/questions/
        # 1097646/number-of-triangles-in-a-triangulation
        nlinear_regions = self.tri.simplices.shape[0]
        assert nlinear_regions <= 2 * self.tri.points.shape[0] \
            - 2 - self.convex_hull_points_idx.shape[0]

        # coordinates of points in each simplex. size: (nsimplex, 3, 2).
        self.tri.simplices_points = self.tri.points[self.tri.simplices]
        assert self.tri.simplices_points.shape == \
            (self.tri.simplices.shape[0], 3, 2)
        # vertex values of points in each simplex. size: (nsimplex, 3, 2).
        self.tri.simplices_values = self.tri.values[self.tri.simplices]
        assert self.tri.simplices_values.shape == \
            (self.tri.simplices.shape[0], 3)

        simplices_centers = (
            np.transpose(self.tri.simplices_points, (0, 2, 1)) @
            self.centers_barycentric_coordinates[:, np.newaxis]
        ).squeeze(axis=-1)

        # affine coefficients of each simplex
        self.tri.simplices_affine_coeff = \
            self.get_affine_coefficients(simplices_centers)
        assert self.tri.simplices_affine_coeff.shape == \
            (self.tri.simplices.shape[0], 3)

    @property
    def is_admissible(self):
        """
        Check if the cpwl function is admissible, i.e., is of the form
        f + ax + b, where f is a zero-boundary function.
        """
        hull_points = self.tri.points[self.convex_hull_points_idx]
        hull_values = self.tri.values[self.convex_hull_points_idx]
        # fit a linear function through three convex hull points
        # solve f(hull_points) = a^(hull_points) + b
        vert = np.concatenate((hull_points,
                               hull_values[:, np.newaxis]),
                              axis=1)
        plane_coeff = Lattice.solve_method(torch.from_numpy(vert).unsqueeze(0))
        affine_coeff = Lattice.get_affine_coeff_from_plane_coeff(plane_coeff)
        affine_coeff = affine_coeff.numpy()
        z_linear = ((affine_coeff[:, 0:2] * hull_points).sum(axis=1) +
                    affine_coeff[:, 2])

        if np.allclose(z_linear, hull_values):
            return True

        return False

    @property
    def has_rectangular_range(self):
        """Check if convex hull of cpwl is rectangular."""
        hull_points = self.tri.points[self.convex_hull_points_idx]

        x_min = hull_points[:, 0].min()
        x_max = hull_points[:, 0].max()
        y_min = hull_points[:, 1].min()
        y_max = hull_points[:, 1].max()

        # test rectangle
        rect = np.array([[x_min, y_min],
                         [x_max, y_min],
                         [x_min, y_max],
                         [x_max, y_max]])

        # check that hull_points contains the 4 corners of the rectangle
        for corner in rect:
            # difference to a corner.
            diff = hull_points - np.array([corner])
            if not np.any(np.all(np.isclose(diff, 0), axis=1)):
                # there isn't any hull point corresponding to this corner
                return False

        # check that all points have
        # x = x_min, or x = x_max, or y = y_min, or y = ymax
        if not np.all(np.isclose(hull_points[:, 0], x_min) +
                      np.isclose(hull_points[:, 0], x_max) +
                      np.isclose(hull_points[:, 1], y_min) +
                      np.isclose(hull_points[:, 1], y_max)):
            return False

        return True

    @staticmethod
    def _verify_input(points, values):
        """
        Verify user input points and values.

        Args:
            points:
                expected np.ndarray of size (n, 2)
            values:
                expected np.ndarray of size (n,)
        """
        assert points.shape[1] == 2, f'{points.shape[1]}'
        assert len(points.shape) == 2, f'{len(points.shape)}'
        assert isinstance(values, np.ndarray), f'{type(values)}'
        assert values.shape == (points.shape[0], ), \
            f'{values.shape} != ({points.shape[0]},)'

    def get_affine_coefficients(self, x):
        """
        Get the affine coefficients (a1, a2, d) of simplex where each datapoint
        leaves, s.t. f(x1, x2) = a1.x1 + a2.x2 + d, using the solve method.

        We have 3 vertices and 3 unkowns (a1, a2, d) for each datapoint.
        So, we can solve the system of 3 equations to get (a1, a2, d).

        Args:
            x (np.ndarray):
                locations where to get affine coefficients. size: (n, 2).

        Returns:
            affine_coeff (np.ndarray):
                affine coefficients of simplices where each datapoint lives.
                size: (n, 3).
        """
        x_simplices, x_simplices_idx = self.get_x_simplices(x)
        try:
            affine_coeff = \
                self.tri.simplices_affine_coeff[x_simplices_idx].copy()
            return affine_coeff
        except AttributeError:
            pass

        x_simplices_points = \
            self.get_x_simplices_points(x_simplices=x_simplices)  # (m, 3, 2)
        x_simplices_values = \
            self.get_x_simplices_values(x_simplices=x_simplices)  # (m, 3)

        # vertices for (x1, x2, z) -> {(x1_K, x2_K, z_K)}, K in {A, B, C}
        # (x1 - x1_C).a1 + (x2 - x2_C).a2 + (z - z_C) = 0
        # plug (x1_A, x2_A, z_A) and (x1_B, x2_B, z_B) above to find a1, a2
        z_diff = (x_simplices_values[:, 0:2] -
                  x_simplices_values[:, 2:3])[:, :, np.newaxis]
        assert z_diff.shape == (x.shape[0], 2, 1)

        x_diff = x_simplices_points[:, 0:2, :] - x_simplices_points[:, 2:3, :]
        assert x_diff.shape == (x.shape[0], 2, 2)

        a1_a2 = np.linalg.solve(x_diff, z_diff)
        assert a1_a2.shape == (x.shape[0], 2, 1)
        a1_a2 = a1_a2.squeeze(-1)

        d = x_simplices_values[:, 2] - \
            (a1_a2 * x_simplices_points[:, 2]).sum(axis=1)
        affine_coeff = np.hstack((a1_a2, d[:, np.newaxis]))

        return affine_coeff

    def get_x_simplices(self, x):
        """
        Get simplices where x lives.

        Args:
            x (np.ndarray):
                input locations. size (n, 2).

        Returns:
            x_simplices (np.ndarray):
                indexes of vertices of simplices where x lives. size: (n, 3).
            x_simplices_idx (np.ndarray):
                indexes of simplices where x lives. size: (n,).
        """
        assert isinstance(x, np.ndarray), f'x is of type: {type(x)}.'
        x_simplices_idx = self.tri.find_simplex(x)
        assert x_simplices_idx.shape == (x.shape[0], )
        if np.any(x_simplices_idx < 0):
            raise ValueError(
                'At least one point is outside the triangulation...')

        x_simplices = self.tri.simplices[x_simplices_idx].copy()
        assert x_simplices.shape == (x.shape[0], 3)

        return x_simplices, x_simplices_idx

    def _check_x_simplices(self, x_simplices, x):
        """
        Verify that either x_simplices is None. If so, returns x_simplices.
        Otherwise, checks if x is not None and computes x_simplices from x.

        Args:
            x (None or np.ndarray):
                input locations. size (n, 2).
            x_simplices (np.ndarray):
                indexes of vertices of simplices where x lives. size: (n, 3).
        Returns:
            x_simplices (np.ndarray):
                indexes of vertices of simplices where x lives. size: (n, 3).
        """
        if x_simplices is not None:
            return x_simplices
        elif x is None:
            raise ValueError('Need to provide either "x" or "x_simplices".')

        x_simplices, _ = self.get_x_simplices(x)
        return x_simplices

    def get_x_simplices_points(self, x_simplices=None, x=None):
        """
        Get locations of vertices from x_simplices (indexes of vertices)
        or x (input locations).

        If x_simplices is not given, x has to be given.

        Args:
            x_simplices (None or np.ndarray):
                indexes of vertices of simplices where x lives. size: (n, 3).
            x (None or np.ndarray):
                input locations. size (n, 2).

        Returns:
            x_simplices_points:
                locations of vertices. size: (n, 3, 2).
        """
        x_simplices = self._check_x_simplices(x_simplices, x)
        x_simplices_points = self.tri.points[x_simplices]
        assert x_simplices_points.shape == (*x_simplices.shape, 2)

        return x_simplices_points

    def get_x_simplices_values(self, x_simplices=None, x=None):
        """
        Get values at the vertices from x_simplices (indexes of vertices)
        or x (input locations).

        If x_simplices is not given, x has to be given.

        Args:
            x_simplices (None or np.ndarray):
                indexes of vertices of simplices where x lives. size: (n, 3).
            x (None or np.ndarray):
                input locations. size (n, 2).

        Returns:
            x_simplices_values:
                values at the vertices. size (n, 3, 2).
        """
        x_simplices = self._check_x_simplices(x_simplices, x)
        x_simplices_values = self.tri.values[x_simplices]
        assert x_simplices_values.shape == x_simplices.shape

        return x_simplices_values

    def evaluate_bar(self, x):
        """
        Evaluate cpwl function at x, using barycentric coordinates method.

        Args:
            x (np.ndarray or torch.Tensor):
                input locations. size (n, 2).

        Returns:
            z (np.ndarray or torch.Tensor):
                values at x. size (n,).
        """
        torchtensor = False
        if isinstance(x, torch.Tensor):
            # convert to numpy
            torchtensor = True
            device, dtype = x.device, x.dtype
            x = x.detach().cpu().numpy()

        x_bar_coord, x_simplices = self.get_x_baryc_coord(x)
        x_simplices_values = \
            self.get_x_simplices_values(x_simplices=x_simplices)
        z = (x_simplices_values * x_bar_coord).sum(axis=1)

        if torchtensor is True:
            # convert to torch
            z = torch.from_numpy(z).to(device=device, dtype=dtype)

        return z

    def get_x_baryc_coord(self, x):
        """
        Get barycentric coordinates of x.

        We use affine coordinates to compute barycentric coordinates
        (more numerically stable):
        x^T = [p1^T p2^T p3^T] @ [bar1, bar2, bar3]^T (with bar3 = 1-bar1-bar2)
        x^T = [(p1-p3)^T (p2-p3)^T] @ [bar1, bar2]^T + p3^T
        <=> (x-p_3)^T = [(p1-p3)^T (p2-p3)^T] @ [bar1, bar2]^T

        Args:
            x (np.ndarray or torch.Tensor):
                input locations. size (n, 2).

        Returns:
            bar_coord (np.ndarray):
                barycentric coordinates of x. size (n, 3).
            x_simplices (np.ndarray):
                indexes of vertices of simplices where x lives. size: (n, 3).
        """
        x_simplices, x_simplices_idx = self.get_x_simplices(x)
        # tri.transform: size: (nsimplex, 3, 2)
        x_affine_coord = (x -
                          self.tri.transform[x_simplices_idx, 2])[:, :,
                                                                  np.newaxis]
        assert x_affine_coord.shape == (x.shape[0], 2, 1)
        p1_p2_affine_coord_inv = self.tri.transform[x_simplices_idx, :2]
        assert p1_p2_affine_coord_inv.shape == (x.shape[0], 2, 2)

        bar1_bar2 = (p1_p2_affine_coord_inv @ x_affine_coord).squeeze(axis=-1)
        assert bar1_bar2.shape == (x.shape[0], 2)
        bar_coord = np.c_[bar1_bar2, 1 - bar1_bar2.sum(axis=1, keepdims=True)]
        assert bar_coord.shape == (x.shape[0], 3)

        return bar_coord, x_simplices

    def evaluate(self, x):
        """
        Evaluate cpwl function at x, using affine coefficients.

        Args:
            x (np.ndarray or torch.Tensor):
                input locations. size (n, 2).

        Returns:
            z (np.ndarray or torch.Tensor):
                values at x. size (n,).
        """
        torchtensor = False
        if isinstance(x, torch.Tensor):
            # convert to numpy
            torchtensor = True
            device, dtype = x.device, x.dtype
            x = x.detach().cpu().numpy()

        affine_coeff = self.get_affine_coefficients(x)
        z = (affine_coeff[:, 0:2] * x).sum(axis=1) + affine_coeff[:, 2]

        if torchtensor is True:
            # convert to torch
            z = torch.from_numpy(z).to(device=device, dtype=dtype)

        return z

    def evaluate_with_grad(self, x):
        """
        Evaluate cpwl function at x, using affine coefficients, and compute
        gradient at x.

        Args:
            x (np.ndarray or torch.Tensor):
                input locations. size (n, 2).

        Returns:
            z (np.ndarray or torch.Tensor):
                values at x. size (n,).
            x_grad (np.ndarray or torch.Tensor):
                gradients at x. size (n, 2)
        """
        torchtensor = False
        if isinstance(x, torch.Tensor):
            # convert to numpy
            torchtensor = True
            device, dtype = x.device, x.dtype
            x = x.detach().cpu().numpy()

        affine_coeff = self.get_affine_coefficients(x)
        z = (affine_coeff[:, 0:2] * x).sum(axis=1) + affine_coeff[:, 2]
        x_grad = affine_coeff[:, 0:2]

        if torchtensor is True:
            # convert to torch
            z = torch.from_numpy(z).to(device=device, dtype=dtype)
            x_grad = torch.from_numpy(z).to(device=device, dtype=dtype)

        return z, x_grad

    def compute_grad(self, x):
        """
        Compute gradient of cpwl function at x.

        Args:
            x (np.ndarray or torch.Tensor):
                input locations. size (n, 2).

        Returns:
            x_grad (np.ndarray or torch.Tensor):
                gradients at x. size (n, 2)
        """
        _, x_grad = self.evaluate_with_grad(x)
        return x_grad

    def get_exact_HTV(self):
        """
        Get exact HTV of cpwl function.

        Returns:
            htv (float)
        """
        grad = self.tri.simplices_affine_coeff[:, 0:2].copy()
        assert grad.shape == (self.tri.simplices.shape[0], 2)

        # size (nsimplex, 3) there are three neighbors of each simplex in 2D.
        neighbors = self.tri.neighbors.copy()  # -1 signals no neighbors
        # tuple (rows, cols)
        no_neighbor_idx = np.where(neighbors == -1)
        has_neighbor_idx = np.where(neighbors > -1)

        ######################################
        # compute norm of gradient differences
        ######################################
        # size: (nsimplex, 3, 2)
        grad_simplices_expand = grad[:, np.newaxis, :].repeat(3, axis=1)
        # size: (nsimplex, 3, 2) --- (simplex idx, neighbor idx, neighbor grad)
        grad_neighbors = grad[neighbors]
        # do not count junctions with neighbors outside delaunay, so make
        # both grads equal to zero, so as to not have a contribution from
        # these.
        grad_neighbors[no_neighbor_idx] = np.array([0., 0.])
        grad_simplices_expand[no_neighbor_idx] = np.array([0., 0.])

        # (nsimplex, 3, 2)
        assert grad_neighbors.shape == (neighbors.shape[0], 3, 2)
        grad_diff_norm = np.linalg.norm(grad_neighbors - grad_simplices_expand,
                                        ord=2,
                                        axis=-1)

        assert grad_diff_norm.shape == neighbors.shape

        ##########################
        # compute junction lengths
        ##########################
        neighbors_simplices = self.tri.simplices[neighbors].copy()
        # (nsimplex, 3, 3)
        assert neighbors_simplices.shape == (neighbors.shape[0], 3, 3)

        simplices_expand = \
            self.tri.simplices[:, np.newaxis, :].repeat(3, axis=1).copy()
        # (nsimplex, 3, 3)
        assert simplices_expand.shape == (neighbors.shape[0], 3, 3)

        # TODO: Comment this section
        neighbors_simplices[no_neighbor_idx] = \
            simplices_expand[no_neighbor_idx]
        new_idx = (*no_neighbor_idx, no_neighbor_idx[1])
        neighbors_simplices[new_idx] = -1

        aux_arr = np.concatenate((simplices_expand, neighbors_simplices),
                                 axis=-1)

        aux_arr = np.sort(aux_arr, axis=-1)
        z = np.diff(aux_arr, axis=-1)
        edges_idx = aux_arr[np.where(z == 0)].reshape((*aux_arr.shape[0:2], 2))
        edges_points = self.tri.points[edges_idx]
        edges_len = np.linalg.norm(np.subtract(edges_points[:, :, 1, :],
                                               edges_points[:, :, 0, :]),
                                   ord=2,
                                   axis=-1)

        assert edges_len.shape == neighbors.shape

        # Divide by 2 but only if repeated
        edges_htv = grad_diff_norm * edges_len
        edges_htv[has_neighbor_idx] = edges_htv[has_neighbor_idx] / 2

        htv = edges_htv.sum()
        assert np.allclose(htv, edges_htv[has_neighbor_idx].sum())

        return htv

    def get_grid(self, h=0.001, to_numpy=True, to_float32=False):
        """
        Get a Grid over the rectangular range of the cpwl function.
        If cpwl does not have rectangular range, throw an Error.

        Args:
            h (float):
                step size.
            to_numpy (bool):
                if True, convert grid to numpy array.
            to_float32 (bool):
                if True, convert grid to float32

        Returns:
            grid (Grid):
                Grid instance (see grid.py).
        """
        # if not self.has_rectangular_range:
        #     raise ValueError(
        #         'The triangulation does not have a rectangular range.')

        # create image
        convex_hull_points = self.tri.points[self.convex_hull_points_idx]
        x1_min, x1_max = \
            convex_hull_points[:, 0].min(), convex_hull_points[:, 0].max()
        x2_min, x2_max = \
            convex_hull_points[:, 1].min(), convex_hull_points[:, 1].max()

        eps = h
        return Grid(x1_min=x1_min + eps,
                    x1_max=x1_max,
                    x2_min=x2_min + eps,
                    x2_max=x2_max,
                    h=h,
                    to_numpy=to_numpy,
                    to_float32=to_float32)

    def get_lefkimiattis_schatten_HTV(self, p=1, h=0.001):
        """
        Get the HTV of the cpwl function via finite second differences for
        computing the Hessian, and then taking its Schatten-p norm.
        All p's should be equivalent from a small enough step.

        Args:
            p (int >= 1):
                p for Schatten norm.
            h (float):
                step size for finite second differences.

        Returns:
            htv (float)
        """
        grid = self.get_grid(h=h)
        Hess = get_finite_second_diff_Hessian(grid, self.evaluate)

        S = np.linalg.svd(Hess, compute_uv=False, hermitian=True)
        assert S.shape == (*Hess.shape[0:2], 2)

        # schatten-p-norm
        points_htv = np.linalg.norm(S, ord=p, axis=-1)
        # sum over locations
        htv = (points_htv * h * h).sum()

        return htv

    def get_exact_grad_schatten_HTV(self, p=1, h=0.001):
        """
        Get the trace-HTV of the cpwl function via finite first-differences on
        the exact gradient for computing the Hessian, and then taking its
        Schatten-p-norm.

        Args:
            p (int >= 1):
                p for Schatten norm.
            h (float):
                step size for finite second differences.

        Returns:
            htv (float)
        """
        grid = self.get_grid(h=h)
        Hess = get_exact_grad_Hessian(grid, self.compute_grad)

        S = np.linalg.svd(Hess, compute_uv=False, hermitian=False)
        assert S.shape == (*Hess.shape[0:2], 2)

        # schatten-p-norm
        points_htv = np.linalg.norm(S, ord=p, axis=-1)
        # sum over locations
        htv = (points_htv * h * h).sum()

        return htv

    def get_lefkimiattis_trace_HTV(self, h=0.001):
        """
        Get the trace-HTV of the cpwl function via finite second-differences
        for computing the Hessian, and then taking its trace.

        Args:
            h (float):
                step size for finite second differences.

        Returns:
            htv (float)
        """
        grid = self.get_grid(h=h)
        Hess = get_finite_second_diff_Hessian(grid, self.evaluate)

        # trace
        points_htv = np.abs(Hess[:, :, 0, 0] + Hess[:, :, 1, 1])
        # sum over locations
        htv = (points_htv * h * h).sum()

        return htv

    def get_exact_grad_trace_HTV(self, h=0.001):
        """
        Get the trace-HTV of cpwl function via finite first differences on
        the exact gradient to compute the Hessian, and then taking its trace.

        Args:
            h (float):
                step size for finite second differences.

        Returns:
            htv (float)
        """
        grid = self.get_grid(h=h)
        Hess = get_exact_grad_Hessian(grid, self.compute_grad)

        # trace
        points_htv = np.abs(Hess[:, :, 0, 0] + Hess[:, :, 1, 1])
        # sum over locations
        htv = (points_htv * h * h).sum()

        return htv
