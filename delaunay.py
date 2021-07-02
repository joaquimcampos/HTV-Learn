import torch
import numpy as np
import scipy.spatial
import warnings
from sklearn.preprocessing import normalize

from base_cpwl import BaseCPWL

class Delaunay(BaseCPWL):
    def __init__(self, npoints=120, x_range=2, z_range=1,
                add_extreme_points=False, pad_factor=0.05,
                points=None, values=None, set_zero_boundary=False, **kwargs):
        """
        Args:
            npoints: number of delaunay triangulation points to generate, if
                    args:points and args:values are not given.
            add_extreme_points: Add extreme points of rectangle to triangulation.
            pad_factor: pad factor for extreme points relative to data domain.
            points, values: Specific set of points, size: (n,2),
                            and corresponding values, size: (n,); if not given,
                            npoints are randomly generated.
            set_zero_boundary: make convex hull points zero (does not have
                                effect if add_extreme_points is True).
        """
        super().__init__()

        if isinstance(points, np.ndarray):
            self._verify_input(points, values)
        else:
            interior_npoints = (npoints-4) if add_extreme_points is True else npoints
            # randomly generate points in [-x_range/2, x_range/2]^2
            points = (np.random.rand(int(interior_npoints), 2) - 0.5) * x_range
            # randomly generate values in [-z_range/2, z_range/2]
            values = (np.random.rand(int(interior_npoints),) - 0.5) * z_range # [-0.5, 0.5]

        if add_extreme_points:
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            # padding
            pf = min(x_max - x_min, y_max - y_min) * pad_factor
            # extreme points of domain
            extreme_points = np.array([[x_min-pf, y_min-pf], [x_max+pf, y_min-pf],
                                        [x_min-pf, y_max+pf], [x_max+pf, y_max+pf]])
            points = np.vstack((points, extreme_points))
            values = np.concatenate((values, np.zeros(4)))

        self.tri = scipy.spatial.Delaunay(points)

        for attstr in ['values', 'simplices_points', 'simplices_values',
                    'simplices_affine_coeff']:
            assert not hasattr(self.tri, attstr), f'{self.tri} has attribute {attstr}...'

        self.convex_hull_points_idx = np.unique(self.tri.convex_hull.flatten())
        self.tri.values = values

        if not add_extreme_points and set_zero_boundary is True:
            self.set_zero_boundary()
        if not self.has_zero_boundary:
            warnings.warn('The constructed delaunay triangulation does '
                        'not have zero boundaries...')

        # sanity check
        # https://math.stackexchange.com/questions/1097646/number-of-triangles-in-a-triangulation
        nlinear_regions = self.tri.simplices.shape[0]
        assert nlinear_regions == 2*self.tri.points.shape[0] - 2 - self.convex_hull_points_idx.shape[0]

        self.tri.simplices_points = self.tri.points[self.tri.simplices] # (nsimplex, 3, 2)
        assert self.tri.simplices_points.shape == (self.tri.simplices.shape[0], 3, 2)
        self.tri.simplices_values = self.tri.values[self.tri.simplices]
        assert self.tri.simplices_values.shape == (self.tri.simplices.shape[0], 3)

        centers_bar_coord = np.array([0.33, 0.33, 0.34])[:,np.newaxis]
        simplices_centers = (np.transpose(self.tri.simplices_points, (0,2,1)) @ \
                            centers_bar_coord).squeeze(axis=-1)

        self.tri.simplices_affine_coeff = self.get_affine_coefficients(simplices_centers)
        assert self.tri.simplices_affine_coeff.shape == (self.tri.simplices.shape[0], 3)


    def set_zero_boundary(self):
        """ """
        # delaunay triangulation always gives you a convex set
        self.tri.values[self.convex_hull_points_idx] = 0.


    @property
    def has_zero_boundary(self):
        """ """
        if self.tri.values[self.convex_hull_points_idx].any():
            return False

        return True


    @property
    def has_rectangular_range(self):
        """ """
        convex_set_points = self.tri.points[self.convex_hull_points_idx]

        x_min = convex_set_points[:, 0].min()
        x_max = convex_set_points[:, 0].max()
        y_min = convex_set_points[:, 1].min()
        y_max = convex_set_points[:, 1].max()

        ones_vec = np.ones((convex_set_points.shape[0],))

        eps = 1e-6
        for val in [x_min, x_max]:
            if np.sum(np.abs(convex_set_points[:, 0] - ones_vec * val) < eps) < 2:
                return False
        for val in [y_min, y_max]:
            if np.sum(np.abs(convex_set_points[:, 1] - ones_vec * val) < eps) < 2:
                return False

        return True


    @staticmethod
    def _verify_input(points, values):
        """ """
        assert points.shape[1] == 2
        assert len(points.shape) == 2
        assert isinstance(values, np.ndarray)
        assert values.shape == (points.shape[0],)


    def get_affine_coefficients(self, x):
        """ Get the affine coefficients for each datapoint from
        simplices points + values, i.e., (x1,x2,z), using the solve method.

        f(x1, x2) = a1.x1 + a2.x2 + d.
        We have 3 vertices and 3 unkowns (a1, a2, d) for each datapoint.
        So, we can solve the system of 3 equations to get (a1, a2, d).
        """
        x_simplices, x_simplices_idx = self.get_x_simplices(x)
        try:
            affine_coeff = self.tri.simplices_affine_coeff[x_simplices_idx]
            return affine_coeff
        except AttributeError:
            pass

        x_simplices_points, _ = self.get_x_simplices_points(x_simplices=x_simplices) # (m,3,2)
        x_simplices_values, _ = self.get_x_simplices_values(x_simplices=x_simplices) # (m,3)

        # vertices for (x1, x2, z) -> {(x1_K, x2_K, z_K)}, K in {A,B,C}
        # (x1-x1_C).a1 + (x2-x2_C).a2 + (z-z_C) = 0
        # plug (x1_A, x2_A, z_A) and (x1_B, x2_B, z_B) above to find a1, a2
        z_diff = (x_simplices_values[:, 0:2] - x_simplices_values[:, 2:3])[:,:,np.newaxis]
        assert z_diff.shape == (x.shape[0],2,1)

        x_diff = x_simplices_points[:, 0:2, :] - x_simplices_points[:, 2:3, :]
        assert x_diff.shape == (x.shape[0],2,2)

        a1_a2 = np.linalg.solve(x_diff, z_diff)
        assert a1_a2.shape == (x.shape[0],2,1)
        a1_a2 = a1_a2.squeeze(-1)

        d = x_simplices_values[:, 2] - (a1_a2 * x_simplices_points[:, 2]).sum(axis=1)
        affine_coeff = np.hstack((a1_a2, d[:, np.newaxis]))

        return affine_coeff


    def get_x_simplices(self, x):
        """ Get simplices (triangle) for x.

        Returns:
            x_simplices: size (x.shape[0], 3)
            x_simplices_idx: size (x.shape[0],)
        """
        assert isinstance(x, np.ndarray), 'x is of type: type(x).'
        x_simplices_idx = self.tri.find_simplex(x)
        assert x_simplices_idx.shape == (x.shape[0],)
        if np.any(x_simplices_idx < 0):
            raise ValueError('At least one point is outside the triangulation...')

        x_simplices = self.tri.simplices[x_simplices_idx]
        assert x_simplices.shape == (x.shape[0], 3)

        return x_simplices, x_simplices_idx


    def _check_x_simplices(self, x, x_simplices):
        """ """
        if x_simplices is not None:
            return x_simplices
        elif x is None:
            raise ValueError('Need to provide either "x" or "x_simplices".')

        x_simplices, _ = self.get_x_simplices(x)
        return x_simplices


    def get_x_simplices_points(self, x_simplices=None, x=None):
        """
        Returns:
            x_simplices_points: size (x.shape[0], 3, 2)
            x_simplices: size (x.shape[0], 3)
        """
        x_simplices = self._check_x_simplices(x, x_simplices)
        x_simplices_points = self.tri.points[x_simplices]
        assert x_simplices_points.shape == (*x_simplices.shape, 2)

        return x_simplices_points, x_simplices


    def get_x_simplices_values(self, x_simplices=None, x=None):
        """
        Returns:
            x_simplices_values: size (x.shape[0], 3)
            x_simplices: size (x.shape[0], 3)
        """
        x_simplices = self._check_x_simplices(x, x_simplices)
        x_simplices_values = self.tri.values[x_simplices]
        assert x_simplices_values.shape == x_simplices.shape

        return x_simplices_values, x_simplices


    def evaluate_bar(self, x):
        """ Evaluate cpwl function at x, using barycentric coordinates method
        """
        torchtensor = False
        if isinstance(x, torch.Tensor):
            torchtensor = True
            x = x.numpy()

        x_bar_coord, x_simplices = self.get_x_baryc_coord(x)
        x_simplices_values, _ = self.get_x_simplices_values(x_simplices=x_simplices)
        z = (x_simplices_values * x_bar_coord).sum(axis=1)

        if torchtensor is True:
            z = torch.from_numpy(z)

        return z


    def evaluate(self, x):
        """ Evaluate cpwl function at x using affine coefficients
        """
        torchtensor = False
        if isinstance(x, torch.Tensor):
            torchtensor = True
            x = x.numpy()

        affine_coeff = self.get_affine_coefficients(x)
        z = (affine_coeff[:, 0:2]*x).sum(axis=1) + affine_coeff[:, 2]

        if torchtensor is True:
            z = torch.from_numpy(z).float()

        return z


    def evaluate_with_grad(self, x):
        """ Evaluate cpwl function at x using affine coefficients and compute gradient.
        """
        affine_coeff = self.get_affine_coefficients(x)
        z = (affine_coeff[:, 0:2]*x).sum(axis=1) + affine_coeff[:, 2]
        x_grad = affine_coeff[:, 0:2]

        return z, x_grad


    def get_x_baryc_coord(self, x):
        """ Get barycentric coordinates of x.

        We use affine coordinates to compute barycentric coordinates
        (more numerically stable):
        x^T = [p1^T p2^T p3^T] @ [bar1, bar2, bar3]^T (with bar3 = 1-bar1-bar2)
        x^T = [(p1-p3)^T (p2-p3)^T] @ [bar1, bar2]^T + p3^T
        <=> (x-p_3)^T = [(p1-p3)^T (p2-p3)^T] @ [bar1, bar2]^T

        Returns:
            x_bar_coord: size (x.shape[0], 3)
            x_simplices: size (x.shape[0], 3)
        """
        x_simplices, x_simplices_idx = self.get_x_simplices(x)
        # tri.transform: size: (nsimplex, 3, 2)
        x_affine_coord = (x - self.tri.transform[x_simplices_idx,2])[:,:,np.newaxis]
        assert x_affine_coord.shape == (x.shape[0], 2, 1)
        p1_p2_affine_coord_inv = self.tri.transform[x_simplices_idx,:2]
        assert p1_p2_affine_coord_inv.shape == (x.shape[0], 2, 2)

        bar1_bar2 = (p1_p2_affine_coord_inv @ x_affine_coord).squeeze(axis=-1)
        assert bar1_bar2.shape == (x.shape[0], 2)
        bar_coord = np.c_[bar1_bar2, 1 - bar1_bar2.sum(axis=1, keepdims=True)]
        assert bar_coord.shape == (x.shape[0], 3)

        return bar_coord, x_simplices
