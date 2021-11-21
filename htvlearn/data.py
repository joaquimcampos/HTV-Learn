import os
import torch
import numpy as np
import math
import scipy

from htvlearn.lattice import Lattice
from htvlearn.delaunay import Delaunay
from htvlearn.grid import Grid


class Hex():
    """Hexagonal lattice vectors"""

    v1 = Lattice.hexagonal_matrix[:, 0].numpy()
    v2 = Lattice.hexagonal_matrix[:, 1].numpy()


class BoxSpline():
    """Three-directional hexagonal box spline"""

    center_points = np.array([0., 0.])
    border_points = np.array([Hex.v1, Hex.v2, -Hex.v1 + Hex.v2,
                              -Hex.v1, -Hex.v2, -Hex.v2 + Hex.v1])
    points = np.vstack((center_points, border_points, 2 * border_points))
    values = np.array([math.sqrt(3) / 2,
                       0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0.])
    htv = 12


class SimplicialSpline():
    """Simplicial spline with randomly positioned vertices"""

    np.random.seed(3)
    center_points = np.array([0., 0.]) + np.random.uniform(-0.2, 0.2, (2, ))
    border_points = np.array([Hex.v1, Hex.v2, -Hex.v1 + Hex.v2,
                              -Hex.v1, -Hex.v2, -Hex.v2 + Hex.v1]) + \
        np.random.uniform(-0.2, 0.2, (6, 2))
    points = np.vstack((center_points, border_points, 2 * border_points))
    values = np.array([math.sqrt(3) / 2,
                       0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0.])


class CutPyramid():
    """Pyramid with flat top"""

    points = np.vstack((BoxSpline.center_points,
                        BoxSpline.border_points,
                        2 * BoxSpline.border_points,
                        3 * BoxSpline.border_points))
    values = np.array([1., 1., 1., 1., 1., 1., 1.,
                       0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., ])
    htv = 16 * math.sqrt(3)


class SimpleJunction():
    """A simple two-polytope junction"""

    points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.],
                       [0., 3. / 4], [1., 1. / 4]])
    values = np.array([0., 2. / 3, 2. / 3, 0., 1., 1.])
    # gradients of each polytope
    a1_affine_coeff = np.array([2. / 3, 4. / 3., 0.])
    a2_affine_coeff = np.array([-2. / 3, -4. / 3., 2.])
    htv = 10. / 3


def init_distorted_grid(size_=(3, 3), range_=(-1, 1)):
    """
    Initialize a distorted grid.

    Args:
        size (2-tuple): grid size.
        range (2-tuple):
            range of points in each dimension before distortion.

    Returns:
        points (np.array):
            distorted grid points. size: (size_[0]*size_[1]) x 2.
    """
    assert isinstance(size_, tuple)
    assert len(size_) == 2
    # initialize undistorted grid points (u, v)
    vec1 = np.linspace(*range_, size_[0]) * 1.
    vec2 = np.linspace(*range_, size_[1]) * 1.
    u, v = np.meshgrid(vec1, vec2)
    u = u.flatten()
    v = v.flatten()

    # add noise to the interior vertices of the grid
    mask = np.ma.mask_or(np.abs(u) == u.max(), np.abs(v) == v.max())
    points = np.hstack((u[:, np.newaxis], v[:, np.newaxis])).copy()
    # the noise is scaled according to the grid spacing
    noise = (np.random.rand(*points.shape) - 0.5) * (u[1] - u[0])
    # don't add noise to boundary vertices
    points[~mask] = points[~mask] + noise[~mask]

    return points


class DistortedGrid:
    """Dataset with random values in a distorted random grid"""

    points = init_distorted_grid(size_=(3, 3))
    values = (np.random.rand(points.shape[0], ) - 0.5)


class Data():
    """Data class for algorithms"""

    def __init__(self,
                 data_from_ckpt=None,
                 dataset_name=None,
                 num_train=None,
                 data_dir='./data',
                 valid_fraction=0.2,
                 test_as_valid=False,
                 noise_ratio=0.,
                 seed=-1,
                 add_lat_vert=False,
                 verbose=False,
                 **kwargs):
        """
        Args:
            data_from_ckpt (dict):
                dictionary with 'train', 'valid' and 'test' data loaded
                from a checkpoint.
            dataset_name (str)
            num_train (int):
                number of training+valid samples. The effective number of
                training samples is a multiple of 1000. Further, if the
                dataset has gaps the data inside the gaps will also removed.
            data_dir (int):
                data directory (for face dataset)
            valid_fraction (float [0,1]):
                fraction of num_train samples that is used for validation
            test_as_valid (bool):
                if True, use test set in validation.
            noise_ratio (float >= 0):
                noise that should be applied to the samples as a fraction of
                the data range.
            seed (int):
                seed for random generation. If negative, no seed is set.
            add_lat_vert (bool):
                if True, add lattice extreme points (face dataset).
            verbose (bool):
                print more info.
        """
        self.data_from_ckpt = data_from_ckpt
        self.dataset_name = dataset_name
        self.num_train = num_train

        if self.data_from_ckpt is None:
            assert self.dataset_name is not None
            if not self.dataset_name.startswith('pyramid'):
                assert self.num_train is not None

        self.data_dir = data_dir
        self.valid_fraction = valid_fraction
        self.test_as_valid = test_as_valid
        self.noise_ratio = noise_ratio
        self.seed = seed
        self.add_lat_vert = add_lat_vert
        self.verbose = verbose
        # if not overwritten, computed in add_noise_to_values()
        # from self.noise_ratio and dataset height range
        self.noise_std = None

        if self.seed >= 0:
            # set seed
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        self.train, self.valid, self.test = {}, {}, {}
        self.delaunay = {}  # points and values for delaunay triangulation

        if self.data_from_ckpt is not None:
            # load data from self.data_from_ckpt
            assert 'train' in self.data_from_ckpt
            assert 'valid' in self.data_from_ckpt
            assert 'test' in self.data_from_ckpt

            self.train = self.data_from_ckpt['train']
            self.valid = self.data_from_ckpt['valid']
            self.test = self.data_from_ckpt['test']

            if 'delaunay' in self.data_from_ckpt:
                assert 'points' in self.data_from_ckpt['delaunay']
                assert 'values' in self.data_from_ckpt['delaunay']

                self.delaunay['points'] = \
                    self.data_from_ckpt['delaunay']['points']
                self.delaunay['values'] = \
                    self.data_from_ckpt['delaunay']['values']

        self.init_data()

    def init_data(self):
        """Initialize cpwl dataset, and train/test/valid sets"""
        if not bool(self.delaunay):

            if self.dataset_name.startswith('pyramid'):
                self.delaunay['points'], self.delaunay['values'] = \
                    self.init_pyramid()

                # training is made of all pyramid vertices except apex
                self.train['input'] = self.delaunay['points'][:-1].clone()
                self.train['values'] = self.delaunay['values'][:-1].clone()
                # force validation set to be equal to test set
                self.test_as_valid = True

            elif self.dataset_name.endswith('planes'):
                self.delaunay['points'], self.delaunay['values'] = \
                    self.init_planes()

            elif 'face' in self.dataset_name:
                self.delaunay['points'], self.delaunay['values'] = \
                    self.init_face(self.data_dir,
                                   cut=True
                                   if 'cut' in self.dataset_name
                                   else False)

        self.cpwl = Delaunay(points=self.delaunay['points'],
                             values=self.delaunay['values'])

        if not bool(self.test):
            if not self.cpwl.has_rectangular_range:
                if self.dataset_name.endswith('planes'):
                    h = (self.cpwl.tri.points[:, 0].max() -
                         self.cpwl.tri.points[:, 0].min()) / 400
                    self.test['input'] = \
                        Grid(x1_min=self.cpwl.tri.points[:, 0].min(),
                             x1_max=self.cpwl.tri.points[:, 0].max(),
                             x2_min=self.cpwl.tri.points[:, 1].min(),
                             x2_max=self.cpwl.tri.points[:, 1].max(),
                             h=h,
                             to_numpy=False,
                             to_float32=True).x

                    # discard samples outside convex set
                    idx = self.cpwl.tri.find_simplex(self.test['input'])
                    self.test['input'] = self.test['input'][idx >= 0]
                else:
                    # generate uniformly distributed samples in cpwl convex set
                    # the final number of test samples will be smaller because
                    # samples outside lattice are discarded
                    nb_samples = 5000
                    self.test['input'] = \
                        self.generate_random_samples(nb_samples)
            else:
                # test set is sampled on a grid inside the convex hull of cpwl
                self.test['input'] = self.cpwl.get_grid(h=0.01,
                                                        to_numpy=False,
                                                        to_float32=True).x

            self.test['values'] = self.cpwl.evaluate(self.test['input'])
            print(f'\nnb. of test data points : {self.test["input"].size(0)}')

            if (not bool(self.valid)) and (self.test_as_valid is True):
                self.valid['input'] = self.test['input'].clone()
                self.valid['values'] = self.test['values'].clone()

        if not bool(self.train):

            num_train_valid_samples = int(self.num_train)

            if self.dataset_name.endswith('planes'):
                # generate grid in lattice reference
                x_lat = torch.empty((num_train_valid_samples, 2))
                x_lat.uniform_(-0.5, 0.5)
                # convert to standard coordinates
                x = (Lattice.hexagonal_matrix @ x_lat.t()).t()
            else:
                # generate num_train_valid_samples uniformly distributed
                # in cpwl convex set
                x = self.generate_random_samples(num_train_valid_samples)

            # training / validation split indices
            if not self.test_as_valid:
                split_idx = int((1 - self.valid_fraction) *
                                x.size(0))
            else:
                # full training set, validation set = test set
                split_idx = x.size(0)

            self.train['input'] = x[0:split_idx]
            self.train['values'] = self.cpwl.evaluate(self.train['input'])

            if self.dataset_name.endswith('gaps'):
                # [(gap_x_min, gap_x_max)...]
                gap_x_range = [[0.108, 0.234], [-0.07, 0.226],
                               [-0.234, -0.108]]
                # [(gap_y_min, gap_y_max)...]
                gap_y_range = [[-0.21, 0.07], [0.19, 0.311], [-0.21, 0.063]]

                # remove data inside gaps
                for i in range(len(gap_x_range)):
                    gap_mask = (
                        (self.train['input'][:, 0] >= gap_x_range[i][0]) *
                        (self.train['input'][:, 0] <= gap_x_range[i][1]) *
                        (self.train['input'][:, 1] >= gap_y_range[i][0]) *
                        (self.train['input'][:, 1] <= gap_y_range[i][1]))

                    self.train['input'] = self.train['input'][~gap_mask]
                    self.train['values'] = self.train['values'][~gap_mask]

            if not np.allclose(self.noise_ratio, 0.):
                # add noise to training data
                self.train['values'] = \
                    self.add_noise_to_values(self.train['values'])

            if self.train['input'].size(0) >= 3000:
                # effective number of samples (rounded to 1000)
                num = int(np.floor(self.train['input'].size(0) / 1000.) * 1000)
                idx = torch.randperm(self.train['input'].size(0))[:num]
                self.train['input'] = self.train['input'][idx]
                self.train['values'] = self.train['values'][idx]

            print('nb. of training data points : '
                  f'{self.train["input"].size(0)}')

            if not bool(self.valid):
                self.valid['input'] = x[(split_idx + 1)::]
                self.valid['values'] = \
                    self.cpwl.evaluate(self.valid['input'])

            if ('face' in self.dataset_name and
                    self.add_lat_vert is True):
                # add lattice corners
                br = Lattice.bottom_right_std
                ur = Lattice.upper_right_std
                lat_points = np.array([[-ur[0], -ur[1]],
                                       [br[0], br[1]],
                                       [-br[0], -br[1]],
                                       [ur[0], ur[1]]])

                self.delaunay['points'] = \
                    np.concatenate((self.delaunay['points'],
                                    lat_points), dim=0)
                self.delaunay['values'] = \
                    np.concatenate((self.delaunay['values'],
                                    np.zeros(4)))
                # refresh self.cpwl
                self.cpwl = Delaunay(points=self.delaunay['points'],
                                     values=self.delaunay['values'])

    def generate_random_samples(self, num_samples):
        """
        Generate uniformly distributed data inside convex set.

        Works by generating num_samples points and then rejecting the
        ones outside the convex set.

        Args:
            num_samples (int) (before possible rejection)

        Returns:
            x (torch.tensor)
        """
        x = torch.empty((num_samples, 2))
        x[:, 0].uniform_(self.cpwl.tri.points[:, 0].min(),
                         self.cpwl.tri.points[:, 0].max())
        x[:, 1].uniform_(self.cpwl.tri.points[:, 1].min(),
                         self.cpwl.tri.points[:, 1].max())
        # reject samples outside convex set
        idx = self.cpwl.tri.find_simplex(x)
        x = x[idx >= 0]

        return x

    def add_noise_to_values(self, values):
        """
        Add gaussian noise to values.

        if self.noise_std exists, it is used as the noise standard deviation,
        otherwise noise_std is computed from self.noise_ratio and the data
        height range.

        Args:
            values (torch.tensor):
                values to add noise to.

        Returns the noisy values.
        """
        noise_std = self.noise_std
        if noise_std is None:
            noise_std = self.noise_ratio * (values.max() - values.min())

        if self.verbose:
            print('Adding noise of standard deviation '
                  'sigma = {:.2E}'.format(noise_std))
        noise = torch.empty_like(values).normal_(std=noise_std)

        return values + noise

    @staticmethod
    def init_pyramid():
        """
        Initialize the pyramid dataset.

        Returns:
            points (np.array): size (M, 2).
            values (np.array): size (M,)
        """
        # points in lattice coordinates
        h = 0.1
        points = torch.tensor([[2 * h, 0.], [0., 2 * h],
                               [2 * h, -2 * h], [0., -2 * h],
                               [-2 * h, 0.], [-2 * h, 2 * h],
                               [h, 0.], [0., h],
                               [h, -h], [0., -h],
                               [-h, 0.], [-h, h],
                               [0., 0.]])  # last element -> apex

        values = torch.tensor([.1, .1, .1, .1, .1, .1,
                               .2, .2, .2, .2, .2, .2,
                               .3])

        if False:
            # extended pyramid
            points_ext = torch.tensor([[3 * h, 0.], [0., 3 * h],
                                       [2.95 * h, -2.85 * h], [0., -3 * h],
                                       [-3 * h, 0.], [-3 * h, 3 * h]])

            values_ext = torch.tensor([.1, .1, .1, .1, .1, .1])

            points = torch.cat((points, points_ext), dim=0)
            values = torch.cat((values, values_ext), dim=0)

        # convert to standard coordinates
        points = (Lattice.hexagonal_matrix @ points.t()).t()

        if False:
            # linear term
            a, b = torch.tensor([.2, .2]) + torch.tensor([.1])
            values = values + (points * a.unsqueeze(0)).sum(1) + b

        return points.numpy(), values.numpy()

    @classmethod
    def init_zero_boundary_planes(cls):
        """
        Initialize the planes dataset with zero boundaries.

        Returns:
            points (torch.tensor): size (M, 2).
            values (torch.tensor): size (M,)
        """
        # fit planes function in the lattice
        pad = 0.08
        x_min, _, x_max, _ = cls.get_data_boundaries(hw_ratio=0.01, pad=pad)
        _, y_min, _, y_max = cls.get_data_boundaries(hw_ratio=100, pad=pad)

        dx = (x_max - x_min) / 100  # delta x step
        dy = (y_max - y_min) / 100  # delta y step

        # control points with values - (x1, x2, val)
        vert = \
            torch.tensor([[x_min + 30 * dx, y_min + 35 * dy, dx * 20],  # 0
                          [x_max - 40 * dx, y_min + 30 * dy, dx * 20],  # 1
                          [x_max - 35 * dx, y_max - 30 * dy, dx * 20],  # 2
                          [x_min + 40 * dx, y_max - 30 * dy, dx * 20],  # 3
                          [x_max - 25 * dx, y_min + 5 * dy, 0.],  # 4
                          [x_min + 25 * dx, y_max - 5 * dy, 0.]])  # 5

        # auxiliary triangulation of the function
        # size (num_simplices, vertices)
        simplices = torch.tensor([[0, 1, 3],
                                  [1, 2, 3],
                                  [4, 1, 0],
                                  [0, 3, 5],
                                  [4, 2, 1],
                                  [3, 2, 5]])

        # check values of vertices so that there is a seamless plane junction
        x_v6 = cls.get_zero_loc(vert, simplices, 2, 3)
        x_v7 = cls.get_zero_loc(vert, simplices, 4, 5)
        br = Lattice.bottom_right_std
        ur = Lattice.upper_right_std

        # add x_v6, x_v7, and lattice corners
        new_vert = torch.tensor([[x_v6[0], x_v6[1], 0.],  # 6
                                 [x_v7[0], x_v7[1], 0.],  # 7
                                 [-ur[0], -ur[1], 0.],  # 8
                                 [br[0], br[1], 0.],  # 9
                                 [-br[0], -br[1], 0.],  # 10
                                 [ur[0], ur[1], 0.]])  # 11

        vert = torch.cat((vert, new_vert), dim=0)
        points, values = vert[:, 0:2], vert[:, 2]

        return points, values

    @staticmethod
    def add_linear_func(points, values):
        """
        Add a linear term to the dataset.

        Args:
            points (torch.tensor): size (M, 2).
            values (torch.tensor): size (M,)
        Returns:
            values (torch.tensor): size (M,).
        """
        # add linear term to vertices
        a = torch.tensor([0.1, 0.05])
        b = torch.tensor([-0.05])
        values += (points * a.unsqueeze(0)).sum(1) + b

        return values

    def init_planes(self):
        """
        Initialize the planes dataset. Set self.noise_std.

        Returns:
            points (torch.tensor): size (M, 2).
            values (torch.tensor): size (M,)
        """
        # initialize planes dataset with zero boundaries
        points, values = self.init_zero_boundary_planes()
        # overwrite noise standard deviation
        self.noise_std = (self.noise_ratio * values.max())
        # add linear function to dataset
        values = self.add_linear_func(points, values)
        # convert to numpy
        points, values = points.numpy(), values.numpy()

        return points, values

    @staticmethod
    def get_zero_loc(vert, simplices, idx1, idx2):
        """
        Get zero locations of vertices for a seamless junction of the planes.

        Args:
            vert (np.array):
                size: (M, 3) (points in the first two columns,
                              values in the third)
            simplices (np.array):
                indexes of vertices for each simplex (row). size: (P, 3).
            idx1, idx2 (int>=0):
                indices of simplices to join.

        Returns:
            x (torch.tensor): size (2,)
        """
        # size (2, 3, 3)
        idx_vec = [idx1, idx2]
        simplices_vert = \
            torch.cat(tuple(vert[simplices[i]].unsqueeze(0)
                            for i in idx_vec), dim=0)

        plane_coeff = Lattice.solve_method(simplices_vert)
        affine_coeff = Lattice.get_affine_coeff_from_plane_coeff(plane_coeff)
        assert affine_coeff.size() == (2, 3)

        B = -affine_coeff[:, -1:]
        A = affine_coeff[:, 0:2]
        x = torch.linalg.solve(A, B)

        return x.squeeze(-1)

    @staticmethod
    def read_face(data_dir, cut_eps=0.6):
        """
        Read the 3D face dataset and construct a function from it by
        cutting and eliminating duplicates.

        Args:
            cut_eps (float in [0,1]):
                what height to cut face relative to its maximum height.

        Returns:
            cleaned_vert (np.array):
                with vertices below cut_eps and duplicates removed and
                zero mean.
                size: (M, 3) (points in the first two columns,
                              values in the third)
        """
        obj_file = os.path.join(data_dir, 'obj_free_male_head.obj')

        V = []
        with open(obj_file, "r") as file1:
            for line in file1.readlines():
                f_list = [i for i in line.split(" ") if i.strip()]
                if len(f_list) == 0:
                    continue
                if f_list[0] != 'v':
                    continue
                V += [float(i) for i in f_list[1::]]

        # vertices
        vert = np.array(V).reshape(-1, 3)
        # sort vertices by z coordinates in descending direction
        sort_vert = vert[vert[:, 2].argsort()][::-1]

        # get unique_idx of first occurences (largest height)
        _, unique_dx = np.unique(sort_vert[:, 0:2], return_index=True, axis=0)
        unique_sort_vert = sort_vert[unique_dx]

        # eliminate vertices whose height is below cutoff
        min_height = unique_sort_vert[:, 2].min()
        max_height = unique_sort_vert[:, 2].max()
        cutoff_val = min_height + (max_height - min_height) * cut_eps
        cutoff_mask = np.where(unique_sort_vert[:, 2] > cutoff_val)[0]
        cleaned_vert = unique_sort_vert[cutoff_mask]
        cleaned_vert[:, 2] = cleaned_vert[:, 2] - \
            cutoff_val  # shift z.min() to z = 0
        x_mean = cleaned_vert[:, 0].min() / 2. + cleaned_vert[:, 0].max() / 2.
        y_mean = cleaned_vert[:, 1].min() / 2. + cleaned_vert[:, 1].max() / 2.
        cleaned_vert[:, 0] = cleaned_vert[:, 0] - x_mean  # shift x around 0
        cleaned_vert[:, 1] = cleaned_vert[:, 1] - y_mean  # shift t around 0

        return cleaned_vert

    @classmethod
    def init_face(cls, data_dir, cut=False):
        """
        Initialize the face dataset.

        Args:
            cut (bool):
                if True, use only a smaller section of the face.
                Otherwise, use full face with zero boundaries.

        Returns:
            points (torch.tensor): size (M, 2).
            values (torch.tensor): size (M,)
        """
        vert = cls.read_face(data_dir)

        # normalize face to fit in [-0.8, 0.8]^2 square
        max_ = max(np.abs(vert[:, 0]).max(), np.abs(vert[:, 1]).max())
        vert = vert / max_ * 0.8

        if cut is True:
            # cut a smaller portion of the face
            cpwl_aux = Delaunay(points=vert[:, 0:2].copy(),
                                values=vert[:, 2].copy())

            x_min, x_max = -0.324, 0.324
            y_min, y_max = -0.45, 0.419

            mask = (vert[:, 0] > x_min) * (vert[:, 0] < x_max) * \
                (vert[:, 1] > y_min) * (vert[:, 1] < y_max)
            vert = vert[mask]

            # add extreme points of the convex hull to vertices
            hull_points = np.array([[x_min, y_min], [x_max, y_min],
                                    [x_max, y_max], [x_min, y_max]])

            hull_values = cpwl_aux.evaluate(hull_points)
            new_vertices = np.concatenate(
                (hull_points, hull_values[:, np.newaxis]), axis=1)
            vert = np.concatenate((vert, new_vertices), axis=0)
        else:
            points = vert[:, 0:2]
            hull = scipy.spatial.ConvexHull(points)
            hull_points = points[hull.vertices]

        # add points along the convex hull
        for i in range(hull_points.shape[0]):
            frac = np.linspace(0.01, 0.99, num=99)[:, np.newaxis]
            next_vert = i + 1 if i != hull_points.shape[0] - 1 else 0
            new_points = hull_points[next_vert][np.newaxis, :] * frac + \
                hull_points[i][np.newaxis, :] * (1 - frac)

            if cut is True:
                # evaluate on convex hull of face
                new_values = cpwl_aux.evaluate(new_points)
            else:
                # add zeros around face (to its convex hull contour)
                new_values = np.zeros(new_points.shape[0])

            new_vertices = np.concatenate(
                (new_points, new_values[:, np.newaxis]), axis=1)
            vert = np.concatenate((vert, new_vertices), axis=0)

        if cut is False:
            # create grid of points with zero value around face
            h = 0.01
            x_r = vert[:, 0].max() * 10. / 8.
            y_r = vert[:, 1].max() * 9.5 / 8.
            fine_grid = Grid(x1_min=-x_r,
                             x1_max=x_r + h,
                             x2_min=-y_r,
                             x2_max=y_r + h,
                             h=h,
                             to_float32=True).x

            # only retain points outside face convex hull
            aux_delaunay = scipy.spatial.Delaunay(points)
            fine_grid = fine_grid[aux_delaunay.find_simplex(fine_grid) < 0]
            # add zeros around face
            new_vertices = np.concatenate(
                (fine_grid, np.zeros((fine_grid.shape[0], 1))), axis=1)
            vert = np.concatenate((vert, new_vertices), axis=0)

        vert = cls.fit_in_lattice(vert)

        points, values = vert[:, 0:2], vert[:, 2]

        return points, values

    @classmethod
    def fit_in_lattice(cls, vert):
        """
        Fit points in lattice.

        Args:
            vert (np.array):
                size: (M, 3) (points in the first two columns,
                              values in the third)

        Returns:
            vert (np.array):
                scaled vertices that fit in lattice.
        """
        # normalize face to fit in lattice
        hw_ratio = (vert[:, 1].max() - vert[:, 1].min()) / \
                   (vert[:, 0].max() - vert[:, 0].min())
        _, _, x_max, y_max = cls.get_data_boundaries(hw_ratio=hw_ratio,
                                                     pad=0.03)

        # recenter data
        x_mean = (vert[:, 0].max() + vert[:, 0].min()) / 2
        y_mean = (vert[:, 1].max() + vert[:, 1].min()) / 2
        vert[:, 0] = vert[:, 0] - x_mean
        vert[:, 1] = vert[:, 1] - y_mean

        # x,y scaling factors
        # vert[i,0] should be within (-x_max, x_max)
        # vert[i,1] should be within (-y_max, y_max)
        x_norm = x_max / vert[:, 0].max()
        y_norm = y_max / vert[:, 1].max()

        if x_norm < y_norm:
            vert = vert * x_norm
        else:
            vert = vert * y_norm

        return vert

    @staticmethod
    def get_data_boundaries(hw_ratio=math.sqrt(3), pad=0.1):
        """
        Get the data boundaries that allow fitting the data in centered
        rectangular region of the lattice with a specified height/width ratio,
        so as to maximize occupied space within the interior lattice.
        Pad a given distance from the limits if pad > 0.

        Takes into account geometry of hexagonal lattice:
        if hw_ratio > math.sqrt(3), the data touches the upper and bottom
        interior border; otherwise, it touch the left and right borders.

        Args:
            hw_ratio (float>0):
                height/width ratio of rectangular region.
            pad (float>=0):
                distance to pad from the limits of the region.

        Returns:
            4-tuple (x_min, x_max, y_min, y_max): data boundaries
        """
        # requires that lattice is hexagonal and lsize*h = 1 (enforced)
        bottom_right_std = Lattice.bottom_right_std

        if hw_ratio > math.sqrt(3):  # from geometry maximize space usage
            y_min = bottom_right_std[1]
            x_min = y_min * (1. / hw_ratio)
        else:
            a = (bottom_right_std[0] * 2) / (1 + hw_ratio * math.sqrt(3) / 3)
            x_min = -a
            y_min = x_min * hw_ratio

        x_min, y_min = x_min + pad, y_min + pad
        x_max, y_max = -x_min, -y_min

        return x_min.item(), y_min.item(), x_max.item(), y_max.item()
