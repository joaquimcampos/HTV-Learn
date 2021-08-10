import os
import torch
import numpy as np
import math
import scipy

from htvlearn.lattice import Lattice
from htvlearn.delaunay import Delaunay
from htvlearn.grid import Grid


class Hex():
    """ Hexagonal lattice vectors """
    v1 = np.array([1., 0.])
    v2 = np.array([0.5, math.sqrt(3) / 2])


class BoxSpline():
    center_points = np.array([0., 0.])
    border_points = np.array([Hex.v1, Hex.v2, -Hex.v1 + Hex.v2,
                              -Hex.v1, -Hex.v2, -Hex.v2 + Hex.v1])
    points = np.vstack((center_points, border_points))
    values = np.array([math.sqrt(3) / 2, 0., 0., 0., 0., 0., 0.])
    htv = 12


class NormalizedBoxSpline(BoxSpline):
    values = BoxSpline.values.copy()
    values[0] = 1.
    htv = 12 / (math.sqrt(3) / 2)


class DistortedBoxSpline():
    np.random.seed(3)
    center_points = np.array([0., 0.]) + np.random.uniform(-0.2, 0.2, (2, ))
    border_points = np.array([Hex.v1, Hex.v2, -Hex.v1 + Hex.v2,
                              -Hex.v1, -Hex.v2, -Hex.v2 + Hex.v1]) + \
        np.random.uniform(-0.2, 0.2, (6, 2))
    points = np.vstack((center_points, border_points))
    values = np.array([math.sqrt(3) / 2, 0., 0., 0., 0., 0., 0.])


class Pyramid():
    points = np.vstack((BoxSpline.center_points,
                        BoxSpline.border_points,
                        2 * BoxSpline.border_points,
                        3 * BoxSpline.border_points))
    values = np.array([2., 1., 1., 1., 1., 1., 1.,
                       0., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., ])
    htv = 16 * math.sqrt(3)


class CutPyramid(Pyramid):
    values = Pyramid.values.copy()
    values[0] = 1.


class SoftPyramid(Pyramid):
    values = Pyramid.values.copy()
    values[7:13] = 0.5
    values[13::] = 0.


class SimpleJunction():
    points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.],
                       [0., 3. / 4], [1., 1. / 4]])
    values = np.array([0., 2. / 3, 2. / 3, 0., 1., 1.])
    # gradients
    a1_affine_coeff = np.array([2. / 3, 4. / 3., 0.])
    a2_affine_coeff = np.array([-2. / 3, -4. / 3., 2.])
    htv = 10. / 3


class AnotherJunction():
    points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    values = np.array([0., 1., 1., 0.])
    htv = 4


class AssymetricSimpleJunction(SimpleJunction):
    values = np.array([1., 1., 2. / 3, 0., 1., 1.])
    a1_affine_coeff = np.array([0., 0., 0.])
    htv = 5. / 3


def init_distorted_grid(size_=(3, 3), range_=(-1, 1)):
    """ """
    assert isinstance(size_, tuple)
    assert len(size_) == 2
    u = np.linspace(*range_, size_[0]) * 1.
    v = np.linspace(*range_, size_[1]) * 1.
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()

    mask = np.ma.mask_or(np.abs(u) == u.max(), np.abs(v) == v.max())
    # noisy
    x, y = u, v
    for vec in [x, y]:
        noise = (np.random.rand(*u.shape) - 0.5) * (u[1] - u[0])
        # don't add noise to boundary vertices
        vec[~mask] = vec[~mask] + noise[~mask]

    return np.hstack((x[:, np.newaxis], y[:, np.newaxis]))


class RealData:
    points = init_distorted_grid(size_=(3, 3))
    values = (np.random.rand(points.shape[0], ) - 0.5)


class Data():
    def __init__(self,
                 data_from_ckpt=None,
                 dataset_name=None,
                 num_train=None,
                 data_dir='./data',
                 valid_fraction=0.2,
                 test_as_valid=False,
                 noise_ratio=0.,
                 seed=-1,
                 verbose=False,
                 **kwargs):
        """
        Args:
            data_from_ckpt:
            dataset_name:
            num_train:
            data_dir:
            valid_fraction:
            test_as_valid:
            noise_ratio:
            seed:
            verbose:
        """
        # TODO: Complete Args in docstring
        self.data_from_ckpt = data_from_ckpt

        self.dataset_name = dataset_name
        self.num_train = num_train

        if self.data_from_ckpt is None:
            assert self.dataset_name is not None, 'self.dataset_name is None.'
            assert self.num_train is not None, 'self.num_train is None.'

        self.data_dir = data_dir
        self.valid_fraction = valid_fraction
        self.test_as_valid = test_as_valid
        self.noise_ratio = noise_ratio
        self.seed = seed
        self.verbose = verbose

        # set seeds
        if self.seed >= 0:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        self.train, self.valid, self.test = {}, {}, {}
        self.delaunay = {}  # points and values for delaunay triangulation

        if self.data_from_ckpt is not None:
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
        """ """
        if not bool(self.delaunay):

            if self.dataset_name.startswith('pyramid'):
                self.delaunay['points'], self.delaunay['values'] = \
                    self.init_pyramid()

            elif self.dataset_name.endswith('planes'):
                self.delaunay['points'], self.delaunay['values'] = \
                    self.init_planes()  # TODO

            elif 'face' in self.dataset_name:
                self.delaunay['points'], self.delaunay['values'] = \
                    self.init_face(cut=True
                                   if 'cut' in self.dataset_name
                                   else False)

        self.cpwl = Delaunay(points=self.delaunay['points'],
                             values=self.delaunay['values'])

        if not bool(self.test):
            if not self.cpwl.has_rectangular_range:
                # generate uniformly distributed samples
                # in cpwl convex set
                self.test['input'] = self.generate_uniform_data(3350)
            else:
                # test set is sampled on a grid inside the convex hull of cpwl
                self.test['input'] = self.cpwl.get_grid(h=0.01,
                                                        to_numpy=False,
                                                        to_float32=True).x

            self.test['values'] = self.cpwl.evaluate(self.test['input'])

            print(f'nb. of test data points : {self.test["input"].size(0)}')

            if not bool(self.valid) and self.test_as_valid:
                self.valid['input'] = self.test['input'].clone()
                self.valid['values'] = self.test['values'].clone()

        if not bool(self.train):
            # generate num_train_valid_samples uniformly distributed
            # in cpwl convex set
            num_train_valid_samples = int(self.num_train)
            x = self.generate_uniform_data(num_train_valid_samples)

            # training / validation split indices
            if not self.test_as_valid:
                split_idx = int((1 - self.valid_fraction) *
                                num_train_valid_samples)
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

                for i in range(len(gap_x_range)):
                    gap_mask = (
                        (self.train['input'][:, 0] >= gap_x_range[i][0]) *
                        (self.train['input'][:, 0] <= gap_x_range[i][1]) *
                        (self.train['input'][:, 1] >= gap_y_range[i][0]) *
                        (self.train['input'][:, 1] <= gap_y_range[i][1]))

                    self.train['input'] = self.train['input'][~gap_mask]
                    self.train['values'] = self.train['values'][~gap_mask]

            if not np.allclose(self.noise_ratio, 0.):
                self.train['values'] = \
                    self.add_noise_to_values(self.train['values'])

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

    def generate_uniform_data(self, num_samples):
        """
        Generate uniformly distributed data inside convex set.

        Works by generating num_samples points and then rejecting the
        ones outside the convex set.

        Args:
            num_samples (int) (before possible rejection)
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
        """  """
        noise_std = self.noise_ratio * (values.max() - values.min())
        if self.verbose:
            print(f'Adding noise of standard deviation s = {noise_std}')
        noise = torch.empty_like(values).normal_(std=noise_std)

        return values + noise

    def init_pyramid(self):
        """ """
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

        points = (Lattice.hexagonal_matrix @ points.t()).t()

        if False:
            # linear term
            a, b = torch.tensor([.2, .2]) + torch.tensor([.1])
            values = values + (points * a.unsqueeze(0)).sum(1) + b

        # training is made of all pyramid vertices except apex
        self.train['input'] = points[:-1].clone()
        self.train['values'] = values[:-1].clone()
        # force validation set to be equal to test set
        self.test_as_valid = True

        return points.numpy(), values.numpy()

    def read_face(self, cut_eps=0.6):
        """
        Args:
            cut_eps:
                what height to cut face relative to its maximum height.
        """
        obj_file = os.path.join(self.data_dir, 'obj_free_male_head.obj')

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

    def init_face(self, cut=False):
        """ """
        vert = self.read_face()

        # normalize face to fit in [-0.8, 0.8]^2 square
        max_ = max(np.abs(vert[:, 0]).max(), np.abs(vert[:, 1]).max())
        vert = vert / max_ * 0.8

        if cut is True:
            cpwl_aux = Delaunay(points=vert[:, 0:2].copy(),
                                values=vert[:, 2].copy())

            x_min, x_max = -0.324, 0.324
            y_min, y_max = -0.45, 0.419

            mask = (vert[:, 0] > x_min) * (vert[:, 0] < x_max) * \
                (vert[:, 1] > y_min) * (vert[:, 1] < y_max)
            vert = vert[mask]

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
            fine_grid = fine_grid[
                aux_delaunay.find_simplex(fine_grid) < 0].numpy()
            # add zeros around face
            new_vertices = np.concatenate(
                (fine_grid, np.zeros((fine_grid.shape[0], 1))), axis=1)
            vert = np.concatenate((vert, new_vertices), axis=0)

        vert = self.fit_in_lattice(vert)

        return vert[:, 0:2], vert[:, 2]

    def fit_in_lattice(self, vert):
        """ """
        # normalize face to fit in lattice
        hw_ratio = (vert[:, 1].max() - vert[:, 1].min()) / \
                   (vert[:, 0].max() - vert[:, 0].min())
        _, _, x_max, y_max = self.get_data_boundaries(hw_ratio=hw_ratio,
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

    def get_data_boundaries(self, hw_ratio=math.sqrt(3), pad=0.1):
        """ Get the data boundaries in standard coordinates for data
        (pad distance from boundary)
        in centered rectangular region with a specified height/width ratio,
        so as to maximize occupied space within the pad-interior lattice.

        Takes into account geometry of hexagonal lattice:
        if hw_ratio > math.sqrt(3), the data touches the upper and bottom
        interior border; otherwise, it touch the left and right borders.
        Args::
            hw_ratio: height/width of rectangular region.
        Returns:
            tuple (x_min, x_max, y_min, y_max)
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
