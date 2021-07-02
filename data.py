import os
import torch
import copy
import math
import numpy as np
from PIL import Image
import scipy.spatial

import htv_utils
from delaunay import Delaunay


class Data():

    def __init__(self, lattice_obj, data_from_ckpt=None, **params):
        """
        Args:
            lattice_obj: Object of class lattice
            params: data parameters
        """
        self.params = params
        self.lat = lattice_obj
        self.data_from_ckpt = data_from_ckpt

        htv_utils.set_attributes(self, ['dataset_name', 'only_vertices', 'num_train',
                                        'valid_fraction', 'add_noise', 'noise_ratio',
                                        'no_linear', 'seed', 'verbose'])

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.train, self.valid, self.test = {}, {}, {}

        if self.data_from_ckpt is not None:
            assert 'train' in self.data_from_ckpt
            assert 'valid' in self.data_from_ckpt
            assert 'test' in self.data_from_ckpt

            self.train = self.data_from_ckpt['train']
            self.valid = self.data_from_ckpt['valid']
            self.test = self.data_from_ckpt['test']

        if self.dataset_name.startswith('pyramid'):
            self.define_pyramid()

        elif self.dataset_name.endswith('planes') or self.dataset_name == 'face':
            self.define_triangulation_data()

        elif self.dataset_name == 'noisy_linear':
            self.define_noisy_linear()

        # Best linear interpolator of the data
        m = self.train['input'].size(0)
        # create extended matrix mth-row: [x1, x2, 1]
        train_A = torch.cat((self.train['input'], torch.ones(m, 1)), dim=1)
        beta_np, _,_,_ = np.linalg.lstsq(train_A.numpy(), self.train['values'], rcond=None)
        beta = torch.from_numpy(beta_np)

        n = self.test['input'].size(0)
        test_A = torch.cat((self.test['input'], torch.ones(n, 1)), dim=1)
        output = torch.mv(test_A, beta)
        mse, _ = htv_utils.compute_mse_psnr(output, self.test['values'])

        if self.verbose:
            print(f'\nBest linear interpolator Test mse: {mse}')


    def define_noisy_linear(self):
        """ """
        self.a = torch.tensor([.3, .2])
        self.b = torch.tensor([.1])

        x_lat = torch.empty((int(self.num_train), 2))
        x_lat.uniform_(self.lat.lmin, self.lat.lmax)
        input = self.lat.lattice_to_standard(x_lat)
        values = self.evaluate_linear(input)

        self.noise_std = self.noise_ratio * (values.max() - values.min())
        noisy_values = self.add_noise_to_values(values)

        if self.data_from_ckpt is None:
            self.train['input'] = input
            self.train['values'] = noisy_values
            self.valid = copy.deepcopy(self.train)
            self.test = copy.deepcopy(self.train)

        mse, _ = htv_utils.compute_mse_psnr(noisy_values, values)
        if self.verbose:
            print(f'Noisy linear noisy-target mse  : {mse}')


    def define_pyramid(self):
        """ """
        assert self.lat.lsize * self.lat.h == 1
        if not self.lat.lsize % 20 == 0:
            raise ValueError(f'lat size {self.lat.lsize} is not a multiple of 20.')
        h = self.lat.lsize/10

        input = torch.tensor([[h, 0.], [0., h], [h, -h],
                                [0., -h], [-h, 0.], [-h, h],
                                [2*h, 0.],  [0., 2*h],  [2*h, -2*h],
                                [0., -2*h], [-2*h, 0.], [-2*h, 2*h]])

        ht = 2*h*self.lat.h
        values = torch.tensor([ht+.1, ht+.1, ht+.1, ht+.1, ht+.1, ht+.1,
                                .1, .1, .1, .1, .1, .1])

        if self.dataset_name == 'pyramid_ext': # extended pyramid dataset
            input_ext = torch.tensor([[3*h, 0.],  [0., 3*h],  [2.95*h, -2.85*h],
                                    [0., -3*h], [-3*h, 0.], [-3*h, 3*h]])
            input = torch.cat((input, input_ext), dim=0)

            values_ext = torch.tensor([.1, .1, .1, .1, .1, .1])
            values = torch.cat((values, values_ext), dim=0)

        self.a, self.b = torch.zeros(2), torch.zeros(1)
        if not self.no_linear:
            self.a, self.b = torch.tensor([.2, .2]), torch.tensor([.1])
            values = values + self.evaluate_linear(self.lat.lattice_to_standard(input))

        if self.data_from_ckpt is None:
            self.train['input'] = input
            self.train['values'] = values
            self.valid = copy.deepcopy(self.train)
            self.test = copy.deepcopy(self.train)

            self.lat.convert_to_standard_basis(self.train, self.valid, self.test)



    def define_triangulation_data(self):
        """ """
        if self.dataset_name.endswith('planes'):
            self.init_planes()

            # for planes, domain is whole lattice
            bottom_left_std = \
                self.lat.lattice_to_standard(torch.tensor([[self.lat.lmin, self.lat.lmin]]).float()).squeeze()
            upper_right_std = \
                self.lat.lattice_to_standard(torch.tensor([[self.lat.lmax, self.lat.lmax]]).float()).squeeze()

            x_min, x_max = bottom_left_std[0], upper_right_std[0]
            y_min, y_max = bottom_left_std[1], upper_right_std[1]

        elif self.dataset_name == 'face':
            # for face, domain is a square around it
            x_min, y_min, x_max, y_max = self.init_face()

        # test data is a fine grid in standard coordinates
        if self.dataset_name == 'face':
            step = (x_max-x_min)/100
        else:
            step = (x_max-x_min)/400

        x1 = torch.arange(x_min, x_max, step)
        x2 = torch.arange(y_min, y_max, step)

        # sample test set on standard grid but only retain
        # the samples inside the lattice
        x_test, _ = htv_utils.get_grid(x1, x2)
        # only select input inside latticce
        test_lattice_mask = \
            self.lat.get_lattice_mask(self.lat.standard_to_lattice(x_test))
        self.test['input'] = x_test[test_lattice_mask, :].clone()

        # training/validation data
        num_train_valid_samples = int(self.num_train)
        x = torch.empty((num_train_valid_samples, 2))

        if self.dataset_name.endswith('planes'):
            x.uniform_(self.lat.lmin, self.lat.lmax)
            x = self.lat.lattice_to_standard(x)
            assert self.lat.inside_lattice(self.lat.standard_to_lattice(x))
        else:
            x[:, 0].uniform_(x_min, x_max)
            x[:, 1].uniform_(y_min, y_max)
            train_lattice_mask = \
                self.lat.get_lattice_mask(self.lat.standard_to_lattice(x))
            x = x[train_lattice_mask, :].clone()

        # split idx for training and validation data
        split_idx = int((1-self.valid_fraction)*num_train_valid_samples)
        self.valid['input'] = x[(split_idx+1)::]

        # training data
        if self.only_vertices:
            # only use vertices of planes as training data
            self.train['input'] = self.vertices[:, 0:2]
        else:
            self.train['input'] = x[0:split_idx]

        print('\nNumber of training data points : {}'.format(self.train['input'].size(0)))
        print('Number of test data points : {}'.format(self.test['input'].size(0)))

        evaluate = self.delaunay.evaluate if self.dataset_name == 'face' else self.planes_function

        if 'values' not in self.test:
            self.test['values'] = evaluate(self.test['input'])

        if 'values' not in self.valid:
            self.valid['values'] = evaluate(self.valid['input'])

        if 'values' not in self.train:
            self.train['values'] = evaluate(self.train['input'])

            if self.add_noise is True:
                self.train['values'] = self.add_noise_to_values(self.train['values'])


    @staticmethod
    def read_face():

        obj_file = 'face_data/obj_free_male_head.obj'

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
        eps = 0.6
        min_height, max_height = unique_sort_vert[:, 2].min(), unique_sort_vert[:, 2].max()
        cutoff_val = min_height + (max_height - min_height) * eps
        cutoff_mask = np.where(unique_sort_vert[:, 2] > cutoff_val)[0]
        cleaned_vert = unique_sort_vert[cutoff_mask]
        cleaned_vert[:, 2] = cleaned_vert[:, 2] - cutoff_val # shift z.min() to z = 0
        cleaned_vert[:, 0] = cleaned_vert[:, 0] - cleaned_vert[:, 0].mean() # shift x around 0
        cleaned_vert[:, 1] = cleaned_vert[:, 1] - cleaned_vert[:, 1].mean() # shift t around 0

        return cleaned_vert



    def init_face(self):
        """ """
        vert = self.read_face()

        # normalize face to fit in lattice
        hw_ratio = vert[:, 1].max()/vert[:, 0].max()
        x_min, y_min, x_max, y_max = self.get_data_boundaries(hw_ratio=hw_ratio, pad=0.03)
        assert x_max > 0
        ratio = x_max/vert[:, 0].max()
        vert = vert * ratio

        # add zeros around face
        points = vert[:, 0:2]
        hull = scipy.spatial.ConvexHull(points)
        hull_points = points[hull.vertices]

        # add zeros to the convex hull contour of face
        for i in range(hull.vertices.shape[0]):
            frac = np.linspace(0.01, 0.99, num=99)[:, np.newaxis]
            next_vert = i+1 if i != hull.vertices.shape[0] - 1 else 0
            new_points = hull_points[next_vert][np.newaxis, :] * frac + \
                        hull_points[i][np.newaxis, :] * (1-frac)
            new_vertices = np.concatenate((new_points, np.zeros((new_points.shape[0],1))), axis=1)
            vert = np.concatenate((vert, new_vertices), axis=0)

        # create grid of points around face and assign zero values to these
        x_min, x_max = hull_points[:, 0].min()-0.04, hull_points[:, 0].max()+0.04
        y_min, y_max = hull_points[:, 1].min()-0.04, hull_points[:, 1].max()+0.04

        step = (x_max-x_min)/100
        x1 = torch.arange(x_min, x_max, step)
        x2 = torch.arange(y_min, y_max, step)

        # only retain the samples inside the lattice
        fine_grid, _ = htv_utils.get_grid(x1, x2)
        # only select input inside latticce
        lattice_mask = self.lat.get_lattice_mask(self.lat.standard_to_lattice(fine_grid))
        fine_grid = fine_grid[lattice_mask, :].clone()

        # only retain points outside face convex hull
        aux_delaunay = scipy.spatial.Delaunay(points)
        fine_grid = fine_grid[aux_delaunay.find_simplex(fine_grid)<0]
        new_vertices = np.concatenate((fine_grid, np.zeros((fine_grid.shape[0],1))), axis=1)
        vert = np.concatenate((vert, new_vertices), axis=0)

        # add lattice vertices
        lat_corners = torch.tensor([
                                    [self.lat.lmin, self.lat.lmin], # 8
                                    [self.lat.lmax, self.lat.lmin], # 9
                                    [self.lat.lmin, self.lat.lmax], # 10
                                    [self.lat.lmax, self.lat.lmax] # 11
                                    ])
        # standard coordinates
        std_corners = self.lat.lattice_to_standard(lat_corners.float())
        new_vertices = torch.cat((std_corners, torch.zeros(std_corners.size(0), 1)), dim=1)
        vert = np.concatenate((vert, new_vertices.numpy()), axis=0)

        self.delaunay = Delaunay(points=vert[:, 0:2], values=vert[:, 2])

        self.triangles = torch.from_numpy(self.delaunay.tri.simplices).long()
        self.vertices = torch.from_numpy(vert)
        self.affine_coeff = torch.from_numpy(self.delaunay.tri.simplices_affine_coeff)

        self.a, self.b = torch.zeros(2), torch.zeros(1)

        # return square around face
        return x_min, y_min, x_max, y_max


    def init_planes(self):
        """ Initialize planes vertice triangulation and values;
        initialize triangle centers, plane coefficients, and affine coefficients.
        Everything is in standard coordinates.
        """
        if self.dataset_name == 'quad_top_planes':

            pad = 0.08
            x_min, _, x_max, _ = self.get_data_boundaries(hw_ratio=0.01, pad=pad)
            _, y_min, _, y_max = self.get_data_boundaries(hw_ratio=100, pad=pad)

            dx = (x_max - x_min) / 100 # delta x step
            dy = (y_max - y_min) / 100 # delta y step

            # control points with values - (x1, x2, val)
            self.vertices = torch.tensor([
                                        [x_min+30*dx,    y_min+35*dy,    dx*20], # 0
                                        [x_max-40*dx,    y_min+30*dy,    dx*20], # 1
                                        [x_max-35*dx,    y_max-30*dy,    dx*20], # 2
                                        [x_min+40*dx,    y_max-30*dy,    dx*20], # 3
                                        [x_max-25*dx,    y_min+5*dy,     0.], # 4
                                        [x_min+25*dx,    y_max-5*dy,    0.] # 5
                                        ])

            # triangulation of the planes function (vertices for each triangle)
            # size (num_triangles, vertices)
            self.triangles = torch.tensor([
                                        [0, 1, 3], # A
                                        [1, 2, 3], # B
                                        [4, 1, 0], # C
                                        [0, 3, 5], # D
                                        [4, 2, 1], # E
                                        [3, 2, 5] # F
                                        ])

            # check values of vertices so that there is a seamless plane junction
            x_v6 = self.get_zero_loc(2, 3, zval=0)
            x_v7 = self.get_zero_loc(4, 5, zval=0)

            new_vertices = torch.tensor([
                                        [x_v6[0], x_v6[1], 0.], # 6
                                        [x_v7[0], x_v7[1], 0.] # 7
                                        ])
            self.vertices = torch.cat((self.vertices, new_vertices), dim=0)

            new_triangles = torch.tensor([
                                        [6, 4, 0],
                                        [6, 0, 5],
                                        [4, 7, 2],
                                        [2, 7, 5]
                                        ])
            self.triangles = torch.cat((self.triangles, new_triangles), dim=0)

            lat_corners = torch.tensor([
                                        [self.lat.lmin, self.lat.lmin], # 8
                                        [self.lat.lmax, self.lat.lmin], # 9
                                        [self.lat.lmin, self.lat.lmax], # 10
                                        [self.lat.lmax, self.lat.lmax] # 11
                                        ])
            # standard coordinates
            std_corners = self.lat.lattice_to_standard(lat_corners.float())
            new_vertices = torch.cat((std_corners, torch.zeros(std_corners.size(0), 1)), dim=1)

            self.vertices = torch.cat((self.vertices, new_vertices), dim=0)

            new_triangles = torch.tensor([
                                        [6, 5, 10],
                                        [5, 7, 10],
                                        [10, 7, 11],
                                        [4, 11, 7],
                                        [8, 4, 6],
                                        [8, 6, 10],
                                        [8, 9, 4],
                                        [9, 11, 4]
                                        ])

            self.triangles = torch.cat((self.triangles, new_triangles), dim=0)

            if self.add_noise:
                self.noise_std = self.noise_ratio * self.vertices[:, 2].max()
                print('noise_std: {:.3f}'.format(self.noise_std))

            self.a, self.b = torch.zeros(2), torch.zeros(1)
            if not self.no_linear:
                # add linear term to first vertices
                self.a, self.b = torch.tensor([0.1, 0.05]), torch.tensor([-0.05])
                self.vertices[:, 2] += self.evaluate_linear(self.vertices[:, 0:2])


        num_triangles = self.triangles.size(0)
        # size (num_triangles, 3, 3)
        self.triangles_with_values = \
            torch.cat(tuple(self.vertices[self.triangles[i]].unsqueeze(0) for i in range(num_triangles)), dim=0)

        # size (num_triangles, 3, 3)
        self.triangles_barycentric_mat = \
            self.lat.append_ones(self.triangles_with_values[:, :, 0:2], dim=-1).transpose(1, 2)

        self.inv_triangles_barycentric_mat = \
            torch.inverse(self.triangles_barycentric_mat)
        assert self.inv_triangles_barycentric_mat.size() == (self.triangles.size(0), 3, 3)

        self.triangle_centers = \
            self.lat.get_triangle_centers(self.triangles_with_values[:, :, 0:2])

        plane_coeff = self.lat.solve_method(self.triangles_with_values)
        self.affine_coeff = self.lat.get_affine_coeff_from_plane_coeff(plane_coeff)


    def get_zero_loc(self, tri1_idx, tri2_idx, zval=0):
        """ """
        # size (2, 3, 3)
        idx_vec = [tri1_idx, tri2_idx]
        triangles_with_values = \
            torch.cat(tuple(self.vertices[self.triangles[i]].unsqueeze(0) for i in idx_vec), dim=0)

        plane_coeff = self.lat.solve_method(triangles_with_values)
        affine_coeff = self.lat.get_affine_coeff_from_plane_coeff(plane_coeff) # (2, 3)
        assert affine_coeff.size() == (2,3)

        B = -affine_coeff[:, -1:] + zval
        A = affine_coeff[:, 0:2]
        x, _ = torch.solve(B, A)

        return x.squeeze(-1)



    def check_my_triangle(self, x):
        """ """
        # size (num_triangles, 3, 3) x (m, 1, 3, 1) = (m, num_triangles, 3, 1) -> (m, num_triangles, 3)
        x_barycentric_tensor = self.lat.append_ones(x, dim=-1).unsqueeze(1).unsqueeze(-1)
        # barycentric coordinates of each point x_m wrt all triangles
        x_barycentric_all = (self.inv_triangles_barycentric_mat @ x_barycentric_tensor).squeeze(-1)

        # check triangle where point x is (all barycentric coordinates >= 0).
        mask = torch.all((x_barycentric_all >= 0), dim=-1)
        inside_mask = torch.any((mask == True), dim=1)

        # get barycentric coordinates wrt triangle in which it is located.
        x_barycentric = x_barycentric_all[mask]
        triangle_idx = torch.argmax(mask.to(torch.int32), 1)
        assert triangle_idx.size() == (x.size(0),)

        return triangle_idx, inside_mask



    def planes_function(self, x):
        """ """
        triangle_idx, inside_mask = self.check_my_triangle(x)

        affine_coeff = torch.zeros(x.size(0), 3)
        triangle_idx_in = triangle_idx[inside_mask]
        affine_coeff_in = self.affine_coeff[triangle_idx_in] # size (m, )
        al1, al2, dl = affine_coeff_in[:, 0], affine_coeff_in[:, 1], affine_coeff_in[:, 2]

        x_values = torch.zeros(x.size(0))
        x_in = x[inside_mask]
        x_values[inside_mask] = al1*x_in[:, 0] + al2*x_in[:, 1] + dl # z = x1*a1' + x2*a2' + d'
        x_values[~inside_mask] = self.evaluate_linear(x[~inside_mask])

        return x_values



    def evaluate_linear(self, x_std):
        """ Evaluate linear term on x_std
        """
        return (x_std * self.a.view(1, 2)).sum(1) + self.b



    def add_noise_to_values(self, values):
        """ """
        if self.verbose:
            print(f'Adding noise of standard deviation = {self.noise_std}')
        self.snr = htv_utils.compute_snr(values, self.noise_std)
        if not np.allclose(self.noise_std, 0.):
            noise = torch.empty_like(values).normal_(std=self.noise_std)
            return values + noise
        else:
            return values



    def get_data_boundaries(self, hw_ratio=math.sqrt(3), pad=0.1):
        """ Get the data boundaries in standard coordinates for data
        (pad distance from boundary)
        in centered rectangular region with a specified height/width ratio, so as
        to maximize occupied space within the pad-interior lattice.

        Takes into account geometry of hexagonal lattice:
        if hw_ratio > math.sqrt(3), the data touches the upper and bottom
        interior border; otherwise, it touch the left and right borders.
        Args::
            hw_ratio: height/width of rectangular region.
        Returns:
            tuple (x_min, x_max, y_min, y_max)
        """
        assert np.allclose(self.lat.lsize * self.lat.h, 1.)
        if not self.lat.is_hexagonal:
            raise ValueError('Cannot use this method with non-hexagonal '
                            'lattices. Please set data boundaries manually...')

        # right bottom interior lattice point in lattice coordinates
        bottom_right_lat = \
            torch.tensor([[self.lat.lmax, self.lat.lmin]])

        # standard coordinates
        bottom_right_std = \
            self.lat.lattice_to_standard(bottom_right_lat.float()).squeeze()

        if hw_ratio > math.sqrt(3): # from geometry maximize space usage
            y_min = bottom_right_std[1]
            x_min = y_min * (1./hw_ratio)
        else:
            a = (bottom_right_std[0]*2) / (1 + hw_ratio*math.sqrt(3)/3)
            x_min = -a
            y_min = x_min * hw_ratio

        x_min, y_min = x_min+pad, y_min+pad
        x_max, y_max = -x_min, -y_min

        return x_min.item(), y_min.item(), x_max.item(), y_max.item()
