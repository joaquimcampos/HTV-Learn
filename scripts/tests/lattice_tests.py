#!/usr/bin/env python3

import sys
import time
import torch
import warnings

from lattice import Lattice


class LatticeTests(Lattice):

    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)



    def transform_tests(self):
        """ """
        print('\nStart transform_tests().')

        x_lat, _, _ = self.generate_fine_square_grid()

        x_std = self.lattice_to_standard(x_lat)
        assert x_std.size() == x_lat.size()

        x_lat_hat = self.standard_to_lattice(x_std)
        assert x_lat_hat.allclose(x_lat, atol=5e-4)

        print('\n----transform_tests(): Passed.----')



    def check_my_triangle_test(self):
        """ """
        print('\nStart check_my_triangle_test().')

        print('Automatic test.')
        x_lat,_,_ = self.generate_fine_square_grid()

        print('1.halfplane method')
        _, vertices_halfplane, second_mask_halfplane = \
            self.check_my_triangle(x_lat, check_method='halfplane')

        print('2.barycentric method')
        _, vertices_barycentric, second_mask_barycentric = \
            self.check_my_triangle(x_lat, check_method='barycentric')

        assert torch.equal(vertices_halfplane, vertices_barycentric)
        assert torch.equal(second_mask_halfplane, second_mask_barycentric)

        print('Manual test.')
        x_lat_manual = torch.tensor([ [-0.9, -1.9],
                                      [-0.48, -1.48],
                                      [0.001, 1.001],
                                      [0.25, 1.25],
                                      [0.48, 1.51],
                                      [0.50, 1.51],
                                      [0.999, 1.999]])

        v1, v2, v3, v4 = [0, 1], [1, 1], [1, 2], [0, 2]
        first = [v1, v2, v4]
        second = [v2, v3, v4]
        neg_first = [[-1, -2], [0, -2], [-1, -1]]
        neg_second = [[0, -2], [0, -1], [-1, -1]]
        vertices_gt = torch.tensor([neg_first, neg_second, first, first, first, second, second]) # ground truth
        assert vertices_gt.size() == (x_lat_manual.size(0), 3, 2)
        mask_gt = torch.tensor([0, 1, 0, 0, 0, 1, 1])

        print('1.halfplane method')
        _, vertices_halfplane, second_mask_halfplane = \
            self.check_my_triangle(x_lat_manual, check_method='halfplane')

        print('2.barycentric method')
        _, vertices_barycentric, second_mask_barycentric = \
            self.check_my_triangle(x_lat_manual, check_method='barycentric')

        assert torch.equal(vertices_halfplane, vertices_barycentric)
        assert torch.equal(second_mask_halfplane, second_mask_barycentric)

        assert torch.equal(vertices_halfplane, vertices_gt)
        assert torch.equal(second_mask_halfplane, mask_gt)

        print('\n----check_my_triangle_test(): Passed.----')



    def get_values_test(self):
        """ """
        print('\nStart get_values_test().')

        print('Automatic test.')
        x_lat, _, _ = self.generate_fine_square_grid()

        print('1.halfplane method')
        halfplane_start = time.time()
        x_values_halfplane, vertices_halfplane = \
            self.get_values_from_interpolation(x_lat, check_method='halfplane',
                                                                    in_origin=True)
        halfplane_end = time.time()

        print('2.barycentric method')
        barycentric_start = time.time()
        x_values_barycentric, vertices_barycentric = \
            self.get_values_from_interpolation(x_lat, check_method='barycentric',
                                                                    in_origin=True)
        barycentric_end = time.time()

        assert torch.equal(x_values_halfplane, x_values_barycentric)
        assert torch.equal(vertices_halfplane, vertices_barycentric)

        time_ratio = (barycentric_start - barycentric_end) / (halfplane_start - halfplane_end)
        print('time barycentric / time halfplane: {:.4f}'.format(time_ratio))

        del x_values_barycentric, vertices_barycentric

        print('3.halfplane method, in_origin=False')
        halfplane2_start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x_values_halfplane2, vertices_halfplane2 = \
                self.get_values_from_interpolation(x_lat, check_method='halfplane',
                                                                    in_origin=False)
        halfplane2_end = time.time()

        assert torch.allclose(x_values_halfplane2, x_values_halfplane, atol=5e-4)
        assert torch.equal(vertices_halfplane2, vertices_halfplane)

        time_ratio2 = (halfplane2_start - halfplane2_end) / (halfplane_start - halfplane_end)
        print('time halfplane, in_origin=False / '
              'time halfplane, in_origin=True: {:.4f}'.format(time_ratio2))

        del x_values_halfplane2, vertices_halfplane2, vertices_halfplane


        for use_saved_affine in [False, True]:

            print(f'\nuse_saved_affine = {use_saved_affine}')

            print('4.affine_solve method')
            affine_solve_start = time.time()
            x_values_affine_solve = \
                self.get_values_from_affine_coefficients(x_lat, plane_method='solve',
                                                        use_saved_affine=use_saved_affine)
            affine_solve_end = time.time()

            assert torch.allclose(x_values_affine_solve, x_values_halfplane, atol=1e-3)

            time_ratio3 = (affine_solve_start - affine_solve_end) / (halfplane_start - halfplane_end)
            print('time affine_solve / time halfplane: {:.4f}'.format(time_ratio3))

            print('5.affine_normal method')
            affine_normal_start = time.time()
            x_values_affine_normal = \
                self.get_values_from_affine_coefficients(x_lat, plane_method='normal',
                                                            use_saved_affine=use_saved_affine)
            affine_normal_end = time.time()

            assert torch.allclose(x_values_affine_normal, x_values_halfplane, atol=1e-3)

            time_ratio4 = (affine_normal_start - affine_normal_end) / (halfplane_start - halfplane_end)
            print('time affine_normal / time halfplane: {:.4f}'.format(time_ratio4))


        print('\n----get_values_test(): Passed.----')



    def get_plane_coefficients_test(self):
        """ """
        print('\nStart get_plane_coefficients_test().')

        x_lat, _, _ = self.generate_fine_square_grid()

        plane_normal_start = time.time()
        plane_coeff_normal = self.get_plane_coefficients(x_lat, plane_method='normal')
        plane_normal_end = time.time()

        plane_solve_start = time.time()
        plane_coeff_solve = self.get_plane_coefficients(x_lat, plane_method='solve')
        plane_solve_end = time.time()

        assert torch.allclose(plane_coeff_normal, plane_coeff_solve, atol=5e-4)

        time_ratio = (plane_solve_end-plane_solve_start) / (plane_normal_end-plane_normal_start)
        print('\ntime plane_solve / time plane_normal: {:.4f}'.format(time_ratio))

        print('\n----get_plane_coefficients_test(): Passed.----')


    def divide_lattice_test(self):
        """ """
        print('\nStart divide_lattice_test().')

        prev_C_mat = self.C_mat
        print('Previous (lsize, lmin, lmax) :', f'({self.lsize}, {self.lmin}, {self.lmax})')

        self.divide_lattice()

        assert self.C_mat.size(0) == (prev_C_mat.size(0) - 1) * 2 + 1
        assert torch.allclose(self.C_mat[0::2, 0::2], prev_C_mat)

        print('New (lsize, lmin, lmax) :', f'({self.lsize}, {self.lmin}, {self.lmax})')

        print('\n----divide_lattice_test(): Passed.----')


#########################################################################

if __name__ == '__main__':

    t = LatticeTests(lsize=500, h=1)
    t.transform_tests()
    t.check_my_triangle_test()
    t.get_values_test()
    t.get_plane_coefficients_test()

    t_padded = LatticeTests(lsize=496)
    t_padded.divide_lattice_test()
