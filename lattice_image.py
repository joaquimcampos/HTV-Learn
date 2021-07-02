import sys
import torch
import numpy as np

import htv_utils


class LatticeImage():
    """ Lattice Image class

    Cretes an image from a padded lattice (so as to have good boundary conditions),
    and allows to compute the discrete htv of that image.
    """

    def __init__(self, lattice_obj, pixel_step=0.01, **kwargs):
        """ """
        self.lat = lattice_obj
        self.pixel_step = pixel_step

        self.create_image()



    def create_image(self):
        """ """
        assert self.lat.is_padded()
        # top right corners of lattice to define image range;
        # notice that for linear maps T([0, 0]) = [0, 0] so bottom left corner
        # has same [0, 0] standard coordinates.
        left_bottom_lat = torch.tensor([[self.lat.lmin, self.lat.lmin]])
        top_right_lat = torch.tensor([[self.lat.lmax, self.lat.lmax]])

        left_bottom_std = self.lat.lattice_to_standard(left_bottom_lat.float()).squeeze()
        top_right_std = self.lat.lattice_to_standard(top_right_lat.float()).squeeze()

        x1_std = torch.arange(left_bottom_std[0], top_right_std[0], self.pixel_step)
        x2_std = torch.arange(left_bottom_std[1], top_right_std[1], self.pixel_step)

        x_std, Xsq_size = htv_utils.get_grid(x1_std, x2_std)
        x_lat = self.lat.standard_to_lattice(x_std)
        x_values = torch.zeros(x_lat.size(0))

        x_lat_min, _ = x_lat.min(1) # minimum along dim=1
        x_lat_max, _ = x_lat.max(1) # maximum along dim=1

        in_lattice_mask = (x_lat_min > self.lat.lmin) * (x_lat_max < self.lat.lmax)

        self.image_lat = x_lat
        self.in_lattice_mask = in_lattice_mask # points within lattice

        x_values[in_lattice_mask], _ = \
            self.lat.get_values_from_interpolation(x_lat[in_lattice_mask])

        self.Xsq_size = Xsq_size
        self.image_std = x_std
        self.image_values = x_values



    def discrete_hessian(self, Z):
        """ Computes the Hessian [Hx]i = [[[delta_r1r1 x]_i, [delta_r1r2 x]_i],
                                          [[delta_r2r1 x]_i, [delta_r2r2 x]_i]]

        at each pixel location.

        Args:
            Z: size (x2size, x1size); input, required to be in standard pixel space
        Returns:
            H - size (x2size, x1size, 2, 2); Hessian matrix on each point

        Note: Indexing (i, j) = (x, y)
        """
        Z_size = Z.size()
        assert len(Z_size) == 2
        Hess = torch.zeros(*Z_size, 2, 2)
        Z_padded = torch.zeros(Z_size[0] + 2, Z_size[1] + 2)
        Z_padded[0:-2, 0:-2] = Z

        Ax0y0 = Z
        Ax1y0 = Z_padded[1:-1, 0:-2]
        Ax0y1 = Z_padded[0:-2, 1:-1]
        Ax1y1 = Z_padded[1:-1, 1:-1]
        Ax2y0 = Z_padded[2::, 0:-2]
        Ax0y2 = Z_padded[0:-2, 2::]

        assert Ax1y0.size() == Z_size
        assert Ax0y1.size() == Z_size
        assert Ax1y1.size() == Z_size
        assert Ax2y0.size() == Z_size
        assert Ax0y2.size() == Z_size

        h = 1 # self.pixel_step * self.pixel_step

        Hess[:, :, 0, 0] = (Ax2y0 - 2*Ax1y0 + Ax0y0).div(h)
        Hess[:, :, 1, 0] = (Ax1y1 - Ax1y0 - Ax0y1 + Ax0y0).div(h)
        Hess[:, :, 0, 1] = Hess[:, :, 1, 0]
        Hess[:, :, 1, 1] = (Ax0y2 - 2*Ax0y1 + Ax0y0).div(h)

        return Hess



    def discrete_htv(self):
        """ Computes discrete htv for p=1, A.K.A. nuclear norm,
        since ||A||_(*) = trace(sqrt(A*A)) = sum_{i=1}^{min{m,n}} singular_i(A)
        in the case of the Hessian (hermitian A=A*), then singular_i = sqrt(eig_i(A*A))
        = sqrt(eig_i(A^2)) = sqrt(eig(A)^2) = |eig(A)|,
        so ||Hx||_Sp = ||H||_(*) = sum_{i=1}^{n} eig_i(A) (sum of eigenvalues)

        Args:
            Z: size (x1size, x2size, 2); "image" values.
        Returns:
            total_discrete_htv - sum of all pixelwise_discrete_htv
        """
        Z = self.image_values.reshape(self.Xsq_size)
        hess = self.discrete_hessian(Z)

        # use numpy to allow batch eigenvalue computation
        hess_eig_np, _ = np.linalg.eig(hess.numpy())
        hess_eig = torch.from_numpy(hess_eig_np)
        assert hess_eig.size() == (*Z.size(), 2)

        pixelwise_discrete_htv = hess_eig.abs().sum(dim=-1)
        assert pixelwise_discrete_htv.size() == Z.size()
        total_discrete_htv = pixelwise_discrete_htv.sum()

        return total_discrete_htv, pixelwise_discrete_htv
