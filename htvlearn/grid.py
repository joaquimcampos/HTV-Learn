import torch
from htvlearn.htv_utils import frange


class Grid:
    def __init__(self,
                 x1_min=-2,
                 x1_max=2,
                 x2_min=-2,
                 x2_max=2,
                 h=0.01,
                 square=False,
                 to_numpy=True,
                 to_float32=False,
                 **kwargs):
        """
        Class for 2D grids.

        Args:
            x1_min, x1_max (float):
                ranges for x coordinate.
            x2_min, x2_max (float)
                ranges for y coordinate.
            h (float):
                grid spacing.
            square (bool):
                if True, make grid square (overrides x2_min, x2_max).
            to_numpy (bool):
                if True, convert grid from torch.tensor to np.array.
            to_float32 (bool):
                if True, convert grid from float64 to float32.
        """
        self.square = square
        self.x1_min = x1_min
        self.x2_min = x1_min if square is True else x2_min
        self.x1_max = x1_max
        self.x2_max = x1_max if square is True else x2_max
        self.h = h
        self.to_numpy = to_numpy
        self.to_float32 = to_float32

        self.generate_grid()

    def generate_grid(self):
        """
        Generate a 2D grid.

        Saves:
            x:
                grid points (size (m, 2))
            x1_vec, x2_vec:
                vectors that originated grid.
            meshgrid_size:
                size of the meshgrid tensor.
        """
        x1_list = list(frange(self.x1_min, self.x1_max, self.h))
        if (isinstance(self.x1_min, int) and
                isinstance(self.x1_max, int) and
                isinstance(self.h, int)):
            self.x1_vec = torch.tensor(x1_list, dtype=torch.int64)
        else:
            self.x1_vec = torch.tensor(x1_list, dtype=torch.float64)

        x2_list = list(frange(self.x2_min, self.x2_max, self.h))
        if (isinstance(self.x2_min, int) and
                isinstance(self.x2_max, int) and
                isinstance(self.h, int)):
            self.x2_vec = torch.tensor(x2_list, dtype=torch.int64)
        else:
            self.x2_vec = torch.tensor(x2_list, dtype=torch.float64)

        # Do not use to_float32=True when computing HTV (precision problems)
        if self.to_float32 is True and self.x1_vec.dtype == torch.float64:
            self.x1_vec = self.x1_vec.float()
            self.x2_vec = self.x2_vec.float()

        X2, X1 = torch.meshgrid(self.x2_vec, self.x1_vec, indexing='ij')
        self.meshgrid_size = tuple(X1.size())
        self.x = torch.cat((X1.reshape(-1, 1), X2.reshape(-1, 1)), dim=1)

        if self.to_numpy is True:
            self.x = self.x.numpy()
            self.x1_vec = self.x1_vec.numpy()
            self.x2_vec = self.x2_vec.numpy()
