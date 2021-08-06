import torch
from htv_utils import frange


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
        """ 2D grid
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

    @property
    def is_int(self):
        """ """
        if isinstance(self.x1_min, int) and isinstance(self.h, int) and \
                isinstance(self.x2_min, int) and isinstance(self.h, int):
            return True
        return False

    def generate_grid(self):
        """ Generates a 2D grid and returns a list of (x1,x2) grid points

        Returns:
            x: list of (x1,x2) grid points (size (m, 2))
            x1_vec, x2_vec: vectors that originated grid.
            mesh_size: size of meshgrid tensor.
        """
        self.x1_vec = torch.tensor(
            list(frange(self.x1_min, self.x1_max, self.h)))
        self.x2_vec = torch.tensor(
            list(frange(self.x2_min, self.x2_max, self.h)))

        assert self.x1_vec.dtype == torch.float64 or \
            self.x1_vec.dtype == torch.int64

        # Do not use to_float32=True when computing HTV (precision problems)
        if self.to_float32 is True and self.x1_vec.dtype == torch.float64:
            self.x1_vec = self.x1_vec.float()
            self.x2_vec = self.x2_vec.float()

        X2, X1 = torch.meshgrid(self.x2_vec, self.x1_vec)
        self.meshgrid_size = tuple(X1.size())
        self.x = torch.cat((X1.reshape(-1, 1), X2.reshape(-1, 1)), dim=1)

        if self.to_numpy is True:
            self.x = self.x.numpy()
            self.x1_vec = self.x1_vec.numpy()
            self.x2_vec = self.x2_vec.numpy()
