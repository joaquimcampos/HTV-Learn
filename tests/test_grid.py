import pytest
import numpy as np
from htvlearn.grid import Grid


@pytest.fixture(scope='module', params=[True, False])
def grid(request):
    grid_params = {
        'x1_min': -2,
        'x1_max': 3,
        'x2_min': -2.5,
        'x2_max': 1.7,
        'h': 0.01
    }
    return Grid(**grid_params, square=request.param)


class TestGrid:
    @pytest.mark.grid
    def test_grid(self, grid):
        assert grid.x.shape[0] == np.prod(np.array(grid.meshgrid_size))
        assert grid.x.shape[1] == 2
        assert grid.is_int is False
        if grid.square is True:
            assert grid.x1_vec.shape == grid.x2_vec.shape
