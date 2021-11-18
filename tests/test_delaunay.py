import pytest
import numpy as np
import torch
import time

from htvlearn.delaunay import Delaunay
from htvlearn.plots.plot_cpwl import Plot
from htvlearn.data import (
    BoxSpline,
    SimplicialSpline,
    CutPyramid,
    SimpleJunction,
    DistortedGrid,
    Data
)


@pytest.fixture(autouse=True)
def set_seed(request):
    """Set random seed."""
    # Code that will run before
    seed = request.config.getoption("--seed")
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))


# toy datasets that have an htv attribute
toy_dataset_list = [BoxSpline, CutPyramid, SimpleJunction]
dataset_dict = {
    'toy': toy_dataset_list,
    'all': toy_dataset_list + [SimplicialSpline, DistortedGrid],
    'simple_junction': [SimpleJunction],
    'distorted_grid': [DistortedGrid]
}

# receives dataset as parameter


@pytest.fixture(scope="module")
def dataset(request):
    dt = request.param
    ret_dict = {
        'name': dt.__name__,
        'points': dt.points.copy(),
        'values': dt.values.copy()
    }

    if hasattr(dt, 'htv'):
        ret_dict['htv'] = dt.htv

    return ret_dict


@pytest.fixture(scope='module')
def skip_plot(request):
    if 'plot' not in request.config.getoption("-m"):
        raise pytest.skip('Skipping!')


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestDelaunay:
    @pytest.mark.plot
    @pytest.mark.parametrize("dataset", dataset_dict["all"], indirect=True)
    def test_plot_delaunay(self, dataset, request):
        """ """
        plot_arg = request.config.getoption("--plot")
        if plot_arg is None or plot_arg not in dataset['name']:
            pytest.skip()

        cpwl = Delaunay(**dataset)
        plot = Plot(log_dir='/tmp')
        plot.plot_delaunay(cpwl)

    def test_is_admissible(self):
        points, values = Data.init_zero_boundary_planes()
        values = Data.add_linear_func(points, values)
        cpwl = Delaunay(points=points, values=values)
        assert cpwl.is_admissible is True

    @pytest.mark.parametrize("dataset", dataset_dict["toy"], indirect=True)
    def test_exact_htv(self, dataset):
        """ """
        cpwl = Delaunay(**dataset)
        assert np.allclose(cpwl.get_exact_HTV(), dataset['htv'])

    @pytest.mark.parametrize("dataset", dataset_dict["all"], indirect=True)
    def test_exact_grad_trace_htv(self, dataset):
        """ """
        if dataset['name'].endswith('Junction') or \
                dataset['name'].endswith('DistortedGrid'):
            cpwl = Delaunay(**dataset)
        else:
            cpwl = Delaunay(**dataset, add_extreme_points=True)

        h = (cpwl.tri.points[:, 0].max() - cpwl.tri.points[:, 0].min()) / 5000
        exact_grad_trace_htv = cpwl.get_exact_grad_trace_HTV(h=h)
        exact_htv = cpwl.get_exact_HTV()
        print('(Discrete, Exact) : ({:.4f}, {:.4f})'
              .format(exact_grad_trace_htv, exact_htv))

        assert np.allclose(exact_grad_trace_htv, exact_htv, rtol=1e-3)

    @pytest.mark.parametrize("dataset", dataset_dict["all"], indirect=True)
    def test_lefkimiattis_HTV(self, dataset):
        """ """
        if dataset['name'].endswith('Junction') or \
                dataset['name'].endswith('DistortedGrid'):
            cpwl = Delaunay(**dataset)
        else:
            cpwl = Delaunay(**dataset, add_extreme_points=True)

        h = (cpwl.tri.points[:, 0].max() - cpwl.tri.points[:, 0].min()) / 5000
        lefkimiattis_htv = cpwl.get_lefkimiattis_schatten_HTV(h=h)
        exact_htv = cpwl.get_exact_HTV()
        print('(Discrete, Exact) : ({:.4f}, {:.4f})'
              .format(lefkimiattis_htv, exact_htv))

        assert not np.allclose(lefkimiattis_htv, exact_htv, rtol=1e-3)

    @pytest.mark.parametrize("dataset", dataset_dict["all"], indirect=True)
    def test_lefkimiattis_trace_HTV(self, dataset):
        """ """
        if dataset['name'].endswith('Junction') or \
                dataset['name'].endswith('DistortedGrid'):
            cpwl = Delaunay(**dataset)
        else:
            cpwl = Delaunay(**dataset, add_extreme_points=True)

        h = (cpwl.tri.points[:, 0].max() - cpwl.tri.points[:, 0].min()) / 5000
        lefkimiattis_trace_htv = cpwl.get_lefkimiattis_trace_HTV(h=h)
        exact_htv = cpwl.get_exact_HTV()
        print('(Discrete, Exact) : ({:.4f}, {:.4f})'
              .format(lefkimiattis_trace_htv, exact_htv))

        assert np.allclose(lefkimiattis_trace_htv, exact_htv, rtol=2e-3)

    @pytest.mark.parametrize("dataset", dataset_dict["all"], indirect=True)
    def test_exact_grad_schatten_HTV(self, dataset):
        """ """
        if dataset['name'].endswith('Junction') or \
                dataset['name'].endswith('DistortedGrid'):
            cpwl = Delaunay(**dataset)
        else:
            cpwl = Delaunay(**dataset, add_extreme_points=True)

        h = (cpwl.tri.points[:, 0].max() - cpwl.tri.points[:, 0].min()) / 5000
        exact_grad_schatten_htv = cpwl.get_exact_grad_schatten_HTV(h=h)
        exact_htv = cpwl.get_exact_HTV()
        print('(Discrete, Exact) : ({:.4f}, {:.4f})'
              .format(exact_grad_schatten_htv, exact_htv))

        assert not np.allclose(exact_grad_schatten_htv, exact_htv, rtol=1e-3)

    @pytest.mark.parametrize("dataset",
                             dataset_dict["simple_junction"],
                             indirect=True)
    def test_simple_junction(self, dataset):
        """ """
        cpwl = Delaunay(**dataset)
        assert np.array_equal(cpwl.tri.points, dataset['points'])
        assert np.array_equal(cpwl.tri.values, dataset['values'])
        pos_mask = (cpwl.tri.simplices_affine_coeff[:, 0] > 0)

        assert np.allclose(
            (cpwl.tri.simplices_affine_coeff[np.where(pos_mask)[0], :] -
             SimpleJunction.a1_affine_coeff[np.newaxis, :]),
            np.zeros((np.sum(pos_mask), 3)))

        assert np.allclose(
            (cpwl.tri.simplices_affine_coeff[np.where(~pos_mask)[0], :] -
             SimpleJunction.a2_affine_coeff[np.newaxis, :]),
            np.zeros((np.sum(pos_mask), 3)))

        grid = cpwl.get_grid(h=0.01)
        z, x_grad = cpwl.evaluate_with_grad(grid.x)

        assert np.allclose(
            (np.abs(x_grad) -
             SimpleJunction.a1_affine_coeff[np.newaxis, 0:2]),
            np.zeros_like(x_grad))

    @pytest.mark.parametrize("dataset",
                             dataset_dict["distorted_grid"],
                             indirect=True)
    def test_evaluate(self, dataset):
        """ """
        cpwl = Delaunay(**dataset, add_extreme_points=True)
        grid = cpwl.get_grid(h=0.01)
        t1 = time.time()
        z, x_grad = cpwl.evaluate_with_grad(grid.x)
        t2 = time.time()
        z_bar = cpwl.evaluate_bar(grid.x)
        t3 = time.time()
        print('affine_coeff/bar time: {:.3f}'
              .format((t2 - t1) / (t3 - t2)))

        assert np.all(np.allclose(z, z_bar))

    @pytest.mark.parametrize("dataset",
                             dataset_dict["distorted_grid"],
                             indirect=True)
    def test_convex_hull_extreme_points(self, dataset):
        """ """
        cpwl = Delaunay(**dataset, add_extreme_points=True)
        npoints = cpwl.tri.points.shape[0]
        expected_convex_hull_points_idx = \
            np.array([npoints - 4, npoints - 3, npoints - 2, npoints - 1])

        assert np.array_equal(cpwl.convex_hull_points_idx,
                              expected_convex_hull_points_idx)
