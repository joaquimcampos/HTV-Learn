# Program Manager

import torch
import numpy as np

from htvlearn.htv_project import HTVProject
from algorithm import Algorithm
from htvlearn.operators import Operators
from htvlearn.hessian import get_finite_second_diff_Hessian
from htvlearn.htv_utils import compute_mse_psnr


class HTVManager(HTVProject):
    """ """
    def __init__(self, params, write=True):
        """ """

        # initializes log, lattice, data, json files
        super().__init__(params, write=write)

        loading_success = self.restore_ckpt_params()
        if loading_success is True:
            self.restore_model_data()

        self.algorithm = Algorithm(self.lat, self.data,
                                   **self.params['algorithm'])

    @property
    def htv_log(self):
        return self.htv

    def train(self):
        """ """
        # updates lattice; creates lattice_dict and results_dict.
        results_dict, lattice_dict = self.algorithm.multires_admm()

        self.evaluate_results()
        self.htv = self.compute_htv()
        results_dict['htv'] = self.htv

        print('\nHTV : {:.2f}'.format(self.htv))
        print('Exact HTV : {:.2f}'.format(self.data.cpwl.get_exact_HTV()))

        # save numeric results in json file
        for info, value in results_dict.items():
            self.update_json(info, value)

        # save params, lattice history and data to checkpoint
        self.save_to_ckpt(lattice_dict)

    def evaluate_func(self, x, batch_size=2000000):
        """ """
        x = torch.from_numpy(x).to(self.device)
        assert x.dtype == torch.float64
        dataloader = x.split(batch_size)
        if self.params['verbose']:
            print('Length dataloader: ', len(dataloader))
        y = torch.tensor([], device=x.device, dtype=x.dtype)

        for batch_idx, input in enumerate(dataloader):
            out = self.forward_data(self.lat, input)
            y = torch.cat((y, out), dim=0)

        return y.detach().cpu().numpy()

    def compute_htv(self):
        """ """
        if self.data.cpwl.has_zero_boundary is True:
            # compute HTV in whole lattice
            z = self.lat.flattened_C.numpy()
            # regularization
            L_mat_sparse = Operators.get_regularization_op(self.lat)
            L_z = L_mat_sparse.dot(z)
            htv = np.linalg.norm(L_z, ord=1)
        else:
            # compute HTV via finite differences in data region
            h = 0.0002
            grid = self.data.cpwl.get_grid(h=h)
            Hess = get_finite_second_diff_Hessian(grid, self.evaluate_func)

            points_htv = np.abs(Hess[:, :, 0, 0] + Hess[:, :, 1, 1])
            htv = (points_htv * h * h).sum()

        return htv

    def evaluate_results(self):
        """ """
        for mode, data_dict in \
                zip(['valid', 'test'], [self.data.valid, self.data.test]):
            output = self.forward_data(self.lat, data_dict['input'])
            # save predictions
            data_dict['predictions'] = output

            # compute mse
            mse, _ = compute_mse_psnr(data_dict['values'], output)
            self.update_json('_'.join([mode, 'mse']), mse)
            print(f'{mode} mse  : {mse}')

    @staticmethod
    def forward_data(lattice_obj, input_std, **kwargs):
        """ """
        input_lat = lattice_obj.standard_to_lattice(input_std)
        output = lattice_obj.get_values_from_affine_coefficients(input_lat)

        return output

    @staticmethod
    def read_htv_log(htv_log):
        """ """
        assert isinstance(htv_log, float) or isinstance(htv_log, np.float32), \
            f'{type(htv_log)}.'
        return np.array([htv_log])
