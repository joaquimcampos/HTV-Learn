# Program Manager

import torch
import numpy as np

from htvlearn.htv_project import HTVProject
from htvlearn.algorithm import Algorithm
from htvlearn.operators import Operators
from htvlearn.hessian import get_finite_second_diff_Hessian
from htvlearn.htv_utils import compute_mse_snr


class HTVManager(HTVProject):
    """ """
    def __init__(self, params, log=True):
        """
        Args:
            params (dict):
                parameter dictionary.
            log (bool):
                if True, log results.
        """
        # initializes log, lattice, data, json files
        super().__init__(params, log=log)

        if self.load_ckpt is True:
            # is_ckpt_loaded=True if a checkpoint was successfully loaded.
            is_ckpt_loaded = self.restore_ckpt_params()
            if is_ckpt_loaded is True:
                self.restore_model_data()

        self.algorithm = Algorithm(self.lat, self.data,
                                   **self.params['algorithm'])

    @property
    def htv_log(self):
        """ """
        return self.htv

    def train(self):
        """Run algorithm and save results."""
        # updates lattice; creates lattice_dict and results_dict.
        results_dict, lattice_dict = self.algorithm.multires_admm()

        # run validation/testing
        for mode in ['valid', 'test']:
            mse, _ = self.evaluate_results(mode)
            self.update_json('_'.join([mode, 'mse']), mse)
            print(f'{mode} mse  : {mse}')

        self.htv = self.compute_htv()
        results_dict['htv'] = self.htv

        print('\nHTV : {:.2f}'.format(self.htv))
        print('Exact HTV : {:.2f}'.format(self.data.cpwl.get_exact_HTV()))

        # save numeric results in json file
        for info, value in results_dict.items():
            self.update_json(info, value)

        # save params, data, htv and lattice history to checkpoint
        self.save_to_ckpt(lattice_dict)

    def compute_htv(self):
        """
        Compute the HTV of the model.

        If ground-truth cpwl function is admissible, compute the HTV over
        the whole lattice. Otherwise, compute it only in the data range.

        Return:
            htv (float).
        """
        if self.data.cpwl.is_admissible is True:
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

    def evaluate_results(self, mode):
        """
        Evaluate train, validation or test results.

        Args:
            mode (str):
                'train', 'valid' or 'test'

        Returns:
            mse (float):
                ``mode`` mean-squared-error.
            output (torch.Tensor):
                result of evaluating model on ``mode`` set.
        """
        assert mode in ['train', 'valid', 'test']

        if mode == 'train':
            data_dict = self.data.train
        elif mode == 'valid':
            data_dict = self.data.valid
        else:
            data_dict = self.data.test

        output = self.forward_data(self.lat, data_dict['input'])
        # save predictions
        data_dict['predictions'] = output
        # compute mse
        mse, _ = compute_mse_snr(data_dict['values'], output)

        return mse, output

    def evaluate_func(self, x, batch_size=2000000):
        """
        Evaluate model function for some input.

        Args:
            x (np.ndarray):
                inputs. size: (n, 2).
            batch_size (int):
                batch size for evaluation.

        Returns:
            y (np.ndarray):
                result of evaluating model at x.
        """
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

    @staticmethod
    def forward_data(lattice_obj, input, **kwargs):
        """
        Compute model output for some input.

        Args:
            lattice_obj (Lattice):
                instance of Lattice() (see lattice.py).
            input (torch.Tensor):
                size: (n, 2).
        """
        input_lat = lattice_obj.standard_to_lattice(input)
        output = lattice_obj.get_values_from_affine_coefficients(input_lat)

        return output

    @staticmethod
    def read_htv_log(htv_log):
        """
        Parse htv_log.

        Args:
            htv_log (float)
        Returns:
            htv (np.ndarray):
                np.array([htv]).
        """
        assert isinstance(htv_log, float) or isinstance(htv_log, np.float32), \
            f'{type(htv_log)}.'
        return np.array([htv_log])
