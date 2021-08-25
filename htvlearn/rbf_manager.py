# Program Manager

import torch
import numpy as np
import time
import datetime
import json

from htvlearn.rbf_project import RBFProject
from htvlearn.rbf import RBF
from htvlearn.hessian import get_finite_second_diff_Hessian
from htvlearn.htv_utils import compute_mse_snr


class RBFManager(RBFProject):
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

        self.rbf = RBF(self.data, **self.params['rbf'])

    @property
    def htv_log(self):
        """ """
        return self.htv_dict

    def train(self):
        """Run algorithm and save results."""
        self.htv_dict = {}

        output = self.forward_data(self.data.train['input'])
        mse, _ = compute_mse_snr(self.data.train['values'].cpu(),
                                 output.cpu())
        self.update_json('train_mse', mse)
        print(f'Train mse : {mse}')

        if not self.params['no_htv']:
            print('\nComputing Hessian...')
            self.htv_dict = self.compute_htv()
            print('HTV dict :',
                  json.dumps(self.htv_dict,
                             indent=4,
                             sort_keys=False))
            print('Exact HTV : {:.2f}'.format(self.data.cpwl.get_exact_HTV()))
            print('Finished.')
            self.update_json('htv', self.htv_dict)

        for mode in ['valid', 'test']:
            mse, _ = self.evaluate_results(mode)
            self.update_json('_'.join([mode, 'mse']), mse)
            print(f'{mode} mse  : {mse}')

        # save params, data and htv to checkpoint
        self.save_to_ckpt()

    def compute_htv(self):
        """
        Compute the HTV of the model in the data range.

        Return:
            htv (dict):
                dictionary of the form {'p': htv_p}
        """
        grid = self.data.cpwl.get_grid(h=0.0005)
        Hess = get_finite_second_diff_Hessian(grid, self.evaluate_func)

        S = np.linalg.svd(Hess, compute_uv=False, hermitian=True)
        assert S.shape == (*Hess.shape[0:2], 2)

        htv = {}
        for p in [1, 2, 5, 10]:
            points_htv = np.linalg.norm(S, ord=p, axis=-1)  # schatten norm
            # value x area (riemman integral)
            htv[str(p)] = (points_htv * grid.h * grid.h).sum()

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

        output = self.forward_data(data_dict['input'])
        # save predictions
        data_dict['predictions'] = output
        # compute mse
        mse, _ = compute_mse_snr(data_dict['values'], output)

        return mse, output

    def evaluate_func(self, x, batch_size=100000):
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
        x = torch.from_numpy(x)
        assert x.dtype == torch.float64
        dataloader = x.split(batch_size)
        y = torch.tensor([], device=x.device, dtype=x.dtype)

        # every 5% progress
        print_step = max(int(len(dataloader) * 1. / 20), 1)
        print('\n==> Starting evaluation...')
        if self.params['verbose']:
            print('=> Length dataloader: ', len(dataloader))

        start_time = time.time()

        for batch_idx, input in enumerate(dataloader):
            out = self.forward_data(input)
            y = torch.cat((y, out), dim=0)
            if batch_idx % print_step == 0:
                progress = int((batch_idx + 1) * 100. / len(dataloader))
                print(f'=> {progress}% complete')

        end_time = time.time()
        run_time = str(datetime.timedelta(seconds=int(end_time - start_time)))

        print(f'==> Finished. Run time: {run_time} (h:min:sec).')

        return y.numpy()

    def forward_data(self, input, *args, **kwargs):
        """
        Compute model output for some input.

        Args:
            input (torch.Tensor):
                size: (n, 2).
        """
        return self.rbf.evaluate(input)

    @staticmethod
    def read_htv_log(htv_log):
        """
        Parse htv_log dictionary.

        Args:
            htv_log (dict).
        Returns:
            htv (dict):
                dictionary of the form {'p': np.array([htv_p])}.
        """
        # e.g. htv_log = {'1': htv_1, '2': htv_2}
        assert isinstance(htv_log, dict)

        # p in schatten-p norms
        p_array = np.array([int(p) for p in htv_log.keys()])
        p_array.sort()

        htv = {}
        for p in p_array:
            htv[str(p)] = np.array([float(htv_log[str(p)])])

        return htv
