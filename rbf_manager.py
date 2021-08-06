# Program Manager

import torch
import numpy as np
import time
import datetime
import json

from rbf_project import RBFProject
from rbf import RBF
import hessian
import htv_utils


class RBFManager(RBFProject):
    """ """
    def __init__(self, params, write=True):
        """
        write: weather to write on log directory
        """
        # initializes log, lattice, data, json files
        super().__init__(params, write=write)

        loading_success = self.restore_ckpt_params()
        if loading_success is True:
            self.restore_model_data()

        self.rbf = RBF(self.data, **self.params['rbf'])

    @property
    def htv_log(self):
        return self.htv_dict

    def train(self):
        """ """
        self.htv_dict = {}

        output = self.forward_data(self.data.train['input'])
        mse, _ = htv_utils.compute_mse_psnr(self.data.train['values'].cpu(),
                                            output.cpu())
        self.update_json('train_mse', mse)
        print(f'Train mse : {mse}')

        if not self.params['no_htv']:
            print('\nComputing Hessian...')
            self.htv_dict['finite_diff_differential'] = self.compute_htv()
            print(
                'HTV dict :',
                json.dumps(self.htv_dict['finite_diff_differential'],
                           indent=4,
                           sort_keys=False))
            print('Exact HTV : {:.2f}'.format(self.data.cpwl.get_exact_HTV()))
            print('Finished.')

        self.evaluate_results()

        # save params and data to checkpoint
        self.save_to_ckpt()

    def evaluate_func(self, x, batch_size=100000):
        """ """
        x = torch.from_numpy(x)
        assert x.dtype == torch.float64
        dataloader = x.split(batch_size)
        if self.params['verbose']:
            print('Length dataloader: ', len(dataloader))
        y = torch.tensor([], device=x.device, dtype=x.dtype)

        # every 5% progress
        print_step = max(int(len(dataloader) * 1. / 20), 1)
        print('Starting evaluation...')

        start_time = time.time()

        for batch_idx, input in enumerate(dataloader):
            out = self.forward_data(input)
            y = torch.cat((y, out), dim=0)
            if batch_idx % print_step == 0:
                progress = int((batch_idx + 1) * 100. / len(dataloader))
                print(f'=> {progress}% complete')

        end_time = time.time()
        run_time = str(datetime.timedelta(seconds=int(end_time - start_time)))

        print(f'Finished. Run time: {run_time} (h:min:sec).')

        return y.numpy()

    def compute_htv(self):
        """ """
        grid = self.data.cpwl.get_grid(h=self.params['htv_grid'])
        Hess = hessian.get_finite_second_diff_Hessian(grid, self.evaluate_func)

        S = np.linalg.svd(Hess, compute_uv=False, hermitian=True)
        assert S.shape == (*Hess.shape[0:2], 2)

        htv = {}
        for p in [1, 2, 5, 10]:
            points_htv = np.linalg.norm(S, ord=p, axis=-1)  # schatten norm
            # value x area (riemman integral)
            htv[str(p)] = (points_htv * grid.h * grid.h).sum()

        return htv

    def evaluate_results(self):
        """ """
        for mode, data_dict in \
                zip(['valid', 'test'], [self.data.valid, self.data.test]):
            output = self.forward_data(data_dict['input'])
            # save predictions
            data_dict['predictions'] = output

            # compute mse
            mse, _ = htv_utils.compute_mse_psnr(data_dict['values'], output)
            self.update_json('_'.join([mode, 'mse']), mse)
            print(f'{mode} mse  : {mse}')

    def forward_data(self, input, *args, **kwargs):
        """ """
        return self.rbf.evaluate(input)

    @staticmethod
    def read_htv_log(htv_log):
        """
        Read htv_log dictionary

        Returns:
            htv: dict('p': htv),
        """
        assert isinstance(htv_log, dict)

        # e.g. htv_log = {'finite_diff_differential': {'1': htv_1, '2': htv_2}}
        # dictio is a dictionary containing 'p': htv_p
        dictio = htv_log['finite_diff_differential']

        assert isinstance(dictio, dict)
        # p in schatten-p norms
        p_array = np.array([int(p) for p in dictio.keys()])
        p_array.sort()

        htv = {}
        for p in p_array:
            htv[str(p)] = np.array([float(dictio[str(p)])])

        return htv
