#!/usr/bin/env python3

import argparse
import json

from htvlearn.lattice import Lattice
from htvlearn.master_project import MasterProject
from htvlearn.nn_manager import NNManager
from htvlearn.rbf_manager import RBFManager
from htvlearn.htv_manager import HTVManager
from htvlearn.htv_utils import (
    compute_mse_psnr,
    get_sigma_from_eps,
    silence_stdout
)


def print_model(args):
    """
    Args:
        args: arguments from argparser
    """
    ckpt = MasterProject.get_loaded_ckpt(args.ckpt_filename)
    params = ckpt['params']
    params['log_dir'] = '/'.join(args.ckpt_filename.split('/')[:-2])
    exact_htv = ckpt['exact_htv']

    print('\nLoading parameters from checkpoint : ',
          args.ckpt_filename,
          sep='\n')
    print('\nParameters : ', params, sep='\n')

    with silence_stdout():

        params['restore'] = True
        htv = None

        if params['method'] == 'neural_net':
            manager = NNManager(params, write=False)
            test_mse = manager.evaluate_results(mode='test')
            train_mse = manager.evaluate_results(mode='train')

            if ckpt['htv_log']:
                _, htv_model = NNManager.read_htv_log(ckpt['htv_log'])
                htv = htv_model[-1]

        elif params['method'] == 'rbf':
            manager = RBFManager(params, write=False)
            data_obj = manager.data
            output = manager.forward_data(data_obj.test['input'])
            test_mse, _ = compute_mse_psnr(data_obj.test['values'],
                                           output)

            output_train = manager.forward_data(data_obj.train['input'])
            train_mse, _ = compute_mse_psnr(data_obj.train['values'],
                                            output_train)

            if ckpt['htv_log']:
                htv = RBFManager.read_htv_log(ckpt['htv_log'])

        elif params['method'] == 'htv':
            manager = HTVManager(params, write=False)
            data_obj = manager.data
            lattice_obj = Lattice(X_mat=ckpt['lattice']['final']['X_mat'],
                                  C_mat=ckpt['lattice']['final']['C_mat'])

            output_test = \
                manager.forward_data(lattice_obj, data_obj.test['input'])
            data_obj.test['predictions'] = output_test
            test_mse, _ = compute_mse_psnr(data_obj.test['values'],
                                           output_test)

            output_train = manager.forward_data(lattice_obj,
                                                data_obj.train['input'])
            train_mse, _ = compute_mse_psnr(data_obj.train['values'],
                                            output_train)

            if ckpt['htv_log']:
                htv = manager.read_htv_log(ckpt['htv_log'])[-1]

    print('\nTrain mse : {:.2E}'.format(train_mse))
    print('Test mse  : {:.2E}'.format(test_mse))

    if htv is not None:
        if isinstance(htv, dict):
            for key, val in htv.items():
                htv[key] = float('{:.2f}'.format(val[0]))
            print('HTV :', json.dumps(htv, indent=4, sort_keys=False))
        else:
            print('HTV : {:.2f}'.format(htv))

    print('Exact HTV : {:.2f}'.format(exact_htv))
    if params['method'] == 'rbf':
        print('sigma : {:.2E}'.format(
            get_sigma_from_eps(params["rbf"]["eps"])))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Load parameters from checkpoint file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('ckpt_filename',
                        metavar='ckpt_filename [STR]',
                        type=str,
                        help='Checkpoint where model is saved.')
    args = parser.parse_args()

    print_model(args)
