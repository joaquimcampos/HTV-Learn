#!/usr/bin/env python3

import os
import argparse
import json

from htvlearn.master_project import MasterProject
from htvlearn.nn_manager import NNManager
from htvlearn.rbf_manager import RBFManager
from htvlearn.htv_manager import HTVManager
from htvlearn.htv_utils import (
    compute_snr,
    json_load,
    get_sigma_from_eps,
    silence_stdout
)


def print_model_from_json(args):
    """
    Args:
        args: arguments from argparser
    """
    ckpt = MasterProject.get_loaded_ckpt(args.ckpt_filename)
    print('\nLoading parameters from checkpoint : ',
          args.ckpt_filename,
          sep='\n')

    params = ckpt['params']
    if args.print_params is True:
        print('\nParameters : ', params, sep='\n')

    with silence_stdout():

        log_dir = '/'.join(args.ckpt_filename.split('/')[:-2])
        json_file = os.path.join(log_dir, "results.json")
        if not os.path.isfile(json_file):
            raise ValueError(f'File {json_file} not found.')

        results_dict = json_load(json_file)
        model_results = results_dict[params["model_name"]]

        train_mse = model_results['train_mse']
        test_mse = model_results['test_mse']
        htv = (model_results['htv']
               if 'htv' in model_results
               else None)

        if params['method'] == 'neural_net':
            manager = NNManager(params, write=False)
        elif params['method'] == 'rbf':
            manager = RBFManager(params, write=False)
        elif params['method'] == 'htv':
            manager = HTVManager(params, write=False)

    data_obj = manager.data
    train_snr = compute_snr(data_obj.train['values'], train_mse)
    test_snr = compute_snr(data_obj.test['values'], test_mse)

    print('\nTrain mse : {:.2E}'.format(train_mse))
    print('Test mse  : {:.2E}'.format(test_mse))
    print('Train snr : {:.2f} dB'.format(train_snr))
    print('Test snr  : {:.2f} dB'.format(test_snr))

    if htv is not None:
        if isinstance(htv, dict):
            if 'finite_diff_differential' in htv.keys() or \
                    'exact_differential' in htv.keys():
                values_list = list(htv.values())
                if isinstance(values_list[0], dict):
                    htv = values_list[0]
                    print('HTV\t  :',
                          json.dumps(htv, indent=4, sort_keys=False))
                else:
                    htv = float('{:.2f}'.format(values_list[0]))
                    print('HTV\t  : {:.2f}'.format(htv))
            else:
                for key, val in htv.items():
                    htv[key] = float('{:.2f}'.format(val))
                print('HTV\t  :',
                      json.dumps(htv, indent=4, sort_keys=False))
        else:
            print('HTV\t  : {:.2f}'.format(htv))

    print('Exact HTV : {:.2f}'.format(ckpt['exact_htv']))
    if params['method'] == 'rbf':
        print('sigma: {:.2E}'.format(
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

    parser.add_argument('--print_params',
                        action='store_true',
                        help='Print model parameters')

    args = parser.parse_args()

    print_model_from_json(args)