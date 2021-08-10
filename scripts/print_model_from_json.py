#!/usr/bin/env python3

import os
import argparse
import json

from htvlearn.master_project import MasterProject
from htvlearn.htv_utils import (
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

    print('\nTrain mse: {:.2E}'.format(train_mse))
    print('Test mse: {:.2E}'.format(test_mse))

    if htv is not None:
        if isinstance(htv, dict):
            for key, val in htv.items():
                htv[key] = float('{:.2f}'.format(val))
            print('HTV:', json.dumps(htv, indent=4, sort_keys=False))
        else:
            print('HTV: {:.2f}'.format(htv))

    print('Exact HTV: {:.2f}'.format(ckpt['exact_htv']))
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
