#!/usr/bin/env python3

import argparse
import os
import glob
import json

from master_project import MasterProject
from nn_manager import NNManager
from rbf_manager import RBFManager
from htv_manager import HTVManager
from htv_utils import ArgCheck, json_load


def get_same_train_mse(args):
    """
    Args:
        args: arguments from argparser
    """
    log_dirs = args.log_dirs
    models = {}

    for i, log_dir in enumerate(log_dirs):
        # search for json file sorted by train_mse
        regexp_ckpt = os.path.join(log_dir, "*train_mse*.json")
        files = list(set(glob.glob(regexp_ckpt)))
        if len(files) == 0:
            raise ValueError('Need to create a json file sorted by train_mse '
                             f'from {os.path.join(log_dir, "results.json")} '
                             'using sorted_results_json.py')
        elif len(files) > 1:
            raise ValueError('More than one train_mse json file in {log_dir1}')

        results_dict = json_load(files[0])

        # check train_mse key
        first = next(iter(results_dict))
        train_mse_key = None
        for key in results_dict[first].keys():
            if 'train_mse' in key:
                train_mse_key = key

        if train_mse_key is None:
            raise ValueError('No "train_mse" key found in results_dict.')

        # model[j-1] train_mse < args.train_mse  < model[j] train_mse
        key_list = list(results_dict.keys())
        for j, key in enumerate(key_list):
            if results_dict[key][train_mse_key] > args.train_mse:
                break

        models[f'{key_list[j - 1]}'] = results_dict[key_list[j - 1]]
        models[f'{key_list[j]}'] = results_dict[key_list[j]]

        for k in range(-1, 1):  # j + k = [j - 1, j]
            log_dir_model = os.path.join(log_dir, f'{key_list[j + k]}')
            ckpt_filename = \
                MasterProject.get_ckpt_from_log_dir_model(log_dir_model)
            ckpt, params = \
                MasterProject.load_ckpt_params(ckpt_filename, flatten=False)

            if params['method'] == 'neural_net':
                _, htv_model = NNManager.read_htv_log(ckpt['htv_log'])
                htv = htv_model[-1]

            elif params['method'] == 'rbf':
                htv = RBFManager.read_htv_log(ckpt['htv_log'])

            elif params['method'] == 'htv':
                htv = HTVManager.read_htv_log(ckpt['htv_log'])[-1]

            if params['method'] == 'rbf':
                for key, val in htv.items():
                    htv[key] = float('{:.2f}'.format(val[0]))
            else:
                htv = float('{:.2f}'.format(htv))

            models[f'{key_list[j + k]}']['htv'] = htv

    print(json.dumps(models, indent=4, sort_keys=False))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Print RBF/HTV/NN models with similar train_mse.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('log_dirs',
                        metavar='log_dirs [LIST[STR]]',
                        nargs='+',
                        type=str,
                        help='List of log directories with saved models.')

    parser.add_argument('train_mse',
                        metavar='train_mse [FLOAT,>0]',
                        type=ArgCheck.p_float,
                        help='Train mse to look for.')

    args = parser.parse_args()

    get_same_train_mse(args)
