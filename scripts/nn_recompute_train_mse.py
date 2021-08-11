#!/usr/bin/env python3

import argparse
import os
import copy
import collections

from htvlearn.nn_manager import NNManager
from htvlearn.master_project import MasterProject
from htvlearn.htv_utils import (
    json_load,
    json_dump,
    silence_stdout
)


def nn_recompute_train_mse(args):
    """
    Args:
        args: arguments from argparser
    """
    path_split = args.json_file.split('/')
    log_dir, json_name = '/'.join(path_split[:-1]), path_split[-1]

    results_dict = json_load(args.json_file)
    new_results_dict = copy.deepcopy(results_dict)

    model_keys = list(results_dict.keys())

    for i, model_name in enumerate(model_keys):

        with silence_stdout():
            log_dir_model = os.path.join(log_dir, model_name)
            ckpt_filename = \
                MasterProject.get_ckpt_from_log_dir_model(log_dir_model)

            ckpt = MasterProject.get_loaded_ckpt(ckpt_filename)
            params = ckpt['params']
            params['log_dir'] = log_dir
            if not params['method'] == 'neural_net':
                raise ValueError(f'{params["method"]} is not neural_net.')

            params['restore'] = True
            manager = NNManager(params, write=False)
            train_mse, _ = manager.evaluate_results(mode='train')
            new_results_dict[model_name]['latest_train_mse'] = \
                float('{:.2E}'.format(train_mse))

        print(f'=> model {i + 1} / {len(model_keys)}')

    sorted_results = sorted(new_results_dict.items(),
                            key=lambda kv: kv[1]['latest_train_mse'])
    sorted_results_dict = collections.OrderedDict(sorted_results)

    sorted_results_json = os.path.join(log_dir,
                                       'latest_train_mse_sorted_' + json_name)

    json_dump(sorted_results_dict, sorted_results_json)

    print('=> Results sorted by latest_train_mse '
          f'written to {sorted_results_json}.')


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='recompute true training mse for neural network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'json_file',
        metavar='json_file [STR]',
        type=str,
        help='path to json file with the train/test runs results.')

    args = parser.parse_args()

    nn_recompute_train_mse(args)
