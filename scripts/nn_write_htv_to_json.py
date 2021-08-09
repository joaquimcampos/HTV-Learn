#!/usr/bin/env python3

import argparse
import os

from htvlearn.nn_manager import NNManager
from htvlearn.master_project import MasterProject
from htvlearn.htv_utils import (
    json_load,
    json_dump,
    silence_stdout
)


def nn_write_htv_to_json(args):
    """
    Args:
        args: arguments from argparser
    """
    path_split = args.json_file.split('/')
    log_dir = '/'.join(path_split[:-1])

    results_dict = json_load(args.json_file)
    model_keys = list(results_dict.keys())

    for i, model_name in enumerate(model_keys):

        with silence_stdout():
            log_dir_model = os.path.join(log_dir, model_name)
            ckpt_filename = \
                MasterProject.get_ckpt_from_log_dir_model(log_dir_model)

            ckpt = MasterProject.get_loaded_ckpt(ckpt_filename)
            params = ckpt['params']
            if not params['method'] == 'neural_net':
                raise ValueError(f'{params["method"]} is not neural_net.')

            if ckpt['htv_log']:
                _, htv_model = NNManager.read_htv_log(ckpt['htv_log'])

            results_dict[model_name]['htv'] = \
                float('{:.2f}'.format(htv_model[-1]))

        print(f'=> model {i + 1} / {len(model_keys)}')

    json_dump(results_dict, args.json_file)

    print(f'=> Results with htv written to {args.json_file}.')


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='write htv in json file for neural network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'json_file',
        metavar='json_file [STR]',
        type=str,
        help='path to json file with the train/test runs results.')

    args = parser.parse_args()

    nn_write_htv_to_json(args)
