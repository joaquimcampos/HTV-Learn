#!/usr/bin/env python3
'''
This script reproduces the results for RBF on the face dataset.
'''

import os
import argparse
import copy
import torch

from htvlearn.main import main_prog


def run_face_rbf(args):
    """
    Args:
        args: verified arguments from arparser
    """
    if not os.path.isdir(args.log_dir):
        print(f'\nLog directory {args.log_dir} not found. Creating it.')
        os.makedirs(args.log_dir)

    params = {
        'method': 'rbf',
        'lmbda': 1e-2,
        'eps': 50,
        'log_dir': args.log_dir,
        'dataset_name': 'face',
        'num_train': 10000,
        'data_dir': './data',
        'noise_ratio': 0.,
        'seed': 8,
        'lsize': 194,
    }

    params['model_name'] = ('face_rbf_lmbda_{:.1E}_'.format(params["lmbda"]) +
                            'eps_{:d}'.format(params["eps"]))

    main_prog(copy.deepcopy(params))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Run RBF on the face dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--log_dir',
                        metavar='[STR]',
                        type=str,
                        default='./ckpt',
                        help='Model log directory.')

    args = parser.parse_args()

    run_face_rbf(args)
