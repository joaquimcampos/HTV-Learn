#!/usr/bin/env python3
"""This script reproduces the results for HTV on the data fitting task."""

import os
import argparse
import copy

from htvlearn.main import main_prog


def run_qtp_htv(args):
    """
    Args:
        args: verified arguments from arparser
    """
    if not os.path.isdir(args.log_dir):
        print(f'\nLog directory {args.log_dir} not found. Creating it.')
        os.makedirs(args.log_dir)

    params = {
        'method': 'htv',
        'lmbda': 4e-3,
        'log_dir': args.log_dir,
        'dataset_name': 'quad_top_planes',
        'num_train': 250,
        'data_dir': './data',
        'noise_ratio': 0.05,
        'seed': 14,
        'lsize': 64,
        'admm_iter': 200000,
        'simplex': True
    }

    params['model_name'] = 'qtp_htv_lmbda_{:.1E}'.format(params["lmbda"])

    main_prog(copy.deepcopy(params))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Run HTV on the data fitting task.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--log_dir',
                        metavar='[STR]',
                        type=str,
                        default='./ckpt',
                        help='Model log directory.')

    args = parser.parse_args()

    run_qtp_htv(args)
