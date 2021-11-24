#!/usr/bin/env python3
"""
This script reproduces the results for ReLU network on the data
fitting task.
"""

import os
import argparse
import copy

from htvlearn.main import main_prog


def run_qtp_nn(args):
    """
    Args:
        args: verified arguments from arparser
    """
    if not os.path.isdir(args.log_dir):
        print(f'\nLog directory {args.log_dir} not found. Creating it.')
        os.makedirs(args.log_dir)

    params = {
        'method': 'nn',
        'log_dir': args.log_dir,
        'dataset_name': 'quad_top_planes',
        'num_train': 250,
        'data_dir': './data',
        'noise_ratio': 0.05,
        'seed': 14,
        'lsize': 64,
        'net_model': 'relufcnet2d',
        'device': 'cpu',
        'num_hidden_layers': 4,
        'num_hidden_neurons': 256,
        'weight_decay': 1e-6,
        'milestones': [375, 450],
        'num_epochs': 500,
        'batch_size': 10,
    }

    params['model_name'] = 'qtp_relu_256'

    main_prog(copy.deepcopy(params))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Run ReLU network on the data fitting task.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--log_dir',
                        metavar='[STR]',
                        type=str,
                        default='./ckpt',
                        help='Model log directory.')

    args = parser.parse_args()

    run_qtp_nn(args)
