#!/usr/bin/env python3

import numpy as np
import sys
import os
import json
import argparse

from htv_utils import ArgCheck
from main import main_prog

if __name__ == "__main__" :

    # parse arguments
    parser = argparse.ArgumentParser(description='gridsearch neural network on the quad_top_planes.')
    parser.add_argument('--log_dir', type=str, help='log directory.')
    parser.add_argument('--dataset_name', choices={'pyramid', 'pyramid_ext', 'noisy_linear'}, type=str,
                        default='pyramid', help='')
    parser.add_argument('--noise_ratio', type=ArgCheck.frac_float, default=0.05)
    parser.add_argument('--weight_decay', type=ArgCheck.p_float, default=1e-6, help='run only for this weight_decay.')
    parser.add_argument('--batch_size', type=ArgCheck.p_int, default=1, help='')
    parser.add_argument('--hidden', type=ArgCheck.p_int, default=50, help='number of hidden neurons.')
    parser.add_argument('--final_lsize', type=ArgCheck.p_int, default=40)
    parser.add_argument('--no_linear', action='store_true', help='Do not learn linear term.')
    args = parser.parse_args()

    if args.log_dir is None:
        sys.stdout.flush()
        print('Need to provide log_dir')
        raise ValueError

    if not os.path.isdir(args.log_dir):
        raise NotADirectoryError


    params = {}
    params['method'] = 'neural_net'
    params['noise_ratio'] = args.noise_ratio
    params['log_dir'] = args.log_dir
    params['dataset_name'] = args.dataset_name
    params['lsize'] = args.final_lsize
    params['weight_decay'] = args.weight_decay
    params['batch_size'] = args.batch_size
    params['hidden'] = args.hidden
    params['milestones'] = [150, 225]
    params['num_epochs'] = 250
    params['no_linear'] = args.no_linear
    params['num_iter'] = 1
    params['device'] = 'cpu'

    params['verbose'] = True
    params['seed'] = 10

    linear_str = '_no_linear' if params['no_linear'] is True else ''

    model_name = (f'nn_{params["dataset_name"]}{linear_str}_'
                    'weight_decay_{:.1E}_'.format(args.weight_decay) +
                    f'batch_size_{args.batch_size}_hidden_{args.hidden}_'
                    f'num_epochs_{params["num_epochs"]}')

    params['model_name'] = model_name
    print('\n\nRunning model: ', params['model_name'])

    main_prog(params.copy())
