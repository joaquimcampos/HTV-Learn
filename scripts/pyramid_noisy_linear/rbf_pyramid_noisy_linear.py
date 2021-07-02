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
    parser = argparse.ArgumentParser(description='gridsearch HTV on the quad_top_planes.')
    parser.add_argument('--log_dir', type=str, help='log directory.')
    parser.add_argument('--dataset_name', choices={'pyramid', 'pyramid_ext', 'noisy_linear'}, type=str,
                    default='pyramid', help='')
    parser.add_argument('--eps', type=ArgCheck.p_float, default=1, help='')
    parser.add_argument('--lmbda', type=ArgCheck.nn_float, default=1e-3, help='')
    parser.add_argument('--noise_ratio', type=ArgCheck.frac_float, default=0.05)
    parser.add_argument('--num_train', type=str, default=650, help='')
    parser.add_argument('--final_lsize', type=ArgCheck.p_int, default=40)
    parser.add_argument('--no_linear', action='store_true', help='Do not learn linear term.')
    parser.add_argument('--plot', action='store_true', help='Plot results.')
    args = parser.parse_args()

    if args.log_dir is None:
        sys.stdout.flush()
        print('Need to provide log_dir')
        raise ValueError

    if not os.path.isdir(args.log_dir):
        raise NotADirectoryError

    params = {}
    params['method'] = 'rbf'
    params['log_dir'] = args.log_dir
    params['dataset_name'] = args.dataset_name
    params['eps'] = args.eps
    params['lmbda'] = args.lmbda
    params['noise_ratio'] = args.noise_ratio
    params['lsize'] = args.final_lsize
    params['no_linear'] = args.no_linear
    params['plot'] = args.plot
    params['html'] = True

    params['verbose'] = True
    params['seed'] = 10

    linear_str = '_no_linear' if params['no_linear'] is True else ''

    model_name = (f'rbf_{params["dataset_name"]}{linear_str}_'
                'eps_{:.3f}_lmbda_{:.1E}'.format(args.eps, args.lmbda))

    params['model_name'] = model_name
    print('\n\nRunning model: ', params['model_name'])

    main_prog(params.copy())
