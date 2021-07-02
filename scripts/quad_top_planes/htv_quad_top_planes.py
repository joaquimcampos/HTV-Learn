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
    parser.add_argument('--lmbda', type=ArgCheck.p_float, default=1, help='')
    parser.add_argument('--num_train', type=str, default=650, help='')
    parser.add_argument('--seed', type=int, default=14, help='')
    parser.add_argument('--admm_iter', type=ArgCheck.p_int, default=100000, help='')
    parser.add_argument('--noise_ratio', type=ArgCheck.frac_float, default=0.05)
    parser.add_argument('--final_lsize', type=ArgCheck.p_int, default=128)
    parser.add_argument('--simplex', action='store_true', help='Perform simplex.')
    parser.add_argument('--no_linear', action='store_true', help='Do not learn linear term.')
    parser.add_argument('--multires', action='store_true', help='Multiresolution.')
    parser.add_argument('--plot', action='store_true', help='Plot results.')
    parser.add_argument('--sigma_rule', type= str, choices=['constant', 'same', 'inverse'])
    args = parser.parse_args()

    if args.log_dir is None:
        sys.stdout.flush()
        print('Need to provide log_dir')
        raise ValueError

    if not os.path.isdir(args.log_dir):
        raise NotADirectoryError

    params = {}
    params['method'] = 'htv'
    params['log_dir'] = args.log_dir
    params['lmbda'] = args.lmbda
    params['num_train'] = args.num_train
    params['seed'] = args.seed
    params['noise_ratio'] = args.noise_ratio
    params['no_linear'] = args.no_linear
    params['plot'] = args.plot
    params['html'] = True
    params['sigma_rule'] = args.sigma_rule

    params['add_noise'] = True
    params['dataset_name'] = 'quad_top_planes'
    params['admm_iter'] = args.admm_iter

    if args.multires:
        params['lsize'] = int(args.final_lsize/(2**3))
        params['num_iter'] = 4 # 32, 64, 128, 256
    else:
        params['lsize'] = args.final_lsize
        params['num_iter'] = 1

    if not args.simplex:
        params['no_simplex'] = True
    params['verbose'] = True
    params['seed'] = 10


    aux_str = '_denoising' if params['add_noise'] else ''
    linear_str = '_no_linear' if params['no_linear'] is True else ''

    model_name = (f'htv_{params["dataset_name"]}{aux_str}_'
                    'lmbda_{:.1E}_'.format(args.lmbda) +
                    f'noise_ratio_{params["noise_ratio"]}{linear_str}_'
                    f'lsize_{params["lsize"]}_'
                    f'admm_iter_{params["admm_iter"]}_'
                    f'num_iter_{params["num_iter"]}_no_simplex')

    params['model_name'] = model_name
    print('\n\nRunning model: ', params['model_name'])

    main_prog(params.copy())
