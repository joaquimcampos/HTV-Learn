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
    parser.add_argument('--eps', type=ArgCheck.p_float, default=1, help='')
    parser.add_argument('--lmbda', type=ArgCheck.nn_float, default=1e-3, help='')
    parser.add_argument('--num_train', type=str, default=20000, help='')
    parser.add_argument('--seed', type=int, default=10, help='')
    parser.add_argument('--final_lsize', type=ArgCheck.p_int, default=128)
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
    params['dataset_name'] = 'face'
    params['log_dir'] = args.log_dir
    params['eps'] = args.eps
    params['lmbda'] = args.lmbda
    params['num_train'] = args.num_train
    params['seed'] = args.seed
    params['plot'] = args.plot
    params['html'] = True
    params['lsize'] = args.final_lsize

    params['add_noise'] = False

    params['verbose'] = True

    model_name = (f'rbf_{params["dataset_name"]}_'
                'eps_{:.3f}_lmbda_{:.1E}'.format(args.eps, args.lmbda))

    params['model_name'] = model_name
    print('\n\nRunning model: ', params['model_name'])

    main_prog(params.copy())
