#!/usr/bin/env python3

import math
import argparse
import sys
import os
from contextlib import contextmanager

from master_project import MasterProject
from htv_manager import HTVManager
from nn_manager import NNManager
from lattice import Lattice
from data import Data
import htv_utils


@contextmanager
def silence_stdout():
    new_target = open(os.devnull, "w")
    old_target, sys.stdout = sys.stdout, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Load parameters from checkpoint file.')
    parser.add_argument('ckpt_filename', type=str, help='')
    args = parser.parse_args()

    ckpt = MasterProject.get_loaded_ckpt(args.ckpt_filename)
    params = ckpt['params']

    if params['data']['add_noise']:
        print('Signal to noise ratio (dB) {:.4f}'.format(10 * math.log10(ckpt['data']['snr'])))

    print('\nLoading parameters from checkpoint : ', args.ckpt_filename, sep='\n')
    print('\nParameters : ', params, sep='\n')

    with silence_stdout():
        if params['method'] == 'neural_net':
            params['restore'] = True
            manager = NNManager(params)
            mse = manager.evaluate_results(mode='test')
            lattice_obj = manager.evaluate_lattice()

        elif params['method'] == 'htv':
            X_mat = ckpt['lattice']['final']['X_mat']
            C_mat = ckpt['lattice']['final']['C_mat']
            lattice_obj = Lattice(X_mat=X_mat, C_mat=C_mat)
            data_obj = Data(lattice_obj, data_from_ckpt=ckpt['data'], **params['data'])

            output = HTVManager.forward_data(lattice_obj, data_obj.test['input'])
            mse, _ = htv_utils.compute_mse_psnr(data_obj.test['values'], output)

    htv_utils.print_algorithm_details(lattice_obj)

    print(f'\nTest mse  : {mse}')
