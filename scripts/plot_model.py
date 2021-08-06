#!/usr/bin/env python3

import os
import sys
import argparse
import json
from contextlib import contextmanager

from master_project import MasterProject
from nn_manager import NNManager
from rbf_manager import RBFManager
from htv_manager import HTVManager
import htv_utils
from delaunay import Delaunay
from lattice import Lattice
from plots.plot_cpwl import Plot


@contextmanager
def silence_stdout():
    new_target = open(os.devnull, "w")
    old_target, sys.stdout = sys.stdout, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


def plot_model(args):
    """
    Args:
        args: arguments from argparser
    """
    ckpt, params = MasterProject.load_ckpt_params(args.ckpt_filename,
                                                  flatten=False)
    params['plots'] = {}
    plot_log_dir = '/'.join(args.ckpt_filename.split('/')[:-1])
    params['plots']['log_dir'] = plot_log_dir
    params['log_dir'] = '/'.join(args.ckpt_filename.split('/')[:-2])
    exact_htv = ckpt['exact_htv']

    if args.log_dir is not None:
        params['plots']['log_dir'] = args.log_dir

    print('Parameters: ', params, sep='\n')

    with silence_stdout():

        params['restore'] = True
        htv = None

        if params['method'] == 'neural_net':
            manager = NNManager(params, write=False)
            data_obj = manager.data
            test_mse = manager.evaluate_results(mode='test')
            train_mse = manager.evaluate_results(mode='train')

            if ckpt['htv_log']:
                _, htv_model = NNManager.read_htv_log(ckpt['htv_log'])
                htv = htv_model[-1]

        if params['method'] == 'rbf':
            manager = RBFManager(params, write=False)
            data_obj = manager.data

            # data_obj.test['input'] = \
            #     data_obj.cpwl.get_grid(h=0.002,
            #                            to_numpy=False,
            #                            to_float32=True).x
            #
            # data_obj.test['values'] = \
            #     data_obj.cpwl.evaluate(data_obj.test['input'])

            output_test = manager.forward_data(data_obj.test['input'])
            data_obj.test['predictions'] = output_test
            test_mse, _ = htv_utils.compute_mse_psnr(data_obj.test['values'],
                                                     output_test)

            output_train = manager.forward_data(data_obj.train['input'])
            train_mse, _ = htv_utils.compute_mse_psnr(data_obj.train['values'],
                                                      output_train)

            if ckpt['htv_log']:
                htv = RBFManager.read_htv_log(ckpt['htv_log'])

        elif params['method'] == 'htv':
            manager = HTVManager(params, write=False)
            data_obj = manager.data
            lattice_obj = Lattice(X_mat=ckpt['lattice']['final']['X_mat'],
                                  C_mat=ckpt['lattice']['final']['C_mat'])

            output_test = manager.forward_data(lattice_obj,
                                               data_obj.test['input'])
            data_obj.test['predictions'] = output_test
            test_mse, _ = htv_utils.compute_mse_psnr(data_obj.test['values'],
                                                     output_test)

            output_train = manager.forward_data(lattice_obj,
                                                data_obj.train['input'])
            train_mse, _ = htv_utils.compute_mse_psnr(data_obj.train['values'],
                                                      output_train)

            if ckpt['htv_log']:
                htv = manager.read_htv_log(ckpt['htv_log'])[-1]

    if not args.no_gt:
        plot = Plot(data_obj, **params['plots'])
        plot.plot_delaunay(data_obj.cpwl,
                           observations=True,
                           top=False,
                           color='normal')
        # plot.plot_delaunay([], observations=True, top=False, color='normal')

    ret_dict = {
        'points': data_obj.test['input'].cpu().numpy(),
        'values': data_obj.test['predictions'].cpu().numpy()
    }

    # construct grid from predictions
    out_cpwl = Delaunay(**ret_dict)

    plot = Plot(data_obj, **params['plots'])
    plot.plot_delaunay(out_cpwl, observations=False, color='normal')
    # plot.plot_delaunay(data_obj.cpwl, out_cpwl,
    #                    observations=False, color='normal')

    print('\nTrain mse : {:.2E}'.format(train_mse))
    print('Test mse  : {:.2E}'.format(test_mse))

    if htv is not None:
        if isinstance(htv, dict):
            for key, val in htv.items():
                htv[key] = float('{:.2f}'.format(val[0]))
            print('HTV :', json.dumps(htv, indent=4, sort_keys=False))
        else:
            print('HTV : {:.2f}'.format(htv))

    print('Exact HTV : {:.2f}'.format(exact_htv))
    if params['method'] == 'rbf':
        print('sigma : {:.2E}'.format(
            htv_utils.get_sigma_from_eps(params["rbf"]["eps"])))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Load parameters from checkpoint file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('ckpt_filename',
                        metavar='ckpt_filename [STR]',
                        type=str,
                        help='Checkpoint where model is saved.')

    parser.add_argument('--log_dir',
                        metavar='[STR]',
                        type=str,
                        help='If not None, override model log '
                             'directory for plots.')

    parser.add_argument('--no_gt',
                        action='store_true',
                        help='If True, do not plot ground truth.')

    args = parser.parse_args()

    plot_model(args)
