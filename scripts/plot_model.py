#!/usr/bin/env python3

import argparse
import json

from htvlearn.master_project import MasterProject
from htvlearn.nn_manager import NNManager
from htvlearn.rbf_manager import RBFManager
from htvlearn.htv_manager import HTVManager
from htvlearn.delaunay import Delaunay
from htvlearn.lattice import Lattice
from htvlearn.plots.plot_cpwl import Plot
from htvlearn.htv_utils import (
    compute_mse_psnr,
    get_sigma_from_eps,
    silence_stdout
)


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
            test_mse, _ = compute_mse_psnr(data_obj.test['values'],
                                           output_test)

            output_train = manager.forward_data(data_obj.train['input'])
            train_mse, _ = compute_mse_psnr(data_obj.train['values'],
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
            test_mse, _ = compute_mse_psnr(data_obj.test['values'],
                                           output_test)

            output_train = manager.forward_data(lattice_obj,
                                                data_obj.train['input'])
            train_mse, _ = compute_mse_psnr(data_obj.train['values'],
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

    print('Exact HTV : {:.2f}'.format(ckpt['exact_htv']))
    if params['method'] == 'rbf':
        print('sigma : {:.2E}'.format(
            get_sigma_from_eps(params["rbf"]["eps"])))


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
