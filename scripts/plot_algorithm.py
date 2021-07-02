#!/usr/bin/env python3

import argparse
from master_project import MasterProject
from htv_manager import HTVManager
from nn_manager import NNManager
from rbf_manager import RBFManager
import htv_utils

from lattice import Lattice
from data import Data
from plots.plot import Plot

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Load parameters from checkpoint file.')
    parser.add_argument('ckpt_filename', type=str, help='')
    parser.add_argument('--log_dir', metavar='STR', type=str,
                        help=f'Override model log directory for plots.')
    parser.add_argument('--color', choices=['z', 'normal'],
                        help='Plot color.', default='normal')
    parser.add_argument('--view', choices=['up', 'up_img', 'side', '3D', '3D_2'], type=str,
                        help=f'Plot view.', default='3D')

    args = parser.parse_args()

    ckpt, params = MasterProject.load_ckpt_params(args.ckpt_filename, flatten=False)
    params['plots']['html'] = True
    log_dir = '/'.join(args.ckpt_filename.split('/')[:-1])
    params['plots']['log_dir'] = log_dir

    if args.log_dir is not None:
        params['plots']['log_dir'] = args.log_dir

    print('Parameters: ' , params, sep='\n')

    if params['method'] == 'htv':
        X_mat = ckpt['lattice']['init']['X_mat']
        C_mat = ckpt['lattice']['init']['C_mat']
        lattice_obj = Lattice(X_mat=X_mat, C_mat=C_mat)
    else:
        lattice_obj = Lattice(**params['lattice'])

    data_obj = Data(lattice_obj, data_from_ckpt=ckpt['data'], **params['data'])

    # plot = Plot(lattice_obj, data_obj, **params['plots'])
    # noise_str = 'noisy_' if data_obj.add_noise else ''
    # plot.show_data_plot(observations=True, mode='train',
    #                     gtruth=True,
    #                     gtruth_color='normal',
    #                     filename=f'{noise_str}planes_data',
    #                     view=args.view)

    # if params['method'] == 'htv':
    #     plot.show_lat_plot(observations=True, mode='train',
    #                         filename='_'.join([params['model_name'], 'lattice_init']),
    #                         color='random',
    #                         marker_size=4,
    #                         eye=dict(x=-0.4, y=-1.3, z=1))

    if params['method'] == 'htv':

        if 'simplex' in list(ckpt['lattice'].keys()):

            X_mat = ckpt['lattice']['admm']['X_mat']
            C_mat = ckpt['lattice']['admm']['C_mat']

            lattice_obj = Lattice(X_mat=X_mat, C_mat=C_mat)

            print('\nPlotting ADMM result...')
            plot = Plot(lattice_obj, data_obj, **params['plots'])
            plot.show_lat_plot(gtruth=False,
                                filename='_'.join([params['model_name'], 'admm']),
                                color=args.color,
                                view=args.view)


        X_mat = ckpt['lattice']['final']['X_mat']
        C_mat = ckpt['lattice']['final']['C_mat']

        lattice_obj = Lattice(X_mat=X_mat, C_mat=C_mat)


    if params['method'] == 'neural_net':
        params['restore'] = True
        manager = NNManager(params)
        mse = manager.evaluate_results(mode='test')
        lattice_obj = manager.evaluate_lattice()

    if params['method'] == 'rbf':
        manager = RBFManager(params)
        output = manager.forward_data(data_obj.test['input'])
        mse, _ = htv_utils.compute_mse_psnr(data_obj.test['values'], output)

        output_train = manager.forward_data(data_obj.train['input'])
        mse_train, _ = htv_utils.compute_mse_psnr(data_obj.train['values'], output_train)
        print(f'Train mse : {mse_train}')
        lattice_obj = manager.evaluate_lattice()

    elif params['method'] == 'htv':
        output = HTVManager.forward_data(lattice_obj, data_obj.test['input'])
        mse, _ = htv_utils.compute_mse_psnr(data_obj.test['values'], output)

    htv_utils.print_algorithm_details(lattice_obj)

    if data_obj.dataset_name.endswith('planes') or data_obj.dataset_name == 'face':
        plot = Plot(lattice_obj, data_obj, **params['plots'])
        noise_str = 'noisy_' if data_obj.add_noise else ''

        plot.show_lat_plot(gtruth=False,
                            filename=f'{noise_str}triangulation_model',
                            color=args.color,
                            opacity=1,
                            view=args.view)

    elif data_obj.dataset_name.startswith('pyramid') or data_obj.dataset_name == 'noisy_linear':
        plot = Plot(lattice_obj, data_obj, **params['plots'])
        plot.show_lat_plot(observations=True,
                            filename='_'.join([params['model_name'], data_obj.dataset_name]),
                            color=args.color,
                            marker_size=1,
                            eye=dict(x=-0.4, y=-1.3, z=1))

    print(f'Test mse  : {mse}')
