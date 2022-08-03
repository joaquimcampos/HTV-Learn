#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from htvlearn.master_project import MasterProject
from htvlearn.nn_manager import NNManager
from htvlearn.rbf_manager import RBFManager
from htvlearn.htv_manager import HTVManager
from htvlearn.rbf import RBF
from htvlearn.htv_utils import add_date_to_filename

from htvlearn.htv_utils import (
    compute_snr,
    silence_stdout
)


lw = 1  # linewidth
ms = 2  # markersize


def load_ckpt_params(log_dir_model):
    """
    Args:
        log_dir_model: model log directory
    """
    ckpt_filename = MasterProject.get_ckpt_from_log_dir_model(log_dir_model)
    ckpt, params = MasterProject.load_ckpt_params(ckpt_filename,
                                                  flatten=False)
    params['log_dir'] = '/'.join(ckpt_filename.split('/')[:-2])
    params['data']['log_dir'] = params['log_dir']
    params['restore'] = True

    return ckpt, params


def plot_test_snr_vs_parameter(args):
    """
    Args:
        args: arguments from argparser
    """
    fig, ax = plt.subplots()

    parameter_array = np.zeros(len(args.log_dirs))
    test_snr = np.zeros_like(parameter_array)

    for i, log_dir in enumerate(args.log_dirs):

        if not os.path.isdir(log_dir):
            raise ValueError(f'log_dir "{log_dir}" is not a '
                             'valid directory.')

        ckpt, params = load_ckpt_params(log_dir)
        print('Parameters: ', params, sep='\n')

        if args.parameter == 'num_hidden_layers':
            if i == 0:
                nhiddenN = params["model"]["num_hidden_neurons"]
                label_str = r"$N_n$ = ${:d}$. ".format(nhiddenN)
                ax.set_xlabel(r"$N_L$", fontsize=20)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            elif params["model"]["num_hidden_neurons"] != nhiddenN:
                raise ValueError(f'{params["model"]["num_hidden_neurons"]}'
                                 ' != {nhiddenN}')

            parameter_array[i] = params["model"]["num_hidden_layers"]

        elif args.parameter == 'num_hidden_neurons':
            if i == 0:
                nhiddenL = params["model"]["num_hidden_layers"]
                label_str = r"$N_L$ = ${:d}$. ".format(nhiddenL)
                ax.set_xlabel(r"$N_n$", fontsize=20)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            elif params["model"]["num_hidden_layers"] != nhiddenL:
                raise ValueError(f'{params["model"]["num_hidden_layers"]} '
                                 '!= {nhiddenL}')

            parameter_array[i] = params["model"]["num_hidden_neurons"]

        elif args.parameter == 'weight_decay':
            if i == 0:
                nhiddenL = params["model"]["num_hidden_layers"]
                nhiddenN = params["model"]["num_hidden_neurons"]
                label_str = r"$N_L$ = ${:d}$. ".format(nhiddenL)
                label_str += r"$N_n$ = ${:d}$.".format(nhiddenN)
                ax.set_xlabel(r"$\mu$", fontsize=20)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            elif params["model"]["num_hidden_layers"] != nhiddenL:
                raise ValueError(f'{params["model"]["num_hidden_layers"]} '
                                 '!= {nhiddenL}')

            elif params["model"]["num_hidden_neurons"] != nhiddenN:
                raise ValueError(f'{params["model"]["num_hidden_neurons"]}'
                                 ' != {nhiddenN}')

            parameter_array[i] = params["weight_decay"]

        elif args.parameter == 'lmbda':
            if i == 0:
                eps = params["rbf"]["eps"]
                label_str = r"$\epsilon$ = ${:.1f}$ ".format(eps) + \
                            "$\sigma$ = ${:.1E}$."\
                            .format(RBF.get_sigma_from_eps(eps))
                ax.set_xlabel(r"$\lambda$", fontsize=20)
            elif not np.allclose(params["rbf"]["eps"], eps):
                raise ValueError(f'{params["rbf"]["eps"]} != {eps}')

            parameter_array[i] = params["rbf"]["lmbda"]

        elif args.parameter == 'sigma':
            if i == 0:
                lmbda = params["rbf"]["lmbda"]
                label_str = r"$\lambda$ = ${:.1E}$.".format(lmbda)
                ax.set_xlabel(r"$\sigma$", fontsize=20)
            elif not np.allclose(params["rbf"]["lmbda"], lmbda):
                raise ValueError(f'{params["rbf"]["lmbda"]} != {lmbda}')

            parameter_array[i] = RBF.get_sigma_from_eps(
                params["rbf"]["eps"])  # sigma

        with silence_stdout():

            if params['method'] == 'neural_net':
                manager = NNManager(params, log=False)

            elif params['method'] == 'rbf':
                params['device'] = 'cpu'
                manager = RBFManager(params, log=False)

            else:
                raise ValueError('Only neural_net and rbf models are allowed.')

            data_obj = manager.data
            test_mse, _ = manager.evaluate_results(mode='test')
            test_snr[i] = compute_snr(data_obj.test['values'], test_mse)

    if args.cmp_log_dirs is not None:
        cmp_test_snr = np.zeros(len(args.cmp_log_dirs))
        cmp_label_str = []
        # to check if at most one model of each method is given
        found = {'htv': False, 'nn': False, 'rbf': False}

        for i, log_dir in enumerate(args.cmp_log_dirs):

            if not os.path.isdir(log_dir):
                raise ValueError(f'log_dir "{log_dir}" is not a '
                                 'valid directory.')

            ckpt, params = load_ckpt_params(log_dir)
            print('Parameters: ', params, sep='\n')

            if params['method'] == 'htv':
                if found['htv'] is False:
                    params['device'] = 'cpu'
                    manager = HTVManager(params, log=False)
                    cmp_label_str.append('HTV')
                    found['htv'] = True
                else:
                    raise ValueError('Only one htv model allowed '
                                     'for comparison.')

            elif params['method'] == 'neural_net':
                if found['nn'] is False:
                    manager = NNManager(params, log=False)
                    cmp_label_str.append('NN')
                    found['nn'] = True
                else:
                    raise ValueError('Only one neural net model allowed '
                                     'for comparison.')

            elif params['method'] == 'rbf':
                if found['rbf'] is False:
                    params['device'] = 'cpu'
                    manager = RBFManager(params, log=False)
                    cmp_label_str.append('RBF')
                    found['rbf'] = True
                else:
                    raise ValueError('Only one rbf model allowed '
                                     'for comparison.')

            else:
                raise ValueError(f"Method {params['method']} is unknown.")

            data_obj = manager.data
            test_mse, _ = manager.evaluate_results(mode='test')
            cmp_test_snr[i] = compute_snr(data_obj.test['values'],
                                          test_mse)

    print(f'Plotting {label_str}')
    # sort array
    idx = np.argsort(parameter_array)
    parameter_array = parameter_array[idx]

    test_snr = test_snr[idx]
    ax.plot(parameter_array,
            test_snr,
            marker='o',
            markersize=ms,
            lw=lw,
            label='NN')
    ax.set_ylabel("Test SNR (dB)", fontsize=20)

    if args.cmp_log_dirs is not None:
        for i in range(len(args.cmp_log_dirs)):
            ax.plot(parameter_array,
                    np.ones_like(parameter_array) * cmp_test_snr[i],
                    linestyle='--',
                    lw=lw,
                    label=cmp_label_str[i])

    if args.parameter in ['weight_decay', 'num_hidden_neurons', 'lmbda']:
        ax.semilogx()

    loc = "upper right"
    ax.legend(loc=loc, prop={'size': 12})

    # extend y range by 10% bottom, 40% top
    bottom, top = plt.ylim()
    ext = 0.1 * (top - bottom)
    plt.ylim(bottom=(bottom - ext), top=(top + ext * 4))

    if args.save_dir is not None:
        filename = os.path.join(args.save_dir,
                                add_date_to_filename(f'{label_str}') + '.pdf')
        plt.savefig(filename, bbox_inches='tight')

    plt.show()
    plt.close()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Plot Test SNR vs parameter for neural net and '
                    'rbf models.')

    param_choices = ['num_hidden_layers',
                     'num_hidden_neurons',
                     'weight_decay',
                     'lmbda',
                     'sigma']
    parser.add_argument(
        'parameter',
        metavar='parameter [STR]',
        choices=param_choices,
        type=str,
        help=f'Parameter to plot Test SNR against. Choices: {param_choices}')

    parser.add_argument(
        'log_dirs',
        nargs='+',
        metavar='[LIST[STR]]',
        type=str,
        help='ckpt log directories.')

    parser.add_argument(
        '--cmp_log_dirs',
        nargs='+',
        metavar='[LIST[STR]]',
        type=str,
        help='comparison ckpt log directories. '
             'Only one model per method allowed.')

    parser.add_argument(
        '--save_dir',
        metavar='[STR]',
        type=str,
        help='Save figure directory. If None, plot is not saved.')

    args = parser.parse_args()
    assert bool(args.log_dirs), f'log_dirs {args.log_dirs} is empty.'

    plot_test_snr_vs_parameter(args)
