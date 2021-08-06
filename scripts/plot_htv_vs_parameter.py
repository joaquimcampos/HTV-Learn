#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

from master_project import MasterProject
from nn_manager import NNManager
from rbf_manager import RBFManager
from htv_utils import add_date_to_filename, get_sigma_from_eps


lw = 1  # linewidth
ms = 2  # markersize


def plot_htv_vs_parameter(args):
    """
    Args:
        args: arguments from argparser
    """
    fig, ax = plt.subplots()

    parameter_array = np.zeros(len(args.log_dirs))

    nn = False  # plotting neural_net checkpoints flag
    zeros = np.zeros_like(parameter_array)
    htv = {'1': zeros.copy(), '2': zeros.copy(), '10': zeros.copy()}
    if args.parameter == 'num_hidden_layers' or \
            args.parameter == 'weight_decay':
        htv = zeros.copy()
        nn = True

    for i, log_dir in enumerate(args.log_dirs):

        if not os.path.isdir(log_dir):
            raise ValueError(f'log_dir "{log_dir}" is not a valid directory.')

        ckpt_filename = MasterProject.get_ckpt_from_log_dir_model(log_dir)
        ckpt, params = MasterProject.load_ckpt_params(ckpt_filename,
                                                      flatten=False)

        if nn is True:
            if not params['method'] == 'neural_net':
                raise ValueError('Method should be "neural_net".')
            if i == 0:
                hidden = params["model"]["hidden"]
                label_str = r"nb_Hneurons = ${:d}$. ".format(hidden)
            elif params["model"]["hidden"] != hidden:
                raise ValueError(f'{params["model"]["hidden"]} != {hidden}')
        elif not params['method'] == 'rbf':
            raise ValueError('Method should be "rbf".')

        if args.parameter == 'num_hidden_layers':
            if i == 0:
                wd = params["weight_decay"]
                label_str += r"$\mu$ = ${:.1E}$".format(wd)
                ax.set_xlabel(r"$N_L$", fontsize=20)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            elif params["weight_decay"] != wd:
                raise ValueError(f'{params["weight_decay"]} != {wd}')

            parameter_array[i] = params["model"]["num_hidden_layers"]

        elif args.parameter == 'weight_decay':
            if i == 0:
                num_hidden_layers = params["model"]["num_hidden_layers"]
                label_str += r"nb_Hlayers = ${:d}$. ".format(num_hidden_layers)
                ax.set_xlabel(r"$\mu$", fontsize=20)
            elif params["model"]["num_hidden_layers"] != num_hidden_layers:
                raise ValueError(
                    f'{params["model"]["num_hidden_layers"]} '
                    f'!= {num_hidden_layers}'
                )

            parameter_array[i] = params["weight_decay"]

        elif args.parameter == 'lmbda':
            if i == 0:
                eps = params["rbf"]["eps"]
                label_str = r"$\epsilon$ = ${:.1f}$ ".format(eps) + \
                            "$\sigma$ = ${:.1E}$."\
                            .format(get_sigma_from_eps(eps))
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

            parameter_array[i] = get_sigma_from_eps(
                params["rbf"]["eps"])  # sigma

        print('Parameters: ', params, sep='\n')

        if i == 0:
            exact_htv = np.ones_like(parameter_array) * ckpt['exact_htv']
        else:
            if not np.allclose(ckpt['exact_htv'], exact_htv):
                raise ValueError('Models do not have the same HTV.')

        if nn is True:
            _, htv_model = NNManager.read_htv_log(ckpt['htv_log'])
            htv[i] = htv_model[0] if args.first else htv_model[-1]
        else:
            htv_model = RBFManager.read_htv_log(ckpt['htv_log'])
            for p in htv.keys():
                htv[p][i] = htv_model[p]

    print(f'Plotting {label_str}')
    # sort array
    idx = np.argsort(parameter_array)
    parameter_array = parameter_array[idx]

    if nn is True:
        htv = htv[idx]
        ax.plot(parameter_array,
                htv,
                marker='o',
                markersize=ms,
                lw=lw,
                label=None)
        ax.set_ylabel(r"$\mathrm{HTV}$", fontsize=20)
    else:
        for p in htv.keys():
            htv[p] = htv[p][idx]
            ax.plot(parameter_array,
                    htv[p],
                    marker='o',
                    markersize=ms,
                    lw=lw,
                    label=f'p = {p}')
        ax.set_ylabel(r"$\mathrm{HTV_p}$", fontsize=20)

    if args.plot_exact_htv:
        exact_htv = exact_htv[idx]
        ax.plot(parameter_array,
                exact_htv,
                linestyle='--',
                lw=lw,
                label='GT HTV')

    if args.parameter in ['weight_decay', 'lmbda']:
        ax.semilogx()

    if nn is False:
        loc = "upper right"
        ax.legend(loc=loc, prop={'size': 12})

    if args.save_dir is not None:
        filename = os.path.join(args.save_dir,
                                add_date_to_filename(f'{label_str}') + '.pdf')
        plt.savefig(filename, bbox_inches='tight')

    plt.show()
    plt.close()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Load parameters from checkpoint file.')

    param_choices = ['num_hidden_layers',
                     'weight_decay',
                     'lmbda',
                     'sigma']
    parser.add_argument(
        'parameter',
        metavar='parameter [STR]',
        choices=param_choices,
        type=str,
        help=f'Parameter to plot HTV against. Choices: {param_choices}')

    parser.add_argument(
        'log_dirs',
        nargs='+',
        metavar='[LIST[STR]]',
        type=str,
        help='ckpt log directories.')

    parser.add_argument(
        '--first',
        action='store_true',
        help='Plot first logged htv and not last.')

    parser.add_argument(
        '--plot_exact_htv',
        action='store_true',
        help='Plot exact ground truth HTV.')

    parser.add_argument(
        '--save_dir',
        metavar='[STR]',
        type=str,
        help='Save figure directory. If None, plot is not saved.')

    args = parser.parse_args()
    assert bool(args.log_dirs), f'log_dirs {args.log_dirs} is empty.'

    plot_htv_vs_parameter(args)
