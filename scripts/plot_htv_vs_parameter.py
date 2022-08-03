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
from htvlearn.htv_utils import silence_stdout


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
            args.parameter == 'num_hidden_neurons' or \
            args.parameter == 'weight_decay':

        htv = zeros.copy()
        nn = True

    # with silence_stdout():

    for i, log_dir in enumerate(args.log_dirs):

        if not os.path.isdir(log_dir):
            raise ValueError(f'log_dir "{log_dir}" is not a '
                             'valid directory.')

        ckpt_filename = MasterProject.get_ckpt_from_log_dir_model(log_dir)
        ckpt, params = MasterProject.load_ckpt_params(ckpt_filename,
                                                      flatten=False)

        if nn is True:
            if not params['method'] == 'neural_net':
                raise ValueError('Method should be "neural_net".')

        elif not params['method'] == 'rbf':
            raise ValueError('Only neural net and rbf models are allowed.')

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

        print('Parameters: ', params, sep='\n')

        if i == 0:
            exact_htv = ckpt['exact_htv']
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

    if args.cmp_log_dirs is not None:
        cmp_htv = np.zeros(len(args.cmp_log_dirs))
        cmp_label_str = []
        # to check if at most one model of each method is given
        found = {'htv': False, 'nn': False, 'rbf': False}

        for i, log_dir in enumerate(args.cmp_log_dirs):

            if not os.path.isdir(log_dir):
                raise ValueError(f'log_dir "{log_dir}" is not a '
                                 'valid directory.')

            ckpt_filename = \
                MasterProject.get_ckpt_from_log_dir_model(log_dir)
            ckpt, params = MasterProject.load_ckpt_params(ckpt_filename,
                                                          flatten=False)

            if params['method'] == 'htv':
                if found['htv'] is False:
                    htv_model = HTVManager.read_htv_log(ckpt['htv_log'])
                    cmp_htv[i] = htv_model
                    cmp_label_str.append('HTV')
                    found['htv'] = True
                else:
                    raise ValueError('Only one htv model allowed '
                                     'for comparison.')

            elif params['method'] == 'neural_net':
                if found['nn'] is False:
                    _, htv_model = NNManager.read_htv_log(ckpt['htv_log'])
                    cmp_htv[i] = \
                        htv_model[0] if args.first else htv_model[-1]
                    cmp_label_str.append('NN')
                    found['nn'] = True
                else:
                    raise ValueError('Only one neural net model allowed '
                                     'for comparison.')

            elif params['method'] == 'rbf':
                if found['rbf'] is False:
                    htv_model = RBFManager.read_htv_log(ckpt['htv_log'])
                    cmp_htv[i] = htv_model['1']  # use htv p=1
                    cmp_label_str.append("RBF " + r"$p = 1$")
                    found['rbf'] = True
                else:
                    raise ValueError('Only one rbf model allowed '
                                     'for comparison.')

            else:
                raise ValueError(f"Method {params['method']} is unknown.")

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
        ax.plot(parameter_array,
                np.ones_like(parameter_array) * exact_htv,
                linestyle='--',
                lw=lw,
                label='GT HTV')

    if args.cmp_log_dirs is not None:
        for i in range(len(args.cmp_log_dirs)):
            ax.plot(parameter_array,
                    np.ones_like(parameter_array) * cmp_htv[i],
                    linestyle='--',
                    lw=lw,
                    label=cmp_label_str[i])

    if args.parameter in ['weight_decay', 'num_hidden_neurons', 'lmbda']:
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
        description='Plot htv vs parameter for neural net and rbf models.')

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
        help=f'Parameter to plot HTV against. Choices: {param_choices}')

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
