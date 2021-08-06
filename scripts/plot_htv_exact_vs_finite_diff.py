#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from master_project import MasterProject
from nn_manager import NNManager
from htv_utils import add_date_to_filename


def plot_htv_exact_vs_finite_diff(args):
    """
    Args:
        args: arguments from argparser
    """
    ckpt_filename = \
        MasterProject.get_ckpt_from_log_dir_model(args.ed_log_dir)
    ed_ckpt, params = \
        MasterProject.load_ckpt_params(ckpt_filename, flatten=False)

    if not params['method'] != 'neural_net':
        raise ValueError(f'method for ed is {params["method"]}, '
                         'but only neural_net is allowed.')

    if params['htv_mode'] != 'exact_differential':
        raise ValueError('Expected params["htv_mode"] = "exact_differential". '
                         f'Got {params["htv_mode"]}...')

    print('Parameters: ', params, sep='\n')

    ckpt_filename = \
        MasterProject.get_ckpt_from_log_dir_model(args.fdd_log_dir)
    fdd_ckpt, params = \
        MasterProject.load_ckpt_params(ckpt_filename, flatten=True)

    if not params['method'] != 'neural_net':
        raise ValueError(f'method for fdd is {params["method"]}, '
                         'but only neural_net is allowed.')

    if params['htv_mode'] != 'finite_diff_differential':
        raise ValueError(
            'Expected params["htv_mode"] = "finite_diff_differential". '
            f'Got {params["htv_mode"]}...')

    epochs, ed_htv = \
        NNManager.read_htv_log(ed_ckpt['htv_log'])
    exact_htv_val = ed_ckpt['exact_htv']

    fdd_epochs, fdd_htv = \
        NNManager.read_htv_log(fdd_ckpt['htv_log'])
    fdd_exact_htv_val = fdd_ckpt['exact_htv']

    if not np.array_equal(epochs, fdd_epochs):
        raise ValueError('Epoch measurements are not equal...')
    if not np.allclose(exact_htv_val, fdd_exact_htv_val):
        raise ValueError('Dataset HTVs are not equal...')

    exact_htv = np.ones_like(epochs) * exact_htv_val

    # Create plots with pre-defined labels.
    fig, ax = plt.subplots()
    ax.plot(epochs, exact_htv, '--', label='GT HTV')

    for label, htv in zip(['ED', 'FDD'], [ed_htv, fdd_htv]):
        ax.plot(epochs,
                htv,
                marker='o',
                markersize=2,
                lw=1,
                label=label)

    ax.set_xlabel("Epochs")
    ax.set_ylabel(r"$\mathrm{HTV}$", fontsize=20)
    ax.legend(loc='lower right')

    if args.save_dir is not None:
        filename = os.path.join(
            args.save_dir,
            add_date_to_filename(params['dataset_name'] + '_discrete_htv') +
            '.pdf')
        plt.savefig(filename, bbox_inches='tight')

    plt.show()
    plt.close()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Load parameters from checkpoint file.')

    parser.add_argument(
        'ed_log_dir',
        metavar='ed_log_dir [STR]',
        type=str,
        help='Neural network ckpt log directory of '
             'exact differential model.')

    parser.add_argument(
        'fdd_log_dir',
        metavar='fdd_log_dir [STR]',
        type=str,
        help='Neural network ckpt log directory of '
             'finite differences differential model.')

    parser.add_argument(
        '--save_dir',
        metavar='[STR]',
        type=str,
        help='save figure directory. If None, plot is not saved.')

    args = parser.parse_args()
    if not os.path.isdir(args.ed_log_dir):
        raise NotADirectoryError(
            f'log_dir "{args.ed_log_dir}" is not a valid directory.')
    if not os.path.isdir(args.fdd_log_dir):
        raise NotADirectoryError(
            f'log_dir "{args.fdd_log_dir}" is not a valid directory.')

    plot_htv_exact_vs_finite_diff(args)
