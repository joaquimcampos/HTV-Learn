#!/usr/bin/env python3

import argparse

from htvlearn.data import Data
from htvlearn.plots.plot_cpwl import Plot
from htvlearn.htv_utils import ArgCheck


def plot_data(args):
    """
    Args:
        args: arguments from argparser
    """
    data_params = {
        'dataset_name': args.dataset_name,
        'num_train': args.num_train,
        'noise_ratio': args.noise_ratio,
        'seed': args.seed
    }

    data_obj = Data(**data_params)

    plot_params = {}
    plot_params['log_dir'] = '/tmp'

    plot = Plot(data_obj, **plot_params)
    plot.plot_delaunay(data_obj.cpwl,
                       observations=True,
                       top=False,
                       color='normal')


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Load parameters from checkpoint file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    dataset_choices = {
        'cpwl', 'face', 'face_gaps', 'cut_face', 'cut_face_gaps'
    }
    parser.add_argument(
        '--dataset_name',
        choices=dataset_choices,
        type=str,
        default='cut_face_gaps',
        help=' ')

    parser.add_argument(
        '--num_train',
        metavar='[INT>0]',
        type=ArgCheck.p_int,
        default=10000,
        help=' ')

    parser.add_argument(
        '--noise_ratio',
        metavar='[FLOAT,>0]',
        type=ArgCheck.nn_float,
        default=0.,
        help=' ')

    parser.add_argument(
        '--seed',
        metavar='[INT]',
        type=int,
        default=-1,
        help=' ')

    parser.add_argument(
        '--lsize',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        default=64,
        help=' ')

    args = parser.parse_args()

    plot_data(args)
