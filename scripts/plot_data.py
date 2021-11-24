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
                       observations=False,
                       color='normal',
                       filename='GT_no_data')

    plot.plot_delaunay(data_obj.cpwl,
                       observations=True,
                       opaque=False,
                       marker_size=(0.65
                                    if 'face' in args.dataset_name
                                    else 2),
                       color='normal',
                       filename='GT_data')


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Load parameters from checkpoint file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    dataset_choices = {
        'face', 'cut_face_gaps', 'pyramid', 'quad_top_planes'
    }
    parser.add_argument(
        '--dataset_name',
        choices=dataset_choices,
        type=str,
        default='face',
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

    args = parser.parse_args()

    plot_data(args)
