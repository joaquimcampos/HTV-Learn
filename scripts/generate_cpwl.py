#!/usr/bin/env python3

import argparse
import torch
import numpy as np

from delaunay import Delaunay
from plots.plot_cpwl import Plot


def generate_cpwl(args):
    """
    Args:
        args: arguments from argparser
    """
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    cpwl = Delaunay()
    plot = Plot(log_dir=args.log_dir)
    plot.plot_delaunay(cpwl,
                       filename=args.filename,
                       color=args.color)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Generate and plot a continuous and '
                    'piecewise-linear function.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--log_dir',
                        metavar='[STR]',
                        type=str,
                        help='Log directory for plot. '
                             'If None, plot is not saved.')

    parser.add_argument('--filename',
                        metavar='[STR]',
                        type=str,
                        default='cpwl',
                        help='Saved plot filename.')

    parser.add_argument('--color',
                        choices=['z', 'normal'],
                        default='normal',
                        help='Plot according to values (z) or normals.')

    parser.add_argument('--seed',
                        metavar='[INT]',
                        type=int,
                        default=-1,
                        help='Seed for numpy/PyTorch. '
                             'If negative, no seed is set.')

    args = parser.parse_args()

    generate_cpwl(args)
