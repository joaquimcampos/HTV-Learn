#!/usr/bin/env python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from htvlearn.delaunay import Delaunay
from htvlearn.data import (
    SimplicialSpline,
    SimpleJunction,
    DistortedGrid
)
from htvlearn.htv_utils import add_date_to_filename


def htv_discretizations(args):
    """
    Args:
        args: arguments from argparser
    """
    if args.dataset == 'SimplicialSpline':
        dataset = SimplicialSpline
    elif args.dataset == 'SimpleJunction':
        dataset = SimpleJunction
    elif args.dataset == 'DistortedGrid':
        dataset = DistortedGrid
    else:
        raise ValueError(f'Dataset {args.dataset} does not exist here...')

    dataset_dict = {
        'name': dataset.__name__,
        'points': dataset.points.copy(),
        'values': dataset.values.copy()
    }

    # add extreme points flag
    aep = True if args.dataset == 'SimplicialSpline' else False

    cpwl = Delaunay(**dataset_dict, add_extreme_points=aep)
    exact_htv = cpwl.get_exact_HTV()

    npoints = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 6000]
    zeros = np.zeros(len(npoints))
    res = {
        'exact_grad_trace_htv': zeros.copy(),
        'exact_grad_schatten_htv': zeros.copy(),
        'lefkimiattis_htv': zeros.copy(),
        'lefkimiattis_trace_htv': zeros.copy()
    }
    grid_size = zeros.copy()

    for i, nump in enumerate(npoints):

        h = (cpwl.tri.points[:, 0].max() - cpwl.tri.points[:, 0].min()) / nump
        grid_size[i] = h
        print(f'-- {i+1}/{len(npoints)}: h = {h} ')

        res['exact_grad_trace_htv'][i] = cpwl.get_exact_grad_trace_HTV(h=h)
        res['exact_grad_schatten_htv'][i] = \
            cpwl.get_exact_grad_schatten_HTV(h=h)
        res['lefkimiattis_htv'][i] = cpwl.get_lefkimiattis_schatten_HTV(h=h)
        res['lefkimiattis_trace_htv'][i] = cpwl.get_lefkimiattis_trace_HTV(h=h)

    # Create plots with pre-defined labels.
    fig, ax = plt.subplots()
    ax.plot(grid_size,
            res['exact_grad_trace_htv'],
            'o-',
            label='Exact gradient + Trace')
    ax.plot(grid_size,
            res['exact_grad_schatten_htv'],
            'o-',
            label='Exact gradient + Schatten')
    ax.plot(grid_size,
            res['lefkimiattis_htv'],
            'o-',
            label='Finite differences + Schatten')
    ax.plot(grid_size,
            res['lefkimiattis_trace_htv'],
            'o-',
            label='Finite differences + Trace')
    ax.plot(grid_size,
            np.ones_like(zeros) * exact_htv,
            '--',
            label='Exact')

    plt.semilogx()
    ax.invert_xaxis()
    ax.set_xlabel("Grid size")
    ax.set_ylabel("HTV")

    ax.legend(loc='lower right')

    if args.log_dir is not None:
        filename = os.path.join(
            args.log_dir,
            add_date_to_filename(dataset_dict['name'] + '_discrete_htv') +
            '.pdf'
        )
        plt.savefig(filename)

    plt.show()
    plt.close()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='HTV discretizations vs Grid size.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--log_dir',
        metavar='[STR]',
        type=str,
        help='Log directory for plot.')

    parser.add_argument(
        '--dataset',
        type=str,
        choices=[
            'SimplicialSpline',
            'SimpleJunction',
            'DistortedGrid'
        ],
        default='SimpleJunction',
        help='')

    args = parser.parse_args()

    if args.log_dir is not None and not os.path.isdir(args.log_dir):
        raise NotADirectoryError(
            f'Directory "{args.log_dir}" does not exist...')

    htv_discretizations(args)
