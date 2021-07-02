#!/usr/bin/env python3

import os
import numpy as np
import copy
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import matplotlib.cm as cm
import argparse
import sys

from plots.base_plot import BasePlot


def plot_two_triangles(params):

    np.random.seed(params['seed'])

    pts = np.array([[0., 1.], [1., 0.], [2.2, 1.5], [1., 2.]])

    z = np.zeros_like(pts[:, 0])
    z_max = 1.5
    if params['eigen'] == 'pos':
        z[0], z[2] = z_max*3/4, z_max*9/20
    elif params['eigen'] == 'neg':
        z[1], z[0], z[3] = z_max*3/4, 0, z_max*3/4
    else:
        z[1], z[0], z[3] = z_max*3/8, z_max*3/4, z_max*3/8

    tri = Delaunay(pts)
    simplices = tri.simplices

    facecolor_uint8 = BasePlot.get_random_facecolor_uint8(simplices.shape[0],
                                                mode='linear', color_range=0.5)
    color_list = BasePlot.get_random_facecolor(facecolor_uint8=facecolor_uint8,
                                                colorscale='coolwarm')

    fig_dict = dict(x=pts[:, 0], y=pts[:, 1], simplices=simplices,
                    edges_color='rgb(200, 200, 200)',
                    title='Two Triangles', showbackground=True,
                    backgroundcolor='rgba(245, 245, 245, 1)')

    fig = ff.create_trisurf(z=z, color_func=color_list, **fig_dict)

    ax_dict = dict(linecolor='#000000', linewidth=4, showgrid=False,
                tickangle=0, tick0=0, dtick=1,
                showticklabels=False,
                zeroline=True, zerolinewidth=0.5, zerolinecolor='#000000',
                tickfont=dict(size=14, family='sans-serif'),
                gridcolor='#000000', gridwidth=0.7,
                title=dict(font=dict(size=38, family='sans-serif'))
    )

    fig_dict = dict(
        scene = dict(
            aspectmode='manual',
            aspectratio=dict(x=pts[:, 0].max(), y=pts[:, 1].max(), z=z_max),
            xaxis={**copy.deepcopy(ax_dict), **{'range': [0, pts[:, 0].max()]}},
            yaxis={**copy.deepcopy(ax_dict), **{'range': [0, pts[:, 1].max()]}},
            zaxis={**copy.deepcopy(ax_dict),  **{'range': [0, 1.5]}},
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            )
        )
    )

    scale=1.8
    fig_dict['scene']['camera']['eye'] = dict(x=0.1*scale, y=-2.2*scale, z=0.6*scale)


    fig.update_layout(**fig_dict)
    if args.html is False:
        fig.show()
    else:
        BasePlot.export_fig(fig, f'eigen_{args.eigen}', args.log_dir)



if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(description='Plot two triangles.')
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--html', action='store_true', help='Save as html.')
    parser.add_argument('--log_dir', type=str, help='Log directory.')
    parser.add_argument('--eigen', choices=['pos', 'neg', 'zero'], type=str,
                        help=f'Eigenvalues of junction Hessian.', default='pos')
    args = parser.parse_args()
    params = vars(args)

    if args.html and args.log_dir is None:
        raise ValueError('Need to provide log directory for saving .')

    plot_two_triangles(params)
