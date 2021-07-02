#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas
import math
import copy
from scipy.spatial import Delaunay
import plotly.figure_factory as ff
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import argparse

from plots.base_plot import BasePlot


def plot_box_spline(params):

    np.random.seed(params['seed'])
    v1 = np.array([1, 0])
    v2 = np.array([.5, math.sqrt(3)/2])

    pts = np.array([v1, v2, -v1+v2, -v1, -v2, -v2+v1, np.array([0., 0.])])

    z = np.zeros_like(pts[:, 0])
    z[pts.shape[0]-1] = 1

    tri = Delaunay(pts)
    simplices = tri.simplices

    facecolor_uint8 = BasePlot.get_random_facecolor_uint8(simplices.shape[0], mode='linear', color_range=0.1)
    color_list = BasePlot.get_random_facecolor(facecolor_uint8=facecolor_uint8, colorscale='Blues')

    fig_dict = dict(x=pts[:, 0], y=pts[:, 1], simplices=simplices,
                    edges_color='rgb(255, 255, 255)',
                    title='Box Spline', showbackground=True,
                    backgroundcolor='rgba(245, 245, 245, 1)')

    fig = ff.create_trisurf(z=z, color_func=color_list, **fig_dict)

    ax_dict = dict(linecolor='#000000', linewidth=4,
                showgrid=True, tickvals=np.array([-1, -0.5, 0., 0.5, 1]),
                tickmode='array', tickangle=0,
                zeroline=True, zerolinewidth=0.5, zerolinecolor='#000000',
                showticklabels=True,
                tickfont=dict(size=14, family='sans-serif'),
                gridcolor='#000000', gridwidth=0.5,
                title=dict(font=dict(size=26, family='sans-serif'))
    )

    fig_dict = dict(
        scene_aspectmode='data',
        scene = dict(
            xaxis=copy.deepcopy(ax_dict),
            yaxis=copy.deepcopy(ax_dict),
            zaxis={**copy.deepcopy(ax_dict),
                    **{ 'tickvals': np.array([1.]),
                        'ticktext': ['1'],
                        'range': [0., 1.1],
                        'zeroline': False}},
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            )
        )
    )

    fig_dict['scene']['camera']['eye'] = dict(x=0.25, y=-2.4, z=0.6)

    fig.update_layout(**fig_dict)
    if args.html is False:
        fig.show()
    else:
        BasePlot.export_fig(fig, f'box_spline', args.log_dir)



if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(description='Plot box spline.')
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--html', action='store_true', help='Save as html.')
    parser.add_argument('--log_dir', type=str, help='Log directory.')
    args = parser.parse_args()
    params = vars(args)

    if args.html and args.log_dir is None:
        raise ValueError('Need to provide log directory for saving.')

    plot_box_spline(params)
