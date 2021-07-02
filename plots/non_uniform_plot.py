#!/usr/bin/env python3

import numpy as np
import copy
from scipy.spatial import Delaunay
import plotly.figure_factory as ff
import matplotlib.cm as cm
import argparse

from plots.base_plot import BasePlot


def plot_delauney(params):

    np.random.seed(params['seed'])

    u = np.linspace(-6, 6, 5) * 1.
    v = np.linspace(-6, 6, 5) * 1.
    u,v = np.meshgrid(u,v)
    u = u.flatten()
    v = v.flatten()

    mask = np.ma.mask_or(np.abs(u) == u.max(), np.abs(v) == v.max())
    # noisy
    x, y = u, v
    for vec in [x, y]:
        noise = (np.random.rand(*u.shape) - 0.5) * (u[1]-u[0]) * 0.5
        vec[~mask] = vec[~mask] + noise[~mask]

    z = np.zeros_like(x)
    z[x.shape[0]//2] = 3.

    points2D = np.vstack([u,v]).T
    tri = Delaunay(points2D)
    simplices = tri.simplices

    fig_dict = dict(x=x, y=y, simplices=simplices,
                    edges_color='rgb(255, 255, 255)',
                    title='Delaunay', showbackground=False)

    if params['view'] == 'up_flat':
        color_list = np.repeat(BasePlot.map_val2color(0., 0., z.max(), colorscale=cm.coolwarm), simplices.shape[0])
        fig = ff.create_trisurf(z=np.zeros_like(x), color_func=color_list, **fig_dict)
    else:
        z_mean = z[simplices].mean(axis=1)
        color_list = np.array([BasePlot.map_val2color(val, 0., z.max(), colorscale=cm.coolwarm) for val in z_mean])
        fig = ff.create_trisurf(z=z, color_func=color_list, **fig_dict)

    ax_dict = dict(linecolor='#000000', linewidth=4, showgrid=True,
                showticklabels=False, gridcolor='#000000', gridwidth=0.3,
                title=dict(font=dict(size=20))
    )
    fig_dict = dict(
        scene_aspectmode='data',
        scene = dict(
            xaxis=copy.deepcopy(ax_dict),
            yaxis=copy.deepcopy(ax_dict),
            zaxis=copy.deepcopy(ax_dict),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )
        ),
        font=dict(size=16)
    )

    if params['view'] == 'side_x':
        fig_dict['scene']['camera']['eye'] = dict(x=2.3, y=0, z=1.1)
    elif params['view'] == 'side_y':
        fig_dict['scene']['camera']['eye'] = dict(x=0, y=2.5, z=1.2)
    else:
        fig_dict['scene']['camera']['up'] = dict(x=0, y=1, z=0)
        fig_dict['scene']['camera']['eye'] = dict(x=0, y=0, z=4)
        fig_dict['scene']['zaxis']['visible'] = False

    fig.update_layout(**fig_dict)
    if args.html is False:
        fig.show()
    else:
        BasePlot.export_fig(fig, f'delaunay', args.log_dir)



if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(description='Plot delauney triangulation.')
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--html', action='store_true', help='Save as html.')
    parser.add_argument('--log_dir', type=str, help='Log directory.')
    parser.add_argument('--view', choices=['up_flat', 'side_x', 'side_y'],
                        type=str, help=f'Plot view.', default='3D')
    args = parser.parse_args()
    params = vars(args)

    if args.html and args.log_dir is None:
        raise ValueError('Need to provide log directory for saving.')

    plot_delauney(params)
