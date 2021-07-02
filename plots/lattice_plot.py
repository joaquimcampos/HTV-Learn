#!/usr/bin/env python3

import os
import copy
import sys
import torch
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.figure_factory as ff
import matplotlib
import matplotlib.pyplot as plt

from lattice import Lattice
from data import Data
from plots.base_plot import BasePlot
from htv_utils import add_date_to_filename

import plotly.express as px


class LatticePlots(BasePlot):

    def __init__(self, lattice_obj, plot_params, data_obj=None, **kwargs):
        """ """
        # the changes in self.lat will also change self.lat (copy)
        super().__init__(lattice_obj, data_obj=data_obj,
                        **plot_params, **kwargs)


    def verify_triangulation_dataset(self):
        """ Verify that a data object exists and dataset is 'planes' or 'face'
        """
        self.verify_data_obj()
        if not (self.data.dataset_name.endswith('planes') or self.data.dataset_name == 'face'):
            raise ValueError('Dataset should be planes or face dataset.')


    def verify_data_obj(self):
        """ Verify that a data object exists
        """
        if self.data is None:
            raise ValueError('A data object does not exist.')


    def add_normals_plot(self, plot_data, normals_scale=0.02):
        """ Get normals and add normal 3Dscatter plot to plot_data.

        Args:
            scale: multiplicative factor of length of normal vectors
        """
        if self.verbose is True:
            print('Adding normals plot.')

        x_lat_centers = self.lat.triangle_centers
        x_values, _ = self.lat.get_values_from_interpolation(x_lat_centers)
        x_std_centers = self.lat.lattice_to_standard(x_lat_centers)
        affine_coeff = self.lat.affine_coeff

        for i in range(x_std_centers.size(0)):
            normal_x = torch.tensor([x_std_centers[i, 0], x_std_centers[i, 0] - affine_coeff[i, 0] * normals_scale])
            normal_y = torch.tensor([x_std_centers[i, 1], x_std_centers[i, 1] - affine_coeff[i, 1] * normals_scale])
            normal_z = torch.tensor([x_values[i], x_values[i].add(1. * normals_scale)])

            normal_x, normal_y, normal_z = normal_x.numpy(), normal_y.numpy(), normal_z.numpy()

            normal_i = self.get_line(x=normal_x, y=normal_y, z=normal_z)
            plot_data.append(normal_i)

        return plot_data



    def add_observations_plot(self, plot_data, mode='train',
                                marker_size=2, **kwargs):
        """ Add observations plot to plot_data

        Args:
            marker_size: marker size for observation points.
        """
        assert mode in ['train', 'valid', 'test']

        self.verify_data_obj()
        if self.verbose is True:
            print('Adding observations plot.')

        data_dict = {'train': self.data.train,
                    'valid': self.data.valid,
                    'test': self.data.test}[mode]

        x_std = data_dict['input'].numpy()
        z = data_dict['values'].numpy()

        observations = self.get_scatter3d(x=x_std[:, 0], y=x_std[:, 1], z=z,
                                    color='black', marker_size=marker_size)
        plot_data.append(observations)

        return plot_data



    def add_mesh_plot(self, plot_data, color='normal',
                            opacity=0.99, **kwargs):
        """ Append large planes plot to plot_data

        Args:
            color: plot triangle colors with random colors,
                according to the average triangle height
            opacity: opacity of planes plot
        """
        self.verify_triangulation_dataset()
        assert color in ['random', 'zero', 'z', 'normal']

        if self.verbose is True:
            print(f'Adding large planes plot.')

        if color == 'zero':
            fc = self.get_zero_facecolor(self.data.triangles.size(0))
        if color == 'z':
            fc = BasePlot.get_z_facecolor(self.data.triangles_with_values[:, :, 2])
        elif color == 'random':
            fc = BasePlot.get_random_facecolor(self.data.triangles.size(0))
        elif color == 'normal':
            fc = BasePlot.get_normal_facecolor(self.data.affine_coeff)

        planes = go.Mesh3d(
                x=self.data.vertices[:, 0].numpy(),
                y=self.data.vertices[:, 1].numpy(),
                z=self.data.vertices[:, 2].numpy(),
                facecolor=fc,
                i=self.data.triangles[:, 0].numpy(),
                j=self.data.triangles[:, 1].numpy(),
                k=self.data.triangles[:, 2].numpy(),
                name='z',
                opacity=opacity,
                showscale=True)

        plot_data.append(planes)

        return plot_data



    def plot_triangulation(self, normals=False, custom=None, observations=False,
                            gtruth=False, filename='lattice_triangulation',
                            color='normal', opacity=0.9, **kwargs):
        """
        Args:
            custom: custom plot to add
            gtruth: plot original data
        """
        assert color in ['random', 'zero', 'z', 'normal']
        if gtruth is True:
            self.verify_triangulation_dataset()

        if self.verbose is True:
            print(f'plot_triangulation(normals={normals}).')

        x_lat = self.lat.lattice_grid
        x_lat_rel = x_lat.sub(self.lat.lmin) # rel = relative to bottom left corner
        x_values = self.lat.C_mat[x_lat_rel[:, 0], x_lat_rel[:, 1]]
        x_std = self.lat.lattice_to_standard(x_lat.float())
        x_std, z = x_std.numpy(), x_values.numpy()

        # check self.lat.unique triangles
        if color == 'zero':
            fc = self.get_zero_facecolor(self.lat.unique_triangles.size(0))
        if color == 'z':
            fc = self.get_z_facecolor(self.lat.unique_triangles_values)
        elif color == 'random':
            fc = self.get_random_facecolor(self.lat.unique_triangles.size(0))
        elif color == 'normal':
            # affine coefficients of each unique triangle
            fc = self.get_normal_facecolor(self.lat.affine_coeff)

        simplices = self.lat.simplices.numpy()
        i = simplices[:, 0]
        j = simplices[:, 1]
        k = simplices[:, 2]

        data = [go.Mesh3d(x=x_std[:, 0], y=x_std[:, 1], z=z,
                            i=i, j=j, k=k,
                            facecolor=fc,
                            opacity=opacity)]

        if normals:
            data = self.add_normals_plot(data)

        if custom is not None:
            assert isinstance(custom, list)
            data = data + custom

        if observations:
            data = self.add_observations_plot(data, **kwargs)

        if gtruth:
            data = self.add_mesh_plot(data, color=color)

        fig = go.Figure(data=data)

        if self.data.dataset_name == 'face':
            fig.update_traces(lighting=dict(ambient=0.1, diffuse=1, specular=0.1),
                        lightposition=dict(x=0, y=0, z=4),
                        selector=dict(type='mesh3d'))
        else:
            fig.update_traces(lighting=dict(ambient=0.95, diffuse=0.),
                            selector=dict(type='mesh3d'))

        self.plot_fig(fig, filename=filename, **kwargs)



    def plot_trisurf_triangulation(self, filename='lattice_trisurf_triangulation',
                                color='normal', **kwargs):
        """
        Args:
        """
        assert color in ['random', 'zero', 'z', 'normal']

        if self.verbose is True:
            print(f'plot_trisurf_triangulation.')

        x_lat = self.lat.lattice_grid
        x_lat_rel = x_lat.sub(self.lat.lmin) # rel = relative to bottom left corner
        x_values = self.lat.C_mat[x_lat_rel[:, 0], x_lat_rel[:, 1]]

        x_std = self.lat.lattice_to_standard(x_lat.float())
        x_std, z = x_std.numpy(), x_values.numpy()

        simplices = self.lat.simplices.numpy()
        trisurf_fig_dict = dict(x=x_std[:, 0],
                                y=x_std[:, 1],
                                simplices=simplices,
                                edges_color='rgb(0, 0, 0)')

        if color == 'normal':
            # affine coefficients of each unique triangle
            color_list = self.get_normal_facecolor(self.lat.affine_coeff)
        elif color == 'z':
            z_mean = z[simplices].mean(axis=1)
            color_list = self.map_array2color(z_mean)
        elif color == 'zero':
            color_list = self.get_zero_facecolor(simplices.shape[0])
        elif color == 'random':
            color_list = self.get_random_facecolor(simplices.shape[0])

        data = ff.create_trisurf(z=z, color_func=color_list, **trisurf_fig_dict)

        fig = go.Figure(data=data)

        filename = 'trisurf_lattice'
        self.plot_fig(fig, filename=filename, **kwargs)



    def plot_data(self, filename='data', view='side',
                    gtruth=True, observations=False, gtruth_color='normal',
                     **kwargs):
        """ """
        assert view in ['up', 'side', '3D', '3D_2']
        self.verify_triangulation_dataset()

        data = []

        if gtruth:
            data = self.add_mesh_plot(data, color=gtruth_color, **kwargs)

        if observations:
            data = self.add_observations_plot(data, **kwargs)

        fig = go.Figure(data=data)
        if self.data.dataset_name == 'face':
            fig.update_traces(lighting=dict(ambient=0.1, diffuse=1, specular=0.1),
                        lightposition=dict(x=0, y=0, z=4),
                        selector=dict(type='mesh3d'))

        self.plot_fig(fig, filename=filename, view=view, **kwargs)


#########################################################################

def lattice_plots(lat_plot):
    """
    Args:
        plot - LatticePlots instance
    """
    # lat_plot.plot_triangulation(color='normal', filename='tmp', normals=True)
    # lat_plot.plot_trisurf_triangulation(color='normal', filename='tmp')
    # lat_plot.plot_data()


if __name__ == '__main__':

    lattice_obj = Lattice(lsize=10, C_init='normal')
    data_obj = Data(lattice_obj, **{})
    lat_plot = LatticePlots(lattice_obj, {'log_dir': './output', 'html': True}, data_obj)

    lattice_plots(lat_plot)
