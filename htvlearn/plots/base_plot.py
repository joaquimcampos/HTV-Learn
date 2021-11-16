import os
import copy
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.cm as cm

from htvlearn.data import Data


class BasePlot():

    colorscale = 'coolwarm'  # default colorscale

    def __init__(self,
                 data_obj=None,
                 log_dir=None,
                 **kwargs):
        """
        Args:
            data_obj (Data):
                None (if not plotting data samples) or
                object of class Data (see htvlearn.data).
            log_dir (str):
                Log directory for html images. If None, the images are only
                shown but not saved in html format.
        """
        self.data = data_obj
        self.log_dir = log_dir

        if self.log_dir is not None and not os.path.isdir(self.log_dir):
            raise NotADirectoryError(
                f'log_dir "{self.log_dir}" is not a valid directory')

    def verify_data_obj(self):
        """Verify that a data object exists and is valid"""
        if self.data is None:
            raise ValueError('A data object does not exist.')
        elif not isinstance(self.data, Data):
            raise ValueError(f'data_obj is of type {type(self.data)}.')

    @classmethod
    def map_val2color(cls, val, vmin, vmax, colorscale=None):
        """
        Map a value to a color in the colormap.

        Args:
            val (float):
                value in [vmin, vmax] to map to color.
            vmin, vmax (float):
                min and max ranges for val.
            colorscale (str):
                matplotlib colorscale or None (use default).

        Returns:
            rgb color.
        """
        colorsc = colorscale if colorscale is not None else cls.colorscale
        cmap = cm.get_cmap(colorsc)
        if vmin > vmax:
            raise ValueError('incorrect relation between vmin and vmax')

        t = 0.
        if not np.allclose(vmin, vmax):
            t = (val - vmin) / float((vmax - vmin))  # normalize val
        R, G, B, alpha = cmap(t)
        return 'rgb(' + '{:d}'.format(int(R * 255 + 0.5)) + ',' + '{:d}'\
            .format(int(G * 255 + 0.5)) +\
            ',' + '{:d}'.format(int(B * 255 + 0.5)) + ')'

    @classmethod
    def map_val2color2d(cls, val, vmin, vmax):
        """
        Map a 2D value to an rgb color. The R and G channels have
        an independent and direct correspondence to each element in
        the 2D value. The B channel is kept fixed.

        Args:
            val (float):
                value s.t. val[i] in [vmin[i], vmax[i]], i=1,2,
                to be mapped to an rgb color. The R, G channels are set
                by val.
            vmin, vmax (2d array):
                min and max ranges for each element in val.

        Returns:
            rgb color.
        """
        if vmin[0] > vmax[0] or vmin[1] > vmax[1]:
            raise ValueError('incorrect relation between vmin and vmax')

        t = np.zeros(2)
        # normalize val
        if not np.allclose(vmin[0], vmax[0]):
            t[0] = (val[0] - vmin[0]) / float((vmax[0] - vmin[0]))
        if not np.allclose(vmin[1], vmax[1]):
            t[1] = (val[1] - vmin[1]) / float((vmax[1] - vmin[1]))

        R, G = t[1], t[0]
        B = 0.4
        return 'rgb(' + '{:d}'.format(int(R * 255 + 0.5)) + ',' + '{:d}'\
            .format(int(G * 255 + 0.5)) +\
            ',' + '{:d}'.format(int(B * 255 + 0.5)) + ')'

    @classmethod
    def map_array2color(cls, array, min=None, max=None):
        """
        Map an array of values to colors.

        Args:
            array (1d array):
                array of values to map to colors. size: (N,)
            min, max (float):
                If not None, set the ranges for the values in array.

        Returns:
            1d array of rgb colors. size: (N,)
        """
        if min is None:
            min = array.min()
        if max is None:
            max = array.max()

        return np.array([cls.map_val2color(val, min, max) for val in array])

    @classmethod
    def map_array2color2d(cls, array, min=None, max=None):
        """
        Map a 2D array of values to colors.

        Args:
            array (2d array):
                array of values to map to colors. size: (N x 2).
            min, max (2d array):
                If not None, sets the ranges for the values in array.

        Returns:
            1d array of rgb colors. size: (N,)
        """
        if array.shape[1] != 2:
            raise ValueError(f"array has shape {array.shape}.")
        if min is None:
            if min.shape != (2, ):
                raise ValueError(f'min has shape {min.shape}')
            min = array.amin(axis=0)
        if max is None:
            if max.shape != (2, ):
                raise ValueError(f'max has shape {max.shape}')
            max = array.amax(axis=0)

        return np.array([cls.map_val2color2d(val, min, max) for val in array])

    @classmethod
    def get_normal_facecolor(cls, affine_coeff, max=None):
        """
        Get facecolor of simplices according to their normals.

        Args:
            affine_coeff (array):
                affine coefficients of the simplices.
                size: (number of simplices, 3).
            max (2d array):
                If not None, sets the max ranges for the values in
                affine_coeff[:, 0:2].

        Returns:
            facecolor (1d array):
                1d array of rgb colors whose size is the number of simplices.
        """
        if not affine_coeff.shape[1] == 3:
            raise ValueError(f"affine_coeff has shape {affine_coeff.shape}.")
        if max is None:
            max = np.array([1.75, 1.75])
        facecolor = \
            cls.map_array2color2d(-affine_coeff[:, 0:2], min=-max, max=max)

        return facecolor

    @staticmethod
    def get_scatter3d(x, y, z, marker_size=2):
        r"""
        Get a scatter 3D plot (f: \R^2 \to \R)

        Args:
            x, y (1d array):
                positions of the samples.
            z (1d array):
                values of the samples.

        Returns:
            A plotly.graph_objects.Scatter3D object.
        """
        data = go.Scatter3d(x=x,
                            y=y,
                            z=z,
                            mode='markers',
                            marker=dict(size=marker_size,
                                        color='black'),
                            opacity=0.8)

        return data

    def plot_fig(self,
                 fig,
                 filename=None,
                 num_subplots=1,
                 view=None,
                 **kwargs):
        """
        Plot figure.

        Args:
            fig:
                instance of plotly.graph_objects.Figure to plot.
            filename (str):
                Figure filename.
            num_subplots (int):
                Number of figure subplots.
            view (str):
                If None, a default view is used.
                Otherwise can be set to "up" "side".
        """
        assert isinstance(fig, go.Figure), f'fig is of type {type(fig)}.'
        assert view in [None, 'up', 'side'], f'view "{view}" is invalid.'

        ax_dict = dict(linecolor='#000000',
                       linewidth=4,
                       showgrid=False,
                       showticklabels=False,
                       tickfont=dict(size=15),
                       gridcolor='#000000',
                       gridwidth=0.3,
                       title='',
                       showbackground=True)

        fig_dict = dict(
            scene_aspectmode='data',
            scene=dict(xaxis=copy.deepcopy(ax_dict),
                       yaxis=copy.deepcopy(ax_dict),
                       zaxis=copy.deepcopy(ax_dict),
                       camera=dict(up=dict(x=0, y=0, z=1),
                                   center=dict(x=0, y=0, z=0))),
            font=dict(size=30),
        )

        if view == 'up':
            fig_dict['scene']['zaxis']['visible'] = False
            fig_dict['scene']['camera']['eye'] = dict(x=0, y=0, z=3)
            fig_dict['scene']['camera']['up'] = dict(x=0, y=1, z=0)
        elif view == 'side':
            fig_dict['scene']['camera']['eye'] = dict(x=1.2, y=0.3, z=0.4)
        else:
            # default
            fig_dict['scene']['camera']['eye'] = dict(x=1.5, y=1.9, z=1.7)

        for i in range(2, num_subplots + 1):
            fig_dict['scene' + str(i)] = fig_dict['scene'].copy()

        fig.update_layout(**fig_dict)
        if self.log_dir is None:
            fig.show()
        else:
            self.export_fig(fig, filename, self.log_dir)

    @staticmethod
    def export_fig(fig, filename, log_dir):
        """
        Plot html figure and export to log_dir.

        Args:
            fig:
                instance of plotly.graph_objects.Figure to plot.
            filename (str):
                Figure filename.
            log_dir (str):
                Log directory where figure is exported to.
        """
        assert isinstance(fig, go.Figure), f'fig is of type {type(fig)}.'
        if not os.path.isdir(log_dir):
            raise NotADirectoryError(
                f'log_dir "{log_dir}" is not a valid directory')

        file_path = os.path.join(f'{log_dir}', f'{filename}')
        pio.write_html(fig, file=f'{file_path}.html', auto_open=True)
