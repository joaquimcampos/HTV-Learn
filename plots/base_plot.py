import os
import copy
import numpy as np
import plotly.graph_objects as go
import chart_studio
import plotly.io as pio
import matplotlib.cm as cm

chart_studio.tools.set_credentials_file(username='jcampos',
                                        api_key='UDIpS9sd0Gd9rHbUcf3e')


class BasePlot():

    colorscale = 'coolwarm'  # 'Blues_r' # 'Greys' # 'Greys_r' # 'Blues_r'

    def __init__(self,
                 data_obj=None,
                 log_dir=None,
                 view=None,
                 verbose=False,
                 **kwargs):
        """
        Args:
            log_dir: if not None, log directory for html images
            view: plot view
        """
        self.data = data_obj
        self.log_dir = log_dir
        self.view = view
        self.verbose = verbose

        if self.log_dir is not None and not os.path.isdir(self.log_dir):
            raise NotADirectoryError(
                f'log_dir "{self.log_dir}" is not a valid directory')

    @classmethod
    def map_val2color(cls, val, vmin, vmax, colorscale=None):
        """ Map the normalized value "val" to a corresponding
        color in the colormap.
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

    @staticmethod
    def map_frac_float_to_rgb(array):
        """ Map colors from 3 values [0,1] to RGB
        """
        if not array.shape[1] == 3:
            raise ValueError(f'array has #cols {array.shape[1]}. Should be 3.')

        color_list = []
        for row in array:
            color_list += [
                'rgb(' + '{:d}'.format(int(row[0] * 255 + 0.5)) + ',' +
                '{:d}'.format(int(row[1] * 255 + 0.5)) + ',' +
                '{:d}'.format(int(row[2] * 255 + 0.5)) + ')'
            ]

        return np.array(color_list)

    @classmethod
    def map_val2color2d(cls, val, vmin, vmax):
        """ Map the normalized 2D value to a corresponding R, B channel.
        val, vmin, vmax: 2d array
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
        """ """
        if min is None:
            min = array.min()
        if max is None:
            max = array.max()

        return np.array([cls.map_val2color(val, min, max) for val in array])

    @classmethod
    def map_array2color2d(cls, array, min=None, max=None):
        """ """
        if min is None:
            min = array.amin(axis=0)
        if max is None:
            max = array.amax(axis=0)

        return np.array([cls.map_val2color2d(val, min, max) for val in array])

    @classmethod
    def get_z_facecolor(cls, tri_z):
        """ Get facecolor according to the mean of the
        heights of the triangle vertices.
        """
        # mean of heights of vertices for each triangle
        tri_zmean = tri_z.mean(1).numpy()
        min_tri_zmean = np.min(tri_zmean)
        max_tri_zmean = np.max(tri_zmean)

        facecolor = cls.map_array2color(tri_zmean,
                                        min=min_tri_zmean,
                                        max=max_tri_zmean)

        return facecolor

    @classmethod
    def get_normal_facecolor(cls, affine_coeff):
        """ Get facecolor of triangles according to
        their normals.
        """
        max = np.array([1.75, 1.75])
        facecolor = \
            cls.map_array2color2d(-affine_coeff[:, 0:2], min=-max, max=max)

        return facecolor

    def get_x_facecolor(self, x_lat):
        """ Get facecolors according to mean height of vertices
        of unique triangles where each point in x_lat lives.
        """
        facecolor = self.get_z_facecolor(self.lat.unique_triangles_values)

        idx_unique_triangles, _ = \
            self.lat.check_my_triangle(x_lat, get='idx_unique')
        color = facecolor[idx_unique_triangles.numpy()]

        return color

    @staticmethod
    def get_scatter3d(x, y, z, color, cmax=None, marker_size=2):
        """ """
        data = go.Scatter3d(x=x,
                            y=y,
                            z=z,
                            mode='markers',
                            marker=dict(size=marker_size,
                                        color=color,
                                        cmax=cmax),
                            opacity=0.8)

        return data

    @classmethod
    def get_surface3d(cls, x, y, z, surfcolor=None, **kwargs):
        """ """
        colorscale = cls.colorscale if surfcolor is None else None
        if surfcolor is not None:
            dcolor = surfcolor.max() - surfcolor.min()
            cmin = surfcolor.min() - 0.2 * dcolor
            cmax = surfcolor.max() + 0.2 * dcolor
            # cmin, cmax = -1, 1
        else:
            dy = z.max() - z.min()
            cmin = z.min() - 0.5 * dy
            cmax = z.max() + 0.3 * dy

        data = go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=surfcolor,
            # cauto=True,
            cmin=cmin,
            cmax=cmax,
            colorscale=colorscale,
            reversescale=True)

        return data

    @staticmethod
    def get_line(x, y, z):
        """ """
        line = go.Scatter3d(x=x,
                            y=y,
                            z=z,
                            marker=dict(size=1,
                                        symbol='circle',
                                        color="rgb(84,48,5)"),
                            line=dict(color="rgb(84,48,5)", width=8),
                            opacity=0.9)

        return line

    def plot_fig(self,
                 fig,
                 filename=None,
                 num_subplots=1,
                 view=None,
                 top=False,
                 **kwargs):
        """ """
        if self.view is not None:
            view = self.view
        elif view is None:
            view = '3D'  # default view

        assert view in ['up', 'side', '3D', '3D_2']

        ax_dict = dict(linecolor='#000000',
                       linewidth=4,
                       showgrid=False,
                       showticklabels=False,
                       tickfont=dict(size=15),
                       gridcolor='#000000',
                       gridwidth=0.3,
                       title='',
                       showbackground=True)
        # title=dict(font=dict(size=35), linecolor='#FFFFFF')

        fig_dict = dict(
            scene_aspectmode='data',
            scene=dict(xaxis=copy.deepcopy(ax_dict),
                       yaxis=copy.deepcopy(ax_dict),
                       zaxis=copy.deepcopy(ax_dict),
                       camera=dict(up=dict(x=0, y=0, z=1),
                                   center=dict(x=0, y=0, z=0))),
            font=dict(size=30),
        )

        if top is True:
            fig_dict['scene']['zaxis']['visible'] = False
            for ax in ['xaxis', 'yaxis']:
                fig_dict['scene'][ax]['showbackground'] = False

        if view == 'side':
            fig_dict['scene']['camera']['eye'] = dict(x=1.2, y=0.3, z=0.4)
        elif view == '3D':
            fig_dict['scene']['camera']['eye'] = dict(x=1.5, y=1.9, z=1.7)
        elif view == '3D_2':
            fig_dict['scene']['camera']['eye'] = dict(x=1.0, y=-2.0, z=0.3)
        elif view == 'up':
            fig_dict['scene']['zaxis']['visible'] = False
            fig_dict['scene']['camera']['eye'] = dict(x=0, y=0, z=3)
            fig_dict['scene']['camera']['up'] = dict(x=0, y=1, z=0)

        for i in range(2, num_subplots + 1):
            fig_dict['scene' + str(i)] = fig_dict['scene'].copy()

        fig.update_layout(**fig_dict)
        if self.log_dir is None:
            fig.show()
        else:
            self.export_fig(fig, filename, self.log_dir)

    @staticmethod
    def export_fig(fig, filename, log_dir):
        """ """
        if not os.path.isdir(log_dir):
            raise NotADirectoryError(
                f'log_dir "{log_dir}" is not a valid directory')

        file_path = os.path.join(f'{log_dir}', f'{filename}')
        pio.write_html(fig, file=f'{file_path}.html', auto_open=True)
