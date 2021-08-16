import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from htvlearn.delaunay import Delaunay
from htvlearn.plots.base_plot import BasePlot


class Plot(BasePlot):

    def __init__(self, data_obj=None, **plot_params):
        """
        Args:
            data_obj (Data):
                None (if not plotting data samples) or
                object of class Data (see htvlearn.data).
        """
        super().__init__(data_obj=data_obj, **plot_params)

    def plot_delaunay(self,
                      *delaunay_obj_list,
                      observations=False,
                      color='normal',
                      opaque=True,
                      filename='trisurf_delaunay',
                      **kwargs):
        """
        Plot a several Delaunay objects from a list.

        Args:
            delaunay_obj_list (Delaunay):
                list of Delaunay objects (see htvlearn.delaunay).
            observations (bool):
                True if plotting data observations.
            color (str):
                Plot colors according to simplex normals (color="normal")
                or mean height of vertices (color="z").
            opaque (bool):
                If True, ground truth is made opaque
                (if True, might make some observations non-visible).
            filename (str):
                Figure filename.
        """
        specs = [[{'type': 'scene'}] * len(delaunay_obj_list)]
        num_subplots = len(delaunay_obj_list)
        fig = make_subplots(cols=num_subplots,
                            specs=specs,
                            shared_xaxes=True,
                            shared_yaxes=True)
        if observations:
            fig.add_trace(self.get_observations_plot(**kwargs), row=1, col=1)

        for i, delaunay in enumerate(delaunay_obj_list):
            if not isinstance(delaunay, Delaunay):
                continue

            delaunay_fig_dict = dict(x=delaunay.tri.points[:, 0],
                                     y=delaunay.tri.points[:, 1],
                                     simplices=delaunay.tri.simplices,
                                     edges_color='rgb(255, 255, 255)',
                                     plot_edges=False)

            if color == 'normal':
                # affine coefficients of each simplex
                color_list = self.get_normal_facecolor(
                    delaunay.tri.simplices_affine_coeff)
            elif color == 'z':
                z_mean = \
                    delaunay.tri.values[delaunay.tri.simplices].mean(axis=1)
                color_list = self.map_array2color(z_mean)
            else:
                raise ValueError(f'color {color} should be "normal" or "z".')

            trisurf_fig = ff.create_trisurf(z=delaunay.tri.values,
                                            color_func=color_list,
                                            **delaunay_fig_dict)

            for trisurf_trace in trisurf_fig.data:
                fig.add_trace(trisurf_trace, row=1, col=i + 1)

            if opaque is False:
                fig['data'][-1].update(opacity=0.85)

        self.plot_fig(fig,
                      filename=filename,
                      num_subplots=num_subplots,
                      **kwargs)

    def get_observations_plot(self,
                              mode='train',
                              marker_size=2,
                              **kwargs):
        """
        Get observations plot.

        Args:
            mode (str):
                'train', 'valid', or 'test'
            marker_size (float):
                marker size for observation points.

        Returns:
            A plotly.graph_objects.Scatter3D object.
        """
        assert mode in ['train', 'valid', 'test']
        self.verify_data_obj()

        data_dict = {
            'train': self.data.train,
            'valid': self.data.valid,
            'test': self.data.test
        }[mode]

        input = data_dict['input'].cpu().numpy()
        values = data_dict['values'].cpu().numpy()

        observations = self.get_scatter3d(x=input[:, 0],
                                          y=input[:, 1],
                                          z=values,
                                          marker_size=marker_size)
        return observations
