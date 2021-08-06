import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from htvlearn.delaunay import Delaunay
from htvlearn.plots.base_plot import BasePlot


class Plot(BasePlot):
    def __init__(self, data_obj=None, **plot_params):
        """ """
        super().__init__(data_obj=data_obj, **plot_params)

    def plot_delaunay(self,
                      *delaunay_obj_list,
                      observations=False,
                      color='normal',
                      opaque=True,
                      filename='trisurf_delaunay',
                      **kwargs):
        """ """
        specs = [[{'type': 'scene'}] * len(delaunay_obj_list)]
        num_subplots = len(delaunay_obj_list)
        fig = make_subplots(cols=num_subplots,
                            specs=specs,
                            shared_xaxes=True,
                            shared_yaxes=True)
        data = []
        if observations:
            data = self.add_observations_plot(data, **kwargs)
            fig.add_trace(data[0], row=1, col=1)
            # fig['data'][0].update(opacity=1)

        for i, delaunay in enumerate(delaunay_obj_list):
            if not isinstance(delaunay, Delaunay):
                continue

            delaunay_fig_dict = dict(x=delaunay.tri.points[:, 0],
                                     y=delaunay.tri.points[:, 1],
                                     simplices=delaunay.tri.simplices,
                                     edges_color='rgb(255, 255, 255)',
                                     plot_edges=False)

            if color == 'normal':
                # affine coefficients of each unique triangle
                color_list = self.get_normal_facecolor(
                    delaunay.tri.simplices_affine_coeff)
            elif color == 'z':
                z_mean = \
                    delaunay.tri.values[delaunay.tri.simplices].mean(axis=1)
                color_list = self.map_array2color(z_mean)

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

    def verify_data_obj(self):
        """ Verify that a data object exists
        """
        if self.data is None:
            raise ValueError('A data object does not exist.')

    def add_observations_plot(self,
                              plot_data,
                              mode='train',
                              marker_size=2,
                              **kwargs):
        """ Add observations plot to plot_data

        Args:
            marker_size: marker size for observation points.
        """
        assert mode in ['train', 'valid', 'test']

        self.verify_data_obj()
        if self.verbose is True:
            print('Adding observations plot.')

        data_dict = {
            'train': self.data.train,
            'valid': self.data.valid,
            'test': self.data.test
        }[mode]

        x_std = data_dict['input'].cpu().numpy()
        z = data_dict['values'].cpu().numpy()

        observations = self.get_scatter3d(x=x_std[:, 0],
                                          y=x_std[:, 1],
                                          z=z,
                                          color='black',
                                          marker_size=marker_size)
        plot_data.append(observations)

        return plot_data
