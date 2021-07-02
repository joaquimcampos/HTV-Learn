############################################
from plots.lattice_plot import LatticePlots


class Plot():

    def __init__(self, lat_obj, data_obj, **params):
        """ """
        self.params = params
        self.dataset_name = data_obj.dataset_name
        self.lat_plot = LatticePlots(lat_obj, params, data_obj=data_obj)


    def show_data_plot(self, **kwargs):
        """ """
        self.lat_plot.plot_data(**kwargs)


    def show_lat_plot(self, *args, **kwargs):
        """ """
        self.lat_plot.plot_triangulation(*args, **kwargs)
