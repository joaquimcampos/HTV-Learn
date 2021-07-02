### Program Manager

import sys
import os
import copy
import torch

from algorithm import Algorithm
from htv_project import HTVProject
import htv_utils

from plots.plot import Plot


class HTVManager(HTVProject):
    """ """

    def __init__(self, params):
        """ """

        # initializes log, lattice, data, json files
        super().__init__(params)

        self.plot = None
        if self.params['plot']:
            self.plot = Plot(self.lat, self.data, **self.params['plots'])
            noise_str = 'noisy_' if self.data.add_noise else ''
            self.plot.show_data_plot(observations=True,
                                    gtruth='True', gtruth_color='normal',
                                    mode='train',
                                    filename=f'{noise_str}data')

        self.algorithm = Algorithm(self.lat, self.data, self.plot,
                                    **self.params['algorithm'])


    def train(self):
        """ """
        # updates lattice; creates lattice_dict and results_dict.
        self.algorithm.multires_admm()

        self.evaluate_results()

        if self.data.dataset_name.endswith('planes') or self.data.dataset_name == 'face':
            if self.params['plot']:
                plot = Plot(self.lat, self.data, **self.params['plots'])
                noise_str = 'noisy_' if self.data.add_noise else ''

                plot.show_lat_plot(gtruth=False,
                                    observations=False,
                                    filename=f'{noise_str}triangulation_model')

        # save numeric results in json file
        for info, value in self.algorithm.results_dict.items():
            self.update_json(info, value)

        # save params, lattice history and data to checkpoint
        self.save_to_ckpt(self.algorithm.lattice_dict)



    def evaluate_results(self):
        """ """
        for mode, data_dict in zip(['valid', 'test'], [self.data.valid, self.data.test]):
            output = self.forward_data(self.lat, data_dict['input'])
            # save predictions
            data_dict['predictions'] = output

            # compute mse
            mse, _ = htv_utils.compute_mse_psnr(data_dict['values'], output)
            self.update_json('_'.join([mode, 'mse']), mse)
            print(f'{mode} mse  : {mse}')



    @staticmethod
    def forward_data(lattice_obj, input_std):
        """ """
        input_lat = lattice_obj.standard_to_lattice(input_std)
        output = lattice_obj.get_values_from_affine_coefficients(input_lat)

        return output
