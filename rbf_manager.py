### Program Manager

import sys
import os
import copy

from rbf_project import RBFProject
import htv_utils

from plots.plot import Plot
from lattice import Lattice
from rbf import RBF


class RBFManager(RBFProject):
    """ """

    def __init__(self, params):
        """ """
        # initializes log, lattice, data, json files
        super().__init__(params)

        self.rbf = RBF(self.data, **self.params['rbf'])

        self.plot = None
        # if self.params['plot']:
        #     self.plot = Plot(self.lat, self.data, **self.params['plots'])
        #     noise_str = 'noisy_' if self.data.add_noise else ''
        #     self.plot.show_data_plot(observations=True,
        #                             gtruth='True', gtruth_color='normal',
        #                             mode='train',
        #                             filename=f'{noise_str}data')


    def train(self):
        """ """
        output = self.forward_data(self.data.train['input'])
        mse, _ = htv_utils.compute_mse_psnr(self.data.train['values'], output)
        print(f'Train mse : {mse}')

        self.evaluate_results()

        if self.data.dataset_name.endswith('planes') or self.data.dataset_name == 'face':
            if self.params['plot']:
                print('\nPlotting.')
                lattice_obj = self.evaluate_lattice()
                plot = Plot(lattice_obj, self.data, **self.params['plots'])
                noise_str = 'noisy_' if self.data.add_noise else ''

                plot.show_lat_plot(gtruth=False,
                                    observations=False,
                                    filename=f'{noise_str}triangulation_model')

        # save params and data to checkpoint
        self.save_to_ckpt()


    def evaluate_lattice(self):
        """ Create and evaluate network on lattice
        """
        lat = Lattice(lsize=self.lat.lsize, h=self.lat.h)
        print(f'Sampled lattice lsize: {lat.lsize}.')

        lattice_grid = lat.lattice_to_standard(lat.lattice_grid.float())
        z = self.forward_data(lattice_grid)

        new_C_mat = lat.flattened_C_to_C_mat(z)
        lat.update_lattice_values(new_C_mat)

        return lat


    def evaluate_results(self):
        """ """
        for mode, data_dict in zip(['valid', 'test'], [self.data.valid, self.data.test]):
            output = self.forward_data(data_dict['input'])
            # save predictions
            data_dict['predictions'] = output

            # compute mse
            mse, _ = htv_utils.compute_mse_psnr(data_dict['values'], output)
            self.update_json('_'.join([mode, 'mse']), mse)
            print(f'{mode} mse  : {mse}')


    def forward_data(self, input_std):
        """ """
        return self.rbf.evaluate(input_std)
