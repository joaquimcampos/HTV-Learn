import os

from master_project import MasterProject
from htv_utils import json_load, json_dump, add_date_to_filename


class HTVProject(MasterProject):

    def __init__(self, params):
        """ """
        super().__init__(params)


    @property
    def info_list(self):
        """ """
        return ['admm', 'simplex', 'df_loss',
                'valid_mse', 'test_mse']


    @property
    def sorting_key(self):
        """ Key for sorting models in json file.
        """
        return 'test_mse'


    def save_to_ckpt(self, lattice_dict):
        """ Save algorithm lattice history and other relevant data to checkpoint.
        """
        ckpt_file = add_date_to_filename(self.params['model_name']) + '.pth'
        ckpt_filename = os.path.join(self.log_dir_model, ckpt_file)

        save_dict = {
            'lattice' : lattice_dict,
        }

        super().save_to_ckpt(ckpt_filename, save_dict)
