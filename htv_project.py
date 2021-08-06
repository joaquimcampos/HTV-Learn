import os

from master_project import MasterProject
from lattice import Lattice
from htv_utils import add_date_to_filename


class HTVProject(MasterProject):
    def __init__(self, params, write=True):
        """ """
        super().__init__(params, write=write)

        self.init_lattice()

    @property
    def info_list(self):
        """ """
        return [
            'train_mse', 'valid_mse', 'test_mse', 'htv', 'percentage_nonzero'
        ]

    @property
    def sorting_key(self):
        """ Key for sorting models in json file.
        """
        return 'test_mse'

    def save_to_ckpt(self, lattice_dict):
        """
        Save algorithm lattice history and other relevant
        data to checkpoint.
        """
        ckpt_file = add_date_to_filename(self.params['model_name']) + '.pth'
        ckpt_filename = os.path.join(self.log_dir_model, ckpt_file)

        save_dict = {
            'lattice': lattice_dict,
        }

        super().save_to_ckpt(ckpt_filename, save_dict)

    def init_lattice(self):
        """ """
        self.lat = Lattice(**self.params['lattice'])

    def restore_model_data(self):
        """ """
        super().restore_model_data()
        self.lat = Lattice(X_mat=self.loaded_ckpt['lattice']['init']['X_mat'],
                           C_mat=self.loaded_ckpt['lattice']['init']['C_mat'])
