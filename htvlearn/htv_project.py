import os

from htvlearn.master_project import MasterProject
from htvlearn.lattice import Lattice
from htvlearn.htv_utils import add_date_to_filename


class HTVProject(MasterProject):
    def __init__(self, params, write=True):
        """ """
        super().__init__(params, write=write)

        self.init_lattice()

    @property
    def info_list(self):
        """ """
        return super().info_list + ['percentage_nonzero']

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
        try:
            self.lat = \
                Lattice(X_mat=self.loaded_ckpt['lattice']['final']['X_mat'],
                        C_mat=self.loaded_ckpt['lattice']['final']['C_mat'])
        except KeyError:
            print('Could not restore final lattice... Using initial one.')
            self.lat = \
                Lattice(X_mat=self.loaded_ckpt['lattice']['init']['X_mat'],
                        C_mat=self.loaded_ckpt['lattice']['init']['C_mat'])
