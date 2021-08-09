import os

from htvlearn.master_project import MasterProject
from htvlearn.htv_utils import add_date_to_filename


class RBFProject(MasterProject):
    def __init__(self, params, write=True):
        """
        write: weather to write on log directory
        """
        super().__init__(params, write=write)

    def save_to_ckpt(self):
        """
        Save algorithm lattice history and other relevant
        data to checkpoint.
        """
        ckpt_file = add_date_to_filename(self.params['model_name']) + '.pth'
        ckpt_filename = os.path.join(self.log_dir_model, ckpt_file)

        super().save_to_ckpt(ckpt_filename)
