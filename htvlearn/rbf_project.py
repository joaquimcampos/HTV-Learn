import os

from htvlearn.master_project import MasterProject
from htvlearn.htv_utils import add_date_to_filename


class RBFProject(MasterProject):
    def __init__(self, params, log=True):
        """
        Args:
            params (dict):
                parameter dictionary.
            log (bool):
                if True, log results.
        """
        super().__init__(params, log=log)

    def save_to_ckpt(self):
        """Save relevant data to checkpoint."""
        ckpt_file = add_date_to_filename(self.params['model_name']) + '.pth'
        ckpt_filename = os.path.join(self.log_dir_model, ckpt_file)

        super().save_to_ckpt(ckpt_filename)
