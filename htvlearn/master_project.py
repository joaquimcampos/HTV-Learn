import os
import collections
import glob
import torch
import numpy as np
from abc import ABC, abstractproperty, abstractmethod

from htvlearn.data import Data
from htvlearn.htv_utils import flatten_structure, dict_recursive_merge
from htvlearn.htv_utils import json_load, json_dump


class MasterProject(ABC):

    # class attribute
    results_json_filename = 'results.json'

    def __init__(self, params, log=True):
        """
        Args:
            params (dict):
                parameter dictionary.
            log (bool):
                if True, log results.
        """
        self.params = params
        if self.params['verbose']:
            print('\n==> Parameters info: ', self.params, sep='\n')

        self.log_dir_model = os.path.join(self.params["log_dir"],
                                          self.params["model_name"])

        self.init_data()
        self.init_device()
        if log is True:
            self.init_log()
            self.init_json()

    def init_data(self):
        """Initialize data"""
        self.params['data']['log_dir'] = self.log_dir_model
        self.data = Data(**self.params['data'])

    def init_device(self):
        """Initialize running device (cpu or gpu)."""
        if self.params['device'].startswith('cuda'):
            if torch.cuda.is_available():
                self.device = 'cuda:0'  # Using GPU0 by default
                print('\nUsing GPU.')
            else:
                self.device = 'cpu'
                print('\nCUDA not available. Using CPU.')
        else:
            self.device = 'cpu'
            print('\nUsing CPU.')

    def init_log(self):
        """
        Create Log directory for training the model as :
        self.params["log_dir"]/self.params["model_name"]/
        """
        if not os.path.isdir(self.log_dir_model):
            os.makedirs(self.log_dir_model)

    @abstractproperty
    def htv_log(self):
        """ """
        pass

    @property
    def info_list(self):
        """List with info to log."""
        return ['train_mse', 'valid_mse', 'test_mse', 'htv']

    @property
    def sorting_key(self):
        """Key for sorting models in json file."""
        return 'test_mse'

    def init_json(self):
        """Init json file for train/test results."""
        # initialize/verify json log file
        self.results_json = os.path.join(self.params['log_dir'],
                                         self.results_json_filename)

        if not os.path.isfile(self.results_json):
            results_dict = {}
        else:
            results_dict = json_load(self.results_json)

        if self.params['model_name'] not in results_dict:
            # initialize model log
            results_dict[self.params['model_name']] = {}

        # add minimal information for sorting models in results_json file
        if self.sorting_key not in results_dict[self.params['model_name']]:
            results_dict[self.params['model_name']][self.sorting_key] = 0.

        json_dump(results_dict, self.results_json)

    def update_json(self, info, value):
        """
        Update json file with latest/best validation/test accuracy/loss,
        if training, and with test accuracy otherwise.

        Args:
            info (str):
                e.g. 'latest_valid_loss', 'best_train_acc'.
            value (float):
                value for the given info.
        """
        assert info in self.info_list, \
            f'{info} should be in {self.info_list}...'

        # save in json
        results_dict = json_load(self.results_json)

        if isinstance(value, dict):
            if info not in results_dict[self.params["model_name"]]:
                results_dict[self.params["model_name"]][info] = {}
            for key, val in value.items():
                update = val
                if isinstance(val, float) or isinstance(val, np.float32):
                    update = float('{:.3E}'.format(val))
                results_dict[self.params["model_name"]][info][key] = update
        else:
            update = value
            if isinstance(value, float) or isinstance(value, np.float32):
                update = float('{:.3E}'.format(value))
            results_dict[self.params["model_name"]][info] = update

        # sort in ascending order (reverse=False)
        sorted_acc = sorted(results_dict.items(),
                            key=lambda kv: kv[1][self.sorting_key])
        sorted_results_dict = collections.OrderedDict(sorted_acc)

        json_dump(sorted_results_dict, self.results_json)

    @abstractmethod
    def save_to_ckpt(self, ckpt_filename, save_dict=None):
        """
        Save params, data and htv to checkpoint.

        Args:
            ckpt_filename (str):
                name of checkpoint where to log results.
            save_dict (dict):
                dictionary with additional info to log.
        """
        save_base_dict = {
            'params': self.params,
            'htv_log': self.htv_log,
            'exact_htv': self.data.cpwl.get_exact_HTV(),
            'data': {
                'train': self.data.train,
                'valid': self.data.valid,
                'delaunay': self.data.delaunay
            },
        }

        if save_dict is None:
            save_dict = save_base_dict
        else:
            save_dict = dict_recursive_merge(save_base_dict, save_dict)

        torch.save(save_dict, ckpt_filename)

    def restore_ckpt_params(self):
        """
        Attempt to restore a checkpoint if resuming training or testing
        a model.

        If successful, it gets the loaded checkpoint and merges the saved
        parameters.

        Returns True if a checkpoint was successfully loaded,
        and False otherwise.
        """
        if self.params['ckpt_filename'] is not None:
            ckpt_filename = self.params['ckpt_filename']
        elif self.params['restore']:
            ckpt_filename = \
                self.get_ckpt_from_log_dir_model(self.log_dir_model)

        if ckpt_filename is not None:
            self.loaded_ckpt, saved_params = \
                self.load_ckpt_params(
                    ckpt_filename, flatten=False, map_loc=self.device)

            # merge w/ saved params
            self.params = dict_recursive_merge(self.params, saved_params)
            self.params['ckpt_filename'] = ckpt_filename

            print('\nSuccessfully loaded ckpt ' + self.params["ckpt_filename"])
            return True

        else:
            print('\nStarting from scratch.')
            return False

    @property
    def load_ckpt(self):
        """
        Return True if loading a checkpoint and restoring its parameters,
        for resuming training or testing a model. Otherwise, returns False.
        """
        if ('ckpt_filename' in self.params and self.params["ckpt_filename"]
                is not None) or (self.params["restore"] is True):
            return True
        else:
            return False

    @classmethod
    def get_ckpt_from_log_dir_model(cls, log_dir_model):
        """Get last checkpoint from log_dir_model (log_dir/model_name).

        Args:
            log_dir_model (str):
                folder of the form log_dir/model_name
        """
        files = cls.get_all_ckpt_from_log_dir_model(log_dir_model)

        if len(files) > 0:
            ckpt_filename = files[-1]
            print(f'Restoring {ckpt_filename}')
            return ckpt_filename
        else:
            print(f'No ckpt found in {log_dir_model}...')
            return None

    @staticmethod
    def get_all_ckpt_from_log_dir_model(log_dir_model):
        """
        Get all checkpoints from log_dir_model (log_dir/model_name).

        Args:
            log_dir_model (str):
                folder of the form log_dir/model_name
        """
        regexp_ckpt = os.path.join(log_dir_model, '*.pth')

        files = glob.glob(regexp_ckpt)
        files.sort(key=os.path.getmtime)  # sort by time from oldest to newest

        return files

    @classmethod
    def load_ckpt_params(cls, ckpt_filename, flatten=True, map_loc=None):
        """
        Return the ckpt dictionary and the parameters saved
        in a checkpoint file.

        Args:
            ckpt_filename (str):
                Name of checkpoint (.pth) file.
            flatten (bool):
                whether to flatten the structure of the parameters dictionary
                into a single level
                (see structure in struct_default_values.py).
        """
        ckpt = cls.get_loaded_ckpt(ckpt_filename, map_loc=map_loc)
        params = ckpt['params']

        if flatten is True:
            params = flatten_structure(params)

        return ckpt, params

    @staticmethod
    def get_loaded_ckpt(ckpt_filename, map_loc=None):
        """
        Return a loaded checkpoint (ckpt dictionary)
        from ckpt_filename, if it exists.

        Args:
            ckpt_filename (str):
                Name of checkpoint (.pth) file.
            map_loc (str):
                for mapping device location.
        """
        assert map_loc is None or \
            map_loc.startswith('cuda') or map_loc.startswith('cpu')
        try:
            # raises an exception if file does not exist
            ckpt = torch.load(ckpt_filename, map_location=map_loc)

        except FileNotFoundError:
            print(
                '\nCheckpoint file not found... Unable to load checkpoint.\n')
            raise
        except BaseException:
            print('\nUnknown error in loading checkpoint parameters.')
            raise

        return ckpt

    @classmethod
    def load_results_dict(cls, log_dir):
        """
        Load results from the results json file in log_dir.

        Args:
            log_dir (str):
                log directory where results json file is located.

        Returns:
            results_dict (dict): dictionary with logged results.
        """
        results_json = os.path.join(log_dir, cls.results_json_filename)
        results_dict = json_load(results_json)

        return results_dict

    @classmethod
    def dump_results_dict(cls, results_dict, log_dir):
        """
        Dump results dictionary in the results json file in log_dir.

        Args:
            results_dict (dict):
                dictionary with logged results.
            log_dir (str):
                log directory where results json file is located.
        """
        results_json = os.path.join(log_dir, cls.results_json_filename)
        json_dump(results_dict, results_json)

    def restore_data(self):
        """Restore data from loaded checkpoint."""
        # Load checkpoint.
        print('\n==> Restoring from checkpoint...')
        self.data = Data(data_from_ckpt=self.loaded_ckpt['data'],
                         **self.params['data'])

        return
