import os
import collections
import sys
import glob
import git
import torch
import numpy as np
from abc import ABC, abstractproperty, abstractmethod

from htvlearn.data import Data
from htvlearn.htv_utils import flatten_structure, dict_recursive_merge
from htvlearn.htv_utils import json_load, json_dump


class MasterProject(ABC):

    # class attribute
    results_json_filename = 'results.json'

    def __init__(self, params, write=True):
        """ """
        self.params = params
        if self.params['verbose']:
            print('\n==> Parameters info: ', self.params, sep='\n')

        self.log_dir_model = os.path.join(self.params["log_dir"],
                                          self.params["model_name"])

        self.init_data()
        self.init_device()
        if write is True:
            self.init_log()
            self.init_json()

    def init_device(self):
        """ """
        if self.params['method'] != 'neural_net' or \
                self.params['device'] == 'cpu':
            self.device = 'cpu'
            print('\nUsing CPU.')

        elif self.params['device'] == 'cuda:0':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                print('\nUsing GPU.')
            else:
                self.device = 'cpu'
                print('\nCUDA not available. Using CPU.')
        else:
            raise ValueError(f'Cannot use device {self.params["device"]}.')

    def init_log(self):
        """ Create Log directory for training the model as :
            self.params["log_dir"]/self.params["model_name"]/
        """
        if not os.path.isdir(self.log_dir_model):
            os.makedirs(self.log_dir_model)

        repo = git.Repo(search_parent_directories=True)
        commit_sha = repo.head.object.hexsha

        cmd_args = ' '.join(sys.argv)

        # save commit sha and cmd line arguments corresponding to this model
        save_str = '\n'.join(['Git commit sha: ', str(commit_sha)])
        save_str += '\n\n'
        save_str += '\n'.join(['Cmd args : ', cmd_args])

        cmd_args_git_commit_filename = os.path.join(self.log_dir_model,
                                                    'cmd_args_git_commit.txt')

        with open(cmd_args_git_commit_filename, 'w') as text_file:
            print(save_str, file=text_file)

    def init_data(self):
        """ Initialize data """
        self.params['data']['log_dir'] = self.log_dir_model
        self.data = Data(**self.params['data'])

    @abstractproperty
    def htv_log(self):
        """ Should return htv log """
        pass

    @property
    def info_list(self):
        """ Should return list of info to log on json file """
        return ['train_mse', 'valid_mse', 'test_mse', 'htv']

    @property
    def sorting_key(self):
        """ Key for sorting models in json file.
        """
        return 'test_mse'

    def init_json(self):
        """ Init json file for train/test results.
        """
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
        """ Update json file with results

        Args:
            info: e.g. 'df_loss'
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
                    update = float('{:.2E}'.format(val))
                results_dict[self.params["model_name"]][info][key] = update
        else:
            update = value
            if isinstance(value, float) or isinstance(value, np.float32):
                update = float('{:.2E}'.format(value))
            results_dict[self.params["model_name"]][info] = update

        # sort in ascending order (reverse=False)
        sorted_acc = sorted(results_dict.items(),
                            key=lambda kv: kv[1][self.sorting_key])
        sorted_results_dict = collections.OrderedDict(sorted_acc)

        json_dump(sorted_results_dict, self.results_json)

    @abstractmethod
    def save_to_ckpt(self, ckpt_filename, save_dict=None):
        """ Save params, data and htv to checkpoint.
        """
        save_base_dict = {
            'params': self.params,
            'htv_log': self.htv_log,
            'exact_htv': self.data.cpwl.get_exact_HTV(),
            'data': {
                'train': self.data.train,
                'valid': self.data.valid,
                'test': self.data.test,
                'delaunay': self.data.delaunay
            },
        }

        if save_dict is None:
            save_dict = save_base_dict
        else:
            save_dict = dict_recursive_merge(save_base_dict, save_dict)

        torch.save(save_dict, ckpt_filename)

    def restore_ckpt_params(self):
        """ Restore the parameters from a previously saved checkpoint
        (either provided via --ckpt_filename or saved in log_dir/model_name)

        Returns:
            True if a checkpoint was successfully loaded and False otherwise.
        """
        ckpt_None = 'ckpt_filename' not in self.params or \
            self.params['ckpt_filename'] is None

        ckpt_filename = None
        if not ckpt_None:
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

    def restore_model_data(self):
        """ """
        # Load checkpoint.
        print('\n==> Restoring from checkpoint...')
        self.data = Data(data_from_ckpt=self.loaded_ckpt['data'],
                         **self.params['data'])

        return

    @staticmethod
    def get_loaded_ckpt(ckpt_filename, map_loc=None):
        """ Returns a loaded checkpoint from ckpt_filename, if it exists.
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
    def load_ckpt_params(cls, ckpt_filename, flatten=True, map_loc=None):
        """ Returns the parameters saved in a checkpoint.

        Args:
            flatten: whether to flatten the structure of the
                     parameters dictionary
            into a single level (see structure in struct_default_values.py)
            map_loc: map_location for torch.load.
        """
        ckpt = cls.get_loaded_ckpt(ckpt_filename, map_loc=map_loc)
        params = ckpt['params']

        if flatten is True:
            params = flatten_structure(params)

        return ckpt, params

    @classmethod
    def get_ckpt_from_log_dir_model(cls, log_dir_model):
        """ Get last ckpt from log_dir_model (log_dir/model_name)
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
        """ Get last ckpt from log_dir_model (log_dir/model_name)
        """
        regexp_ckpt = os.path.join(log_dir_model, '*.pth')

        files = glob.glob(regexp_ckpt)
        files.sort(key=os.path.getmtime)  # sort by time from oldest to newest

        return files

    @classmethod
    def load_results_dict(cls, log_dir):
        """ Load results dictionary from results json file in log_dir.
        """
        results_json = os.path.join(log_dir, cls.results_json_filename)
        results_dict = json_load(results_json)

        return results_dict

    @classmethod
    def dump_results_dict(cls, results_dict, log_dir):
        """ Dump results dictionary in results json file in log_dir.
        """
        results_json = os.path.join(log_dir, cls.results_json_filename)
        json_dump(results_dict, results_json)

    @classmethod
    def get_best_model(cls, log_dir, mode='train'):
        """
        Get the name and checkpoint of the best model (best validation/test)
        from the train/test results (saved in log_dir/[mode]_results.json).
        """
        results_dict = cls.load_results_dict(log_dir, mode)

        # models are ordered by validation accuracy; choose first one.
        best_model_name = next(iter(results_dict))
        log_dir_best_model = os.path.join(log_dir, best_model_name)
        ckpt_filename = cls.get_ckpt_from_log_dir_model(log_dir_best_model)

        return best_model_name, ckpt_filename
