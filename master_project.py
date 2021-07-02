import os
import collections
import sys
import glob
import git
import torch

from lattice import Lattice
from data import Data
from htv_utils import flatten_structure
from htv_utils import json_load, json_dump
import numpy as np

from abc import ABC, abstractproperty, abstractmethod


class MasterProject(ABC):

    # class attribute
    results_json_filename = 'results.json'

    def __init__(self, params):
        """ """
        self.params = params
        if self.params['verbose']:
            print('\n==> Parameters info: ', self.params, sep='\n')

        self.init_log()
        self.init_json()
        self.init_lattice()
        self.init_data()



    def init_log(self):
        """ Create Log directory for training the model as :
            self.params["log_dir"]/self.params["model_name"]/
        """
        self.log_dir_model = os.path.join(self.params["log_dir"],
                                            self.params["model_name"])
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



    def init_lattice(self, **kwargs):
        """ """
        self.lat = Lattice(**self.params['lattice'], **kwargs)



    def init_data(self):
        """ """
        self.params['data']['log_dir'] = self.log_dir_model
        self.data = Data(self.lat, **self.params['data'])



    @abstractproperty
    def sorting_key(self):
        """ Key for sorting models in json file.
        """
        pass



    @abstractproperty
    def info_list(self):
        """ Should return list of info to log on json file """
        pass


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
            results_dict[self.params['model_name']] = {} # initialize model log

        # add minimal information for sorting models in results_json file
        if self.sorting_key not in results_dict[self.params['model_name']]:
            results_dict[self.params['model_name']][self.sorting_key] = 0.

        json_dump(results_dict, self.results_json)



    def update_json(self, info, value):
        """ Update json file with results

        Args:
            info: e.g. 'df_loss'
        """
        assert info in self.info_list, f'{info} should be in {self.info_list}...'

        # save in json
        results_dict = json_load(self.results_json)

        if isinstance(value, dict):
            if info not in results_dict[self.params["model_name"]]:
                results_dict[self.params["model_name"]][info] = {}
            for key, val in value.items():
                update = val
                if isinstance(val, float) or isinstance(val, np.float32):
                    update = float('{:.7f}'.format(val))
                results_dict[self.params["model_name"]][info][key] = update
        else:
            update = value
            if isinstance(value, float) or isinstance(val, np.float32):
                update = float('{:.7f}'.format(value))
            results_dict[self.params["model_name"]][info] = update

        # sort in ascending order (reverse=False)
        sorted_acc = sorted(results_dict.items(), key=lambda kv : kv[1][self.sorting_key])
        sorted_results_dict = collections.OrderedDict(sorted_acc)

        json_dump(sorted_results_dict, self.results_json)



    @abstractmethod
    def save_to_ckpt(self, ckpt_filename, save_dict):
        """ Save params, lattice and data to checkpoint.
        """
        save_base_dict = {
            'params': self.params,
            'data'  : {
                'train' : self.data.train,
                'valid' : self.data.valid,
                'test'  : self.data.test
            }
        }

        save_dict = {**save_dict, **save_base_dict}

        if self.data.add_noise:
            save_dict['data']['snr'] = self.data.snr

        torch.save(save_dict, ckpt_filename)



    @staticmethod
    def get_loaded_ckpt(ckpt_filename):
        """ Returns a loaded checkpoint from ckpt_filename, if it exists.
        """
        try:
            ckpt = torch.load(ckpt_filename) # raises an exception if file does not exist

        except FileNotFoundError:
            print('\nCheckpoint file not found... Unable to load checkpoint.\n')
            raise
        except:
            print('\nUnknown error in loading checkpoint parameters.')
            raise

        return ckpt



    @classmethod
    def load_ckpt_params(cls, ckpt_filename, flatten=True):
        """ Returns the parameters saved in a checkpoint.

        Args:
            flatten - whether to flatten the structure of the parameters dictionary
            into a single level (see structure in struct_default_values.py)
        """
        ckpt = cls.get_loaded_ckpt(ckpt_filename)
        params = ckpt['params']

        if flatten is True:
            params = flatten_structure(params)

        return ckpt, params



    @staticmethod
    def get_ckpt_from_log_dir_model(log_dir_model):
        """ Get last ckpt from log_dir_model (log_dir/model_name)
        """
        regexp_ckpt = os.path.join(log_dir_model, '*.pth')

        files = glob.glob(regexp_ckpt)
        files.sort(key=os.path.getmtime) # sort by time from oldest to newest

        if len(files) > 0:
            ckpt_filename = files[-1]
            print(f'Restoring {ckpt_filename}')
            return ckpt_filename
        else:
            print(f'No ckpt found in {log_dir_model}...')
            return None



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
        """ Get the name and checkpoint of the best model (best validation/test)
        from the train/test results (saved in log_dir/[mode]_results.json).
        """
        results_dict = cls.load_results_dict(log_dir, mode)

        # models are ordered by validation accuracy; choose first one.
        best_model_name = next(iter(results_dict))
        log_dir_best_model = os.path.join(log_dir, best_model_name)
        ckpt_filename = cls.get_ckpt_from_log_dir_model(log_dir_best_model)

        return best_model_name, ckpt_filename
