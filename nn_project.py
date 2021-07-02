import torch
import os
import glob
import math

from htv_utils import size_str, dict_recursive_merge
from master_project import MasterProject
from nn_dataloader import NNDataLoader
from lattice import Lattice
from data import Data

from abc import abstractproperty


class NNProject(MasterProject):

    def __init__(self, params):
        """ """
        super().__init__(params)

        self.init_device()
        self.init_dataloader()


    def init_device(self):
        """ """
        if self.params['device'] == 'cuda:0':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                print('\nUsing GPU.')
            else:
                self.device = 'cpu'
                print('\nCUDA not available. Using CPU.')
        else:
            self.device = 'cpu'
            print('\nUsing CPU.')



    def init_dataloader(self):
        """ """
        print('\n==> Preparing data..')
        self.dataloader = NNDataLoader(self.data, **self.params['dataloader'])
        self.trainloader, self.validloader = self.dataloader.get_train_valid_loader()
        self.testloader = self.dataloader.get_test_loader()

        # saves num_samples, num_batches for train, valid, test
        self.save_train_test_info()



    def save_train_test_info(self):
        """ """
        if self.params['verbose']:
            sample_data, sample_target = self.trainloader[0]
            print(f'batch (data, target) size : ({size_str(sample_data)}, {size_str(sample_target)})')

        self.num_samples, self.num_batches = {}, {}
        for mode, loader in zip(['train', 'valid', 'test'],
                            [self.trainloader, self.validloader, self.testloader]):

            self.num_samples[mode] = sum(inputs.size(0) for inputs,_ in loader)
            self.num_batches[mode] = math.ceil(self.num_samples[mode] / self.dataloader.batch_size)

            if self.params['verbose']:
                print(f'no.  of {mode} samples : {self.num_samples[mode]}')
                print(f'\nNumber of {mode} batches per epoch : {self.num_batches[mode]}')



    @staticmethod
    def build_model(NetworkModule, params, device='cuda:0'):
        """ """
        print('\n==> Building model..')

        net = NetworkModule(**params['model'], device=device)

        net = net.to(device)
        if device == 'cuda':
            cudnn.benchmark = True

        print('\n[Network] Total number of parameters : {}'.format(net.num_params))

        return net



    @abstractproperty
    def model_state(self):
        """ Should return model state """
        pass

    @abstractproperty
    def optimizer_state(self):
        """ Should return optimizer state """
        pass


    @property
    def info_list(self):
        """ """
        return ['latest_train_mse', 'latest_valid_mse', 'test_mse']



    @property
    def sorting_key(self):
        """ Key for sorting models in json file.
        """
        return 'test_mse'



    def train_log_step(self, epoch, batch_idx, losses_dict):
        """
        Args:
            losses_dict - a dictionary {loss name : loss value}
        """
        assert isinstance(losses_dict, dict)

        print('[{:3d}, {:6d} / {:6d}] '.format(epoch + 1, batch_idx + 1, self.num_batches['train']), end='')
        for key, value in losses_dict.items():
            print('{}: {:7.7f} | '.format(key, value), end='')

        for key, val in losses_dict.items():
            info = '_'.join(['latest_train', key])
            self.update_json(info, val)



    def valid_log_step(self, losses_dict):
        """
        Args:
            losses_dict - a dictionary {loss name : loss value}
        """
        assert isinstance(losses_dict, dict)

        print('\nvalidation_step : ', end='')
        for key, value in losses_dict.items():
            print('{}: {:7.3f} | '.format(key, value), end='')

        for key, val in losses_dict.items():
            info = '_'.join(['latest_valid', key])
            self.update_json(info, val)



    def ckpt_log_step(self, epoch):
        """ """
        base_ckpt_filename = os.path.join(self.log_dir_model,
                                        self.params['model_name'] + '_net_{:04d}'.format(epoch+1))
        regexp_ckpt = os.path.join(self.log_dir_model, "*_net_*.pth")

        # save checkpoint as *_net_{epoch+1}.pth
        ckpt_filename = base_ckpt_filename + '.pth'

        files = list(set(glob.glob(regexp_ckpt)))
        files.sort(key=os.path.getmtime, reverse=True) # sort from newest to oldest

        if (not self.params['ckpt_nmax_files'] < 0) and (len(files) >= self.params['ckpt_nmax_files']):
            assert len(files) == (self.params['ckpt_nmax_files']), 'There are more than (ckpt_nmax_files+1) *_net_*.pth checkpoints.'
            filename = files[-1]
            os.remove(filename)

        self.save_to_ckpt(ckpt_filename)

        return



    def save_to_ckpt(self, ckpt_filename):
        """ Save model, optimizer, and other relevant data to checkpoint.
        """
        save_dict = {
            'model_state'         : self.model_state,
        }

        super().save_to_ckpt(ckpt_filename, save_dict)



    def restore_ckpt_params(self):
        """ Restore the parameters from a previously saved checkpoint
        (either provided via --ckpt_filename or saved in log_dir/model_name)

        Returns:
            True if a checkpoint was successfully loaded and False otherwise.
        """
        ckpt_None = 'ckpt_filename' not in self.params or self.params['ckpt_filename'] is None

        ckpt_filename = None
        if not ckpt_None:
            ckpt_filename = self.params['ckpt_filename']
        elif self.params['restore']:
            ckpt_filename = self.get_ckpt_from_log_dir_model(self.log_dir_model)

        if ckpt_filename is not None:
            self.loaded_ckpt, saved_params = \
                    self.load_ckpt_params(ckpt_filename, flatten=False)

            self.params = dict_recursive_merge(self.params, saved_params) # merge w/ saved params
            self.params['ckpt_filename'] = ckpt_filename

            print('\nSuccessfully loaded ckpt ' + self.params["ckpt_filename"])
            return True

        else:
            print('\nStarting from scratch.')
            return False



    def restore_model(self):
        """ """
        # Load checkpoint.
        print('\n==> Restoring from checkpoint...')

        ckpt = self.loaded_ckpt
        self.net.load_state_dict(ckpt['model_state'], strict=False)

        return
