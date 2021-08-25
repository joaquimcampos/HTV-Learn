import os
import glob
import math
from abc import abstractproperty

from htvlearn.htv_utils import size_str
from htvlearn.master_project import MasterProject
from htvlearn.nn_dataloader import NNDataLoader


class NNProject(MasterProject):
    def __init__(self, params, log=True):
        """
        Args:
            params (dict):
                parameter dictionary.
            log (bool):
                if True, log results.
        """
        super().__init__(params, log=log)

        self.init_dataloader()

    def init_dataloader(self):
        """Initialize train, valid and test dataloaders."""
        print('\n==> Preparing data..')
        self.dataloader = NNDataLoader(self.data, **self.params['dataloader'])
        self.trainloader, self.validloader = \
            self.dataloader.get_train_valid_loader()
        self.testloader = self.dataloader.get_test_loader()

        self.save_train_info()

    def save_train_info(self):
        """ """
        assert (self.trainloader is not None)
        self.num_train_samples = sum(
            inputs.size(0) for inputs, _ in self.trainloader)
        self.num_train_batches = \
            math.ceil(self.num_train_samples / self.dataloader.batch_size)

    def print_train_info(self):
        """ """
        assert (self.validloader is not None)
        assert hasattr(self, 'num_train_samples')
        assert hasattr(self, 'num_train_batches')

        num_valid_samples = sum(
            inputs.size(0) for inputs, _ in self.validloader)
        sample_data, sample_target = self.trainloader[0]

        num_valid_batches = \
            math.ceil(num_valid_samples / self.dataloader.batch_size)

        print('\n==> Train info:')
        print('batch (data, target) size : '
              f'({size_str(sample_data)}, {size_str(sample_target)}).')
        print('no. of (train, valid) samples : '
              f'({self.num_train_samples}, {num_valid_samples}).')
        print('no. of (train, valid) batches : '
              f'({self.num_train_batches}, {num_valid_batches}).')

    def print_test_info(self):
        """ """
        assert (self.testloader is not None)

        num_test_samples = sum(
            inputs.size(0) for inputs, _ in self.testloader)
        sample_data, sample_target = self.testloader[0]

        num_test_batches = math.ceil(num_test_samples /
                                     self.dataloader.batch_size)

        print('\n==> Test info:')
        print('batch (data, target) size : '
              f'({size_str(sample_data)}, {size_str(sample_target)}).')
        print(f'no. of test samples : {num_test_samples}.')
        print(f'no. of test batches : {num_test_batches}.')

    @abstractproperty
    def model_state(self):
        """Returns model state."""
        pass

    @abstractproperty
    def optimizer_state(self):
        """Returns optimizer state."""
        pass

    @abstractproperty
    def train_loss_log(self):
        """Returns train loss log accross epochs."""
        pass

    @abstractproperty
    def valid_loss_log(self):
        """Returns valid loss log accross epochs."""
        pass

    def train_log_step(self, epoch, batch_idx, losses_dict):
        """
        Log the training.

        Args:
            epoch (int):
                current epoch.
            batch_idx (int):
                current batch.
            losses_dict (dict):
                A dictionary of the form {loss name (str) : loss value (float)}
        """
        assert isinstance(losses_dict, dict)

        print('[{:3d}, {:6d} / {:6d}] '
              .format(epoch + 1, batch_idx + 1, self.num_train_batches),
              end='')
        for key, value in losses_dict.items():
            print('{}: {:.3E} | '.format(key, value), end='')

    def valid_log_step(self, losses_dict):
        """
        Log the validation.

        Args:
            epoch (int):
                current epoch.
            losses_dict (dict):
                A dictionary of the form {loss name (str) : loss value (float)}
        """
        assert isinstance(losses_dict, dict)

        print('\nvalidation_step : ', end='')
        for key, value in losses_dict.items():
            print('{}: {:.3E} | '.format(key, value), end='')

        for key, val in losses_dict.items():
            self.update_json(key, val)

    def ckpt_log_step(self, epoch):
        """
        Save the model in a checkpoint.

        Only allow at most params['ckpt_nmax_files'] checkpoints.
        Delete the oldest checkpoint, if necessary.
        Also log the best results so far in a separate checkpoint.

        Args:
            epoch (int):
                current epoch.
        """
        base_ckpt_filename = os.path.join(
            self.log_dir_model,
            self.params['model_name'] + '_net_{:04d}'.format(epoch + 1))
        regexp_ckpt = os.path.join(self.log_dir_model, "*_net_*.pth")

        # save checkpoint as *_net_{epoch+1}.pth
        ckpt_filename = base_ckpt_filename + '.pth'

        files = list(set(glob.glob(regexp_ckpt)))
        # sort from newest to oldest
        files.sort(key=os.path.getmtime, reverse=True)

        if (not self.params['ckpt_nmax_files'] < 0) and \
                (len(files) >= self.params['ckpt_nmax_files']):
            assert len(files) == (self.params['ckpt_nmax_files']), 'There '\
                'are more than (ckpt_nmax_files+1) *_net_*.pth checkpoints.'
            filename = files[-1]
            os.remove(filename)

        self.save_to_ckpt(ckpt_filename)

        return

    def save_to_ckpt(self, ckpt_filename):
        """
        Save model, train/valid loss and other relevant data
        to checkpoint.
        """
        save_dict = {
            'model_state': self.model_state,
            'train_loss_log': self.train_loss_log,
            'valid_loss_log': self.valid_loss_log,
        }

        super().save_to_ckpt(ckpt_filename, save_dict)

    def restore_model_data(self, net):
        """
        Restore model and data, and initialize dataloader.

        Args:
            net (nn.Module):
                model to restore state of.
        """
        self.restore_data()
        net.load_state_dict(self.loaded_ckpt['model_state'], strict=False)
        self.init_dataloader()

        return
