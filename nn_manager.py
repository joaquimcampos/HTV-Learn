import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import math
import copy

import htv_utils
from models import *
from plots.plot import Plot
from nn_project import NNProject
from lattice import Lattice


##################################################################################################
#### MANAGER

class NNManager(NNProject):
    """ """

    def __init__(self, params):
        """ """
        super().__init__(params)

        loading_success = self.restore_ckpt_params()

        self.net = self.build_model(self.params, self.device)
        self.set_optimization()

        if loading_success is True:
            self.restore_model()

        # During testing, average the loss only at the end to get accurate value
        # of the loss per sample. If we used reduction='mean', when
        # nb_test_samples % batch_size != 0, we could only average the loss per batch
        # (as done in training for printing the losses), but not per sample.
        self.criterion = nn.MSELoss(reduction='mean')
        self.test_criterion = nn.MSELoss(reduction='sum')

        self.criterion.to(self.device)
        self.test_criterion.to(self.device)

        # print(self.net)
        return


    @classmethod
    def build_model(cls, params, device, *args, **kwargs):
        """ Returns network instance """
        if params['verbose']:
            print(f'\nUsing network model: {params["net_model"]}.')

        if params['net_model'] == 'fcnet2d':
            NetworkModule = fcNet2D
        elif params['net_model'] == 'deepfcnet2d':
            NetworkModule = DeepfcNet2D
        else:
            raise ValueError(f'Model {params["net_model"]} not available.')

        return super(NNManager, cls).build_model(NetworkModule, params, device, *args, **kwargs)



    def set_optimization(self):
        """ """
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001,
                                    weight_decay=self.params['weight_decay'])

        print('\nOptimizer :', self.optimizer, sep='\n')

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        self.params['milestones'])


    @property
    def model_state(self):
        """ """
        return self.net.state_dict()

    @property
    def optimizer_state(self):
        """ """
        return self.optimizer.state_dict()


##################################################################################################
#### TRAIN

    def train(self):
        """ """
        self.net.train()

        # log training at every epoch and validate in the middle and at the end of training.
        if self.params['log_step'] is None:
            # log at every epoch
            self.params['log_step'] = self.num_batches['train']
            if self.params['valid_log_step'] is None:
                # validation done halfway and at the end of training
                self.params['valid_log_step'] = \
                    self.num_batches['train'] * int(self.params['num_epochs'] / 2)

        if self.params['plot']:
            self.plot = Plot(Lattice(), self.data, **self.params['plots'])
            noise_str = 'noisy_' if self.data.add_noise else ''
            self.plot.show_data_plot(observations=True, mode='train',
                                    filename=f'{noise_str}planes_data')


        print('\n\nStarting training...')
        self.global_step = 0

        for epoch in range(0, self.params['num_epochs']):

            self.train_epoch(epoch)
            # shuffle training data
            self.trainloader = self.dataloader.get_shuffled_trainloader_in_memory()

        print('\nFinished training.')

        # test model
        self.test()



    def train_epoch(self, epoch):
        """ """
        print(f'\nEpoch: {epoch}\n')

        running_loss = 0.

        for batch_idx, (inputs, labels) in enumerate(self.trainloader):

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if batch_idx % self.params['log_step'] == (self.params['log_step'] - 1):
                mse = (running_loss / self.params['log_step'])
                losses_dict = {'mse': mse}
                self.train_log_step(epoch, batch_idx, losses_dict)
                running_loss = 0. # reset running loss

            if self.global_step % self.params['valid_log_step'] == (self.params['valid_log_step'] - 1):
                self.validation_step(epoch)
                self.net.train()

            self.global_step += 1


        if self.scheduler is not None:
            self.scheduler.step()
            if self.params['verbose']:
                lr = [group['lr'] for group in self.optimizer.param_groups]
                print(f'scheduler: epoch - {self.scheduler.last_epoch}; '
                    f'learning rate - {lr}')



    def validation_step(self, epoch):
        """ """
        loss = self.evaluate_results(mode='valid')

        print(f'\nvalidation mse : {loss}')
        losses_dict = {'mse'  : loss}

        self.valid_log_step(losses_dict)
        self.ckpt_log_step(epoch) # save checkpoint



    def test(self):
        """ """
        loss = self.evaluate_results(mode='test')

        print(f'\ntest mse : {loss}')
        self.update_json('test_mse', loss)

        if self.data.dataset_name.endswith('planes'):
            if self.params['plot']:
                plot = Plot(Lattice(), self.data, **self.params['plots'])
                noise_str = 'noisy_' if self.data.add_noise else ''
                plot.show_data_plot(predictions=True,
                                    filename=f'{noise_str}planes_model')


        # save test prediction to last checkpoint
        self.ckpt_log_step(self.params['num_epochs']-1)

        print('\nFinished testing.')



    def evaluate_results(self, mode='valid'):
        """ """
        assert mode in ['valid', 'test']

        if mode == 'valid':
            dataloader = self.validloader
            data_dict = self.data.valid
        else:
            dataloader = self.testloader
            data_dict = self.data.test

        self.net.eval()
        running_loss = 0.
        total = 0
        predictions = torch.tensor([])

        with torch.no_grad():

            for batch_idx, (inputs, labels) in enumerate(dataloader):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                predictions = torch.cat((predictions, outputs.cpu()), dim=0)

                loss = self.test_criterion(outputs, labels)
                running_loss += loss.item()
                total += labels.size(0)


        data_dict['predictions'] = predictions

        loss = running_loss / total
        # sanity check
        mse, _ = htv_utils.compute_mse_psnr(data_dict['values'], predictions)
        assert np.allclose(mse, loss), '(mse: {:.7f}, loss: {:.7f})'.format(mse, loss)

        return loss


    def evaluate_lattice(self):
        """ Create and evaluate network on lattice
        """
        lat = Lattice(lsize=self.lat.lsize, h=self.lat.h)
        print(f'Sampled lattice lsize: {lat.lsize}')

        lattice_grid = lat.lattice_to_standard(lat.lattice_grid.float())
        z = self.forward_data(lattice_grid)

        new_C_mat = lat.flattened_C_to_C_mat(z)
        lat.update_lattice_values(new_C_mat)

        return lat



    def forward_data(self, inputs, batch_size=64):
        """ """
        self.net.eval()
        predictions = torch.tensor([])

        with torch.no_grad():

            for batch_idx, inputs in enumerate(inputs.split(batch_size)):

                inputs = inputs.to(self.device)
                outputs = self.net(inputs)
                predictions = torch.cat((predictions, outputs.cpu()), dim=0)

        return predictions
