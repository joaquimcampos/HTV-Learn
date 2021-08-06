import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from functools import partial
import torch.autograd.functional as AF

import htv_utils
import hessian
from models import GELUfcNet2D, ReLUfcNet2D
from nn_project import NNProject

#########################################################################
# MANAGER


class NNManager(NNProject):
    """ """
    def __init__(self, params, write=True):
        """
        write: weather to write on log directory
        """
        super().__init__(params, write=write)

        loading_success = self.restore_ckpt_params()

        self.net = self.build_model(self.params, self.device)
        self.net.double()
        self.net.dtype = next(self.net.parameters()).dtype

        self.set_optimization()

        if loading_success is True:
            self.restore_model_data()

        # During testing, average the loss only at the end to get accurate
        # value of the loss per sample. If we used reduction='mean', when
        # nb_test_samples % batch_size != 0, we could only average the loss
        # per batch (as done in training for printing the losses), but not
        # per sample.
        self.criterion = nn.MSELoss(reduction='mean')
        self.test_criterion = nn.MSELoss(reduction='sum')

        self.criterion.to(self.device)
        self.test_criterion.to(self.device)

        print(self.net)
        return

    @classmethod
    def build_model(cls, params, device, *args, **kwargs):
        """ Returns network instance """
        if params['verbose']:
            print(f'\nUsing network model: {params["net_model"]}.')

        if params['net_model'] == 'relufcnet2d':
            NetworkModule = ReLUfcNet2D
        elif params['net_model'] == 'gelufcnet2d':
            NetworkModule = GELUfcNet2D
        else:
            raise ValueError(f'Model {params["net_model"]} not available.')

        return super(NNManager, cls).build_model(NetworkModule, params, device,
                                                 *args, **kwargs)

    def set_optimization(self):
        """ """
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=0.001,
                                    weight_decay=self.params['weight_decay'])

        print('\nOptimizer :', self.optimizer, sep='\n')

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.params['milestones'])

    @property
    def model_state(self):
        """ """
        return self.net.state_dict()

    @property
    def optimizer_state(self):
        """ """
        return self.optimizer.state_dict()

    @property
    def htv_log(self):
        return self.htv_dict

    @property
    def train_loss_log(self):
        return self.train_loss_dict

    @property
    def valid_loss_log(self):
        return self.valid_loss_dict

    #########################################################################
    # TRAIN

    def train(self):
        """ Training loop """
        self.net.train()

        if self.params['log_step'] is None:  # default
            # log at every epoch
            self.params['log_step'] = self.num_batches['train']

        if self.params['valid_log_step'] is None:  # default
            # validation done halfway and at the end of training
            self.params['valid_log_step'] = \
                int(self.num_batches['train'] *
                    self.params['num_epochs'] * 1. / 2.)

        elif self.params['valid_log_step'] < 0:
            # validation at every epoch
            self.params['valid_log_step'] = self.num_batches['train']

        print('\n\nStarting training...')
        self.global_step = 0
        ###
        self.htv_dict = {}
        self.train_loss_dict = {}
        self.valid_loss_dict = {}
        ###

        # signaling that training was not performed
        self.latest_train_loss = -1
        self.validation_step(-1)
        self.net.train()

        for epoch in range(0, self.params['num_epochs']):

            self.train_epoch(epoch)
            # shuffle training data
            self.trainloader = \
                self.dataloader.get_shuffled_trainloader_in_memory()

        print('\nFinished training.')

        # test model
        self.test()

    def train_epoch(self, epoch):
        """ """
        print(f'\nEpoch: {epoch}\n')

        running_loss = 0.

        for batch_idx, (inputs, labels) in enumerate(self.trainloader):

            inputs = inputs.to(device=self.device, dtype=self.net.dtype)
            labels = labels.to(device=self.device, dtype=self.net.dtype)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if batch_idx % self.params['log_step'] == (
                    self.params['log_step'] - 1):
                mse = (running_loss / self.params['log_step'])
                self.latest_train_loss = mse
                losses_dict = {'mse': mse}
                self.train_log_step(epoch, batch_idx, losses_dict)
                running_loss = 0.  # reset running loss

            if self.global_step % self.params['valid_log_step'] == (
                    self.params['valid_log_step'] - 1):
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

        #####
        if not self.params['no_htv']:
            print('\nComputing Hessian...')
            self.htv_dict[str(epoch + 1)] = self.compute_network_htv()
            print('HTV dict :', self.htv_dict[str(epoch + 1)])
            print('Exact HTV :', self.data.cpwl.get_exact_HTV())
            print('Finished.')
        #####
        self.train_loss_dict[str(epoch + 1)] = self.latest_train_loss
        self.valid_loss_dict[str(epoch + 1)] = loss

        print(f'\nvalidation mse : {loss}')
        losses_dict = {'mse': loss}

        self.valid_log_step(losses_dict)
        self.ckpt_log_step(epoch)  # save checkpoint

    def evaluate_func(self, x, batch_size=2000000):
        """ """
        x = torch.from_numpy(x).to(self.device)
        assert x.dtype == torch.float64
        dataloader = x.split(batch_size)
        if self.params['verbose']:
            print('Length dataloader: ', len(dataloader))
        y = torch.tensor([], device=x.device, dtype=x.dtype)

        for batch_idx, input in enumerate(dataloader):
            out = self.net(input)
            y = torch.cat((y, out), dim=0)

        return y.detach().cpu().numpy()

    def differentiate_func(self, mode, x):
        """ """
        assert mode in ['jacobian', 'hessian']
        inputs = tuple(torch.from_numpy(x).to(self.device))
        autograd_func = AF.jacobian if mode == 'jacobian' else AF.hessian

        g = partial(autograd_func, lambda x: self.net(x.unsqueeze(0)))
        x_diff = torch.stack(tuple(map(g, inputs))).detach().cpu().numpy()

        return x_diff

    @staticmethod
    def get_htv_from_Hess(Hess, h, cpwl=False):
        """
        Args:
            h: grid size
            cpwl: True if network is CPWL.
        """
        if cpwl is False:
            # schatten-p-norm -> sum
            S = np.linalg.svd(Hess, compute_uv=False, hermitian=True)
            assert S.shape == (*Hess.shape[0:2], 2)

            htv = {}
            for p in [1, 2, 5, 10]:
                points_htv = np.linalg.norm(S, ord=p, axis=-1)  # schatten norm
                # value x area (riemman integral)
                htv[str(p)] = (points_htv * h * h).sum()

        else:
            points_htv = np.abs(Hess[:, :, 0, 0] + Hess[:, :, 1, 1])
            htv = (points_htv * h * h).sum()

        return htv

    def compute_network_htv(self):
        """ """
        cpwl = True if 'relu' in self.params['net_model'] else False
        htv = {}

        if self.params['htv_mode'] == 'exact_differential':
            grid = self.data.cpwl.get_grid(h=0.01)
            if self.params['net_model'] == 'relufcnet2d':
                Hess = hessian.get_exact_grad_Hessian(
                    grid, partial(self.differentiate_func, 'jacobian'))
            else:
                Hess = hessian.get_exact_Hessian(
                    grid, partial(self.differentiate_func, 'hessian'))

            htv['exact_differential'] = self.get_htv_from_Hess(Hess,
                                                               grid.h,
                                                               cpwl=cpwl)

        elif self.params['htv_mode'] == 'finite_diff_differential':
            grid = self.data.cpwl.get_grid(h=0.0002)
            with torch.no_grad():
                Hess = hessian.get_finite_second_diff_Hessian(
                    grid, self.evaluate_func)

            htv['finite_diff_differential'] = self.get_htv_from_Hess(Hess,
                                                                     grid.h,
                                                                     cpwl=cpwl)

        return htv

    def test(self):
        """ """
        loss = self.evaluate_results(mode='test')

        print(f'\ntest mse : {loss}')
        self.update_json('test_mse', loss)

        # save test prediction to last checkpoint
        self.ckpt_log_step(self.params['num_epochs'] - 1)

        print('\nFinished testing.')

    def evaluate_results(self, mode='valid'):
        """ """
        assert mode in ['train', 'valid', 'test']

        if mode == 'train':
            dataloader = self.trainloader
            data_dict = self.data.train
        elif mode == 'valid':
            dataloader = self.validloader
            data_dict = self.data.valid
        else:
            dataloader = self.testloader
            data_dict = self.data.test

        self.net.eval()
        running_loss = 0.
        total = 0
        predictions = torch.tensor([]).to(device=self.device)
        values = torch.tensor([]).to(device=self.device)

        with torch.no_grad():

            for batch_idx, (inputs, labels) in enumerate(dataloader):

                inputs = inputs.to(device=self.device, dtype=self.net.dtype)
                labels = labels.to(device=self.device, dtype=self.net.dtype)
                outputs = self.net(inputs)
                predictions = torch.cat((predictions, outputs), dim=0)
                values = torch.cat((values, labels), dim=0)

                loss = self.test_criterion(outputs, labels)
                running_loss += loss.item()
                total += labels.size(0)

        data_dict['predictions'] = predictions

        loss = running_loss / total
        # sanity check
        mse, _ = htv_utils.compute_mse_psnr(values.cpu(), predictions.cpu())
        assert np.allclose(mse, loss), \
            '(mse: {:.7f}, loss: {:.7f})'.format(mse, loss)

        return loss

    def forward_data(self, inputs, batch_size=64):
        """ """
        self.net.eval()
        predictions = torch.tensor([])

        with torch.no_grad():

            for batch_idx, inputs in enumerate(inputs.split(batch_size)):

                inputs = inputs.to(device=self.device, dtype=self.net.dtype)
                outputs = self.net(inputs)
                predictions = torch.cat((predictions, outputs.cpu()), dim=0)

        return predictions

    @staticmethod
    def read_htv_log(htv_log):
        """
        Read htv_log dictionary

        Returns:
            epochs: array with saved epochs. size: (E,).
            htv: dict/array with saved htv accross training:
                if non-cpwl: dict('p': array of size = #epochs),
                if cpwl: array of size = #epochs.
        """
        assert isinstance(htv_log, dict)
        # keys of htv_log are epochs
        epochs = np.array([int(epoch) for epoch in htv_log.keys()])
        idx = np.argsort(epochs)
        epochs = epochs[idx]

        # e.g. epoch 1
        # non-cpwl network:
        # htv_log['1'] = {'exact_differential': {'1': htv_1, '2': htv_2}}
        # relu network: epoch 1: htv_log['1'] = {'exact_differential': htv}

        # dictio is list of dictionaries containing 'p': htv_p for each epoch
        # or a single value in the case of CPWL networks
        dictio = [val[list(val)[0]] for val in htv_log.values()]

        if isinstance(dictio[0], dict):
            # p in schatten-p norms
            p_array = np.array([int(p) for p in dictio[0].keys()])
            p_array.sort()

            # e.g.:
            # {'1': np.array of size (epochs,),
            #  '2': np.array of size (epochs,)}
            htv = {}
            for p in p_array:
                htv[str(p)] = np.array(
                    [float(dictio[j][str(p)]) for j in range(0, len(dictio))])
                htv[str(p)] = htv[str(p)][idx]  # sort array according to epoch
        else:
            htv = np.array([float(dictio[j]) for j in range(0, len(dictio))])
            htv = htv[idx]

        return epochs, htv

    @staticmethod
    def read_loss_log(loss_log):
        """
        Read loss_log (train or valid) dictionary

        Returns:
            epochs: array with saved epochs. size: (E,).
            loss: array with saved loss across training. size: (E,)
        """
        assert isinstance(loss_log, dict)
        # keys of loss_log are epochs
        epochs = np.array([int(epoch) for epoch in loss_log.keys()])
        idx = np.argsort(epochs)
        epochs = epochs[idx]

        # e.g. loss_log = {'1': loss_epoch_1, ..., '300': loss_epoch_300}
        loss = np.array([float(val) for val in loss_log.values()])
        loss = loss[idx]  # sort

        return epochs, loss
