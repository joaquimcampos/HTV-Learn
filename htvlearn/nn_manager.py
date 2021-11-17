import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from functools import partial
import torch.autograd.functional as AF

from htvlearn.htv_utils import compute_mse_snr
from htvlearn.networks import (
    ReLUfcNet2D,
    LeakyReLUfcNet2D,
    GELUfcNet2D
)
from htvlearn.nn_project import NNProject
from htvlearn.hessian import (
    get_exact_grad_Hessian,
    get_exact_Hessian,
    get_finite_second_diff_Hessian
)
from htvlearn.lattice import Lattice

#########################################################################
# MANAGER


class NNManager(NNProject):
    """ """
    def __init__(self, params, log=True):
        """
        Args:
            params (dict):
                parameter dictionary.
            log (bool):
                if True, log results.
        """
        super().__init__(params, log=log)

        is_ckpt_loaded = False
        if self.load_ckpt is True:
            # is_ckpt_loaded=True if a checkpoint was successfully loaded.
            is_ckpt_loaded = self.restore_ckpt_params()

        self.net = self.build_model(self.params, self.device)
        self.net.double()
        self.net.dtype = next(self.net.parameters()).dtype

        self.optimizer, self.scheduler = self.set_optimization()

        if is_ckpt_loaded is True:
            self.restore_model_data(self.net)

        # During testing, average the loss only at the end to get accurate
        # value of the loss per sample. If using reduction='mean', when
        # nb_test_samples % batch_size != 0 we can only average the loss per
        # batch (as in training for printing the losses) but not per sample.
        self.criterion = nn.MSELoss(reduction='mean')
        self.test_criterion = nn.MSELoss(reduction='sum')

        self.criterion.to(self.device)
        self.test_criterion.to(self.device)

        # # uncomment for printing network architecture
        # print(self.net)

    @classmethod
    def build_model(cls, params, device, *args, **kwargs):
        """
        Build the network model.

        Args:
            params (dict):
                contains the network name and the model parameters.
            device (str):
                'cpu' or 'cuda:0'.
        Returns:
            net (nn.Module)
        """
        print('\n==> Building model..')

        if params['net_model'] == 'relufcnet2d':
            NetworkModule = ReLUfcNet2D
        elif params['net_model'] == 'leakyrelufcnet2d':
            NetworkModule = LeakyReLUfcNet2D
        elif params['net_model'] == 'gelufcnet2d':
            NetworkModule = GELUfcNet2D
        else:
            raise ValueError(f'Model {params["net_model"]} not available.')

        net = NetworkModule(**params['model'], device=device)

        net = net.to(device)
        if device == 'cuda':
            cudnn.benchmark = True

        print(f'[Network] Total number of parameters : {net.num_params}.')

        return net

    def set_optimization(self):
        """Initialize optimizer and scheduler."""
        optimizer = optim.Adam(self.net.parameters(),
                               lr=0.001,
                               weight_decay=self.params['weight_decay'])

        print('\nOptimizer :', optimizer, sep='\n')

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   self.params['milestones'])

        return optimizer, scheduler

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
        """ """
        return self.htv_dict

    @property
    def train_loss_log(self):
        """ """
        return self.train_loss_dict

    @property
    def valid_loss_log(self):
        """ """
        return self.valid_loss_dict

    #########################################################################
    # TRAIN

    def train(self):
        """Training loop."""
        self.net.train()  # set the network in training mode

        if self.params['verbose']:
            self.print_train_info()

        if self.params['log_step'] is None:  # default
            # log at every epoch
            self.params['log_step'] = self.num_train_batches

        if self.params['valid_log_step'] is None:  # default
            # validation done halfway and at the end of training
            self.params['valid_log_step'] = int(
                self.num_train_batches * self.params['num_epochs'] * 1. / 2.)

        elif self.params['valid_log_step'] < 0:
            # validation at every epoch
            self.params['valid_log_step'] = self.num_train_batches

        self.global_step = 0
        ###
        # initialize log dictionaries
        self.htv_dict = {}
        self.train_loss_dict = {}
        self.valid_loss_dict = {}
        ###

        # get initial model performance
        # -1 signals that training was not performed
        self.latest_train_loss = -1
        self.validation_step(-1)
        self.net.train()

        print('\n==> Starting training...')

        for epoch in range(0, self.params['num_epochs']):

            self.train_epoch(epoch)
            # shuffle training data
            self.trainloader = \
                self.dataloader.get_shuffled_trainloader_in_memory()

        print('\nFinished training.')

        # test model
        self.test()

    def train_epoch(self, epoch):
        """
        Training for one epoch.

        Args:
            epoch (int).
        """
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
                # training log step
                mse = (running_loss / self.params['log_step'])
                self.latest_train_loss = mse
                losses_dict = {'avg. mse': mse}
                self.train_log_step(epoch, batch_idx, losses_dict)
                running_loss = 0.  # reset running loss

            if self.global_step % self.params['valid_log_step'] == (
                    self.params['valid_log_step'] - 1):
                # validation log step
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
        """
        Perform one validation step. Saves results on checkpoint.

        Args:
            epoch (int).
        """
        train_loss, _ = self.evaluate_results(mode='train')
        valid_loss, _ = self.evaluate_results(mode='valid')

        #####
        if not self.params['no_htv']:
            print('\n==> Computing Hessian...')
            self.htv_dict[str(epoch + 1)] = self.compute_network_htv()
            print('HTV dict :', self.htv_dict[str(epoch + 1)])
            print('Exact HTV :', self.data.cpwl.get_exact_HTV())
            print('Finished.')
            self.update_json('htv', self.htv_dict[str(epoch + 1)])
        #####
        self.train_loss_dict[str(epoch + 1)] = train_loss
        self.valid_loss_dict[str(epoch + 1)] = valid_loss

        print(f'\nvalidation mse : {valid_loss}')

        losses_dict = {'train_mse': train_loss,
                       'valid_mse': valid_loss}
        self.valid_log_step(losses_dict)
        self.ckpt_log_step(epoch)  # save checkpoint

    def compute_network_htv(self):
        """
        Compute the network's HTV using the 'finite_diff_differential' or
        'exact_differential' mode.

        Returns:
            htv (float).
        """
        cpwl = True if 'relu' in self.params['net_model'] else False
        htv = {}

        if self.params['htv_mode'] == 'finite_diff_differential':
            # use finite second differences to compute the hessian
            grid = self.data.cpwl.get_grid(h=0.0002)
            with torch.no_grad():
                Hess = get_finite_second_diff_Hessian(
                    grid, self.evaluate_func)

            htv = self.get_htv_from_Hess(Hess, grid.h, cpwl=cpwl)

        elif self.params['htv_mode'] == 'exact_differential':
            warnings.warn('"exact_differential" mode is computationally '
                          'expensive and does not lead to to precise '
                          'computations, in general. Prefer setting '
                          '"htv_mode" to "finite_diff_differential".')
            grid = self.data.cpwl.get_grid(h=0.01)
            if self.params['net_model'].endswith('relufcnet2d'):
                # cpwl function -> exact gradient + finite first differences
                #  to compute hessian
                Hess = get_exact_grad_Hessian(
                    grid, partial(self.differentiate_func, 'jacobian'))
            else:
                # cpwl function -> exact hessian + sum over the grid locations
                # to compute hessian
                Hess = get_exact_Hessian(
                    grid, partial(self.differentiate_func, 'hessian'))

            htv = self.get_htv_from_Hess(Hess, grid.h, cpwl=cpwl)

        return htv

    def evaluate_func(self, x, batch_size=2000000):
        """
        Evaluate model function for some input.

        Args:
            x (np.ndarray):
                inputs. size: (n, 2).
            batch_size (int):
                batch size for evaluation.

        Returns:
            y (np.ndarray):
                result of evaluating model at x.
        """
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
        """
        Evaluate model Jacobian/Hessian at some input.

        Args:
            mode (str):
                "jacobian" or "hessian"
            x (np.ndarray):
                inputs. size: (n, 2).

        Returns:
            x_diff (np.ndarray):
                result of evaluating model ``mode`` at x.
        """
        assert mode in ['jacobian', 'hessian']
        inputs = tuple(torch.from_numpy(x).to(self.device))
        autograd_func = AF.jacobian if mode == 'jacobian' else AF.hessian

        g = partial(autograd_func, lambda x: self.net(x.unsqueeze(0)))
        x_diff = torch.stack(tuple(map(g, inputs))).detach().cpu().numpy()

        return x_diff

    @staticmethod
    def get_htv_from_Hess(Hess, h, cpwl=False):
        """
        Get the HTV from the hessian at grid locations.

        Args:
            h (float):
                grid size.
            cpwl (bool):
                True if network is CPWL.

        Returns:
            htv (float).
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

    def test(self):
        """Test model."""
        if self.params['verbose']:
            self.print_test_info()

        test_loss, _ = self.evaluate_results(mode='test')

        print(f'\ntest mse : {test_loss}')
        self.update_json('test_mse', test_loss)

        # save test prediction to last checkpoint
        self.ckpt_log_step(self.params['num_epochs'] - 1)

        print('\nFinished testing.')

    def evaluate_results(self, mode):
        """
        Evaluate train, validation or test results.

        Args:
            mode (str):
                'train', 'valid' or 'test'

        Returns:
            mse (float):
                ``mode`` mean-squared-error.
            output (torch.Tensor):
                result of evaluating model on ``mode`` set.
        """
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
        output = torch.tensor([]).to(device=self.device)
        values = torch.tensor([]).to(device=self.device)

        with torch.no_grad():

            # notation: _b = 'batch'
            for batch_idx, (inputs_b, labels_b) in enumerate(dataloader):
                inputs_b = inputs_b.to(device=self.device,
                                       dtype=self.net.dtype)
                labels_b = labels_b.to(device=self.device,
                                       dtype=self.net.dtype)
                outputs_b = self.net(inputs_b)
                output = torch.cat((output, outputs_b), dim=0)
                values = torch.cat((values, labels_b), dim=0)

                loss = self.test_criterion(outputs_b, labels_b)
                running_loss += loss.item()
                total += labels_b.size(0)

        data_dict['predictions'] = output

        loss = running_loss / total
        # sanity check
        mse, _ = compute_mse_snr(values.cpu(), output.cpu())
        assert np.allclose(mse, loss), \
            '(mse: {:.7f}, loss: {:.7f})'.format(mse, loss)

        return mse, output

    def evaluate_lattice(self):
        """Create and evaluate network on lattice."""
        lat = Lattice(**self.params['lattice'])
        print(f'Sampled lattice lsize: {lat.lsize}.')

        lattice_grid = lat.lattice_to_standard(lat.lattice_grid.float())
        z = self.forward_data(lattice_grid)

        new_C_mat = lat.flattened_C_to_C_mat(z)
        lat.update_coefficients(new_C_mat)

        return lat

    def forward_data(self, inputs, batch_size=64):
        """
        Compute model output for some input.

        Args:
            input (torch.Tensor):
                size: (n, 2).
            batch_size (int)
        """
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
        Parse htv_log dictionary.

        Args:
            htv_log (dict).

        Returns:
            epochs (np.ndarray):
                array with saved epochs. size: (E,).
            htv: (dict/np.ndarray):
                saved htv accross training:
                if non-cpwl: dict('p': array of size: (E,)),
                if cpwl: array of size: (E,).
        """
        assert isinstance(htv_log, dict)
        # keys of htv_log are epochs
        epochs = np.array([int(epoch) for epoch in htv_log.keys()])
        idx = np.argsort(epochs)
        epochs = epochs[idx]

        # e.g. epoch 1
        # non-cpwl network:
        # htv_log['1'] = {'1': htv_1, '2': htv_2}
        # relu network: epoch 1: htv_log['1'] = htv

        # dictio is list of dictionaries containing 'p': htv_p for each epoch
        # or a single value in the case of CPWL networks
        values_list = list(htv_log.values())
        if 'finite_diff_differential' in values_list[0] or \
                'exact_differential' in values_list[0]:
            # TODO: Remove backward compatibility
            dictio = [val[list(val)[0]] for val in values_list]
        else:
            dictio = values_list

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
        Read loss_log (train or valid) dictionary.

        Returns:
            epochs (np.ndarray):
                array with saved epochs. size: (E,).
            loss (np.ndarray):
                array with saved loss across training. size: (E,)
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
