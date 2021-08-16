import torch
import numpy as np
import odl
from odl.operator.tensor_ops import MatrixOperator

from htvlearn.operators import Operators
from htvlearn.data import Data
from htvlearn.lattice import Lattice


class Algorithm():
    """ Class for the HTV minimization algorithm """

    eps = 1e-4  # epsilon to assess sparsity

    def __init__(self,
                 lattice_obj,
                 data_obj,
                 model_name,
                 lmbda,
                 admm_iter=100000,
                 verbose=False,
                 **kwargs):
        """
        Args:
            lattice_obj (Lattice):
                object of lattice type (see htvlearn.lattice)
            data_obj (Data):
                object of data type (see htvlearn.data)
            model_name (str)
            lmbda (float):
                regularization weight
            admm_iter (int):
                number of admm iterations to run
            verbose (bool):
                Print more info.
        """
        if not isinstance(lattice_obj, Lattice):
            raise ValueError(f'lattice_obj is of type {type(lattice_obj)}.')
        if not isinstance(lattice_obj, Data):
            raise ValueError(f'lattice_obj is of type {type(lattice_obj)}.')
        self.lat = lattice_obj
        self.data = data_obj

        self.model_name = model_name
        self.lmbda = lmbda
        self.admm_iter = admm_iter
        self.verbose = verbose

        # log_step for admm iterations
        self.log_step = min(self.admm_iter, 200000)

        # verify inputs
        self.input = self.data.train['input']
        self.values = self.data.train['values']

        assert len(self.input.size()) == 2, \
            f'Found invalid size {self.input.size()}...'
        assert len(self.values.size()) == 1, \
            f'Found invalid size {self.values.size()}...'

        if self.input.size(0) != self.values.size(0):
            raise ValueError('input and values have different lengthtv: '
                             f'{self.input.size(0)}, {self.values.size(0)}...')

        # results_dict logs the numerical results
        self.results_dict = {}

    def admm(self):
        """ Learning using ADMM.

        In this example we solve the optimization problem
            min_x  ||H_op(z) - x_values||_2^2 + lam * ||L_op(z)||_1

        Where ``H_op`` is the forward operator, ``L_op`` is the regularization
        operator based on the HTV, ``x_values`` are the observations at
        the datapoints, ``z`` is the vectorized lattice vertices values
        (to be updated), which are initialized with the current values.

        The problem is rewritten in decoupled form as min_x g(L(z)),
        with a separable sum ``g`` of functionals and the stacked
        operator ``L``:
            g(z) = ||k_1 - x_values||_2^2 + lam * ||k_2||_1,

                       ( H_op(z) )
            k = L(z) = ( L_op(z) ).

        Link: https://github.com/odlgroup/odl/blob/master/
        examples/solvers/admm_tomography.py
        See the documentation of the `admm_linearized` solver
        for further details.
        """
        z_odl, f, g, stack_op, tau, sigma, callback = self.setup_admm()

        # Run the algorithm
        odl.solvers.admm_linearized(z_odl,
                                    f,
                                    g,
                                    stack_op,
                                    tau,
                                    sigma,
                                    self.log_step,
                                    callback=callback)

        # update lattice values with admm result stored in z_odl
        z = z_odl.asarray()
        z_torch = torch.from_numpy(z)
        new_C_mat = self.lat.flattened_C_to_C_mat(z_torch)
        self.lat.update_lattice_values(new_C_mat)

        self.z_admm = z
        self.y_lmbda_admm, self.L_z_admm = self.update_results_dict(z)

    def setup_admm(self):
        """
        Setup for the admm algorithm.

        Returns:
            z_odl (odl.set.space.LinearSpaceElement)
                odl array of values.
            f, g (odl.solvers.functional.functional.Functional):
                odl functionals for admm.
            stack_op:
                forward and regularization odl linear Operator stacking.
            tau, sigma (positive float):
                step sizes for the admm.
            callback:
                function called with the current iterate after each iteration.
        """
        # construct H, L operators
        self.op = Operators(self.lat, self.input, lmbda=self.lmbda)

        # --- Set up the inverse problem --- #

        # TODO: Construct matrix-free operator
        # odlgroup: [MatrixOperator] is in general a rather slow
        # and memory-inefficient approach, and users are recommended
        # to use other alternatives if possible.

        # L_mat_sparse is multiplied by lmbda
        L_op = MatrixOperator(self.op.L_mat_sparse)
        H_op = MatrixOperator(self.op.H_mat_sparse)

        # Stacking of the two operators
        stack_op = odl.BroadcastOperator(H_op, L_op)

        x_values_np = self.values.numpy()
        data_fit = \
            odl.solvers.L2NormSquared(H_op.range).translated(x_values_np)
        reg_func = odl.solvers.L1Norm(L_op.range)  # lmbda is inside L_op

        g = odl.solvers.SeparableSum(data_fit, reg_func)
        # We don't use the f functional, setting it to zero
        f = odl.solvers.ZeroFunctional(stack_op.domain)

        # Estimated operator norm, add 10 percent for some safety margin
        op_norm = 1.1 * odl.power_method_opnorm(stack_op, maxiter=1000)

        sigma = 2.0
        tau = sigma / op_norm**2  # Step size for f.proximal

        if self.verbose:
            print('admm (tau, sigma): ({:.3f}, {:.4f})'.format(tau, sigma))

        callback = (odl.solvers.CallbackPrintIteration(step=2000))

        z = self.lat.flattened_C.numpy()
        rn_space = odl.rn(z.shape[0], dtype='float32')
        z_odl = rn_space.element(z)

        return z_odl, f, g, stack_op, tau, sigma, callback

    def update_results_dict(self, z):
        """
        Update the "results" dictionary.

        Args:
            z (1d array):
                new array of lattice parameters.
        """
        # construct H, L operators. L without lmbda multiplication.
        self.op = Operators(self.lat, self.input)

        # data fildelity
        y_lmbda = self.op.H_mat_sparse.dot(z)
        df_loss = ((y_lmbda - self.values.numpy())**2).sum()
        train_mse = df_loss / y_lmbda.shape[0]
        self.results_dict['train_mse'] = train_mse

        # regularization
        # L_mat_sparse is not multiplied by lmbda
        L_z = self.op.L_mat_sparse.dot(z)
        htv_loss = np.linalg.norm(L_z, ord=1)

        total_loss = df_loss + self.lmbda * htv_loss

        # sparsity
        L_z_zero_idx = np.where(np.absolute(L_z) <= self.eps)[0]
        percentage_nonzero = 0.
        if L_z.shape[0] != 0:
            fraction_zero = L_z_zero_idx.shape[0] / L_z.shape[0]
            percentage_nonzero = (100. - fraction_zero * 100)
        self.results_dict['percentage_nonzero'] = percentage_nonzero

        if self.verbose:
            print('\nTrain MSE: {:.3E}'.format(train_mse))
            print('Full lattice HTV: {:.3f}'.format(htv_loss))
            print('Total Loss: {:.3f}'.format(total_loss))
            print('nonzero slopes (%): {:.2f}%\n'.format(percentage_nonzero))

        return y_lmbda, L_z

    def multires_admm(self):
        """
        Perform multiresolution admm.

        Returns:
            results_dict (dict):
                dictionary with saved results.
            lattice_dict (dict):
                dictionary with saved lattice states across iterations.
        """

        print('\n\nStart multires admm.')
        lattice_dict = self.lat.save('init')

        num_iter = 1  # increase for multiresolution
        for i in range(num_iter):

            for j in range(0, self.admm_iter // self.log_step):
                print(f'\nsubrun {j} / {self.admm_iter // self.log_step - 1}:')
                self.admm()

            print(f'\n--> admm iteration {i+1}/{num_iter} completed.')

            if i < num_iter - 1:  # don't divide in last iteration
                lattice_dict = self.lat.save(f'admm_iteration_{i+1}',
                                             lattice_dict)
                self.lat.divide_lattice()
            else:
                lattice_dict = self.lat.save('final', lattice_dict)

        return self.results_dict, lattice_dict
