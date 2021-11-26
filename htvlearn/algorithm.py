import warnings
import torch
import numpy as np
import cvxopt
from scipy import sparse
import odl
from odl.operator.tensor_ops import MatrixOperator

from htvlearn.operators import Operators
from htvlearn.data import Data
from htvlearn.lattice import Lattice
from htvlearn.htv_utils import csr_to_spmatrix
from htvlearn.struct_default_values import SPARSITY_EPS


class Algorithm():
    """Class for the HTV minimization algorithm"""

    def __init__(self,
                 lattice_obj,
                 data_obj,
                 model_name,
                 lmbda,
                 admm_iter=100000,
                 sigma=None,
                 simplex=False,
                 verbose=False,
                 **kwargs):
        """
        Args:
            lattice_obj (Lattice):
                object of lattice type (see htvlearn.lattice).
            data_obj (Data):
                object of data type (see htvlearn.data).
            model_name (str)
            lmbda (float):
                regularization weight.
            admm_iter (int):
                number of admm iterations to run.
            sigma (None or float):
                step size for admm. If None, sigma = lmbda.
            simplex (bool):
                If True, perform simplex after admm.
            verbose (bool):
                Print more info.
        """
        if not isinstance(lattice_obj, Lattice):
            raise ValueError(f'lattice_obj is of type {type(lattice_obj)}.')
        if not isinstance(data_obj, Data):
            raise ValueError(f'lattice_obj is of type {type(lattice_obj)}.')
        self.lat = lattice_obj
        self.data = data_obj

        self.model_name = model_name
        self.lmbda = lmbda
        self.admm_iter = admm_iter
        self.sigma = sigma
        self.simplex = simplex
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

    def run_admm(self):
        """Learning using ADMM.

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
        self.lat.update_coefficients(new_C_mat)

        self.z_admm = z
        self.y_lmbda_admm, self.htv_loss_admm = \
            self.update_results_dict(z, 'admm')

    def setup_admm(self):
        """
        Perform setup for the admm algorithm.

        Returns:
            z_odl (odl.set.space.LinearSpaceElement)
                odl array of values.
            f, g (odl.solvers.functional.functional.Functional):
                odl functionals for admm.
            stack_op:
                forward and regularization odl linear Operator stacking.
            tau, sigma (positive float):
                proximal step sizes for the admm.
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

        # Estimated operator norm, add 20 percent for some safety margin
        op_norm = 1.2 * odl.power_method_opnorm(stack_op, maxiter=1000)
        # set step size for g.proximal
        if self.sigma is None:
            # if self.lmbda = 0, need to set a positive sigma
            sigma = self.lmbda if self.lmbda > 0 else 2.0
        else:
            sigma = float(self.sigma)  # constant
        # set step size for f.proximal
        tau = sigma / op_norm**2

        if self.verbose:
            print('admm (tau, sigma): ({:.3f}, {:.4f})'.format(tau, sigma))

        callback = (odl.solvers.CallbackPrintIteration(step=2000))

        z = self.lat.flattened_C.numpy()
        rn_space = odl.rn(z.shape[0], dtype='float32')
        z_odl = rn_space.element(z)

        return z_odl, f, g, stack_op, tau, sigma, callback

    def run_simplex(self):
        """Perform simplex to sparsify ADMM solution

        Solves the problem:
        minimize(s, x) c_matrix^T x
        subject to A_ub_sp(x) + s = zeros_ub_matrix
                   A_eq_sp(x) = b_matrix
                   s => 0

        See https://cvxopt.org/userguide/coneprog.html#linear-programming
        """
        c_matrix, A_ub_sp, zeros_ub_matrix, A_eq_sp, b_matrix, L_mat_sparse = \
            self.setup_simplex()

        sol = cvxopt.solvers.lp(c_matrix,
                                A_ub_sp,
                                zeros_ub_matrix,
                                A_eq_sp,
                                b_matrix,
                                solver='glpk')

        if self.verbose:
            print('\nSimplex status: ', sol['status'])
        self.results_dict['simplex_status'] = sol['status']

        x = np.array(sol['x']).reshape(-1)
        # checks on solution
        assert x.dtype == np.float64
        assert x.shape == (L_mat_sparse.shape[1] + L_mat_sparse.shape[0],)

        u = x[L_mat_sparse.shape[1]::]
        assert u.shape[0] == L_mat_sparse.shape[0]

        # update lattice values
        z = x[0:L_mat_sparse.shape[1]].astype(np.float32)
        z_torch = torch.from_numpy(z)
        new_C_mat = self.lat.flattened_C_to_C_mat(z_torch)

        self.lat.update_coefficients(new_C_mat)

        y_lmbda_simplex, htv_loss_simplex = \
            self.update_results_dict(z, 'simplex')

        if np.allclose(z, self.z_admm, atol=1e-5):
            print_msg = 'No change after simplex: z_simplex = z_admm'
            print('\n=> ' + print_msg)

        # checks
        assert np.allclose(y_lmbda_simplex, self.y_lmbda_admm, atol=1e-5)
        if np.allclose(self.lmbda, 0.):
            assert htv_loss_simplex <= self.htv_loss_admm, \
                f'{htv_loss_simplex} > {self.htv_loss_admm}'
        else:
            if not np.allclose(htv_loss_simplex,
                               self.htv_loss_admm,
                               atol=1e-3):
                print_msg = ('ADMM did not fully converge! \nReg. loss: \n' +
                             'simplex: {:.3E} != '.format(htv_loss_simplex) +
                             'admm: {:.3E}'.format(self.htv_loss_admm))
                warnings.warn(print_msg, UserWarning)

    def setup_simplex(self):
        """
        Perform setup for the simplex algorithm.

        Setup for the problem:
        minimize(s, x) c_matrix^T x
        subject to A_ub_sp(x) + s = zeros_ub_matrix
                   A_eq_sp(x) = b_matrix
                   s => 0

        See https://cvxopt.org/userguide/coneprog.html#linear-programming

        Returns:  # see problem above
            c_matrix (cvxopt.matrix)
            A_ub_sp (scipy.sparse.spmatrix)
            zeros_ub_matrix (cvxopt.matrix)
            A_eq_sp (scipy.sparse.spmatrix)
            b_matrix (cvxopt.matrix)
            L_mat_sparse (scipy.sparse.csr_matrix):
                regularization matrix
        """
        # construct H, L operators
        self.op = Operators(self.lat, self.input)

        # convert z, L, H to np.float64 (simplex requires this)
        H_mat_sparse = self.op.H_mat_sparse.astype(np.float64)
        L_mat_sparse = self.op.L_mat_sparse.astype(np.float64)

        assert self.z_admm.shape[0] == L_mat_sparse.shape[1]

        c = np.concatenate((np.zeros(L_mat_sparse.shape[1], dtype=np.float64),
                            np.ones(L_mat_sparse.shape[0], dtype=np.float64)),
                           axis=0)

        identity = sparse.eye(L_mat_sparse.shape[0],
                              format='csr',
                              dtype=np.float64)
        L_block = sparse.vstack((L_mat_sparse * (-1), L_mat_sparse))
        identity_block = sparse.vstack((identity, identity))
        # check reference in docstring
        A_ub = sparse.hstack((L_block, identity_block * (-1))).tocsr()

        zeros_ub = np.zeros(L_mat_sparse.shape[0] * 2, dtype=np.float64)

        A_ub_sp = csr_to_spmatrix(A_ub)

        # Here we just need to extend H_mat_sparse with zeros
        # (for the slack variable u)
        ext_shape = (H_mat_sparse.shape[0],
                     H_mat_sparse.shape[1] + L_mat_sparse.shape[0])
        A_eq_sp = csr_to_spmatrix(H_mat_sparse, extended_shape=ext_shape)

        c_matrix = cvxopt.matrix(c)
        zeros_ub_matrix = cvxopt.matrix(zeros_ub)
        b_matrix = cvxopt.matrix(self.y_lmbda_admm.astype(np.float64))

        if self.verbose:
            print('\nShapes for simplex :')
            print('c        : {}'.format(c_matrix.size),
                  'A_ub_sp  : {}'.format(A_ub_sp.size),
                  'zeros_ub : {}'.format(zeros_ub_matrix.size),
                  'A_eq_sp  : {}'.format(A_eq_sp.size),
                  'b_matrix : {}\n'.format(b_matrix.size),
                  sep='\n')

        return (c_matrix,
                A_ub_sp,
                zeros_ub_matrix,
                A_eq_sp,
                b_matrix,
                L_mat_sparse)

    def update_results_dict(self, z, mode):
        """
        Update the "results" dictionary.

        Args:
            z (1d array):
                new array of lattice parameters.
            mode (str):
                "admm" or "simplex"

        Returns:
            y_lmbda (1d array):
                observations at the datapoints.
            htv_loss (float):
                regularization loss.
        """
        assert mode in ['admm', 'simplex'], f'mode: {mode}.'
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
        self.results_dict['_'.join(mode, 'htv')] = htv_loss

        total_loss = df_loss + self.lmbda * htv_loss

        # sparsity
        L_z_zero_idx = np.where(np.absolute(L_z) <= SPARSITY_EPS)[0]
        percentage_nonzero = 0.
        if L_z.shape[0] != 0:
            fraction_zero = L_z_zero_idx.shape[0] / L_z.shape[0]
            percentage_nonzero = (100. - fraction_zero * 100)
        self.results_dict['percentage_nonzero'] = percentage_nonzero

        if self.verbose:
            print(f'\n{mode} results:')
            print('Train MSE: {:.3E}'.format(train_mse))
            print('Full lattice HTV: {:.3f}'.format(htv_loss))
            print('Total Loss: {:.3f}'.format(total_loss))
            print('nonzero slopes (%): {:.2f}%\n'.format(percentage_nonzero))

        return y_lmbda, htv_loss

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
                self.run_admm()

            print(f'\n--> admm iteration {i+1}/{num_iter} completed.')

            if i < num_iter - 1:  # don't divide in last iteration
                lattice_dict = self.lat.save(f'admm_iteration_{i+1}',
                                             lattice_dict)
                self.lat.refine_lattice()

            elif self.simplex is True:
                # save final admm lattice before running simplex
                lattice_dict = self.lat.save('admm_final', lattice_dict)

        if self.simplex is True:
            print('\nStart simplex.')
            self.run_simplex()
            print('\n---> simplex completed.')

        lattice_dict = self.lat.save('final', lattice_dict)

        return self.results_dict, lattice_dict
