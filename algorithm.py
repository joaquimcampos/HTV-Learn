import sys
import os
import copy
import math
import torch
import warnings
import numpy as np
import scipy
from scipy.optimize import linprog
import odl
from odl.operator.tensor_ops import MatrixOperator

import cvxopt
from scipy import sparse
from scipy.spatial import ConvexHull
from matplotlib.path import Path

from lattice_image import LatticeImage
import htv_utils
from operators import Operators


class Algorithm():

    eps = 1e-4 # epsilon to assess sparsity

    def __init__(self, lattice_obj, data_obj, plot_obj, **params):
        """
        Args:
            data_obj: -> see data.py.
            num_iter: number of ADMM iterations (with grid division) before simplex.
            plot: plot object or None (see plots/plot.py).
                    If not None, plots results after each ADMM iteration and simplex.
        """
        self.params = params
        self.lat = lattice_obj
        self.data = data_obj
        self.plot = plot_obj

        htv_utils.set_attributes(self, ['model_name', 'num_iter', 'admm_iter',
                                'lmbda', 'verbose', 'reduced',
                                'pos_constraint', 'no_simplex', 'sigma_rule'])

        self.niter = min(self.admm_iter, 200000) # admm iterations step

        # verify inputs
        self.input = self.data.train['input']
        self.values = self.data.train['values']

        assert len(self.input.size()) == 2, f'Found invalid size {self.input.size()}...'
        assert len(self.values.size()) == 1, f'Found invalid size {self.values.size()}...'

        if self.input.size(0) != self.values.size(0):
            raise ValueError('input and values have different lengthtv: '
                            f'{self.input.size(0)}, {self.values.size(0)}...')

        if not self.lat.is_padded(pad_size=1):
            raise ValueError('Lattice does not have zero boundaries...')

        # lattice_dict contains lattice "history"
        self.lattice_dict = {}
        # results_dict logs the numerical results
        self.results_dict = {}

        self.index_eps = 1e-9 # for get_index_masks()
        self.sign_atol = 1e-9
        self.zero_atol = 1e-7



    def admm(self):
        """ Learning using ADMM.

        In this example we solve the optimization problem
            min_x  ||H_op(z) - x_values||_2^2 + lam * ||L_op(z)||_1

        Where ``H_op`` is the forward operator, ``L_op`` is the regularization operator
        based on the HTV, ``x_values`` are the observations at
        the x_lat points (in lattice coordinates), ``z`` is the vectorized
        lattice vertices values (to be updated), which are initialized with
        current values.

        The problem is rewritten in decoupled form as min_x g(L(z)),
        with a separable sum ``g`` of functionals and the stacked operator ``L``:
            g(z) = ||k_1 - x_values||_2^2 + lam * ||k_2||_1,

                       ( H_op(z) )
            k = L(z) = ( L_op(z) ).

        Link: https://github.com/odlgroup/odl/blob/master/examples/solvers/admm_tomography.py
        See the documentation of the `admm_linearized` solver for further details.
        """
        z_odl, f, g, stack_op, tau, sigma, callback = self.setup_admm()

        # Run the algorithm
        odl.solvers.admm_linearized(z_odl, f, g, stack_op, tau,
                                    sigma, self.niter, callback=callback)

        # update lattice values with admm result stored in z_odl
        z = z_odl.asarray()
        z_torch = torch.from_numpy(z)
        new_C_mat = self.lat.flattened_C_to_C_mat(z_torch)
        self.lat.update_lattice_values(new_C_mat)

        self.z_admm = z
        self.y_lmbda_admm, self.L_z_admm = self.update_results_dict('admm', z)

        # checks
        total_loss_admm = self.results_dict['admm']['total_loss']
        df_loss_admm = self.results_dict['admm']['df_loss']
        normalized_df_loss_admm = self.results_dict['admm']['normalized_df_loss']
        htv_loss_admm = self.results_dict['admm']['htv_loss']
        nz_admm = self.results_dict['admm']['percentage_nonzero'] # nz=nonzero

        if self.verbose:
            print(f'\ntotal_loss: admm - {total_loss_admm}')
            print(f'df_loss: admm - {df_loss_admm}')
            print(f'normalized_df_loss: admm - {normalized_df_loss_admm}')
            print(f'htv_loss: admm - {htv_loss_admm}')
            print(f'nonzero slopes (%) : admm - {nz_admm}%\n')


    def setup_admm(self):
        """ """
        # construct H, L operators
        self.op = Operators(self.lat, self.input, lmbda=self.lmbda)

        # --- Set up the inverse problem --- #
        L_op = MatrixOperator(self.op.L_mat_sparse)
        H_op = MatrixOperator(self.op.H_mat_sparse)

        # Stacking of the two operators
        stack_op = odl.BroadcastOperator(H_op, L_op)

        x_values_np = self.values.numpy()
        data_fit = odl.solvers.L2NormSquared(H_op.range).translated(x_values_np)
        reg_func = odl.solvers.L1Norm(L_op.range) # lmbda is inside L_op

        g = odl.solvers.SeparableSum(data_fit, reg_func)
        # We don't use the f functional, setting it to zero
        f = odl.solvers.ZeroFunctional(stack_op.domain)

        # --- Select parameters and solve using LADMM --- #
        # Estimated operator norm, add 10 percent for some safety margin
        op_norm = 1.1 * odl.power_method_opnorm(stack_op, maxiter=1000)

        if self.sigma_rule == 'constant':
            sigma = 2.0
        elif self.sigma_rule == 'same':
            sigma = self.lmbda
        elif self.sigma_rule == 'inverse':
            sigma = 1./self.lmbda
        else:
            raise ValueError(f'sigma rule "{self.sigma_rule}" is invalid.')

        tau = sigma / op_norm ** 2  # Step size for f.proximal

        print(f'admm (tau, sigma): ({tau}, {sigma})')

        callback = None
        if self.verbose and False:
            # Optionally pass a callback to the solver to display intermediate results
            callback = (odl.solvers.CallbackPrintIteration(step=2000) &
                        odl.solvers.CallbackShow(step=10))
        else:
            callback = (odl.solvers.CallbackPrintIteration(step=2000))

        z = self.lat.flattened_C.numpy()
        rn_space = odl.rn(z.shape[0], dtype='float32')
        z_odl = rn_space.element(z)

        return z_odl, f, g, stack_op, tau, sigma, callback


    def update_results_dict(self, mode, z):
        """ """
        assert mode in ['admm', 'simplex']

        if mode not in self.results_dict:
            self.results_dict[mode] = {}

        # construct H, L operators. L without weight multiplication.
        self.op = Operators(self.lat, self.input)

        # data fildelity (should not change after simplex)
        y_lmbda = self.op.H_mat_sparse.dot(z)
        df_loss = ((y_lmbda - self.values.numpy()) ** 2).sum()
        self.results_dict[mode]['df_loss'] = df_loss
        normalized_df_loss = df_loss/y_lmbda.shape[0]
        self.results_dict[mode]['normalized_df_loss'] = normalized_df_loss

        # regularization (should not change after simplex)
        L_z = self.op.L_mat_sparse.dot(z)
        htv_loss = np.linalg.norm(L_z, ord=1)
        self.results_dict[mode]['htv_loss'] = htv_loss

        total_loss = df_loss + self.lmbda*htv_loss
        self.results_dict[mode]['total_loss'] = total_loss

        # sparsity
        L_z_zero_idx = np.where(np.absolute(L_z) <= self.eps)[0]
        sparsity = L_z.shape[0] - L_z_zero_idx.shape[0]
        self.results_dict[mode]['sparsity'] = sparsity

        fraction_zero = 1.
        if L_z.shape[0] != 0:
            fraction_zero = L_z_zero_idx.shape[0] / L_z.shape[0]
        self.results_dict[mode]['percentage_nonzero'] = \
                                    (100. - fraction_zero * 100)

        return y_lmbda, L_z



    def get_index_masks(self, L_z, epsilon_tol=True):
        """ Gets indexes where sign(L.dot(z)) = 0/1/-1

        epsilon_tol: if true, a tolerance of epsilon is used for zeros.
        """
        if epsilon_tol is True:
            I_pos = np.where(L_z > self.index_eps)[0]
            I_neg = np.where(L_z < -self.index_eps)[0]
            I_zero = np.where(np.absolute(L_z) <= self.index_eps)[0]
        else:
            I_pos = np.where(L_z > 0)[0]
            I_neg = np.where(L_z <= 0)[0]
            I_zero = np.empty_like(I_pos, shape=(0,))

        assert (I_zero.shape[0] + I_pos.shape[0] + I_neg.shape[0]) == L_z.shape[0], \
        f'{I_zero.shape[0]}, {I_pos.shape[0]}, {I_neg.shape[0]}, {L_z.shape[0]}.'

        return I_zero, I_pos, I_neg



    def simplex(self):
        """ Performs simplex to sparsify ADMM solution

        See https://cvxopt.org/userguide/coneprog.html#linear-programming
        """
        # construct H, L operators
        self.op = Operators(self.lat, self.input)

        # convert z, L, H to np.float64 (simplex requires this)
        H_mat_sparse = self.op.H_mat_sparse.astype(np.float64)
        L_mat_sparse = self.op.L_mat_sparse.astype(np.float64)

        assert self.z_admm.shape[0] == L_mat_sparse.shape[1]

        # perform reduced simplex (updates lattice values)
        if self.reduced is True:
            z, status = self.reduced_simplex(H_mat_sparse, L_mat_sparse)
        else:
            z, status = self.regular_simplex(H_mat_sparse, L_mat_sparse)

        if self.verbose:
            print('\nSimplex status: ', status)

        if np.allclose(z, self.z_admm, atol=1e-5):
            print('\nNo change after simplex: z_simplex = z_admm')

        y_lmbda_simplex, L_z_simplex = self.update_results_dict('simplex', z)
        self.results_dict['simplex']['status'] = status

        df_loss_admm = self.results_dict['admm']['df_loss']
        htv_loss_admm = self.results_dict['admm']['htv_loss']
        nz_admm = self.results_dict['admm']['percentage_nonzero'] # nz=nonzero
        df_loss_simplex = self.results_dict['simplex']['df_loss']
        htv_loss_simplex = self.results_dict['simplex']['htv_loss']
        nz_simplex = self.results_dict['simplex']['percentage_nonzero']

        if self.verbose:
            print(f'\ndf_loss simplex - {df_loss_simplex}')
            print(f'htv_loss simplex - {htv_loss_simplex}')
            print(f'nonzero slopes (%) : simplex - {nz_simplex}%')

        # checks
        assert np.allclose(y_lmbda_simplex, self.y_lmbda_admm, atol=1e-5)
        assert np.allclose(df_loss_simplex, df_loss_admm, atol=1e-5), \
                                f'{df_loss_simplex} != {df_loss_admm}'
        if np.allclose(self.lmbda, 0.):
            assert htv_loss_simplex <= htv_loss_admm, \
                                f'{htv_loss_simplex} > {htv_loss_admm}'
        else:
            if not np.allclose(htv_loss_simplex, htv_loss_admm, atol=1e-3):
                warnings.warn(f'ADMM did not fully converge! \nReg. loss: \n'
                    f'simplex: {htv_loss_simplex} != admm: {htv_loss_admm}',
                    UserWarning)



    def reduced_simplex(self, H_mat_sparse, L_mat_sparse):
        """ Reduced simplex version.

        Args:
            H_mat_sparse, L_mat_sparse: np.float64 operators
        """
        I_zero, I_pos, I_neg = self.get_index_masks(self.L_z_admm, epsilon_tol=True)

        if self.verbose:
            print('Reduced simplex - sizes of (I_zero, I_pos, I_neg): '
                f'({I_zero.shape[0]}, {I_pos.shape[0]}, {I_neg.shape[0]})')

        k = (L_mat_sparse[I_pos].sum(0) - L_mat_sparse[I_neg].sum(0))
        k = np.squeeze(np.asarray(k))
        k_z_admm = k.dot(self.z_admm)

        assert np.allclose(k_z_admm, self.results_dict['admm']['htv_loss'],
                        atol=self.index_eps*L_mat_sparse.shape[0]), \
            f'{k_z_admm} != {self.results_dict["admm"]["htv_loss"]}'

        # A_eq = H_mat_sparse
        k_sparse = sparse.csr_matrix(k[np.newaxis, :].astype(np.float64))
        A_eq = sparse.vstack((H_mat_sparse, k_sparse)).tocsr()
        A_eq_sp = htv_utils.csr_to_spmatrix(A_eq)

        b = np.concatenate((self.y_lmbda_admm, np.array([k_z_admm]))).astype(np.float64)

        # enforce (Lz)_{I_pos} > -sign_atol, (Lz)_{I_neg} < sign_atol,
        # |(Lz)_{I_zero}| < zero_atol
        A_ub = sparse.vstack(((-1) * L_mat_sparse[I_pos], L_mat_sparse[I_neg],
                        L_mat_sparse[I_zero], (-1) * L_mat_sparse[I_zero])).tocsr()
        A_ub_sp = htv_utils.csr_to_spmatrix(A_ub)

        eps_ub = np.ones(A_ub.shape[0], dtype=np.float64) * self.sign_atol
        zeros_start = I_pos.shape[0] + I_neg.shape[0]
        eps_ub[zeros_start::] = \
            np.ones(I_zero.shape[0]*2, dtype=np.float64) * self.zero_atol

        # c_matrix = cvxopt.matrix(k.astype(np.float64))
        c_matrix = cvxopt.matrix(np.zeros_like(k, dtype=np.float64))
        eps_ub_matrix = cvxopt.matrix(eps_ub)
        b_matrix = cvxopt.matrix(b)

        print('\nShapes for simplex :')
        print('c        : {}'.format(c_matrix.size),
              'A_ub_sp  : {}'.format(A_ub_sp.size),
              'eps_ub : {}'.format(eps_ub_matrix.size),
              'A_eq_sp  : {}'.format(A_eq_sp.size),
              'b_matrix : {}\n\n'.format(b_matrix.size),
              sep='\n')

        sol = cvxopt.solvers.lp(c_matrix, A_ub_sp, eps_ub_matrix, A_eq_sp,
                        b_matrix, solver='glpk')

        x = np.array(sol['x']).reshape(-1)
        # checks on solution
        assert x.dtype == np.float64
        assert x.shape == (L_mat_sparse.shape[1],)

        # update parameters
        z = x.astype(np.float32)
        z_torch = torch.from_numpy(z)
        new_C_mat = self.lat.flattened_C_to_C_mat(z_torch)

        self.lat.update_lattice_values(new_C_mat)

        return z, sol['status']



    def regular_simplex(self, H_mat_sparse, L_mat_sparse):
        """ Regular simplex (possibly with positivity constraint).

        Args:
            H_mat_sparse, L_mat_sparse: np.float64 operators
        """
        c = np.concatenate((np.zeros(L_mat_sparse.shape[1], dtype=np.float64),
                            np.ones(L_mat_sparse.shape[0], dtype=np.float64)),
                            axis=0)

        identity = sparse.eye(L_mat_sparse.shape[0], format='csr', dtype=np.float64)
        L_block = sparse.vstack((L_mat_sparse * (-1), L_mat_sparse))
        identity_block = sparse.vstack((identity, identity))
        A_ub = sparse.hstack((L_block, identity_block * (-1))).tocsr() # check reference above

        zeros_ub = np.zeros(L_mat_sparse.shape[0] * 2, dtype=np.float64)

        # positivity constrain on slack variables (may help)
        if self.pos_constraint:
            # enforce positivity constraint on slack variable u
            # extend (-1)*identity with zeros first
            pos_block = (-1) * identity
            ext_shape = (L_mat_sparse.shape[0], L_mat_sparse.shape[1] + L_mat_sparse.shape[0])
            pos_block.indices += L_mat_sparse.shape[1] # put (-1)*identity in the right part
            pos_block.resize(ext_shape)

            A_ub = sparse.vstack((A_ub, pos_block))
            zeros_ub = np.zeros(L_mat_sparse.shape[0] * 3)

        A_ub_sp = htv_utils.csr_to_spmatrix(A_ub)

        # Here we just need to extend H_mat_sparse with zeros (for slack variable u)
        ext_shape = (H_mat_sparse.shape[0], H_mat_sparse.shape[1] + L_mat_sparse.shape[0])
        A_eq_sp = htv_utils.csr_to_spmatrix(H_mat_sparse, extended_shape=ext_shape)

        c_matrix = cvxopt.matrix(c)
        zeros_ub_matrix = cvxopt.matrix(zeros_ub)
        b_matrix = cvxopt.matrix(self.y_lmbda_admm.astype(np.float64))

        if self.verbose:
            print('\nShapes for simplex :')
            print('c        : {}'.format(c_matrix.size),
                  'A_ub_sp  : {}'.format(A_ub_sp.size),
                  'zeros_ub : {}'.format(zeros_ub_matrix.size),
                  'A_eq_sp  : {}'.format(A_eq_sp.size),
                  'b_matrix : {}\n\n'.format(b_matrix.size),
                  sep='\n')

        sol = cvxopt.solvers.lp(c_matrix, A_ub_sp, zeros_ub_matrix, A_eq_sp,
                                b_matrix, solver='glpk')

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

        self.lat.update_lattice_values(new_C_mat)

        return z, sol['status']



    def save_lattice(self, save_str, **kwargs):
        """ """
        if save_str not in self.lattice_dict:
            self.lattice_dict[save_str] = {}

        self.lattice_dict[save_str]['X_mat'] = self.lat.X_mat.clone()
        self.lattice_dict[save_str]['C_mat'] = self.lat.C_mat.clone()

        if self.plot is not None:
            self.plot.show_lat_plot(observations=True, mode='train',
                filename='_'.join([self.model_name, save_str]), **kwargs)



    def multires_admm(self):
        """ multires_admm """

        self.save_lattice('init', color='random')
        print('\n\nStart multires admm.')

        for i in range(self.num_iter):

            self.admm_iter  # Number of iterations
            for j in range(0, self.admm_iter//self.niter):
                print(f'\nsubrun {j}/{self.admm_iter//self.niter-1}:')
                self.admm()

            print(f'\n--> admm iteration {i+1}/{self.num_iter} completed.')

            if i < self.num_iter - 1: # don't divide in last iteration
                self.save_lattice(f'admm_iteration_{i+1}')
                self.lat.divide_lattice()
            else:
                self.save_lattice(f'admm')

        if not self.no_simplex:
            print('\nStart simplex.')
            self.simplex()
            print('\nSimplex completed.\n')
            self.save_lattice('simplex')

        self.save_lattice('final')
