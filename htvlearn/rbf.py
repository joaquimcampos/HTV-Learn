import torch
import numpy as np


class RBF():
    def __init__(self, data_obj, eps, lmbda, **params):
        """
        Args:
            data_obj (Data):
                instance of Data class (data.py), containing:
                data_obj['train']['input']: size: (M,2);
                data_obj['train']['values']: size: (M,).
            eps (float):
                rbf kernel shape parameter (the higher, the more localized)
            lmbda (float):
                regularization weight
        """
        self.params = params
        self.data = data_obj
        self.eps = eps
        self.lmbda = lmbda

        self.input = self.data.train['input']
        assert self.input.size(1) == 2
        self.values = self.data.train['values']
        assert self.input.size(0) == self.values.size(0)

        self.init_gram_mat()
        self.init_coeffs()

    @staticmethod
    def get_sigma_from_eps(eps):
        """
        Get sigma (standard deviation) from rbf shape parameter eps.

        Args:
            eps (float):
                rbf kernel shape parameter (the higher, the more localized).

        Returns:
            sigma (float):
                standard deviation of the kernel.
        """
        return np.sqrt(1. / (2. * eps))

    def init_gram_mat(self):
        """
        Init Gram interpolation matrix with gaussian kernels.

        ``gram_mat = exp(-(eps * distance_pairs)**2)``, where
        ``distance pairs`` is the gram matrix of parwise distances of training
        datapoints.
        """
        # (1, M, 2) - (M, 1, 2) = (M, M, 2)
        # location pairwise differences
        loc_diff_pairs = self.input.unsqueeze(0) - self.input.unsqueeze(1)
        assert loc_diff_pairs.size() == (self.input.size(0),
                                         self.input.size(0), 2)
        # distance pairs
        distance_pairs = torch.norm(loc_diff_pairs, dim=-1)
        assert distance_pairs.size() == (self.input.size(0),
                                         self.input.size(0))

        # RBF interpolation matrix with gaussian kernels
        self.gram_mat = torch.exp(-(self.eps * distance_pairs.double())**2)

        # check if it is symmetric
        assert torch.equal(self.gram_mat.T, self.gram_mat)
        assert self.gram_mat.size() == (self.input.size(0), self.input.size(0))

    def init_coeffs(self):
        """
        Solve ``(gram_mat + lambda*I)a = y`` for ``a`` (coefficients),
        and save the result in self.coeffs.
        """
        A = self.gram_mat + self.lmbda * \
            torch.ones_like(self.gram_mat[:, 0]).diag()
        B = self.values.unsqueeze(-1).to(dtype=self.gram_mat.dtype,
                                         device=self.gram_mat.device)

        # Solve AX = B
        X, _ = torch.lstsq(B, A)
        self.coeffs = X.squeeze(-1).float()

        assert self.coeffs.size() == self.values.size()

    def construct_H_mat(self, x, **kwargs):
        """
        Construct the forward matrix H from x (the locations where we wish
        to evaluate the RBF), such that f(x) = H_mat @ self.coeffs.

        Args:
            x (torch.Tensor):
                locations where we wish to evaluate the RBF.

        Returns:
            H_mat (torch.Tensor):
                size: (x.size(0), self.coeffs.size(0))
        """
        # (N, 1, 2) - (1, M, 2) = (N, M, 2)
        # location pairwise differences \varphi(\norm{\V x - \V x_i})
        loc_diff_pairs = x.unsqueeze(1) - self.input.unsqueeze(0)

        assert loc_diff_pairs.size() == (x.size(0), self.input.size(0), 2)
        # distance pairs
        distance_pairs = torch.norm(loc_diff_pairs, dim=-1)
        assert distance_pairs.size() == (x.size(0), self.input.size(0))

        # x interpolation matrix with gaussian kernels
        H_mat = torch.exp(-(self.eps * distance_pairs)**2)

        assert H_mat.size() == (x.size(0), self.coeffs.size(0))

        return H_mat

    def evaluate(self, x, **kwargs):
        """
        Evaluate RBF output at locations x.

        f(\V x) = self.H_mat @ self.coeffs, where self.H_mat is constructed
        from x.

        Args:
            x (torch.Tensor):
                locations where we wish to evaluate the RBF.

        Returns:
            output (torch.Tensor):
                evaluation of the RBF at x.
        """
        try:
            H_mat = self.construct_H_mat(x)
        except RuntimeError:
            print('\nError: need to reduce size of input batches.\n')
            raise

        coeffs = self.coeffs.to(dtype=H_mat.dtype)
        output = torch.mv(H_mat, coeffs)

        return output
