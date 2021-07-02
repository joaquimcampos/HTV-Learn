import argparse
import copy
import math
from datetime import datetime

import torch
import numpy as np
import json
from scipy import sparse
import cvxopt
from matplotlib import cm

from struct_default_values import default_values
from operators import Operators
import algorithm


class ArgCheck():
    """ Class for input argument verification """

    @staticmethod
    def p_int(value):
        """ Check if int value got from argparse is positive
            and raise error if not.
        """
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f'{value} is an invalid positive int value')
        return ivalue


    @staticmethod
    def p_odd_int(value):
        """ Check if int value got from argparse is positive
            and raise error if not.
        """
        ivalue = int(value)
        if (ivalue <= 0) or ((ivalue + 1) % 2 != 0) :
            raise argparse.ArgumentTypeError(f'{value} is an invalid positive odd int value')
        return ivalue


    @staticmethod
    def nn_int(value):
        """ Check if int value got from argparse is non-negative
            and raise error if not.
        """
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(f'{value} is an invalid non-negative int value')
        return ivalue



    @staticmethod
    def p_float(value):
        """ Check if float value got from argparse is positive
            and raise error if not.
        """
        ivalue = float(value)
        if ivalue <= 0:
             raise argparse.ArgumentTypeError(f'{value} is an invalid positive float value')
        return ivalue


    @staticmethod
    def frac_float(value):
        """ Check if float value got from argparse is >= 0 and <= 1
            and raise error if not.
        """
        ivalue = float(value)
        if ivalue < 0 or ivalue > 1:
             raise argparse.ArgumentTypeError(f'{value} is an invalid fraction float value (should be in [0, 1])')
        return ivalue


    @staticmethod
    def nn_float(value):
        """ Check if float value got from argparse is non-negative
            and raise error if not.
        """
        ivalue = float(value)
        if not np.allclose(np.clip(ivalue, -1.0, 0.0), 0.0):
            raise argparse.ArgumentTypeError(f'{value} is an invalid non-negative float value')
        return ivalue


### Parameters

def size_str(input):
    """ Returns a string with the size of the input pytorch tensor
    """
    out_str = '[' + ', '.join(str(i) for i in input.size()) + ']'
    return out_str



def dict_recursive_merge(params_root, merger_root, base=True):
    """ Recursively merges merger_root into params_root giving precedence
    to the second as in z = {**x, **y}
    """
    if base:
        assert isinstance(params_root, dict)
        assert isinstance(merger_root, dict)
        merger_root = copy.deepcopy(merger_root)

    if merger_root: # non-empty dict
        for key, val in merger_root.items():
            if isinstance(val, dict):
                merger_root[key] = dict_recursive_merge(params_root[key], merger_root[key], base=False)

        merger_root = {**params_root, **merger_root}

    return merger_root



def assign_structure_recursive(assign_root, structure, base=True):
    """ Recursively assigns values to assign_root according to structure
    (see structure variable in default_struct_values.py)
    """
    if base:
        assert isinstance(assign_root, dict)
        assert isinstance(structure, dict)
        assign_root = copy.deepcopy(assign_root)
        global assign_orig
        assign_orig = assign_root # keep the original dict
        global leaves
        leaves = [] # leaves on dict tree levels deeper than base

    if structure: # non-empty dict
        for key, val in structure.items():
            if isinstance(val, dict):
                assign_root[key] = {}
                assign_root[key] = assign_structure_recursive(assign_root[key], structure[key], base=False)
                if len(assign_root[key]) < 1: # do not have empty dictionaries in assign_root
                    del assign_root[key]
            else:
                assert val is None, 'leaf values in structure should be None'
                if key in assign_orig:
                    assign_root[key] = assign_orig[key]
                    if not base:
                        leaves.append(key)

    # delete duplicated leaves in base root if they are not
    # in first level of structure dict
    if base:
        for key in leaves:
            if key not in structure and key in assign_root:
                del assign_root[key]


    return assign_root



def flatten_structure(assign_root, base=True):
    """ Flattens the structure created with assign_structure_recursive()
    """
    if base:
        assert isinstance(assign_root, dict)
        global flattened
        flattened = {}
        assign_root = copy.deepcopy(assign_root)

    for key, val in assign_root.items():
        if isinstance(val, dict):
            flatten_structure(assign_root[key], base=False)
        elif key not in flattened:
            flattened[key] = val

    return flattened



def get_colorscale(colorscale='YlOrRd', pl_entries=256):
    """ Gets a matplotlib color map and returns a colorscale
    numpy array os shape (256, 2) where in the first column are
    the ascending normalized color values between 0 and 1 and
    in the second column the corresponding rgb color tuple.
    """
    cmap = cm.get_cmap(colorscale)

    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return np.array(pl_colorscale)[:, 1]



def generate_custom_square_grid(start, end, step=1):
    """ Generates square 2D grid and returns a list of (x1,x2) grid points

    Returns:
        x: list of (x1,x2) grid points (size (m, 2) = (a*a, 2))
        x_vec: vector that originated grid (size (a,)).
        Xsq_size: size of meshgrid tensor (size (a, a)).
    """
    if isinstance(start, int) and isinstance(step, int):
        x_vec = torch.arange(start, end, step)
    else:
        x_vec = torch.arange(start, end, step, dtype=torch.float32)

    x, Xsq_size = get_grid(x_vec, x_vec)

    return x, x_vec, Xsq_size



def get_grid(x1_vec, x2_vec):
    """ Get grid from x1_vec, x2_vec 1D vectors

    returns:
        Xsq_size = (x2_vec_size, x1_vec_size)
    """
    X2, X1 = torch.meshgrid(x2_vec, x1_vec)
    Xsq_size = X1.size()

    X1_flat, X2_flat = X1.flatten(), X2.flatten()
    x = torch.cat((X1_flat.view(-1, 1), X2_flat.view(-1, 1)), dim=1)
    assert x.size(1) == 2

    return x, Xsq_size



def csr_get_nonzero_rows(M):
    """ Get indexes of nonzero rows of matrix (sparse.csr_matrix)
    """
    assert type(M) == sparse.csr_matrix, 'M should be a sparse csr matrix...'
    if M.dtype in [np.float32, np.float64]:
        # check scipy.optimize._remove_redundancy._remove_zero_rows()
        zero_tol = 1e-9
        row_count = np.array((abs(M) > zero_tol).sum(axis=1)).flatten()
        zero_mask = np.logical_not(row_count == 0)
        nonzero_rows_idx = np.where(zero_mask == 1)[0]
        zero_rows_idx = np.where(zero_mask == 0)[0]
        assert (nonzero_rows_idx.shape[0] + zero_rows_idx.shape[0] == \
                row_count.shape[0])
    else:
        row_diff = np.diff(M.indptr)
        nonzero_rows_idx = np.where(row_diff != 0)[0]
        zero_rows_idx = np.where(row_diff == 0)[0]
        assert (nonzero_rows_idx.shape[0] + zero_rows_idx.shape[0] == \
                row_diff.shape[0])

    return nonzero_rows_idx, zero_rows_idx



def csr_get_nonzero_cols(M):
    """ """
    assert type(M) == sparse.csr_matrix, 'M should be a sparse csr matrix'
    rows, columns = M.nonzero() # a tuple of two arrays: 0th is row indices, 1st is cols
    nonzero_cols_idx = np.array(sorted(set(columns)), dtype=np.int64)

    return nonzero_cols_idx



def csr_to_spmatrix(M, extended_shape=None):
    """ Convert csr matrix to spmatrix

    Args:
        M - csr matrix
        extended_shape - extended shape for matrix.
    """
    # convert torows, cols = M.nonzero() spmatrix
    if extended_shape is not None:
        shape = extended_shape
    else:
        shape = M.get_shape()

    rows, cols = M.nonzero()
    data = np.squeeze(np.asarray(M[rows, cols]))
    M_sp = cvxopt.spmatrix(data, rows, cols, shape)

    return M_sp



def first_nonzero(mask, axis=0, invalid_val=-1):
    """ """
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)



def last_nonzero(mask, axis=0, invalid_val=-1):
    """ """
    val = mask.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)



def compute_snr(x_values, noise_std):
    """ """
    return x_values.pow(2).mean().div(noise_std ** 2)



def compute_psnr_from_mse(mse, pixel_max=255):
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * math.log10(pixel_max / math.sqrt(mse))

    return psnr



def compute_mse_psnr(x_values, x_values_hat, pixel_max=255):
    """ """
    mse = ((x_values - x_values_hat)**2).mean().item()
    psnr = compute_psnr_from_mse(mse, pixel_max)

    return mse, psnr



def json_load(json_filename):
    """ """
    try:
        with open(json_filename) as jsonfile:
            results_dict = json.load(jsonfile)
    except FileNotFoundError:
        print(f'File {json_filename} not found...')
        raise

    return results_dict



def json_dump(results_dict, json_filename):
    """ """
    try:
        with open(json_filename, 'w') as jsonfile:
            json.dump(results_dict, jsonfile, sort_keys=False, indent=4)
    except FileNotFoundError:
        print(f'File {json_filename} not found...')
        raise



def set_attributes(obj, names):
    """ set object attributes specified by names, whose values
    are in params dictionary else use default_values in struct_default_values.py
    """
    if not isinstance(names, list):
        raise ValueError(f'names is of type {type(names)} and not list...')

    if not hasattr(obj, 'params'):
        raise ValueError(f'object {obj} does have a "params" attribute...')

    if not isinstance(obj.params, dict):
        raise ValueError(f'params is of type {type(obj.params)} and not dictionary...')

    for name in names:
        assert isinstance(name, str), f'{name} is not string.'
        if name in obj.params:
            setattr(obj, name, obj.params[name])
        else:
            try:
                setattr(obj, name, default_values[name])
            except KeyError:
                print(f'{name} not in default_values dictionary')
                raise



def add_date_to_filename(filename):
    """ """
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H:%M")
    new_filename = '_'.join([filename, dt_string])

    return new_filename



def print_algorithm_details(lat):
    # Initialize operators
    placeholder_input = torch.tensor([[0., 0]])
    op = Operators(lat, placeholder_input)
    # convert z, L, H to np.float64 (simplex requires this)
    L_mat_sparse = op.L_mat_sparse.astype(np.float64)
    z = lat.flattened_C

    # compute ||Lz||_0 and ||Lz||_1
    L_z = L_mat_sparse.dot(z.numpy())
    L_z_zero_idx = np.where(np.absolute(L_z) <= algorithm.Algorithm.eps)[0]
    sparsity = L_z.shape[0] - L_z_zero_idx.shape[0]

    fraction_zero = 1.
    if L_z.shape[0] != 0:
        fraction_zero = L_z_zero_idx.shape[0] / L_z.shape[0]
    percentage_nonzero = (100. - fraction_zero * 100)
    print('\nPercentage_nonzero: {:.3f}'.format(percentage_nonzero))

    htv_loss = np.linalg.norm(L_z, ord=1)
    print(f'htv loss: {htv_loss}')
