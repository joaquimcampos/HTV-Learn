"""Module with project utilities"""

import os
import sys
import argparse
import copy
import math
from datetime import datetime
from contextlib import contextmanager

import torch
import numpy as np
import json
import cvxopt

from htvlearn.operators import Operators
from htvlearn.algorithm import Algorithm


class ArgCheck():
    """Class for input argument verification"""

    @staticmethod
    def p_int(value):
        """
        Check if int value got from argparse is positive
        and raise error if not.
        """
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                f'{value} is an invalid positive int value')
        return ivalue

    @staticmethod
    def p_float(value):
        """
        Check if float value got from argparse is positive
        and raise error if not.
        """
        ivalue = float(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                f'{value} is an invalid positive float value')
        return ivalue

    @staticmethod
    def nn_float(value):
        """
        Check if float value got from argparse is non-negative
        and raise error if not.
        """
        ivalue = float(value)
        if not np.allclose(np.clip(ivalue, -1.0, 0.0), 0.0):
            raise argparse.ArgumentTypeError(
                f'{value} is an invalid non-negative float value')
        return ivalue


def size_str(input):
    """
    Return a string with the size of the input tensor

    Args:
        input (torch.Tensor)
    Returns:
        out_str (str)
    """
    out_str = '[' + ', '.join(str(i) for i in input.size()) + ']'
    return out_str


def dict_recursive_merge(params_root, merger_root, base=True):
    """
    Recursively merges merger_root into params_root giving precedence
    to the second dictionary as in z = {**x, **y}

    Args:
        params_root (dict)
        merger_root (dict):
            dictionary with parameters to be merged into params root;
            overwrites values of params_root for the same keys and level.
        base (bool):
            True for the first level of the recursion
    """
    if base:
        assert isinstance(params_root, dict)
        assert isinstance(merger_root, dict)
        merger_root = copy.deepcopy(merger_root)

    if merger_root:  # non-empty dict
        for key, val in merger_root.items():
            if isinstance(val, dict) and key in params_root:
                merger_root[key] = dict_recursive_merge(params_root[key],
                                                        merger_root[key],
                                                        base=False)

        merger_root = {**params_root, **merger_root}

    return merger_root


def assign_tree_structure(assign_root, structure, base=True):
    """
    Assign a tree structure to dictionary according to structure.
    (see structure variable in default_struct_values.py)

    Args:
        assign_root (dict):
            dictionary to be assigned a tree structure
        base (bool):
            True for the first level of the recursion
    """
    if base:
        assert isinstance(assign_root, dict)
        assert isinstance(structure, dict)
        assign_root = copy.deepcopy(assign_root)
        global assign_orig
        assign_orig = assign_root  # keep the original dict
        global leaves
        leaves = []  # leaves on dict tree levels deeper than base

    if structure:  # non-empty dict
        for key, val in structure.items():
            if isinstance(val, dict):
                assign_root[key] = {}
                assign_root[key] = assign_tree_structure(assign_root[key],
                                                         structure[key],
                                                         base=False)
                if len(assign_root[key]) < 1:
                    # do not have empty dictionaries in assign_root
                    del assign_root[key]
            else:
                assert val is None, 'leaf values in structure should be None'
                if key in assign_orig:
                    assign_root[key] = assign_orig[key]
                    if not base:
                        leaves.append(key)

    # delete duplicated leaves in base root if they are not in first level of
    # structure dict
    if base:
        for key in leaves:
            if key not in structure and key in assign_root:
                del assign_root[key]

    return assign_root


def flatten_structure(assign_root, base=True):
    """
    Reverses the operation of assign_tree_structure()
    by flattening input dictionary.

    Args:
        assign_root (dict):
            dictionary to be flattened
        base (bool):
            True for the first level of the recursion
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


def csr_to_spmatrix(M, extended_shape=None):
    """
    Convert scipy.sparse.csr_matrix to scipy.sparse.spmatrix.

    Args:
        M (scipy.sparse.csr_matrix)
        extended_shape:
            extended shape for matrix.

    Returns:
        M_sp (scipy.sparse.spmatrix)
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


def compute_snr(x_values, mse):
    """
    Compute snr from gtruth values and mse of prediction.

    Args:
        x_values (torch.Tensor):
            gtruth values for a given input.
        mse (float):
            MSE(x_values, x_values_hat), where x_values_hat are the
            predictions for the same input.

    Returns:
        snr (dB)
    """
    gt_energy = (x_values ** 2).mean().item()
    snr = 10 * math.log10(gt_energy / mse)

    return snr


def compute_mse_snr(x_values, x_values_hat):
    """
    Compute mse and snr from gtruth values and predictions.

    Args:
        x_values (torch.Tensor):
            gtruth values at a given location.
        x_values_hat (torch.Tensor):
            predictions for the same input.

    Returns:
        mse (float)
        snr (db)
    """
    mse = ((x_values - x_values_hat)**2).mean().item()
    snr = compute_snr(x_values, mse)

    return mse, snr


def json_load(json_filename):
    """
    Load a json file.

    Args:
        json_filename (str):
            Path of the .json file with results.
    Returns:
        results_dict (dict):
            dictionary with results stored in the json file.
    """
    try:
        with open(json_filename) as jsonfile:
            results_dict = json.load(jsonfile)
    except FileNotFoundError:
        print(f'File {json_filename} not found...')
        raise

    return results_dict


def json_dump(results_dict, json_filename):
    """
    Save results in a json file.

    Args:
        results_dict (dict):
            dictionary with the results to be stored in the json file.
        json_filename (str):
            Path of the .json file where results are stored.
    """
    try:
        with open(json_filename, 'w') as jsonfile:
            json.dump(results_dict, jsonfile, sort_keys=False, indent=4)
    except FileNotFoundError:
        print(f'File {json_filename} not found...')
        raise


def add_date_to_filename(filename):
    """
    Add current date to a filename.

    Args:
        filename (str)
    Returns:
        new_filename (str):
            filename with added date.
    """
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H:%M")
    new_filename = '_'.join([filename, dt_string])

    return new_filename


def frange(start, stop, step, n=None):
    """
    Return a WYSIWYG series of float values that mimic range behavior
    by excluding the end point and not printing extraneous digits beyond
    the precision of the input numbers (controlled by n and automatically
    detected based on the string representation of the numbers passed).

    EXAMPLES
    ========

    non-WYSIWYS simple list-comprehension

    >>> [.11 + i*.1 for i in range(3)]
    [0.11, 0.21000000000000002, 0.31]

    WYSIWYG result for increasing sequence

    >>> list(frange(0.11, .33, .1))
    [0.11, 0.21, 0.31]

    and decreasing sequences

    >>> list(frange(.345, .1, -.1))
    [0.345, 0.245, 0.145]

    To hit the end point for a sequence that is divisibe by
    the step size, make the end point a little bigger by
    adding half the step size:

    >>> dx = .2
    >>> list(frange(0, 1 + dx/2, dx))
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    """
    if step == 0:
        raise ValueError('step must not be 0')
    # how many decimal places are showing?
    if n is None:
        n = max([
            0 if '.' not in str(i) else len(str(i).split('.')[1])
            for i in (start, stop, step)
        ])
    if step * (stop - start) > 0:  # a non-null incr/decr range
        if step < 0:
            for i in frange(-start, -stop, -step, n):
                yield -i
        else:
            steps = round((stop - start) / step)
            while round(step * steps + start, n) < stop:
                steps += 1
            for i in range(steps):
                yield round(start + i * step, n)


@contextmanager
def silence_stdout():
    """
    Contextmanager for silencing stdout.

    Usage:
        with silence_stdout():
            code_block that cannot print to stdout.
    """
    new_target = open(os.devnull, "w")
    old_target, sys.stdout = sys.stdout, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


def get_sparsity(lat):
    """Return percentage of nonzero slopes in lat.

    Args:
        lat (Lattice): instance of Lattice class
    """
    # Initialize operators
    placeholder_input = torch.tensor([[0., 0]])
    op = Operators(lat, placeholder_input)
    # convert z, L, H to np.float64 (simplex requires this)
    L_mat_sparse = op.L_mat_sparse.astype(np.float64)
    z = lat.flattened_C

    # # compute ||Lz||_1
    # htv_loss = np.linalg.norm(L_z, ord=1)
    # print('HTV: {:.2f}'.format(htv_loss))

    # compute ||Lz||_0
    L_z = L_mat_sparse.dot(z.numpy())
    L_z_zero_idx = np.where(np.absolute(L_z) <= Algorithm.eps)[0]

    fraction_zero = 1.
    if L_z.shape[0] != 0:
        fraction_zero = L_z_zero_idx.shape[0] / L_z.shape[0]
    percentage_nonzero = (100. - fraction_zero * 100)

    return percentage_nonzero
