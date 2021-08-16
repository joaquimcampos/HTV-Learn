#!/usr/bin/env python3

import argparse

from htvlearn.htv_manager import HTVManager
from htvlearn.nn_manager import NNManager
from htvlearn.rbf_manager import RBFManager
from htvlearn.htv_utils import ArgCheck, assign_structure_recursive
from htvlearn.struct_default_values import structure, default_values


# Fix the Acknowledgements (new Grant)
def get_arg_parser():
    """
    Parses command-line arguments.

    The default values are fetched from the 'default_values' dictionary.
    (see struct_default_values.py)

    Returns:
        parser (argparse.ArgumentParser)
    """

    parser = argparse.ArgumentParser(
        description='HTV for Supervised Learning and '
        'Measuring Model Complexity.')

    parser.add_argument(
        '--method',
        choices=['htv', 'neural_net', 'rbf'],
        type=str,
        help='Method used for learning. '
        f'(default: {default_values["method"]})')

    parser.add_argument(
        '--lmbda',
        metavar='[FLOAT,>=0]',
        type=ArgCheck.nn_float,
        help='Regularization weight for HTV/RBF. '
        f'(default: {default_values["lmbda"]})')

    parser.add_argument(
        '--no_htv',
        action='store_true',
        help='If True, do not compute HTV of model. '
        f'(default: {default_values["no_htv"]})')

    # logs-related
    parser.add_argument(
        '--log_dir',
        metavar='[STR]',
        type=str,
        help='Directory for saving results. '
        f'(default: {default_values["log_dir"]})')

    parser.add_argument(
        '--model_name',
        metavar='[STR]',
        type=str,
        help='Directory under --log_dir where checkpoints are saved. '
        f'(default: {default_values["model_name"]})')

    # data
    dataset_choices = {
        'pyramid',
        'quad_top_planes',
        'face',
        'cut_face_gaps'
    }
    parser.add_argument(
        '--dataset_name',
        choices=dataset_choices,
        type=str,
        help='Dataset to train/test on. '
        f'(default: {default_values["dataset_name"]})')

    parser.add_argument(
        '--num_train',
        metavar='[INT>0]',
        type=ArgCheck.p_int,
        help='Number of training samples. '
        f'(default: {default_values["num_train"]})')

    parser.add_argument(
        '--test_as_valid',
        action='store_true',
        help='Use test set as validation. '
        f'(default: {default_values["test_as_valid"]})')

    parser.add_argument(
        '--noise_ratio',
        metavar='[FLOAT,>0]',
        type=ArgCheck.nn_float,
        help='Add random noise sampled from zero-mean '
        'normal distribution with std = noise_ratio * max_z. '
        f'(default: {default_values["noise_ratio"]})')

    parser.add_argument(
        '--seed',
        metavar='INT',
        type=int,
        help='Seed for random generation of dataset. '
        'If negative, set no seed. '
        f'(default: {default_values["seed"]})')

    # Lattice
    parser.add_argument(
        '--lsize',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help=f'Lattice size. (default: {default_values["lsize"]})')

    # HTV minimization algorithm
    parser.add_argument(
        '--admm_iter',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help='Number of admm iterations. '
        f'(default: {default_values["admm_iter"]})')

    parser.add_argument(
        '--sigma_rule',
        type=str,
        choices=['constant', 'same', 'inverse'],
        help='Rule to set step size for g proximal in admm; '
        'either constant or same or inverse of lmbda. '
        f'(default: {default_values["sigma_rule"]})')

    # RBF
    parser.add_argument(
        '--eps',
        metavar='[FLOAT>0]',
        type=ArgCheck.p_float,
        help='RBF shape parameter (higher -> more localized). '
        f'(default: {default_values["eps"]})')

    # neural net
    parser.add_argument(
        '--net_model',
        type=str,
        choices=['relufcnet2d', 'gelufcnet2d'],
        help='Neural network model to train. '
        f'(default: {default_values["net_model"]})')

    parser.add_argument(
        '--htv_mode',
        type=str,
        choices=['finite_diff_differential', 'exact_differential'],
        help='Mode for computing gradients for HTV. '
        'Only relevant if --no_htv is not set. '
        f'(default: {default_values["htv_mode"]})')

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda:0'],
        help=f'(default: {default_values["device"]})')

    parser.add_argument(
        '--num_epochs',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help='Number of training epochs. '
        f'(default: {default_values["num_epochs"]})')

    parser.add_argument(
        '--num_hidden_layers',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help='Number of hidden layers. '
        f'(default: {default_values["num_hidden_layers"]})')

    parser.add_argument(
        '--num_hidden_neurons',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help='Number of hidden neurons per layer. '
        f'(default: {default_values["num_hidden_neurons"]})')

    parser.add_argument(
        '--weight_decay',
        metavar='[FLOAT,>0]',
        type=ArgCheck.p_float,
        help='Weight decay to be applied. '
        f'(default: {default_values["weight_decay"]})')

    parser.add_argument(
        '--milestones',
        metavar='[INT,>0]',
        nargs='+',
        type=ArgCheck.p_int,
        help='List of epochs in which learning rate is decreased by 10. '
        f'(default: {default_values["milestones"]})')

    parser.add_argument(
        '--log_step',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help='Train log step in number of batches. '
        'If None, done at every epoch. '
        f'(default: {default_values["log_step"]})')

    parser.add_argument(
        '--valid_log_step',
        metavar='[INT]',
        type=int,
        help='Train log step in number of batches. '
        'If None, done halfway and at the end of training. '
        'If negative, done at every epoch. '
        f'(default: {default_values["valid_log_step"]})')

    # dataloader
    parser.add_argument(
        '--batch_size',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help='Batch size for training. '
        f'(default: {default_values["batch_size"]})')

    # verbose output
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help=f'Print more info. (default: {default_values["verbose"]})')

    return parser


def verify_params(params):
    """
    Verifies the parameters (e.g. checks for mutual inclusivity/exclusivity).

    If not specified by the user via the command-line, a parameter
    gets the default value from the 'default_values' dictionary.
    (see struct_default_values.py).

    Args:
        params (dict):
            dictionary with parameter names (keys) and values.

    Returns:
        params (dict):
            dictionary with all parameters. If not specified by the user,
            a parameter gets the default value in the 'default_values'
            dictionary.
    """
    # Check parameters input by user and set default values
    for key, value in default_values.items():
        if key not in params or params[key] is None:
            params[key] = value  # param which was not set by user

    return params


def main_prog(params, isloaded_params=False):
    """
    Main program that initializes the Manager with the parameters
    and runs the training.

    It first verifies the params dictionary, if necessary.

    'params' is then assigned a tree structure according to the
    'structure' dictionary (see struct_default_values.py).

    Finally, 'params' is used to instantiate a Manager() object
    and the training is ran.

    Args:
        params (dict):
            dictionary with the parameters from the parser.
        isloaded_params :
            True if params were loaded from ckpt (no need to verify) and
            are flattened (htv_utils), i.e., don't have a tree structure.
    """
    if not isloaded_params:
        params = verify_params(params)

    # assign recursive structure to params according
    # to structure in struct_default_values.py
    params = assign_structure_recursive(params, structure)

    if params['method'] == 'htv':
        manager = HTVManager(params)
    elif params['method'] == 'neural_net':
        manager = NNManager(params)
    elif params['method'] == 'rbf':
        manager = RBFManager(params)

    print('\n==> Training model.')
    manager.train()


if __name__ == "__main__":

    # parse arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    params = vars(args)  # transform to dictionary

    main_prog(params)
