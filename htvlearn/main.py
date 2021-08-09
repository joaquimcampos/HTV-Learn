#!/usr/bin/env python3

import argparse

from htvlearn.htv_manager import HTVManager
from htvlearn.nn_manager import NNManager
from htvlearn.rbf_manager import RBFManager
from htvlearn.htv_utils import ArgCheck, assign_structure_recursive
from htvlearn.struct_default_values import structure, default_values


# Fix the Acknowledgements (new Grant)
def get_arg_parser():
    """ Define argument parser """

    parser = argparse.ArgumentParser(
        description='HTV for Supervised Learning and '
        'Measuring Model Complexity.')

    parser.add_argument(
        '--method',
        choices=['htv', 'neural_net', 'rbf'],
        type=str,
        help='Method to solve problem. '
             f'(default: {default_values["method"]})')

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
        help=f'(default: {default_values["model_name"]})')

    parser.add_argument(
        '--lmbda',
        metavar='[FLOAT,>=0]',
        type=ArgCheck.nn_float,
        help='Regularization weight for HTV/RBF. '
             f'(default: {default_values["lmbda"]})')

    parser.add_argument(
        '--htv_mode',
        type=str,
        choices=['finite_diff_differential', 'exact_differential'],
        help='How to compute gradients for HTV. '
             f'(default: {default_values["htv_mode"]})')

    parser.add_argument(
        '--no_htv',
        action='store_true',
        help='If True, do not compute HTV of model. '
             f'(default: {default_values["no_htv"]})')

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help=f'Print more info. (default: {default_values["verbose"]})')

    # data
    dataset_choices = {
        'cpwl', 'face', 'face_gaps', 'cut_face', 'cut_face_gaps'
    }
    parser.add_argument(
        '--dataset_name',
        choices=dataset_choices,
        type=str,
        help=f'(default: {default_values["dataset_name"]})')

    parser.add_argument(
        '--num_train',
        metavar='[INT>0]',
        type=ArgCheck.p_int,
        help='number of training samples. '
             f'(default: {default_values["num_train"]})')

    parser.add_argument(
        '--valid_fraction',
        metavar='[FLOAT,(0,1)]',
        type=ArgCheck.frac_float,
        help='Fraction of train samples to use for validation. '
             f'(default: {default_values["valid_fraction"]})')

    parser.add_argument(
        '--test_as_valid',
        action='store_true',
        help='Use test as validation. '
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

    # lattice
    parser.add_argument(
        '--lsize',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help=f'Lattice size. (default: {default_values["lsize"]})')

    # algorithm
    parser.add_argument(
        '--num_iter',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help='Number of iterations (grid divisions). '
             f'(default: {default_values["num_iter"]})')

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

    parser.add_argument(
        '--htv_grid',
        metavar='[FLOAT,>=0]',
        type=ArgCheck.nn_float,
        help='Grid size for computing htv for RBF. '
             f'(default: {default_values["htv_grid"]})')

    # neural net
    # model
    parser.add_argument(
        '--net_model',
        type=str,
        choices=['relufcnet2d', 'gelufcnet2d'],
        help='Neural network model to train. '
             f'(default: {default_values["net_model"]})')

    parser.add_argument(
        '--hidden',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help='Number of hidden neurons per layer. '
             f'(default: {default_values["hidden"]})')

    parser.add_argument(
        '--num_hidden_layers',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help='Number of hidden layers. '
             f'(default: {default_values["num_hidden_layers"]})')

    # dataloader
    parser.add_argument(
        '--batch_size',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help='Batch size for training. '
             f'(default: {default_values["batch_size"]})')

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
        '--num_epochs',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help='Number of training epochs. '
             f'(default: {default_values["num_epochs"]})')

    parser.add_argument(
        '--log_step',
        metavar='[INT,>0]',
        type=ArgCheck.p_int,
        help='If None, done at every epoch. '
             f'(default: {default_values["log_step"]})')

    parser.add_argument(
        '--valid_log_step',
        metavar='[INT]',
        type=int,
        help='If None, done halfway and at the end of training. '
             'If negative, done at every epoch. '
             f'(default: {default_values["valid_log_step"]})')

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda:0'],
        help=f'(default: {default_values["device"]})')

    return parser


def verify_params(params):
    """ Verify parameters (e.g. mutual inclusivity or exclusivity) and
    assign recursive structure to parameters.
    """
    # Check parameters input by user and set default values
    for key, value in default_values.items():
        if key not in params or params[key] is None:
            params[key] = value  # param which was not set by user

    return params


def main_prog(params, isloaded_params=False):
    """ Main program
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
