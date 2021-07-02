#!/usr/bin/env python3

import argparse

from htv_manager import HTVManager
from nn_manager import NNManager
from rbf_manager import RBFManager
from master_project import MasterProject
from htv_utils import ArgCheck, assign_structure_recursive
from struct_default_values import structure, default_values


def get_arg_parser():
    """ Define argument parser """

    parser = argparse.ArgumentParser(description='Variational HTV Project.')

    parser.add_argument('--method', choices=['htv', 'neural_net', 'rbf'], type=str,
                    help=f'Method to solve problem. Default: {default_values["method"]}.')

    parser.add_argument('--log_dir', metavar='STR', type=str,
                    help=f'Directory for saving results. Default: {default_values["log_dir"]}.')
    parser.add_argument('--model_name', metavar='STR', type=str,
                    help=f'Default: {default_values["model_name"]}')
    parser.add_argument('--verbose', '-v', action='store_true',
                    help='Print more info.')

    # data
    dataset_choices={'pyramid', 'pyramid_ext', 'quad_top_planes', 'face', 'noisy_linear'}
    parser.add_argument('--dataset_name', choices=dataset_choices, type=str,
                    help=f'Default: {default_values["dataset_name"]}.')

    parser.add_argument('--only_vertices', action='store_true',
                        help='Only give vertices as training data.')

    parser.add_argument('--num_train', metavar='INT>0', type=ArgCheck.p_int,
                        help='number of training samples. '
                            f'Default: {default_values["num_train"]}.')
    parser.add_argument('--valid_fraction', metavar='FLOAT,[0,1]', type=ArgCheck.frac_float,
                        help='Fraction of train samples to use for validation. '
                            f'Default: {default_values["valid_fraction"]}.')
    parser.add_argument('--add_noise', action='store_true', help='Add noise to dataset.')
    parser.add_argument('--noise_ratio', metavar='FLOAT,>0', type=ArgCheck.nn_float,
                        help='Add random noise sampled from zero-mean normal distribution with '
                            f'std=noise_ratio*max_z. Default: {default_values["noise_ratio"]}.')
    parser.add_argument('--seed', metavar='INT,>=0', type=ArgCheck.nn_int,
                        help=f'Seed for random generation of dataset. Default: {default_values["seed"]}.')

    # lattice
    parser.add_argument('--lsize', metavar='INT,>0', type=ArgCheck.p_int,
                        help='Lattice size without possible padding. '
                            'Note: each element of the data should fall in [lsize+1, lsize-1]. '
                            f'Default: {default_values["lsize"]}.')
    parser.add_argument('--C_init', type=str, choices=['zero', 'normal'],
                        help=f'Initalization for lattice values. Default: {default_values["C_init"]}.')

    # algorithm
    parser.add_argument('--num_iter', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Number of iterations (grid divisions). Default: {default_values["num_iter"]}.')
    parser.add_argument('--admm_iter', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Number of admm iterations. Default: {default_values["admm_iter"]}.')
    parser.add_argument('--lmbda', metavar='FLOAT,>=0', type=ArgCheck.nn_float,
                        help=f'Regularization weight for htv or rbf. Default: {default_values["lmbda"]}.')
    parser.add_argument('--no_linear', action='store_true', help='Do not add linear term to dataset.')
    parser.add_argument('--reduced', action='store_true', help='Solve reduced problem.')
    parser.add_argument('--pos_constraint', action='store_true', help='Add positivity constraint.')
    parser.add_argument('--no_simplex', action='store_true', help='Do not perform simplex after multires_admm.')
    parser.add_argument('--sigma_rule', type=str, choices=['constant', 'same', 'inverse'],
                        help='Rule to set step size for g proximal in admm;'
                            'either constant or same or inverse of lmbda.'
                            f'Default: {default_values["sigma_rule"]}.')

    # RBF
    parser.add_argument('--eps', metavar='FLOAT>0', type=ArgCheck.p_float,
                        help='RBF shape parameter (higher -> more localized). '
                        f'Default: {default_values["eps"]}.')

    # plots
    parser.add_argument('--plot', action='store_true', help='Plot results.')
    parser.add_argument('--hmtl', action='store_true', help="Save plots to as html.")

    ### neural net
    # model
    parser.add_argument('--net_model', type=str, choices=['fcnet2d', 'deepfcnet2d'],
                        help=f'Neural network model to train. Default: {default_values["net_model"]}.')
    parser.add_argument('--hidden', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Number of hidden neurons in network. Default: {default_values["hidden"]}.')

    # dataloader
    parser.add_argument('--batch_size', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Number of hidden neurons in network. Default: {default_values["batch_size"]}.')

    parser.add_argument('--weight_decay', metavar='FLOAT,>0', type=ArgCheck.p_float,
                        help=f'Weight decay to be applied. Default: {default_values["weight_decay"]}.')
    parser.add_argument('--milestones', metavar='LIST[INT,>0]', nargs='+', type=ArgCheck.p_int,
                        help=f'Default: {default_values["milestones"]}.')
    parser.add_argument('--num_epochs', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Default: {default_values["num_epochs"]}.')
    parser.add_argument('--log_step', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Default: {default_values["log_step"]}.')
    parser.add_argument('--valid_log_step', metavar='INT,>0', type=ArgCheck.p_int,
                        help=f'Default: {default_values["valid_log_step"]}.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0'],
                        help=f'Default: {default_values["device"]}.')


    return parser



def verify_params(params):
    """ Verify parameters (e.g. mutual inclusivity or exclusivity) and
    assign recursive structure to parameters.
    """
    # Check parameters input by user and set default values
    for key, value in default_values.items():
        if key not in params or params[key] is None:
            params[key] = value # param which was not set by user

    # check parameter dependecies
    if params['only_vertices'] and ('planes' not in params['dataset_name']):
        print('Can only set --only_vertices for planes datasets.')
        raise ValueError

    return params



def main_prog(params, isloaded_params=False):
    """ Main program
    """
    if not isloaded_params:
        params = verify_params(params)

    # assign recursive structure to params according to structure in struct_default_values.py
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
    params = vars(args) # transform to dictionary

    main_prog(params)
