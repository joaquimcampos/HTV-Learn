# Define default values of parameters and desired structure for the
# parameters dictionary

# Example of default_values dictionary:
# default_values = {'combination' : None,
#                   'num_epochs' : 300,
#                   'train' : False}

default_values = {
    'method': 'neural_net',
    'log_dir': 'output',
    'model_name': 'htv-complexity',
    'lmbda': 0.01,
    'htv_mode': 'finite_diff_differential',
    'no_htv': False,
    'dataset_name': 'cpwl',
    'num_train': 1000,
    'data_dir': './data',
    'valid_fraction': 0.2,
    'test_as_valid': False,
    'noise_ratio': 0.,
    'seed': -1,
    'verbose': False,
    # lattice
    'lsize': 60,
    'C_init': 'zero',
    # algorithm
    'num_iter': 1,
    'admm_iter': 200000,
    'sigma_rule': 'constant',
    # rbf
    'eps': 5,  # rbf. kernel size
    'htv_grid': 0.0005,
    # neural net
    'net_model': 'relufcnet2d',
    'hidden': 50,
    'num_hidden_layers': 5,
    'batch_size': 50,
    'num_workers': 4,
    'weight_decay': 1e-6,
    'milestones': [540, 580],
    'num_epochs': 600,
    'log_step': None,
    'valid_log_step': None,
    'ckpt_filename': None,
    'restore': False,  # restore model
    'ckpt_nmax_files': 3,  # max number of saved *_net_*.ckpt
    # checkpoint files at a time. Set to -1 if not restricted. '
    'device': 'cuda:0'
}

# This tree defines the strcuture of self.params in the Project() class.
# if it is desired to keep an entry in the first level that is also a leaf of
# deeper levels of the structure, this entry should be added to the first
# level too (as done for 'dataloader')

structure = {
    'log_dir': None,
    'model_name': None,
    'lmbda': None,
    'verbose': None,
    'data': {
        'dataset_name': None,
        'num_train': None,
        'data_dir': None,
        'valid_fraction': None,
        'test_as_valid': None,
        'noise_ratio': None,
        'seed': None,
        'verbose': None
    },
    'lattice': {
        'lsize': None,
        'C_init': None,
        'verbose': None
    },
    'algorithm': {
        'model_name': None,
        'lmbda': None,
        'num_iter': None,
        'admm_iter': None,
        'sigma_rule': None,
        'verbose': None
    },
    'rbf': {
        'eps': None,
        'lmbda': None
    },
    # neural net
    'model': {
        'hidden': None,
        'num_hidden_layers': None
    },
    'dataloader': {
        'batch_size': None,
        'num_workers': None
    }
}
