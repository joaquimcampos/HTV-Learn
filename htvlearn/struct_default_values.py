"""
Define default values of parameters and desired structure for the
parameters dictionary.
"""

default_values = {
    'method': 'neural_net',
    'lmbda': 0.01,
    'no_htv': False,
    # logs-related
    'log_dir': 'output',
    'model_name': 'htv-output',
    # data
    'dataset_name': 'cut_face_gaps',
    'num_train': 5000,
    'data_dir': './data',
    'test_as_valid': False,
    'noise_ratio': 0.,
    'seed': -1,
    'valid_fraction': 0.2,
    # Lattice
    'lsize': 256,
    'C_init': 'zero',  # coefficient initialization
    # HTV minimization algorithm
    'admm_iter': 100000,
    'simplex': False,
    # RBF
    'eps': 15,  # kernel size
    # Neural Net
    'net_model': 'relufcnet2d',
    'htv_mode': 'finite_diff_differential',
    'device': 'cuda:0',
    'num_hidden_layers': 5,
    'num_hidden_neurons': 50,
    'weight_decay': 1e-6,
    'milestones': [850, 920],
    'num_epochs': 1000,
    'log_step': None,
    'valid_log_step': None,
    'ckpt_filename': None,
    'restore': False,  # restore model
    'ckpt_nmax_files': 3,  # max number of saved *_net_*.ckpt
    # checkpoint files at a time. Set to -1 if not restricted. '
    # dataloader
    'batch_size': 100,
    'num_workers': 4,  # number of subprocesses to use for data loading.
    # verbose
    'verbose': False,
}

# epsilon to assess sparsity
SPARSITY_EPS = 1e-4

# This tree defines the strcuture of self.params in the Project() class.
# if it is desired to keep an entry in the first level that is also a
# leaf of deeper levels of the structure, this entry should be added to
# the first level too (e.g. as done for 'log_dir')
structure = {
    'lmbda': None,
    'log_dir': None,
    'model_name': None,
    'verbose': None,
    'data': {
        'dataset_name': None,
        'num_train': None,
        'data_dir': None,
        'test_as_valid': None,
        'noise_ratio': None,
        'seed': None,
        'valid_fraction': None,
        'verbose': None
    },
    'lattice': {
        'lsize': None,
        'C_init': None,
        'verbose': None
    },
    'algorithm': {
        'lmbda': None,
        'model_name': None,
        'admm_iter': None,
        'simplex': None,
        'verbose': None
    },
    'rbf': {
        'eps': None,
        'lmbda': None
    },
    # neural net
    'model': {
        'num_hidden_layers': None,
        'num_hidden_neurons': None
    },
    'dataloader': {
        'batch_size': None,
        'num_workers': None
    }
}
