# Define default values of parameters and desired structure for the
# parameters dictionary

# Example of default_values dictionary:
# default_values = {'combination' : None,
#                   'num_epochs' : 300,
#                   'train' : False}

default_values = { 'method'          : 'htv',
                   'log_dir'         : 'output',
                   'model_name'      : 'htv',
                   'verbose'         : False,
                   'dataset_name'    : 'quad_top_planes',
                   'only_vertices'   : False,
                   'num_train'       : 650,
                   'valid_fraction'  : 0.2,
                   'add_noise'       : False,
                   'noise_ratio'     : 0.05,
                   'seed'            : -1,
                   'data_dir'        : './data',
                   'lsize'           : 60,
                   'C_init'          : 'zero',
                   'num_iter'        : 1,
                   'admm_iter'       : 60000,
                   'lmbda'           : 0.01,
                   'no_linear'       : False,
                   'reduced'         : False,
                   'pos_constraint'  : False,
                   'no_simplex'      : False,
                   'sigma_rule'      : 'same',
                   'eps'             : 1,
                   'plot'            : False,
                   'html'            : False,
                   # neural net
                   'net_model'       : 'fcnet2d',
                   'hidden'          : 20,
                   'batch_size'      : 64,
                   'num_workers'     : 4,
                   'weight_decay'    : 1e-4,
                   'milestones'      : [750, 900],
                   'num_epochs'      : 1000,
                   'log_step'        : None,
                   'valid_log_step'  : None,
                   'ckpt_filename'   : None,
                   'restore'         : False, # restore model
                   'ckpt_nmax_files' : -1, # max number of saved *_net_*.ckpt
                                            # checkpoint files at a time. Set to -1 if not restricted. '
                   'device'          : 'cuda:0'
                 }


# This tree defines the strcuture of self.params in the Project() class.
# if it is desired to keep an entry in the first level that is also a leaf of deeper
# levels of the structure, this entry should be added to the first level too
# (as done for 'dataloader')

structure = {   'verbose'    : None,
                'plot'       : None,
                'model_name' : None,
                'log_dir'    : None,
                'num_iter'   : None,
                'data' :
                    {'dataset_name'   : None,
                     'only_vertices'  : None,
                     'num_train'      : None,
                     'valid_fraction' : None,
                     'add_noise'      : None,
                     'noise_ratio'    : None,
                     'no_linear'      : None,
                     'seed'           : None,
                     'verbose'        : None
                    },
                'lattice' :
                    {'lsize'       : None,
                     'C_init'      : None,
                     'verbose'     : None
                     },
                'plots' :
                    {'html'    : None,
                     'log_dir' : None
                    },
                'algorithm' :
                    {'num_iter'        : None,
                     'admm_iter'       : None,
                     'lmbda'           : None,
                     'reduced'         : None,
                     'pos_constraint'  : None,
                     'no_simplex'      : None,
                     'sigma_rule'      : None,
                     'verbose'         : None,
                     'model_name'      : None,
                     },
                'rbf' :
                    {'eps' : None,
                     'lmbda': None
                     },
                # neural net
                'model':
                    {'hidden' : None},
                'dataloader':
                    {'batch_size'  : None,
                     'num_workers' : None,
                     }
            }
