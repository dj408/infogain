"""
Script to run cross-validation for (P and/or
spectral wavelet) MFCN or GNN baseline models, 
on various learning tasks. On completion, pickles 
final metrics records and train timing dictionary.
"""
import sys
sys.path.insert(0, '../')
import os
import time
import pickle
import argparse

import torch
import numpy as np
from numpy.random import RandomState
import pandas as pd

import MFCN.mfcn as mfcn
import LEGS.LEGS_module_DJedit as legs
import baselines.gnn as gnn
import utilities as u
import dataset_creation as dc
import data_utilities as du
import cv


"""
clargs

! Note: ensure 'spectral' or 'p' is the last substring when 
model bool clargs are split by '_'. If an alternative to
'mcn_spectral' is introduced, a more sophisticated
tagging scheme will need to be coded where MFCN model kwargs
are assigned and parsed by downstream functions / spectral filters 
are precomputed in `dataset_creation.py`.
"""
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--machine', default='desktop', type=str, 
                    help='key for the machine in use (default: desktop): see args_template.py')
# parser.add_argument('--experiment_design', default="cv", type=str, 
#                     help="Experimental design: 'cv' (default; cross-validation) or 'tvt' (train/valid/test)")
# MFCNs with wavelet filters
parser.add_argument('--mfcn_p', default=False, action='store_true')
parser.add_argument('--p_wavelet_scales_type', default='dyadic', type=str,
                   help="P-wavelet type: 'dyadic' (default) or 'custom' or 'precalc'")
parser.add_argument('--use_k_drop', default=False, action='store_true')
parser.add_argument('--mfcn_spectral', default=False, action='store_true')

# MCNs with simple, non-wavelet filter
parser.add_argument('--mcn_p', default=False, action='store_true')
parser.add_argument('--mcn_spectral', default=False, action='store_true')

# Comparative scattering models
parser.add_argument('--legs', default=False, action='store_true')

# Baseline GNNs
parser.add_argument('--gcn', default=False, action='store_true')
parser.add_argument('--sage', default=False, action='store_true')
parser.add_argument('--gin', default=False, action='store_true')
parser.add_argument('--gat', default=False, action='store_true')

parser.add_argument('-d', '--dataset', default=None, type=str, 
                    help='dataset key, e.g. dd, nci1, etc.')
parser.add_argument('-u', '--multi_dataset_dir', default=None, type=str, 
                    help='dataset directory (default: None)')
parser.add_argument('--use_args_excl_dataset_indices', default=False, action='store_true')

parser.add_argument('-i', '--max_n_datasets', default=None, type=int, 
                    help='for testing: max number of multi-datasets (in --multi_dataset_dir) to actually use (default: None, to use all)')
parser.add_argument('-j', '--J', default=4, type=int, 
                    help='Reference number of wavelets and diffusion steps, e.g. J=4 gives T = 2**J = 16 (default: 4)')

parser.add_argument('-f', '--n_folds', default='10', type=int, 
                    help='number of cross-validation folds to run (default: 10)')
parser.add_argument('-e', '--n_epochs', default='10000', type=int, 
                    help='max num. of epochs to run in each fold (default: 10000)')
parser.add_argument('-b', '--burn_in', default='4', type=int, 
                    help='min num. of epochs to run before enforcing early stopping (default: 4)')
parser.add_argument('-p', '--patience', default='32', type=int, 
                    help='if args.STOP_RULE is no_improvement, max num. of epochs'
                    'without improvement in validation loss (default: 32)')
parser.add_argument('-l', '--learn_rate', default='0.003', type=float, 
                    help='learning rate hyperparameter (default: 0.003)')
parser.add_argument('-t', '--batch_size', default='32', type=int, 
                    help='train set minibatch size hyperparameter (default: 32)')
parser.add_argument('-v', '--verbosity', default='0', type=int, 
                    help='integer controlling volume of print output during execution')
clargs = parser.parse_args()


"""
args
"""
# interpret clargs
dataset_key = clargs.dataset.lower()
pyg_data_list_filepaths = [None] # placeholder
model_bools_keys = (
    (clargs.mfcn_p, 'mfcn_p'),
    (clargs.mcn_p, 'mcn_p'),
    (clargs.mfcn_spectral, 'mfcn_spectral'),
    (clargs.mcn_spectral, 'mcn_spectral'),
    (clargs.gcn, 'gcn'),
    (clargs.sage, 'sage'),
    (clargs.gin, 'gin'),
    (clargs.gat, 'gat'),
    (clargs.legs, 'legs')
)
model_keys = []
for model_bool, model_key in model_bools_keys:
    if model_bool:
        model_keys.append(model_key)

# import correct task args file
new_subdir = None
if 'nci1' in dataset_key:
    import infogain_testing.nci1s_args as a
elif 'ptc_mr' in dataset_key:
    import infogain_testing.ptc_args as a
elif ('proteins' in dataset_key):
    import infogain_testing.proteins_args as a
elif ('dd' in dataset_key):
    import infogain_testing.dd_args as a
elif ('mutag' in dataset_key):
    import infogain_testing.mutag_args as a
    
# init args (with command-line overrides)
args = a.Args(
    MACHINE=clargs.machine,
    MODEL_NAME='model', # placeholder; renamed in models loop
    # MODEL_SAVE_SUBDIR=model_save_subdir,
    J=clargs.J,
    P_WAVELET_SCALES=clargs.p_wavelet_scales_type,
    USE_PRECALC_P_SCALES="precalc" in clargs.p_wavelet_scales_type,
    USE_K_DROP=clargs.use_k_drop,
    N_FOLDS=clargs.n_folds,
    N_EPOCHS=clargs.n_epochs,
    BURNIN_N_EPOCHS=clargs.burn_in,
    NO_VALID_LOSS_IMPROVE_PATIENCE=clargs.patience,
    LEARN_RATE=clargs.learn_rate,
    # >0 prints epoch-by-epoch stats in train_fn
    VERBOSITY=clargs.verbosity
)
args.set_model_save_dirs(new_subdir=new_subdir)
args.set_batch_sizes(train_size=clargs.batch_size)

if ('nci1' in dataset_key) \
or ('ptc_mr' in dataset_key) \
or ('proteins' in dataset_key) \
or ('dd' in dataset_key) \
or ('mutag' in dataset_key):
    num_nodes_one_graph = None
    model_save_subdir = args.MODEL_SAVE_SUBDIR
    pyg_data_list_filepaths = [f"{args.DATA_DIR}"]

"""
[optional]: only run on a subset of a multi-dataset 
"""
if clargs.max_n_datasets is not None:
    pyg_data_list_filepaths = pyg_data_list_filepaths[:(clargs.max_n_datasets)]
    

"""
cv splits
"""
cv_idxs_unexpanded = cv.get_cv_idxs_for_dataset(args)


"""
training objects' kwargs
"""
base_module_kwargs = {
    'task': args.TASK,
    'target_name': args.TARGET_NAME,
    'device': args.DEVICE
}
metrics_kwargs = {}

# all graph-level classification task: needs MLP classifier head
if ('nci1' in dataset_key) \
or ('ptc_mr' in dataset_key) \
or ('proteins' in dataset_key) \
or ('dd' in dataset_key) \
or ('mutag' in dataset_key):
    node_pool_out_channels = None 
    fc_model_kwargs = {
        'base_module_kwargs': base_module_kwargs,
        'output_dim': args.OUTPUT_DIM,
        'hidden_dims_list': args.NN_HIDDEN_DIMS,
        'use_batch_normalization': args.USE_BATCH_NORMALIZATION,
        'use_dropout': args.MLP_USE_DROPOUT,
        'dropout_p': args.MLP_DROPOUT_P
    }

# different datasets may require different data containers
# DataLoader for multi-graph tasks; Data object for one-graph
# (e.g. node-level) tasks
if ('nci1' in dataset_key) \
or ('ptc_mr' in dataset_key) \
or ('proteins' in dataset_key) \
or ('dd' in dataset_key) \
or ('mutag' in dataset_key):
    pyg_data_container_creator = dc.PyG_Dataset_DataContainer_Creator


"""
run k-fold cross validation, for models x datasets
"""
# since results are saved-overwritten after each model, 
# fix the save file prefix
results_file_prefix = "-".join([k for k in model_keys])
if new_subdir is not None:
    save_dir = f'{args.RESULTS_SAVE_DIR}/{new_subdir}/{results_file_prefix}_{args.ts}'
else:
    save_dir = f'{args.RESULTS_SAVE_DIR}/{results_file_prefix}_{args.ts}'
os.makedirs(save_dir, exist_ok=True)


# loop models
for i, model_key in enumerate(model_keys):

    # init model's results container dicts
    # for each model, we save metrics and timing results containers
    model_metrics_records = []
    model_timing_dict = {}
    args.MODEL_NAME = model_key
    model_timing_dict[model_key] = []

    # set model's save filenames for results containers
    save_filenames = {}
    for result_key in ('metrics', 'times'):
        if len(model_keys) > 1:
            save_filenames[result_key] = f'cv_{result_key}_{model_key}.pkl'
        else:
            save_filenames[result_key] = f'cv_{result_key}.pkl'

    # print 'new model starting CVs' message
    print('-' * 50)
    print(
        f"Starting {args.N_FOLDS}-fold CVs of {model_key} on"
        f" {len(pyg_data_list_filepaths)} \'{dataset_key}\'\ndatasets")
    print(f"\tstarted {time.ctime()}")
    print('-' * 50)

    # reset 'model_kwargs' for each new model
    model_kwargs = {
        'base_module_kwargs': base_module_kwargs,
        'fc_kwargs': fc_model_kwargs,
        'verbosity': args.VERBOSITY
    }

    in_channels = max(args.MANIFOLD_N_AXES, args.AMBIENT_DIM) \
        if (args.N_NODE_FEATURES is None) else args.N_NODE_FEATURES

        
    if ('mfcn' in model_key) or ('mcn' in model_key):
        model_class = mfcn.MFCN_Module
        model_kwargs = model_kwargs \
            | mfcn.set_mfcn_model_kwargs(
                args,
                model_key,
                in_channels,
                num_nodes_one_graph, 
                node_pool_out_channels,
                k_drop_kwargs,
                k_drop_generator 
            )
        

    elif model_key == 'legs':
        model_class = legs.LEGS_MLP
        model_kwargs['legs_module_kwargs'] = {
            'in_channels': in_channels,
            'channel_pool_key': args.MFCN_FINAL_CHANNEL_POOLING,
            'J': args.J,
            'n_moments': args.Q,
            'trainable_laziness': False,
            # 'selector_matrix_save_path': args.MODEL_SAVE_DIR
        }
        
    else: # GCN, GAT, GIN, GraphSAGE
        model_class = gnn.GNN_FC
        model_kwargs = model_kwargs | {
            'gnn_type': model_key,
            'in_channels': -1, # -1 is 'auto'; inferred from first batch
            # if 'out_channels' is not None, pytorch geometric GNNs will apply
            # a final linear layer to convert hidden embeddings to size 'out_channels'
            'out_channels': node_pool_out_channels,
            'channel_pool_key': args.GNN_FINAL_CHANNEL_POOLING,
            'dropout_p': args.GNN_DROPOUT_P if args.GNN_USE_DROPOUT else 0.
        }
        
    args.set_model_name_timestamp(new_timestamp=False)
    

    # loop datasets
    for dataset_j, data_filepath in enumerate(pyg_data_list_filepaths):

        print('-' * 50)
        print(
            f'Running {args.N_FOLDS}-fold CV of {model_key} on dataset'
            f' {dataset_j + 1} of {len(pyg_data_list_filepaths)}')
        print(f'\tstarted {time.ctime()}')
        print('-' * 50)

        if data_filepath is not None:
            # create dataset identifier, e.g., "comb_0",
            # under 'dataset' key; make new subsubdir for
            # cv runs of each dataset
            dataset_id = data_filepath \
                .split("/")[-1] \
                .split(".")[0] # remove '.pkl'
        else:
            dataset_id = dataset_key

        # update train_history filename prefix for model-dataset combo
        args.set_model_save_dirs(train_hist_prefix=dataset_id)

        if clargs.experiment_design.lower() == 'cv':
            # run k-fold cross validation of one MFCN model
            metrics_records, epoch_times_l = cv.run_cv(
                args,
                dataset_id,
                n_folds=args.N_FOLDS,
                model_name=args.MODEL_NAME,
                model_class=model_class,
                model_kwargs=model_kwargs,
                optimizer_class=torch.optim.AdamW,
                # optimizer_kwargs='args',
                cv_idxs=cv_idxs_unexpanded,
                n_oversamples=args.N_OVERSAMPLES,
                # train_kwargs='args',
                metrics_kwargs=metrics_kwargs,
                using_pytorch_geo=True,
                pyg_data_container_creator=pyg_data_container_creator,
                pyg_data_list_filepath=data_filepath,
                verbosity=args.VERBOSITY,
            )
        # elif clargs.experiment_design.lower() == 'tvt':
        #     pass
        
        # save model's metrics and runtimes in all-model containers
        model_metrics_records += metrics_records
        model_timing_dict[args.MODEL_NAME] += epoch_times_l

        """
        pickle all-model metrics and times containers
        - overwrite saved files after each dataset completes all its folds
        (we hence overwrite model's results files 'n_datasets' times)
        """
        for result_key, obj \
        in (('metrics', model_metrics_records), ('times', model_timing_dict)):
            result_filename = save_filenames[result_key]
            full_save_path = f'{save_dir}/{result_filename}'
            u.pickle_obj(full_save_path, obj, overwrite=True)
            print(
                f"{model_key} {result_key} CV results saved (overwrite {dataset_j})"
                f"in\n\'{full_save_path}\'."
            )


