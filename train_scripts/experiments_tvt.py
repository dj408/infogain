"""
Script to run a train-valid-test (TVT)
experiment for one model on one dataset.

Supported datasets:
- 'peptides-func'

Supported models:
- MFCN
- LEGS

Example script call:
! python3.11 "../code/train_scripts/experiments_tvt.py" \
--machine "desktop" \
--model_key "mfcn_p" \
--p_wavelet_scales_type "custom" \
--J 4 \
--dataset "peptides-func" \
--n_epochs 250 \
--burn_in 50 \
--validate_every_n_epochs 5 \
--patience 32 \
--learn_rate 0.001 \
--batch_size 512 \
--verbosity 0

"""
import sys
sys.path.insert(0, '../')
import os
import time
import pickle
import argparse
from torch.optim import AdamW

from utilities import pickle_obj
from tvt import run_tvt
import dataset_creation as dc


"""
hardcoded vars (for now; may add flexible functionality later...)
"""
new_subdir = None
using_pytorch_geo = True
optimizer_class = AdamW
num_nodes_one_graph = None # need this if task is with one large graph
node_pool_out_channels = None # set to None if no node pooling
# gnn_out_channels = None
k_drop_generator = None
k_drop_kwargs = {}
# k_drop_generator = torch.Generator(device=args.DEVICE) # needs to follow args init!
# k_drop_generator.manual_seed(args.K_DROP_SEED)
legs_trainable_laziness = False


"""
clargs
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--machine', default='desktop', type=str, 
    help='key for the machine in use (default: desktop): see args_template.py'
)

# data args
parser.add_argument(
    '-d', '--dataset', default=None, type=str, 
    help="dataset key, e.g. 'peptides-func' etc."
)

# models
parser.add_argument('--model_key', default=None, type=str, 
    help="model key, e.g. 'mfcn_p' etc."
)
parser.add_argument(
    '--p_wavelet_scales_type', default='dyadic', type=str,
    help="P-wavelet type: 'dyadic' (default) or 'custom' or 'precalc'"
)
parser.add_argument('--use_all_orig_feats', default=False, action='store_true')

# scattering args
parser.add_argument(
    '-j', '--J', default=4, type=int, 
    help='Reference number of wavelets/diffusion steps, e.g. J=4 gives T=2**J=16 (default: 4)'
)

# training args
parser.add_argument(
    '-e', '--n_epochs', default='1000', type=int, 
    help='max num. of epochs to run in each fold (default: 1000)'
)
parser.add_argument(
    '--validate_every_n_epochs', default='5', type=int, 
    help='run model validation every nth epoch (default: 5)'
)
parser.add_argument(
    '-b', '--burn_in', default='4', type=int, 
    help='min num. of epochs to run before enforcing early stopping (default: 4)'
)
parser.add_argument(
    '-p', '--patience', default='32', type=int, 
    help='if args.STOP_RULE is no_improvement, max num. of epochs'
    'without improvement in validation loss (default: 32)'
)
parser.add_argument(
    '-l', '--learn_rate', default='0.003', type=float, 
    help='learning rate hyperparameter (default: 0.003)'
)
parser.add_argument(
    '-t', '--batch_size', default='32', type=int, 
    help='train set minibatch size hyperparameter (default: 32)'
)
parser.add_argument('--use_k_drop', default=False, action='store_true')
parser.add_argument(
    '-v', '--verbosity', default='0', type=int, 
    help='integer controlling volume of print output during execution'
)
clargs = parser.parse_args()


"""
args (dataset dependent)
"""
if clargs.dataset.lower() == 'peptides-func':
    import infogain_testing.peptides_func_args as a
    
# init args (with command-line overrides)
args = a.Args(
    MACHINE=clargs.machine,
    MODEL_NAME=clargs.model_key,
    # MODEL_SAVE_SUBDIR=model_save_subdir,
    J=clargs.J,
    P_WAVELET_SCALES=clargs.p_wavelet_scales_type,
    USE_PRECALC_P_SCALES="precalc" in clargs.p_wavelet_scales_type,
    USE_K_DROP=clargs.use_k_drop,
    N_EPOCHS=clargs.n_epochs,
    VALIDATE_EVERY_N_EPOCHS=clargs.validate_every_n_epochs,
    BURNIN_N_EPOCHS=clargs.burn_in,
    NO_VALID_LOSS_IMPROVE_PATIENCE=clargs.patience,
    LEARN_RATE=clargs.learn_rate,
    # >0 prints epoch-by-epoch stats in train_fn
    VERBOSITY=clargs.verbosity
)
if clargs.use_all_orig_feats:
    args.EXCLUDED_FEAT_IDX = None
    num_excl_feats = 0
else:
    num_excl_feats = len(args.EXCLUDED_FEAT_IDX)
args.set_model_save_dirs(new_subdir=new_subdir)
args.set_batch_sizes(train_size=clargs.batch_size)
# model_keys = [model_key]
# args.set_model_name_timestamp(new_timestamp=False)


"""
data
"""
in_channels = max(args.MANIFOLD_N_AXES, args.AMBIENT_DIM) \
    if (args.N_NODE_FEATURES is None) \
    else (args.N_NODE_FEATURES - num_excl_feats)

# for MFCNs where custom wavelet scales have been precalculated,
# default to the number of features indicated by args.P_WAVELET_SCALES_PRECALC,
# over args.N_NODE_FEATURES (which may reflect the original feature count,
# before InfoGain drops features)
# if (clargs.model_key == 'mfcn_p') \
# and (args.P_WAVELET_SCALES == 'custom') \
# and (args.P_WAVELET_SCALES_PRECALC is not None):
#     args.N_NODE_FEATURES = len(args.P_WAVELET_SCALES_PRECALC)
#     in_channels = args.N_NODE_FEATURES

if clargs.dataset.lower() == 'peptides-func':
    # create DataLoaders dict from downloaded datasets on disk
    # note that sparsifying features for this dataset results in slower training
    data_container, pos_weight = dc.get_peptides_func_dataloaders_dict(args)


"""
training objects' kwargs
"""
if pos_weight is not None:
    loss_fn_kwargs = {'pos_weight': pos_weight}
else:
    loss_fn_kwargs = {}
    
base_module_kwargs = {
    'task': args.TASK,
    'target_name': args.TARGET_NAME,
    'device': args.DEVICE,
    'loss_fn_kwargs': loss_fn_kwargs,
}

# metrics_kwargs: needed for, e.g., multiclass classification
# metrics_kwargs = {'num_classes': args.NUM_TARGET_CLASSES}
metrics_kwargs = {} 

fc_model_kwargs = {
    'base_module_kwargs': base_module_kwargs,
    'output_dim': args.OUTPUT_DIM,
    'hidden_dims_list': args.NN_HIDDEN_DIMS,
    'use_batch_normalization': args.USE_BATCH_NORMALIZATION,
    'use_dropout': args.MLP_USE_DROPOUT,
    'dropout_p': args.MLP_DROPOUT_P
}

model_kwargs = {
    'base_module_kwargs': base_module_kwargs,
    'fc_kwargs': fc_model_kwargs,
    'verbosity': args.VERBOSITY
}

    
if ('mfcn' in clargs.model_key) \
or ('mcn' in clargs.model_key):
    from MFCN.mfcn import (
        MFCN_Module, 
        set_mfcn_model_kwargs
    )
    model_class = MFCN_Module
    mfcn_model_kwargs = set_mfcn_model_kwargs(
        args,
        clargs.model_key,
        in_channels,
        num_nodes_one_graph, 
        node_pool_out_channels,
        k_drop_kwargs,
        k_drop_generator 
    )
    model_kwargs = model_kwargs | mfcn_model_kwargs
    

elif clargs.model_key == 'legs':
    from LEGS.LEGS_module_DJedit import LEGS_MLP
    model_class = LEGS_MLP
    model_kwargs['legs_module_kwargs'] = {
        'in_channels': in_channels,
        'channel_pool_key': args.MFCN_FINAL_CHANNEL_POOLING,
        'J': args.J,
        'n_moments': args.Q,
        'trainable_laziness': legs_trainable_laziness,
        'selector_matrix_save_path': args.MODEL_SAVE_DIR
    }


"""
init results container and call training wrapper function
"""
model_metrics_records = []
model_timing_dict = {}
model_timing_dict[clargs.model_key] = []

# set model's save filenames for results containers
# we keep these general for multi-model/dataset runs
# for future compatibility
save_filenames = {}
for result_key in ('metrics', 'times'):
    # if len(model_keys) > 1:
    #     save_filenames[result_key] = f'cv_{result_key}_{clargs.model_key}.pkl'
    # else:
    save_filenames[result_key] = f'cv_{result_key}.pkl'

# set results save directory
results_file_prefix = clargs.model_key # "-".join([k for k in model_keys])
if new_subdir is not None:
    save_dir = f'{args.RESULTS_SAVE_DIR}/{new_subdir}/{results_file_prefix}_{args.ts}'
else:
    save_dir = f'{args.RESULTS_SAVE_DIR}/{results_file_prefix}_{args.ts}'
os.makedirs(save_dir, exist_ok=True)

# run experiment
metrics_records, epoch_times_l = run_tvt(
    args,
    clargs.dataset,
    data_container=data_container,
    model_name=args.MODEL_NAME,
    model_class=model_class,
    model_kwargs=model_kwargs,
    optimizer_class=optimizer_class,
    # optimizer_kwargs='args',
    # train_kwargs='args',
    metrics_kwargs=metrics_kwargs,
    using_pytorch_geo=using_pytorch_geo,
    verbosity=args.VERBOSITY,
)

# save model's metrics and runtimes in all-model containers
model_metrics_records += metrics_records
model_timing_dict[args.MODEL_NAME] += epoch_times_l


"""
pickle results (metrics and times containers)
"""
for result_key, obj \
in (('metrics', model_metrics_records), ('times', model_timing_dict)):
    result_filename = save_filenames[result_key]
    full_save_path = f'{save_dir}/{result_filename}'
    pickle_obj(full_save_path, obj, overwrite=False)
    print(
        f"{clargs.model_key} {result_key} results saved"
        f"in\n\'{full_save_path}\'."
    )

