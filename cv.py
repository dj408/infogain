"""
Functions for executing k-fold cross-
validation of models that extend the
vanilla_nn.BaseModule class.
"""
from train_fn import train_model
import utilities as u
import data_utilities as du
import base_module as bm
import dataset_creation as dc
from infogain import calc_custom_P_wavelet_scales
from args_template import ArgsTemplate
# import infogain

import torch
import numpy as np
from numpy.random import RandomState
from typing import (
    Optional,
    Tuple,
    List,
    Dict,
    Any,
    Type
)
import time
import datetime
import pickle


def run_cv(
    args: ArgsTemplate,
    dataset_id: str,
    n_folds: int = 10,
    model_name: str = '',
    model_class: torch.nn.Module = None,
    model_kwargs: Dict[str, Any] = {},
    optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
    optimizer_kwargs: Optional[Dict[str, Any] | str] = 'args',
    cv_idxs: List[np.ndarray] = None,
    n_oversamples: Optional[int] = None,
    train_kwargs: Optional[Dict[str, Any] | str] = 'args',
    metrics_kwargs: Dict[str, Any] = {},
    using_pytorch_geo: bool = True,
    pyg_data_container_creator: Type[dc.PyG_DataContainer_Creator] \
        = dc.PyG_DataContainer_Creator,
    pyg_data_list_filepath: Optional[str] = None,
    verbosity: int = 0
) -> Tuple[List[Dict[str, float]], List]:
    """
    Runs k-fold cross validation, 
    If 'n_oversamples' is not None, then we 
    generate oversampled fold train/valid set indexes
    using 'cv_idxs' and 'n_oversamples'
    (when oversampling, it's important to keep oversamples
    from the same base sample in the same train/valid set).
    
    Args:
        args: ArgsTemplate set of experiment arguments.
        dataset_id: identifier string for dataset.
        n_folds: number of folds for the cross-validation
            ('k').
        model_name: string key of the model being evaluated.
        model_class: class of the model being evaluated
            (initialized here with 'model_kwargs').
        model_kwargs: keyword arguments to initalize the
            model with.
        optimizer_class: class of the optimizer object to
            use for training; initialized here with
            'optimizer_kwargs'.
        optimizer_kwargs: keyword arguments to initalize the
            optimizer with; or if 'args', use those in args.
        cv_idxs: list of sample index arrays for each fold.
        n_oversamples: if not None, indicates how many oversamples
            were taken from each observation indexed by each
            integer in the 'cv_idxs' arrays. This function will
            then 'expand' the fold indices to ensure that oversamples
            from the same observation are kept in the same 'train' 
            or 'valid' set within each fold during cross-validation.
        train_kwargs: keyword arguments to pass to train_fn.train_model,
            or if 'args', use those in args.
        metrics_kwargs: kwargs for setting up metric calcu-
            lator objects, e.g., num_classes for multiclass
            accuracy.
        using_pytorch_geo: whether the data is in pytorch geometric
            data classes (Data, Batch, etc.).
        pyg_data_container_creator: class or subclass to use that creates
            the PyG data container (Data object for one graph, or
            DataLoader object for multi-graph dataset). Subclass the
            default `dc.PyG_DataContainer_Creator`, overriding the 
            `_load_data_list` method if needed.
        pyg_data_list_filepath: optional string path to a list of PyG
            Data objects, from which to construct the data container.
            Passed to the pyg_data_container_creator object.
        verbosity: integer controlling the volume of print output
            as this function executes.
    Returns:
        Tuple of (1) metric records (list of dicts, with 
        metric name keys and float score values, as well
        as model and fold keys and values), and (2) a
        timing list (each value is seconds elapsed per 
        epoch).
    """
    # init empty records (list of dicts) to hold a dict
    # for each fold's metric results (and model, fold ids)
    metrics_records = []
    
    # init empty list to hold 'sec per epoch' times
    # (note this is fold-agnostic; times for all folds
    # are collected in this flat list, since time stats
    # are calculated across all folds)
    epoch_times_ll = []

    # use args' optimizer and train kwargs if passed 'args'
    if optimizer_kwargs == 'args':
        optimizer_kwargs = {
            'lr': args.LEARN_RATE,
            'betas': args.ADAM_BETAS,
            'weight_decay': args.ADAM_WEIGHT_DECAY
        }
    if train_kwargs == 'args':
        train_kwargs = {
            'save_states': args.SAVE_STATES,
        }
    
    # mark cv start time
    if verbosity > -1:
        time_0 = time.time()

    # optional: seed pytorch
    # torch.manual_seed(args.TORCH_MANUAL_SEEDS[0])

    # if needed, init RandomState for subsampling graphs used to calculate
    # custom P wavelets (outside of CV loop, so we only need one)
    if (args.CUSTOM_P_SUBSAMPLE_PROP is not None) \
    and ('mfcn_p' in args.MODEL_NAME) \
    and (args.P_WAVELET_SCALES == 'custom'):
        data_subsample_random_state = RandomState(
            seed=args.CUSTOM_P_SUBSAMPLE_SEED
        )
    else:
        data_subsample_random_state = None

    # generate train/valid/[test] set index lists for each fold of CV
    fold_set_idxs_dicts = get_fold_set_idxs_dicts(
        args, 
        n_folds, 
        cv_idxs,
        n_oversamples
    )

    """
    Define inner function to process data into
    a data_container (DataLoader or Data). Will 
    have to be called twice if Infogain drops
    channels.
    """
    def proc_dataset(fold_i: int) -> Tuple[Any]:
            
        if using_pytorch_geo:
            data_container_creator = pyg_data_container_creator(
                args=args, 
                pyg_data_list_filepath=pyg_data_list_filepath,
                set_idxs_dict=fold_set_idxs_dicts[fold_i],
                verbosity=verbosity
            )
            data_container = data_container_creator.get_data_container()
            pos_weight = data_container_creator.get_pos_class_bal_wt() \
                if args.REWEIGHT_LOSS_FOR_IMBALANCED_CLASSES else None
            
            # if a classifier class-rebalancing weight has been returned,
            # insert it into the model_kwargs
            if pos_weight is not None:
                if 'loss_fn_kwargs' not in model_kwargs:
                    model_kwargs['base_module_kwargs']['loss_fn_kwargs'] = {}
                model_kwargs['base_module_kwargs']['loss_fn_kwargs'] \
                    ['pos_weight'] = pos_weight
        else:
            raise NotImplementedError(
                "DataLoader creation function not yet implemented"
                " for non-pytorch_geometric data."
            )
        return data_container, pos_weight, model_kwargs

    """
    Loop through CV folds.
    """
    for fold_i in range(n_folds):
        if verbosity > -1:
            print(f'\nCV fold {fold_i + 1}:')
            
        # load this fold's train/valid sets into DataLoaders
        # if autoprocessing custom P wavelet scales, make sure
        # no features are residually excluded
        if args.CUSTOM_P_AUTOPROCESS_UNINFORM_CHAN:
            args.EXCLUDED_FEAT_IDX = None
        data_container, pos_weight, model_kwargs = proc_dataset(fold_i)

        # for MFCN-P 'custom' P-wavelet scales, compute new scales
        # from each fold's new train set; else MFCN-P defaults to dyadic
        # wavelet scales
        # note that the custom scales are only implemented for the 
        # first filter step in each MFCN cycle, for the data's original
        # number of channels (since unique scales are calculated for each
        # channel)
        if ('mfcn_p' in args.MODEL_NAME) \
        and (args.P_WAVELET_SCALES == 'custom'):
            
            if args.USE_PRECALC_P_SCALES \
            and (args.P_WAVELET_SCALES_PRECALC is not None):
                P_wavelets_channels_t_is = torch.tensor(
                    args.P_WAVELET_SCALES_PRECALC,
                    device=args.DEVICE
                )
            else:
                T = int(2 ** args.J) # max diffusion step (e.g. J=5 -> T=32)
                print(f"Calculating 'custom' P-wavelet scales (T={T})...")
                print(f"\tquantiles={args.CUSTOM_P_CMLTV_KLD_QUANTILES}")
                time_custom_scales = time.time()
                uninform_chan_is, P_wavelets_channels_t_is = \
                infogain.calc_custom_P_wavelet_scales(
                    data_container,
                    task=args.TASK,
                    device=args.DEVICE,
                    fixed_above_zero_floor=args.CUSTOM_P_ZERO_FLOOR,
                    data_subsample_prop=args.CUSTOM_P_SUBSAMPLE_PROP,
                    data_subsample_random_state=data_subsample_random_state,
                    reweight_klds=args.CUSTOM_P_REWEIGHT_KLDS_FOR_CLASS_IMBALANCE,
                    T=T,
                    cmltv_kld_quantiles=args.CUSTOM_P_CMLTV_KLD_QUANTILES,
                    auto_process_uninformative_channels= \
                        args.CUSTOM_P_AUTOPROCESS_UNINFORM_CHAN,
                    uninformative_channel_strategy= \
                        args.CUSTOM_P_AUTOPROCESS_UNINFORM_CHAN_STRATEGY,
                    savepath_kld_by_channel_plot=f"{args.MODEL_SAVE_DIR}"
                )
                print(f"\tCustom scales calculated successfully.")

                # if channels were dropped by infogain, update args and reprocess
                # dataset
                if uninform_chan_is is not None:
                    uninform_cs = [t.item() for t in uninform_chan_is]
                    print(
                        f"\t{len(uninform_chan_is)} uninformative channels found:"
                        f" {uninform_cs}"
                    )
                    if (args.CUSTOM_P_AUTOPROCESS_UNINFORM_CHAN_STRATEGY == 'drop'):
                        new_num_channels = args.N_NODE_FEATURES - len(uninform_chan_is)
                        args.EXCLUDED_FEAT_IDX = uninform_chan_is
                        data_container, pos_weight, model_kwargs = proc_dataset(fold_i)
                        model_kwargs['in_channels'] = new_num_channels
    
                    elif (args.CUSTOM_P_AUTOPROCESS_UNINFORM_CHAN_STRATEGY == 'drop'):
                        print(f" assigning median custom scales to these channels.")
            # load custom scales into model_kwargs
            model_kwargs['P_wavelets_channels_t_is'] = P_wavelets_channels_t_is

            # print time elapsed to get custom scales and reprocess dataset
            time_custom_scales = time.time() - time_custom_scales
            print(f"\tDone ({time_custom_scales:.4f} sec).")
        
        # init. new model in each fold
        model = model_class(**model_kwargs)
        optimizer = optimizer_class(
            model.parameters(), 
            **optimizer_kwargs
        )

        # train model
        trained_model, train_records, epoch_ctr = train_model(
            args=args,
            model=model,
            data_container=data_container,
            optimizer=optimizer,
            using_pytorch_geo=using_pytorch_geo,
            return_best=True,
            fold_i=fold_i,
            verbosity=verbosity,
            **train_kwargs
        )

        # append this fold's list of epoch times elapsed (in timing dict)
        # only include epochs up through the best epoch (not burn-in, patience, etc.)
        best_epoch = epoch_ctr.best[args.MAIN_METRIC]['epoch']
        fold_times_l = [
            r['sec_elapsed'] for r in train_records \
            if r['epoch'] <= best_epoch
        ]
        epoch_times_ll.append(fold_times_l) 
        
        """
        fold testing on hold-out set (valid or test)
        """
        set_key = 'test' if args.USE_CV_TEST_SET else 'valid'
        # print('trained_model.device', trained_model.device)
        metric_scores_dict = bm.test_nn(
            trained_model=trained_model.to(args.DEVICE),
            data_container=data_container,
            task=args.TASK,
            device=args.DEVICE,
            target_name=args.TARGET_NAME,
            set_key=set_key,
            metrics_kwargs=metrics_kwargs,
            using_pytorch_geo=using_pytorch_geo
        )

        # save and print test set performance metrics
        if verbosity > 0:
            print('-' * 40)
            print(
                f'\'{model_name.upper()}\' {set_key} set metrics on'
                f' fold {fold_i + 1}:'
            )

        # init container for fold_i's metrics
        fold_dict = {
            'model': model_name,
            'dataset': dataset_id,
            'fold_i': fold_i
        }
        
        for metric, score in metric_scores_dict.items():
            score = score.detach().cpu().numpy()
            if score.size == 1:
                score = score.item()
                if verbosity > -1:
                    print(f'\t{metric} = {score:.4f}')
            fold_dict[metric] = score
        metrics_records.append(fold_dict)
        if verbosity > -1:
            print('-' * 50, '\n')

    # print total time elapsed
    if verbosity > -1:
        t_min, t_sec = u.get_time_min_sec(time.time(), time_0)
        out = f'\n{n_folds}-fold CV complete in {t_min:.0f}min, {t_sec:.1f}sec.\n'
        print(out)
    
    return metrics_records, epoch_times_ll



def get_cv_idxs_for_dataset(
    args,
    # dataset_key: str
) -> List[np.ndarray]:
    """
    Helper function to generate cross-validation
    fold index sets, for a given dataset. Passing 
    args.CV_SPLIT_SEED ensures the split is the same
    (e.g. for baseline models and MFCN models) for
    the given dataset.
    
    Args:
        args: ArgsTemplate subclass object for the 
            dataset.
    Returns:
        List of np arrays holding CV folds' hold-out
        set data indices.
    """
    # if 'melan' in dataset_key:
    if args.STRATIF_SAMPLING_KEYS is not None:
        
        # stratify-sample so that the outcome class ratios and stage_at_dx
        # are roughly the same across cv folds
        with open(f"{args.DATA_DIR}/{args.RAW_DATA_FILENAME}", "rb") as f:
            data_dictl = pickle.load(f)
    
        strat_l_dict = {key: [] for key in args.STRATIF_SAMPLING_KEYS}
        for data_dict in data_dictl:
            for key in args.STRATIF_SAMPLING_KEYS:
                strat_l_dict[key].append(data_dict[key])
        strata_label_ll = [l for l in strat_l_dict.values()]
    
        # generate cross-validation train/valid set indexes
        # (ignoring oversampling, as in the melanoma dataset)
        cv_idxs_unexpanded = du.multi_strat_multi_fold_idx_sample(
            strata_label_ll, 
            n_folds=args.N_FOLDS,
            seed=args.CV_SPLIT_SEED,
            return_np_arrays_l=True,
            verbosity=0
        )
    
        # old: 540-manifold (10x-oversampled) melanoma dataset
        # for manifold classification, random sets of manifolds are held
        # out, and oversampled manifolds means manifolds originating from
        # same patient should stay in same train/hold-out set in each cv fold
        # cv_idxs_unexpanded = du.get_cv_idx_l(
        #     seed=args.CV_SPLIT_SEED, 
        #     dataset_size=args.N_PATIENTS,
        #     n_folds=args.N_FOLDS,
        #     strat_labels=strat_labels
        # )   
        # strat_labels = np.load(f'{args.DATA_DIR}/labels.npy') # should be dtype='int'
        # # 'labels.npy' has oversampled labels (i.e. 10 for every patient)
        # strat_labels = strat_labels[0::args.N_OVERSAMPLES]
        # print(strat_labels)
    else:
        dataset_size = args.N_PTS_ON_MANIFOLD \
            if 'node' in args.TASK \
            else args.N_MANIFOLDS
        
        cv_idxs_unexpanded = du.get_cv_idx_l(
            seed=args.CV_SPLIT_SEED, 
            dataset_size=dataset_size,
            n_folds=args.N_FOLDS,
            strat_labels=None
        )   
    # print(cv_idxs_unexpanded)

    return cv_idxs_unexpanded


def get_fold_set_idxs_dicts(
    args: ArgsTemplate,
    n_folds: int,
    cv_idxs: List[np.ndarray],
    n_oversamples: Optional[bool] = None
) -> List[Dict[str, List[int]]]:
    """
    Args:
        args: ArgsTemplate set of experiment 
            arguments.
        n_folds: number of folds in the cross-
            validation procedure.
        cv_idxs: list of sample index arrays for 
            each fold.
        n_oversamples: number of oversamples
            taken from each original sample; used
            to keep oversamples in the same train/
            valid/test set, where oversampling is
            utilized (set to None if not).
    Returns:
        A list of dictionaries, each
        holding set keys ('train', 'valid', and
        possibly 'test') for index sets for each
        fold of cross-validation.
    """
    # init empty container to populate
    fold_set_idxs_dicts = [None] * n_folds
    
    # if using train/valid/test sets: randomly shuffle valid set fold indexes
    # this assumes test and valid sets get 1 fold each, of equal size 
    # (e.g. in 80/10/10 split)
    if args.USE_CV_TEST_SET:
        valid_shuffle_rs = RandomState(seed=args.VALID_SET_SHUFFLE_SEED)
        valid_is = list(range(n_folds))
        valid_shuffle_rs.shuffle(valid_is)

    # loop through cv folds
    for fold_i in range(n_folds):

        if args.USE_CV_TEST_SET:
            # use fold_i for test set, and to look up random valid set
            valid_i = valid_is[fold_i]
            train_idx = np.concatenate([
                fold_idx for i, fold_idx \
                in enumerate(cv_idxs) \
                if ((i != fold_i) and (i != valid_i))
            ])
    
            # create this fold's train/valid/test idxs dict
            fold_set_idxs_dict = {
                'train': train_idx, 
                'valid': cv_idxs[valid_i],
                'test': cv_idxs[fold_i]
            }
            
        else: # if no test set:
            # use fold_i for valid set
            train_idx = np.concatenate([
                fold_idx for i, fold_idx \
                in enumerate(cv_idxs) \
                if i != fold_i
            ])
    
            # create this fold's train/valid idxs dict
            fold_set_idxs_dict = {
                'train': train_idx, 
                'valid': cv_idxs[fold_i]
            }

        # if 'n_oversamples' is provided (is not None), expand 
        # fold_set_idxs_dict for oversampled fold train/valid set indexes
        if (n_oversamples is not None) and (n_oversamples > 1):
            if verbosity > 0:
                print(f'\texpanding fold data samples for {n_oversamples}x-oversampling.')
            # expand train/valid idx sets for oversampling
            # (to make sure each patient's samples are in only ONE of the sets)
            fold_set_idxs_dict = du.expand_set_idxs_dict_for_oversampling(
                set_idxs_dict=fold_set_idxs_dict,
                n_oversamples=n_oversamples
            )
        # print(fold_set_idxs_dict)
        fold_set_idxs_dicts[fold_i] = fold_set_idxs_dict

    return fold_set_idxs_dicts

