"""
Utility classes and functions
for data processing.
"""
import pickle
import numpy as np
from numpy.random import RandomState
import torch
from torch import Generator
from torch.utils.data import (
    Dataset, 
    DataLoader
)
from typing import (
    Tuple,
    List,
    Dict,
    Any,
    Optional
)
from itertools import product
import warnings


def get_binary_class_rebalance_weight(
    train_targets: List[int]
) -> torch.Tensor:
    """
    Given a list of integer targets for a binary
    classification task, calculates the positive
    class weight for rebalancing a torch loss
    function to weight binary classes equally.

    Args: 
        train_targets: list of integer binary
        targets.
    Returns:
        Scalar tensor containing the positive-class
        weight for, e.g., cross-entropy loss to
        be re-weighted for balanced class contribution
        in a binary classification learning task.
    """
    n = len(train_targets)
    ct_1s = np.sum(train_targets)
    rebalance_pos_wt = torch.tensor((n - ct_1s) / ct_1s)
    return rebalance_pos_wt
    

def possible_pd_df_to_array(
    poss_df: Any, 
    array_type_key: str = 'np', # scipy-sparse
    dtype = np.float64,
    warn: bool = True,
) -> Any:
    """
    Checks if passed object is a pandas dataframe, and
    if so, keeps only numeric columns and converts to
    a numpy array or scipy sparse matrix. If not a dataframe, 
    the passed object is simply returned.

    Args:
        poss_df: an object that will be checked if
            it's an instance of a pandas dataframe.
        array_type_key: the type of matrix to convert the
            numeric content of the dataframe to: numpy ('np')
            or 'scipy-sparse'.
        dtype: the data type (e.g., np.float64) in which
            the matrix values are stored.
        warn: whether to raise a warning if non-numeric
            columns are dropped when converting the dataframe.
    Returns:
        Either the original non-dataframe object, or a numpy array
        or scipy-sparse csr matrix of the numeric columns of the
        dataframe passed.
    """
    import pandas as pd
    if isinstance(poss_df, pd.DataFrame):
        # Select numeric columns
        numeric_df = poss_df.select_dtypes(include=[np.number])
        
        # Check if any columns were removed
        if warn:
            removed_columns = set(poss_df.columns) - set(numeric_df.columns)
            if removed_columns:
                warnings.warn(
                    f"Non-numeric columns removed: {', '.join(removed_columns)}", 
                    UserWarning
                )
        
        # Convert to NumPy array
        if array_type_key == 'np':
            return numeric_df.to_numpy(dtype=dtype)
        elif array_type_key == 'scipy-sparse':
            from scipy.sparse import csr_matrix
            return csr_matrix(numeric_df, dtype=dtype)
    else:
        return poss_df


def get_cv_idx_l(
    seed: int, 
    dataset_size: int, 
    n_folds: int = 5,
    strat_labels: Optional[List[int] | np.ndarray] = None
) -> List[np.ndarray]:
    r"""
    Generates a list of index arrays for $k$ folds.
    E.g., if $k = 5$, get 80/20 test/valid split in
    each fold; $k = 10$ gives 90/10 splits.

    Can naively stratify by a single stratum, if a
    'strat_labels' list or array is passed. The
    function 'multi_strat_multi_fold_idx_sample' 
    below is more robust and preferred.

    Args:
        seed: integer seed for np RandomState.
        dataset_size: number of samples in the
            dataset.
        n_folds: number of folds/splits.
        strat_labels: indexed class labels for (single-stratum)
            stratified sampling, to get approximately the 
            same proportion of classes in each fold (can
            be multiclass). Leave 'None' for simple 
            random sampling / no stratification.
    Returns:
        List (of length n_folds) of np arrays holding 
        folds' (unique) validation set indexes.
    """
    # rs = RandomState(seed)
    # idx = np.arange(dataset_size)
    # rs.shuffle(idx)
    # folds_idx_l = np.array_split(idx, n_folds)
    # return folds_idx_l
    rs = RandomState(seed)
    idx = np.arange(dataset_size)

    # if not stratifying, simply shuffle indices and chunk into 
    # a list of n_folds arrays
    if strat_labels is None:
        rs.shuffle(idx)
        folds_idx_l = np.array_split(idx, n_folds)

    # if stratifying (single-stratum): divide indices by 
    # stratum label, shuffle, chunk into n_folds arrays, and
    # concatenate chunks of different labels back together
    # for each fold
    else:
        unique_class_is = np.unique(strat_labels)
        class_idx_ll = [None] * len(unique_class_is)
        for j, class_i in enumerate(unique_class_is):
            class_mask = np.argwhere(strat_labels == class_i)
            class_idx = idx[class_mask].flatten()
            rs.shuffle(class_idx)
            class_idx_l = np.array_split(class_idx, n_folds)
            
            # check if stratum's fold's index list has <1 samples
            for class_fold_idx in class_idx_l:
                if len(class_fold_idx) < 2:
                    warnings.warn(
                        f"<2 samples of class {class_i} were added to a cv fold!"
                    )
            class_idx_ll[j] = class_idx_l
        folds_idx_l = [np.concatenate(arrays) for arrays in zip(*class_idx_ll)]

    return folds_idx_l


def multi_strat_multi_fold_idx_sample(
    strata_label_ll: List[List[int]],
    n_folds: int,
    seed: int,
    return_np_arrays_l: bool = True,
    verbosity: int = 0
) -> List[List[int]] | List[np.ndarray]:
    """
    Generates lists of indexes that are (roughly) 
    stratified-sampled under the strata labels 
    passed in 'strata_label_ll'. Unlike 'get_cv_idx_l',
    this function can stratify based on multiple strata.
    The strategy used in this function is:

    1. Compute all combinatorial 'bins' of strata,
        and associate lists of sample indexes that
        fit each bin.
    2. Move any bins with counts < 'n_folds'
        to a 'leftovers' container.
    2. Create 'n_folds' stratified folds from bins
        with > 'n_folds' counts.
    3. Append leftover indexes to shortest
        fold index list successively, emptying bins
        before moving to the next 'leftover' bin.

    Args:
        strata_label_ll: a list of lists of integer
            labels, indexed so that each sample's
            labels in each list is at the same
            index.
        n_folds: number of folds / stratified
            index sets to create.
        seed: seed for np.RandomState
            object, for sampling repro-
            ducibility.
        return_np_arrays_l: if True, return fold idx
            sets as list of np arrays, not list of lists.
        verbosity: controls volume of print output
            as function runs.
    Returns:
        A list of index lists or np arrays for each 
        fold (i.e., validation sets in cross-
        validation).
    """

    rs = RandomState(seed)
    
    """
    generate all strata combo bins
    """
    # get unique labels within each stratum list
    n_samples = len(strata_label_ll[0])
    n_strata = len(strata_label_ll)
    unique_class_is = [None] * n_strata
    for i, l in enumerate(strata_label_ll):
        unique_class_is[i] = np.unique(l).tolist()
    
    # generate all stratum label combos
    strata_label_combos = list(product(*unique_class_is))
    if verbosity > 0:
        print(strata_label_combos)
    
    # collect indices into stratified bins
    strata_bins = {c: [] for c in strata_label_combos}
    # strata_cts = {c: 0 for c in strata_label_combos}
    for i in range(n_samples):
        ith_strata_labels = [None] * n_strata
        for j in range(n_strata):
            ith_strata_labels[j] = strata_label_ll[j][i]
        tup_key = tuple(ith_strata_labels)
        strata_bins[tup_key].append(i)
    
    # check bin counts
    strata_cts = {k: len(v) for k, v in strata_bins.items()}
    min_strata_ct = min(list(strata_cts.values()))
    if verbosity > 0:
        print('min_strata_ct', min_strata_ct)
    
    if verbosity > 0:
        print('\nstrata_bins\n', strata_bins)
    underrep_strata_l = []
    for comb, ct in strata_cts.items():
        if ct < n_folds:
            underrep_strata_l.append(comb)
    if len(underrep_strata_l) > 0:
        warnings.warn(
            f"Found strata with bin count < {n_folds} = n_folds, hence stratifying by"
            f" {n_strata} strata means at least 1 fold(s) will not have sample(s)"
            f" from the following strata bins: {underrep_strata_l}"
        )

    """
    sample idx from strata bins;
    separate 'fully stratified' folds from 'leftover' indices
    """
    # shuffle all bins in place (before splitting up strata_bins)
    strata_bins = {k: rs.permutation(v) for k, v in strata_bins.items()}
    if verbosity > 0:
        print('\nshuffled strata_bins\n', strata_bins)

    # move bins with idx counts < n_folds to 'leftover' dict, and then remove
    # these bins from strata_bins
    leftover_bins = {k: v.tolist() for k, v in strata_bins.items() if len(v) < n_folds}
    strata_bins = {k: v for k, v in strata_bins.items() if len(v) >= n_folds}

    # generate n_folds stratified idx lists from the bins with sufficient counts
    full_strata_bins = np.stack([v[:n_folds] for v in strata_bins.values()], axis=1)
    if verbosity > 0:
        print('\nfull_strata_bins\n', full_strata_bins)

    # trim indices used in 'full_strata_bins' from 'strata_bins', and merge with 
    # 'leftover_bins'
    leftover_bins = leftover_bins | \
        {k: v[n_folds:].tolist() for k, v in strata_bins.items()}
    if verbosity > 0:
        print('\nleftover_bins\n', leftover_bins)
        

    '''
    # subset each bin to size of min_strata_ct, and stack in dim 1
    # -> rows are 'balanced' multi-stratum stratified sample idxs
    full_strata_bins = np.stack([v[:min_strata_ct] for v in strata_bins.values()], axis=1)
    
    # print(leftover_bins)
    '''
    
    # separate 'balanced' idxs, and append first non-empty bin's leftovers to 
    # any idx list < full length, then first idx list, then second non-empty bin 
    # to second idx list, until all leftovers are empty -> this prioritizes equal size folds
    fold_idx_ll = [a.tolist() for a in full_strata_bins]
    full_idx_len = len(fold_idx_ll[0])
    # print('len(fold_idx_ll)', len(fold_idx_ll))
    
    # if there are fewer idx lists than n_folds, append empty idx lists
    if len(fold_idx_ll) < n_folds:
        n_miss_idx_l = n_folds - len(fold_idx_ll)
        for miss_idx_l in range(n_miss_idx_l):
            fold_idx_ll.append([])
        # print('len(fold_idx_ll) with new', len(fold_idx_ll))
    
    leftover_ll = [v for v in leftover_bins.values() if len(v) > 0]
    if verbosity > 0:
        print('\nleftover_ll\n')
        for l in leftover_ll:
            print(l)
    num_leftovers = sum([len(l) for l in leftover_ll])

    """
    append 'leftover' indices by:
    (1) filling any empty folds first, from sequential strata bins; then
    (2) appending to sequential 'full'-length folds, emptying strata bins
    """
    # init
    fold_idx_i, leftover_l_i = 0, 0

    def get_shortest_fold_idx_l():
        return np.argmin([len(l) for l in fold_idx_ll])

    def get_leftover_bins_next_idx(leftover_l_i):
        if leftover_l_i is None:
            return None
        if leftover_l_i > len(leftover_ll) - 1:
            leftover_l_i = 0
        if len(leftover_ll[leftover_l_i]) == 0:
            leftover_l_i = increment_leftover_l_i(leftover_l_i)
        leftover_idx = leftover_ll[leftover_l_i][0]
        del leftover_ll[leftover_l_i][0]
        return leftover_idx

    def leftovers_are_left():
        return sum([len(l) for l in leftover_ll]) > 0

    def increment_leftover_l_i(leftover_l_i):
        leftover_l_i += 1
        # if i is beyond len of leftover_ll, reset to 0
        if leftover_l_i > (len(leftover_ll) - 1):
            leftover_l_i = 0
        # if i lands on empty list, increment until non-empty
        # is found
        if len(leftover_ll[leftover_l_i]) == 0:
            if leftovers_are_left():
                leftover_l_i = increment_leftover_l_i(leftover_l_i)
            else:
                leftover_l_i = None
        # else:
        #     leftover_l_i = None
        return leftover_l_i

    
    while True:
        if not leftovers_are_left():
            break

        # always fill shortest fold_idx list first
        fold_idx_i = get_shortest_fold_idx_l()
        
        # if shortest is shorter than the 'fully stratified' fold idx lists,
        # fill it to 'full' with idxs from successive strata bins
        shortest_diff_from_full = full_idx_len - len(fold_idx_ll[fold_idx_i])
        # print('shortest_diff_from_full', shortest_diff_from_full)
        if shortest_diff_from_full > 0:
            shortest_addl_idx_l = [None] * shortest_diff_from_full
            for i in range(shortest_diff_from_full):
                idx = get_leftover_bins_next_idx(leftover_l_i)
                leftover_l_i = increment_leftover_l_i(leftover_l_i)
                if idx is not None:
                    shortest_addl_idx_l[i] = idx
                else:
                    break
                # print('shortest_addl_idx_l', shortest_addl_idx_l)
            fold_idx_ll[fold_idx_i] += shortest_addl_idx_l

        # if all fold_idx_l have 'full' length, append leftovers by 
        # depleting bins into different (successive) fold_idx_l
        else:
            # print('\nall of full length; depeleting bins')
            # print('leftover_l_i', leftover_l_i)
            # print('leftover_ll', leftover_ll)
            
            idx = get_leftover_bins_next_idx(leftover_l_i)
            # (recall: always appending to shortest fold_idx_l)
            fold_idx_ll[fold_idx_i].append(idx)
            
            # fold_idx_i = increment_fold_idx_i(fold_idx_i)
            # print('len(leftover_ll)', len(leftover_ll))
    
    if verbosity > 0:
        print('\nfinal fold_idx_ll')
        for l in fold_idx_ll:
            print(l, f"len = {len(l)}")

    if return_np_arrays_l:
        fold_idx_ll = [np.array(l) for l in fold_idx_ll]
    
    return fold_idx_ll


def expand_set_idxs_dict_for_oversampling(
    set_idxs_dict: Dict[str, int],
    n_oversamples: int
) -> Dict[str, int]:
    """
    Expands each i in index arrays in set_idxs_dict,
    by the number of oversamples. For example, if
    'train' idx = [1, 2, 3] and n_oversamples = 10,
    the new 'train' idx = [10, 11, ... 19, 20, 21,
    ..., 29, 30, 31, ..., 39].

    Args:
        set_idxs_dict: Dictionary of unexpanded
            index sets, keyed by (e.g.) 'train'/'valid'/
            'test'.
        n_oversamples: number of new indexes to create
            for each existing index in each set index list.
    Returns:
        Dictionary of expanded index sets, keyed by
        set.
    """
    for set, idx in set_idxs_dict.items():
        expanded_idx = np.stack(np.array([
            np.arange(
                i * n_oversamples,
                (i + 1) * n_oversamples, 
                1
            ) for i in idx
        ]), axis=0)
        set_idxs_dict[set] = expanded_idx.flatten()
    return set_idxs_dict


def split_and_pickle_DictDataset_dict(
    args,
    feat_tensor: torch.Tensor,
    target_dictl: List[dict],
    flatten_sample_input: bool = False,
    flatten_sample_kwargs: dict = {},
    set_idxs_dict: Dict[str, List[int]] = None
) -> None:
    """
    From a master feature tensor and list of target
    dictionaries, creates DictDataset objects, splits
    by train/valid/test set, contains in a dictionary
    keyed by set, and pickles the result. (A DictDataset
    extends torch's Dataset class to contain features and
    target(s) within a dictionary).

    Args:
        args: an ArgsTemplate object.
        feat_tensor: a master feature tensor where 
            samples are rows and features are 
            columns.
        flatten_sample_input: bool whether to flatten
            each sample's individual feature tensor into
            a vector, passed to 'DictDataset' init.
        flatten_sample_kwargs: kwargs regarding sample
            flattening, passed to 'DictDataset' init.
        target_dictl: a list of dictionaries, one for each
            sample, containing that sample's dictionary of
            keyed target(s), at the same index as the 
            sample's row in 'feat_tensor'.
        set_idxs_dict: a dictionary of 'train'/'valid'/
            'test' keys and list of index integer 
            values assigning samples to these sets.
    Returns:
        None; pickles a dictionary of datasets of
        type Dict[DictDataset].
    """
    # get train/valid/test split idxs
    if set_idxs_dict is None:
        set_idxs_dict = get_train_valid_test_idxs(
            seed=args.TRAIN_VALID_TEST_SPLIT_SEED,
            n=feat_tensor.shape[0],
            train_prop=args.TRAIN_PROP,
            valid_prop=args.VALID_PROP
        )
    
    # print dataset sizes
    print('Dataset sizes:')
    for set, idx in set_idxs_dict.items():
        print(f'{set}: {len(idx)}')
    
    # optional: rescale input/feature data
    if args.FEATURE_SCALING_TYPE is not None:
        set_tensors_dict = scale_tensor_feature_sets(
            features_tensor=feat_tensor,
            set_idxs_dict=set_idxs_dict,
            scaler_type=args.FEATURE_SCALING_TYPE,
            drop_features_zero_var=args.DROP_ZERO_VAR_FEATURES
        )
    else:
        set_tensors_dict = {
            set: feat_tensor[idx] \
            for set, idx in set_idxs_dict.items()
        }
    
    # create dictionary of un/scaled train/valid/test sets
    # sets are 'data_utilities.DictDatasets' objects
    datasets_dict = {
        set: DictDataset(
            inputs_tensor=set_tensors_dict[set], 
            flatten_sample_input=flatten_sample_input,
            flatten_sample_kwargs=flatten_sample_kwargs,
            targets_dictl=[target_dictl[i] for i in idx]
        ) \
        for set, idx in set_idxs_dict.items()
    }
    
    # pickle the dataset dict
    save_path = f'{args.DATA_DIR}/{args.DATASETS_DICT_FILENAME}'
    with open(save_path, "wb") as f:
        pickle.dump(datasets_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Datasets saved.\n')

    

def prep_scat_mom_for_Dataset(
    args,
    data_dictl: List[dict],
) -> torch.Tensor:
    """
    Converts 'data_dictl''s scattering moment inputs to a 
    master tensor that can be indexed by a Dataset (subclass's) 
    __getitem__ method.

    Args:
        args: Args dataclass object with experiment parameters.
        data_dictl: dictionary list holding keyed inputs
            (e.g., scattering moments) and targets.
    Returns:
        Master scattering moments feature tensor; 
        shape (n_samples, ...).
    """
    
    n = len(data_dictl)
    sm_tensors_l = [None] * n
    stack_keys = []

    # for each sample:
    for i, manifold_dict in enumerate(data_dictl):

        # extract dict of scat. moments by wavelet type and channel
        channel_dict = manifold_dict['scattering_moments'][args.WAVELET_TYPE]
        channel_vectors = [None] * len(channel_dict)

        # for each channel, stack scattering q-moments into 1d (vector) tensors
        for j, (k, sm_dict) in enumerate(channel_dict.items()):
            channel_vectors[j] = torch.tensor(
                np.concatenate([
                    sm_arr[list(args.Q_IDX)].flatten() \
                    for mom, sm_arr in sm_dict.items()
                ]),
            requires_grad=False,
            dtype=args.FEAT_TENSOR_DTYPE)

        # stack channels' vector tensors in a new dim.
        stacked_channels = np.stack(channel_vectors, axis=0)
        sm_tensors_l[i] = torch.tensor(stacked_channels)
        
    # stack all sample's feature tensors into one master tensor
    # where axis 0 indexes into samples
    sm_feat_tensor = torch.stack(sm_tensors_l, axis=0)
    
    # free memory
    del sm_tensors_l
    
    return sm_feat_tensor


def tensorize_Wjs(
    data_dictl: List[dict],
    tensor_dtype,
    wavelet_type: str
) -> Dict[str, torch.Tensor]:
    """
    Gather individual samples' wavelet operators (and, for
    spectral wavelet's, graph Laplacians) into master
    tensor(s), to feed the WaveletMFCNDataset class.
    
    Args:
        data_dictl: dictionary list holding keyed inputs
            (e.g., scattering moments) and targets.
        tensor_dtype: output type of tensor, e.g. tensor.float.
        wavelet_type: 'spectral' or 'P'.
    Returns:
        Dict of tensors needed by the MFCN model
        for its repeated cycles of wavelet filtration.
    """
    out_dict = {}
    if wavelet_type == 'spectral': 
        # these are 'np.ndarray's
        for key in ('Wjs_spectral', 'L_eigenvecs'):
            out_dict[key] = torch.stack([
                torch.tensor(
                    manifold_dict[key], 
                    requires_grad=False,
                    dtype=tensor_dtype
                ) \
                for manifold_dict in data_dictl
            ])
        # out_dict['L_eigenvecs'] = torch.stack([
        #     torch.tensor(
        #         manifold_dict['L_eigenvecs'], 
        #         dtype= args.FEAT_TENSOR_DTYPE
        #     ) \
        #     for manifold_dict in data_dictl
        # ])

    elif wavelet_type == 'P':
        # data_dictl[i]['P'] is a SPARSE (scipy csr) array
        # -> tensor of (n_samples, n_filters)
        sparse_Ps = [None] * len(data_dictl)
        for i, manifold_dict in enumerate(data_dictl):
            P = manifold_dict['P']
            sparse_Ps[i] = torch.sparse_coo_tensor(
                indices=np.stack(P.nonzero()), # shape (2, n_values)
                values=P.data,
                size=P.shape,
                requires_grad=False,
                dtype=tensor_dtype
            ) # .coalesce()
        out_dict['Ps'] = torch.stack(sparse_Ps)
    
    return out_dict


def gather_targets_dictl(
    args,
    data_dictl: List[dict],
    target_keys: List[str]
) -> List[Dict[str, torch.Tensor]]:
    """
    Gather targets/labels into list of dicts holding 
    <key: scalar tensor> pairs that can be indexed by 
    a Dataset (subclass's) __getitem__ method.

    Args:
        data_dictl: dictionary list holding keyed targets.
        target_keys: tuple of string keys for the targets.
    Returns:
        List of target str: tensor dictionaries.
    """
    target_dictl = [None] * len(data_dictl)
    
    for i, manifold_dict in enumerate(data_dictl):
        target_dictl[i] = {}
        for target_key in target_keys:
            target_tensor = torch.tensor(
                manifold_dict[target_key], 
                requires_grad=False,
                dtype=args.TARGET_TENSOR_DTYPE
            )
            target_dictl[i][target_key] = target_tensor
    return target_dictl


def get_sklearn_dataset_dict(
    dataset_name: str = 'iris',
    train_prop: float = 0.7,
    valid_prop: float = 0.15,
    split_seed: int = 444386,
    x_tensor_dtype = torch.float32, # MPS (Apple silicon) doesn't do float64
    y_tensor_dtype = torch.float32
) -> Dict[str, DataLoader]:
    """
    Convenience function to create a train/valid/test-split
    dataset from one of sklearn's iris, diabetes, wine, or
    breast cancer toy datasets.
    
    Ref:
    https://scikit-learn.org/stable/datasets/toy_dataset.html
    """
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler

    # load dataset
    if dataset_name == 'iris': # classification
        data = datasets.load_iris()
    elif dataset_name == 'diabetes': # regression
        data = datasets.load_diabetes()
    elif dataset_name == 'wine': # classification
        data = datasets.load_wine()
    elif dataset_name == 'breast_cancer': # classification
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    # get train/valid/test split idxs
    set_idxs_dict = get_train_valid_test_idxs(
        seed=split_seed,
        n=len(y),
        train_prop=train_prop,
        valid_prop=valid_prop
    )

    # split, rescale, and tensorize X data
    set_tensors_dict = {}
    set_targets_dict = {}
    scaler = StandardScaler()
    scaler.fit(X[set_idxs_dict['train']])
    
    for set in ('train', 'valid', 'test'):
        set_idx = set_idxs_dict[set]
        set_tensors_dict[set] = torch.tensor(
            scaler.transform(X[set_idx]),
            requires_grad=False,
            dtype=x_tensor_dtype
        )

    # put y targets into a list of dicts
    target_dictl = [
        {'y': torch.tensor(
                yval, 
                requires_grad=False,
                dtype=y_tensor_dtype
            )
        } \
        for yval in y
    ]
    
    # create dict of DictDatasets by set
    datasets_dict = {
        set: DictDataset(
            set_tensors_dict[set], 
            [target_dictl[i] for i in idx]
        ) \
        for set, idx in set_idxs_dict.items()
    }
    return datasets_dict
    


class DictDataset(Dataset):
    """
    Subclass of `torch.utils.data.Dataset` that
    contains inputs and targets in dictionaries,
    for abstraction that allows for a generic PyTorch
    training function.

    __init__ args:
        inputs_tensor: tensors for x/input, first dimension
            of which indexes into one sample/input:
            hence shape (n, , [n_channels], n_pts_per_sample).
        flatten_sample_input: whether to flatten a sample's
            input features tensor in some way, such as stacking
            multiple channels' vector tensors into one.
        flatten_sample_kwargs: kwargs to pass to torch.flatten
            (e.g., {'start_dim': 1}).
        targets_dictl: list of dictionaries holding
            target(s') keys and values.

    __getitem__ returns:
        A dictionary of one sample's input tensor ('x' key) and
        a sub-dictionary of targets ('target' key).
    """
    def __init__(self, 
                 inputs_tensor: torch.Tensor,
                 targets_dictl: List[Dict[str, Any]],
                 flatten_sample_input: bool = False,
                 flatten_sample_kwargs: dict = {}
                ) -> None:
        super(DictDataset, self).__init__()
        if flatten_sample_input:
            self.inputs_tensor = inputs_tensor \
                .flatten(**flatten_sample_kwargs)
        else:
            self.inputs_tensor = inputs_tensor
        print(f'DictDataset: self.inputs_tensor.shape = {self.inputs_tensor.shape}')
        self.targets_dictl = targets_dictl

    def __len__(self):
        return len(self.inputs_tensor)

    def __getitem__(self, idx):
        data_obj_dict = {
            'x': self.inputs_tensor[idx], 
            'target': self.targets_dictl[idx]
        }
        return data_obj_dict
        

def get_train_valid_test_idxs(
    seed: int,
    n: int,
    train_prop: float = 0.7, 
    valid_prop: float = 0.15
) -> Dict[str, List[int]]:
    """
    Generates a dictionary holding random index
    sets for train/valid/test dataset splits.

    Args:
        seed: seed for numpy RandomState.
        n: count of samples in entire dataset.
        train_prop: proportion of data in train
            set (0. < train_prop < 1.)
        valid_prop: proportion of data in validation
            set (0. < valid_prop < 1.) Any proportion
            of the data not in train + valid sets goes
            into the test set.

    Returns:
        Dictionary with 'train'/'valid'/'test' keys
        and index list values.
    """
    if (train_prop < 0.) | (valid_prop < 0.) \
    | (train_prop > 1.) | (valid_prop > 1.):
        print('Error: train_prop and valid_prop must be between 0 and 1!')
        return None
    rs = np.random.RandomState(seed=seed)
    idx = np.arange(n)
    rs.shuffle(idx) 
    train_valid_cut_i = int(train_prop * n)
    valid_test_cut_i = train_valid_cut_i + int(valid_prop * n)
    idx_dict = {
        'train': idx[:train_valid_cut_i],
        'valid': idx[train_valid_cut_i:valid_test_cut_i],
        'test': idx[valid_test_cut_i:]
    }
    return idx_dict


def get_fixed_shape_blockdiag_coo_indices(
    batch_size: int,
    n_row: int, 
    n_col: int
) -> np.ndarray:
    """
    Calculates row and col (COO) indices for a sparse
    block-diagonal matrix constructed of a batch of 
    dense matrices of the same shape -- that is, where
    only the off-block-diagonal entries are sparse.

    Args:
        batch_size: number of graphs in batch.
        n_row: fixed number of rows, same in each graph.
        n_col: fixed number of cols, same in each graph.
    Returns:
        Array of indices for a block-diagonal sparse
        COO matrix.
    """
    # since each sample in a batch is the same size,
    # grab sample matrix dimensions from the first sample
    base_row_indices = []
    base_col_indices = []
    for i in range(n_row):
        for j in range(n_col):
            base_row_indices.append(i)
            base_col_indices.append(j)
    # convert to numpy arrays (to use array operations below)
    base_row_indices = np.array(base_row_indices)
    base_col_indices = np.array(base_col_indices)
    
    row_indices = []
    col_indices = []
    for b in range(batch_size):
        row_indices.extend(base_row_indices + b * n_row)
        col_indices.extend(base_col_indices + b * n_col)
    indices = torch.tensor(
        np.stack((row_indices, col_indices))
    )
    return indices


def sparse_block_diagonalize_dense_3d_tensor(
    x: torch.Tensor,
    dtype: torch.Tensor.type = torch.float
) -> torch.sparse_coo_tensor:
    """
    Creates a 2-d block-diagonal sparse (COO) 
    tensor from a 3-d tensor (e.g., a batch of
    2-d tensors).
    
    Args:
        x: dense 3-d tensor of batched/stacked
            tensors/graphs.
        dtype: data type for output sparse COO
            tensor.
    Returns:
        Sparse COO 2-d block diagonal tensor of
        the tensor batch.
    """
    batch_size = x.shape[0]
    n_row = x.shape[1]
    n_col = x.shape[2]
    
    indices = get_fixed_shape_blockdiag_coo_indices(
        batch_size,
        n_row, 
        n_col
    )
    # print(indices.shape)
    
    # values are simply the raveled 3-d batch tensor
    values = x.ravel()
    # print(values.shape)
    
    # size
    size = (
        batch_size * n_row,
        batch_size * n_col
    )
    
    # construct sparse COO tensor
    x = torch.sparse_coo_tensor(
        indices=indices, # shape (2, n_values)
        values=values, # shape (n_values, )
        size=size, # shape (batch_size * n_col, batch_size * n_row)
        requires_grad=False,
        dtype=dtype
    )
    return x


def sparse_block_diagonalize_3d_coo_tensor(
    coo_3d_tensor: torch.sparse_coo_tensor,
    dtype: torch.Tensor.dtype = torch.float,
    verbosity: int = 0
) -> torch.sparse_coo_tensor:
    """
    Given stacked 3-d sparse COO tensor where each 2-d constituent
    tensor is the same size, this method creates a 2-d 'doubly sparse'
    block-diagonal tensor, where off-blocks are sparse, and within
    blocks are (possibly) sparse.
    
    Note that sparse COO tensors must be coalesced for accurate
    getting operations on their indices and values, which we
    use for constructing block-diagonal P batches during MFCN
    training. See:
    https://pytorch.org/docs/stable/sparse.html#sparse-uncoalesced-coo-docs
    
    Args:
        coo_3d_tensor: 3-d sparse coalesced COO tensor 
        of batched 2-d sparse tensors of same 2-d dense 
        size.
    Returns:
        2-d sparse COO block-diagonal tensor.
    """
    # if not coo_3d_tensor.is_coalesced:
    #     print(f"Warning: \'coo_3d_tensor\' was not coalesced: coalescing...")
    #     coo_3d_tensor = coo_3d_tensor.coalesce()
        
    # given stacked 2-d tensors are same size, extract 
    # first for size variable usage
    t_0 = torch.index_select(
        input=coo_3d_tensor, 
        dim=0, 
        index=torch.tensor([0], dtype=torch.long)
    ) # shape (1, n, n)
    # block-diag. matrix is square with side length batch_size * n_nodes
    sbd_side_len = coo_3d_tensor.shape[0] * coo_3d_tensor.shape[-1]
    sbd_size = (sbd_side_len, sbd_side_len)
    
    # grab indices() of each, add marginal t_0 size to 
    # both row and col indices and concatenate to create full
    # block-diagonal indices set
    # sbd_indices = torch.cat([
    #     torch.index_select(
    #         input=coo_3d_tensor, 
    #         dim=0, 
    #         index=torch.tensor([i], dtype=torch.long)
    #     ).indices()[1:] \
    #     + (i * t_0.shape[1]) \
    #     for i in range(coo_3d_tensor.shape[0])
    # ], dim=1)

    sbd_indices = [None] * coo_3d_tensor.shape[0]
    sbd_values = [None] * coo_3d_tensor.shape[0]
    for i in range(coo_3d_tensor.shape[0]):
        # select out each sparse 2d tensor
        t_i = torch.index_select(
            input=coo_3d_tensor, 
            dim=0, 
            index=torch.tensor([i], dtype=torch.long)
        ) # .coalesce()
        # grab its indices and add marginal block-diagonalizing
        # scalar to both row and col indices
        # note: indices() here gives shape (3, nnz)
        sbd_indices[i] = t_i._indices()[1:] + (i * t_0.shape[1]) 
        # grab its values
        # note: torch.ravel() doesn't work on 3d sparse tensors,
        # else we could just use that without looping
        sbd_values[i] = t_i._values() # shape (nnz, )
    sbd_indices = torch.cat(sbd_indices, dim=1) # cat wider
    sbd_values = torch.cat(sbd_values, dim=0)
    
    if verbosity > 1:
        print(
            '\tt_0.shape =', 
            t_0.shape
        )
        print(
            '\tsbd_indices.shape =', 
            sbd_indices.shape
        )
        print(
            '\tsbd_values.shape =', 
            sbd_values.shape
        )
        print('\tsbd_size =', sbd_size)
        
    blockdiag = torch.sparse_coo_tensor(
        indices=sbd_indices,
        values=sbd_values, 
        size=sbd_size,
        requires_grad=False,
        dtype=dtype
    )
    return blockdiag


def get_rebatched_3d_dense_indices(
    shape: Tuple[int] | torch.Size
) -> np.ndarray:
    """
    Since 'torch.reshape' is not implemented for 
    sparse tensors, this utility function generates
    the (3-row) dense COO indices array for a 3-d tensor.
    Useful for reshaping the product of sparse block-
    diagonal tensor matrix multiplication back into
    a 3-d (e.g. batched) dense tensor.

    Args:
        shape: tuple of ints holding the shape of the
            dense 3-d tensor.
    Returns:
        3-row array of dense 3-d tensor COO (coordinate)
        indices.
    """
    batch_indices = np.repeat(
        range(shape[0]),
        int(shape[1] * shape[2])
    )
    row_indices = np.tile(
        np.repeat(range(shape[1]), shape[2]),
        shape[0]
    )
    col_indices = np.tile(
        range(shape[2]),
        int(shape[0] * shape[1])
    )
    rebatched_dense_indices = np.stack(
        (batch_indices,
         row_indices,
         col_indices)
    )
    return rebatched_dense_indices
    

def scale_tensor_feature_sets(
    features_tensor: torch.Tensor,
    set_idxs_dict: dict,
    scaler_type: str = 'minmax',
    zero_float_threshold: float = 1e-12,
    drop_features_zero_var: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Rescales a feature matrix tensor, training the
    scaler on the train set only. Note that this only works
    for feature vector tensors stacked into a matrix; features 
    with higher-dimensional tensors will need a more complex 
    treatment. See: https://stackoverflow.com/a/72441806.

    Args:
        features_tensor: (n x p)-matrix tensor holding
        the feature values for n samples with p features.
        set_idxs_dict: dictionary with 'train', 'valid'
            and 'test' keying lists of integer indexes
            for the rows in 'features_tensor' belonging
            to the respective sets.
        scaler_type: type of data re-scaling to do. 
        zero_float_threshold: if a number is between 0. and
             this float, it's 0. within our threshold of
             precision. Used for 0-denominator checking.

    Returns:
        Dictionary of rescaled feature matrix 
        tensors by set.
    """
    if features_tensor.ndim > 2:
        print(
            'ERROR: features_tensor.ndim > 2. Scaling not implemented'
            ' for high-dimensional tensors.'
        )
        return None
        
    scaled_set_dict = {}

    # parameterize scaler using train set only
    if scaler_type == 'minmax':
        train_matrix = features_tensor[set_idxs_dict['train']]
        train_mins, _ = torch.min(train_matrix, dim=0, keepdim=True)
        train_maxs, _ = torch.max(train_matrix, dim=0, keepdim=True)
        train_ranges = (train_maxs - train_mins)

        # check if any feature has a max == min (0 range and 0 variance)
        zero_var_feat_idx = np.argwhere(train_ranges < 1e-12).squeeze()
        if zero_var_feat_idx.shape[0] > 0:
            # if indicated, drop such features
            if drop_features_zero_var:
                print(
                    '\nWARNING: zero denominator(s) created during minmax'
                    ' scaling; these features with zero variance will be'
                    ' removed.\n'
                )
                features_tensor = np.delete(
                    arr=features_tensor,
                    obj=zero_var_feat_idx,
                    axis=1
                )
                train_mins = np.delete(
                    arr=train_mins,
                    obj=zero_var_feat_idx,
                    axis=None
                )
                train_maxs = np.delete(
                    arr=train_maxs,
                    obj=zero_var_feat_idx,
                    axis=None
                )
                train_ranges = np.delete(
                    arr=train_ranges,
                    obj=zero_var_feat_idx,
                    axis=None
                )
            else:
                print(
                    '\nERROR: zero denominator(s) created during minmax'
                    ' scaling. This will create NaNs in feature tensors.'
                    ' Consider dropping features with zero variance.\n'
                )
                return None
        # print('train_mins', train_mins)
        # print('train_maxs', train_maxs)
    
        # apply scaler to train/valid/test sets
        for set in set_idxs_dict.keys():
            # subset features matrix by set 
            set_matrix = features_tensor[set_idxs_dict[set]]
            # rescale set and contain in output dict
            scaled_set_dict[set] = (set_matrix - train_mins) \
                / train_ranges
    else:
        print(f'\nERROR: scaler type \'{scaler_type}\' not implemented!\n')
        return None
        
    return scaled_set_dict


def get_dataloaders_dict(
    datasets_dict: DictDataset, # dictionary
    seed: int,
    batch_sizes_dict: dict, 
    dataloader_subclass: bool = None,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False
) -> Dict[str, DataLoader]:
    """
    Loads dict of dataset (with 'train'/'valid'/'test' keys)
    into a similar dict of torch DataLoaders (or subclass thereof).

    References:
    Torch's DataLoader class: 
    https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    pin_memory param: 
    https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

    Args:
        datasets_dict: dict of train/valid/test DictDatasets.
        seed: seed for Dataloader's Generator.
        batch_sizes_dict: dictionary of train/valid/test 
            batch sizes.
        dataloader_subclass: optional subclass of torch's DataLoader
            class to use instead of the main class.
        num_workers: number of workers to use when loading data.
        drop_last: whether to drop leftover samples when batch_size
            doesn't split a dataset evenly.
        pin_memory: whether to pin data in device memory; may improve 
            loading speeds with GPUs.
    Returns:
        Dictionary of train/valid/test dataloaders.
    """
    # if no DataLoader subclass is provided, use main class
    if dataloader_subclass is None:
        dataloader_subclass = DataLoader
        generator = Generator()
        generator.manual_seed(seed)
        dataloader_kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': drop_last,
            'generator': generator
        }
        
    dataloaders_dict = {}
    for set_name, dataset in datasets_dict.items():
        shuffle = 'train' in set_name # (set_name == 'train') # only shuffle train set
        # print(f'shuffle: {shuffle}')
        dataloaders_dict[set_name] = dataloader_subclass(
            dataset, 
            batch_size=batch_sizes_dict[set_name],
            shuffle=shuffle,
            **dataloader_kwargs
        )
    return dataloaders_dict


def subset_pyg_Data_edge_index(
    edge_index: torch.Tensor,
    set_mask: torch.Tensor,
) -> torch.Tensor:
    """
    For a PyTorch Geometric Data object, subsets its 
    edge_index for the train/valid/test set 

    Args:
        edge_index: 'edge_index' attribute of the Data
            object, shape [x, n_edges].
        set_mask: e.g. 'train_index' attribute of the 
            Data object, shape [n_nodes].
    Returns:
        New edge_index tensor, shape [2, new_n_edges].
    """
    # -> remove any edge_index elements (in both rows)
    # where either row's entry isn't in the set mask
    # (note: True * False = False)
    set_idx = torch.argwhere(set_mask).squeeze()
    mask = torch.isin(edge_index[0], set_idx) \
        * torch.isin(edge_index[1], set_idx)
    set_edge_index = torch.stack((
        edge_index[0][mask],
        edge_index[1][mask]
    ))
    
    return set_edge_index

