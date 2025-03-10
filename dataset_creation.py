"""
Function to create a dataset for
MFCN / wavelet filtration of manifolds.

This function can also compute and store
geometric scattering moments, for use
as features (e.g., to feed a regressor/
classifier), if desired.
    - set 'SAVE_SCAT_MOMENTS_OBJS' to True
    in the 'args' passed; also set 'SAVE_SPECTRAL_OBJS'
    to true to save spectral scattering moments,
    or set 'WAVELET_TYPE' to 'p' to save diffusion
    scattering moments.
"""
import sys
sys.path.insert(0, '../')

import manifold_sampling as ms
import graph_construction as gc
import wavelets as w
import utilities as u
import data_utilities as du
import pyg_utilities as pygu

import pickle
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.base import spmatrix
from scipy.sparse.linalg import eigsh
from typing import (
    Tuple,
    List,
    Dict,
    Optional,
    Any
)
import torch
import torch_geometric
from torch_geometric.data import (
    Data,
    Dataset,
    InMemoryDataset
)
from torch_geometric.loader import DataLoader


def create_manifold_wavelet_dataset(
    args,
    manifolds_dictl: List[dict],
    pickle_dictl_out: bool = True
) -> None | List[Dict[str, Any]]: # pickles or returns
    """
    
    Args:
        args: an ArgsTemplate object.
        manifolds_dictl: list of dictionaries holding
            keyed objects for each manifold in a 
            manifold dataset (e.g., features tensor,
            target value, etc.).
        pickle_dictl_out: bool whether to pickle the
             output list of dictionary, or else return
             it.
    Returns:
        Either None (and pickles the output),
        or returns a list of dictionaries, each
        dictionary being one data sample's record,
        containing its id, target, channel signals,
        etc.
    """

    if args.COORD_FN_AXES == 'all':
        coord_fn_axes = tuple(range(args.MANIFOLD_N_AXES))
    
    # if args.SAVE_SPECTRAL_OBJS:
    #     args.RAW_DATA_FILENAME += '_spectral'
    # if args.SAVE_P_OBJS:
    #     args.RAW_DATA_FILENAME += '_P'
    # if args.SAVE_SCAT_MOMENTS_OBJS:
    #     args.RAW_DATA_FILENAME += '_SMs'
    save_filename = args.RAW_DATA_FILENAME + '.pkl'

    """
    time overall dataset generation
    """
    print('Generating manifold dataset...')
    t_0 = time.time()
    
    
    """
    init empty list of dicts to hold manifolds'
    parameters, targets, and scattering moments
    
    final structure:
    'key': value [e.g. training targets for the manifold]
    'scattering_moments':
        |-> wavelet type: 'spectral' | 'P'
            |-> scattering moment order: 0 | 1 | 2
                |-> arrays of scattering moment values:
                    rows = Wjs (args.J/jprime...args.J); 
                    cols = qth-order norms (1...args.Q)
    """
    # init empty lists to populate
    out_dictl = [None] * args.N_MANIFOLDS
    Pnfs = [None] * args.N_MANIFOLDS
    graph_Laplacians = [None] * args.N_MANIFOLDS
    # graph_Ps = [None] * args.N_MANIFOLDS
    
    """
    function-on-manifold values
    """
    for i, manifold_dict in enumerate(manifolds_dictl):

        # loop results container
        out_dictl[i] = {}
        
        # manifold = manifold_dict[args.MANIFOLD_KEY]
        manifold_signals = manifold_dict[args.FN_ON_MANIFOLD_KEY]
        manifold_geometry = manifold_dict[args.MANIFOLD_GEOMETRY_KEY]

        # in case pandas dataframes with non-numeric cols were passed
        # -> convert to scipy sparse matrices (dropping non-numeric cols)
        manifold_signals = du.possible_pd_df_to_array(
            manifold_signals,
            array_type_key='scipy-sparse',
            warn=False
        )
        manifold_geometry = du.possible_pd_df_to_array(
            manifold_geometry,
            array_type_key='scipy-sparse',
            warn=False
        )
        
        # compute normalized-evaluated signals on manifolds
        Pnfs[i] = ms.get_manifold_coords_as_fn_vals(
            manifold_signals,
            args.COORD_FN_AXES,
            norm_evaluate=True
        )
        if i == 0:
            print('Pnfs[0].shape:', Pnfs[i].shape)
    
        # construct graph
        # 'W' is the sparse, eta-kernelized weighted adjacency matrix
        # (but where eta = indicator (default), weights are all 1 or 0)
        if args.GRAPH_TYPE == 'knn':
            graph = gc.KNNGraph(
                x=manifold_geometry,
                n=manifold_geometry.shape[0], # args.N_PTS_ON_MANIFOLD
                k=args.K_OR_EPS, # 'auto',
                d_manifold=args.D_MANIFOLD,
                eta_type=args.ETA_TYPE
            )
        elif args.GRAPH_TYPE == 'epsilon':
            graph = gc.EpsilonGraph(
                x=manifold_geometry,
                n=manifold_geometry.shape[0], # args.N_PTS_ON_MANIFOLD 
                eps=args.K_OR_EPS, # 'auto'
                d_manifold=args.D_MANIFOLD,
                eta_type=args.ETA_TYPE
            )

        # manifold coords array no longer needed -> delete
        # del manifold_dict[args.MANIFOLD_KEY]
        del manifold_signals

        if args.SAVE_SPECTRAL_OBJS:
            # compute Laplacians and Ps (lazy random walk matrices)
            # save graph Laplacians (for spectral wavelets) and LRWMs to list
            graph.calc_Laplacian()
            # print(type(graph.L))
            graph_Laplacians[i] = graph.L

        # if args.SAVE_P_OBJS:
        if args.WAVELET_TYPE.lower() == 'p':
            # Ws are already calculated when calculating Laplacians
            if not args.SAVE_SPECTRAL_OBJS:
                graph.calc_W()
            # print('graph.W.shape:', graph.W.shape)
            P = gc.calc_LRWM(graph.W, normalize=False)
            # print(P.shape)
            # graph_Ps[i] = P
            out_dictl[i]['P'] = P

        # save manifold objects in 'out_dictl'
        out_dictl[i] = out_dictl[i] | manifold_dict
        if args.SAVE_GRAPH:
            out_dictl[i]['W'] = graph.W
        if args.SAVE_FN_VALS:
            out_dictl[i]['Pnfs'] = Pnfs[i]
            
    
    """
    spectral wavelet operators and/or scattering moments
    """
    if args.SAVE_SPECTRAL_OBJS:
        print(
            f'Working on spectral decomp. of L objects...'
            f'\n\tkappa = {args.KAPPA}'
        )

        # track total time for all sparse eigendecompositions
        if args.VERBOSITY > 0:
            t_eigendecomp_start = time.time()
            
        for i, L in enumerate(graph_Laplacians):
            # track time for each sparse eigendecomposition
            if args.VERBOSITY > 1:
                print(f"eigendecomposing graph {i}")
                t_i_0 = time.time()
                
            spectral_sm_dict = {}
            # decompose kappa + 1 eigenpairs, since we discard first
            # (trivial / constant, with lambda = 0)
            eigenvals, eigenvecs = eigsh(L, k=args.KAPPA + 1, which='SM')
            # eigenvals[0] = 0.
            # print('eigenvecs.shape:', eigenvecs.shape) # shape (N, k)
            # don't include first (trivial) eigenpair
            eigenvals, eigenvecs = eigenvals[1:], eigenvecs[:, 1:]
            out_dictl[i]['L_eigenvals'] = eigenvals
            out_dictl[i]['L_eigenvecs'] = eigenvecs
            

            # if args.WAVELET_TYPE is not None:
                # Wjs_spectral = w.spectral_wavelets(
                #     eigenvals=eigenvals, 
                #     J=args.J,
                #     include_low_pass=args.INCLUDE_LOWPASS_WAVELET
                # )
                # out_dictl[i]['Wjs_spectral'] = Wjs_spectral
                
            # if args.NON_WAVELET_FILTER_TYPE == 'spectral':
            #     lowpass_filters = w.spectral_lowpass_filter(eigenvals)
            #     out_dictl[i]['spectral_lowpass_filters'] = lowpass_filters
        
            if args.SAVE_SCAT_MOMENTS_OBJS:
                # print('Calculating spectral-based scattering moments...')
                
                # spectral_sm_dict['all_feats'] = w.get_spectral_wavelets_scat_moms_dict(
                #     L=L,
                #     eigenvals=eigenvals,
                #     eigenvecs=eigenvecs,
                #     Pnf=Pnfs[i],
                #     J=args.J,
                #     Q=args.Q,
                #     include_low_pass=args.INCLUDE_LOWPASS_WAVELET,
                #     verbosity=args.VERBOSITY
                # )
                Pnf = Pnfs[i]
                if sp.isspmatrix(Pnf):
                    Pnf = Pnf.toarray()
                for j in coord_fn_axes:
                    spectral_axis_sm_dict = w.get_spectral_wavelets_scat_moms_dict(
                        L=L,
                        eigenvals=eigenvals,
                        eigenvecs=eigenvecs,
                        Pnf=Pnf[:, j],
                        J=args.J,
                        Q=args.Q,
                        include_low_pass=args.INCLUDE_LOWPASS_WAVELET,
                        verbosity=args.VERBOSITY
                    )
                    key = f'feat_{j+1}'
                    spectral_sm_dict[key] = spectral_axis_sm_dict
                
                # save in manifold's existing dict
                out_dictl[i]['scattering_moments'] = {}
                out_dictl[i]['scattering_moments']['spectral'] = spectral_sm_dict
            if args.VERBOSITY > 1:
                t_i_eigendecomp = time.time() - t_i_0
                t_min, t_sec = u.get_time_min_sec(t_i_eigendecomp)
                print(f'\t{t_min:.0f}min, {t_sec:.4f}sec.')
                
        # after all graphs have been eigendecomposed
        if args.VERBOSITY > 0:
            t_eigendecomp = time.time() - t_eigendecomp_start
            t_min, t_sec = u.get_time_min_sec(t_eigendecomp)
            print(f'Data processing time ({args.N_MANIFOLDS} graphs):')
            print(f'\t{t_min:.0f}min, {t_sec:.4f}sec. total')
            print(f'\t{t_eigendecomp / args.N_MANIFOLDS:.4f} sec/graph')
    
    
    """
    P (lazy random walk / diffusion) wavelet scattering moments
    """
    if args.WAVELET_TYPE.lower() == 'p' and args.SAVE_SCAT_MOMENTS_OBJS:
        
        print('Calculating P-based scattering moments...')
        for i, out_dict in enumerate(out_dictl):

            # init scattering moments results container
            P_wavelets_sm_dict = {}

            # grab ith manifold's function values
            Pnf = Pnfs[i]
            if sp.isspmatrix(Pnf):
                Pnf = Pnf.toarray()

            # calc scattering moments for each function/signal channel
            for j in coord_fn_axes:
                P_chan_sm_dict = w.get_P_wavelets_scat_moms_dict(
                    P=out_dict['P'],
                    Pnf=Pnf[:, j],
                    J=args.J,
                    Q=args.Q,
                    include_lowpass=args.INCLUDE_LOWPASS_WAVELET,
                    verbosity=args.VERBOSITY
                )
                key = f'feat_{j+1}'            
                P_wavelets_sm_dict[key] = P_chan_sm_dict
            
            # enter into dict
            if 'scattering_moments' not in out_dictl[i]:
                out_dictl[i]['scattering_moments'] = {}
            out_dictl[i]['scattering_moments']['p'] = P_wavelets_sm_dict


    if pickle_dictl_out:
        """
        pickle dataset (list of dicts)
        """
        save_path = f'{args.DATA_DIR}/{save_filename}'
        with open(save_path, "wb") as f:
            pickle.dump(out_dictl, f, protocol=pickle.HIGHEST_PROTOCOL)  
        print(f'Data saved as \'{save_filename}\'.')
        out_dictl = None
    else:
        pass # returns non-None 'out_dictl' below
    
    t_overall = time.time() - t_0
    t_min, t_sec = u.get_time_min_sec(t_overall)
    print(
        f'{args.N_MANIFOLDS} manifold'
        f' {args.GRAPH_TYPE} graphs and associated objects generated'
        f' in:\n\t{t_min:.0f}min, {t_sec:.4f}sec.'
    )

    return out_dictl


class WaveletPyGData(Data):
    """
    Subclass of PyG 'Data' class that collates
    wavelet filter attributes in new (first) dimension,
    instead of concatenating in dim 0 (default).

    Note that `mfcn.get_Batch_spectral_Wjxs` expects 
    `Wjs_spectral` (spectral filter tensors) in the shape 
    (n_graphs, n_eigenpairs, n_filters).
    
    Reference:
    https://pytorch-geometric.readthedocs.io/en/2.5.0/advanced/batching.html
    """
    def __cat_dim__(self, key, value, *args, **kwargs):
        # note: these keys have been deprecated, and
        # shape of 'Wjs_spectral' from w.spectral_wavelets transposed
        if (key =='L_eigenvals'):
        # or (key =='Wjs_spectral') \
        # or (key == 'spectral_lowpass_filters'):
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


def pickle_as_pytorch_geo_Data(
    args,
    data_dictl: Optional[List[Dict[str, Any]]] = None,
    manual_save_path: Optional[str] = None
) -> None:
    """
    Loads a manifold wavelet dataset
    into a list of PyTorch Geometric 
    Data Objects, and pickles it.

    Args:
        args: ArgsTemplate subclass, holding
            experiment arguments.
        data_dictl: if a list of dictionaries of 
            data objects already exists, use what's
            passed to this arg; else create it from
            data at args.RAW_DATA_FILENAME.
        manual_save_path: if not None, save
            result to this filename; else use what's at
            args.PYG_DATA_LIST_FILENAME.
    Returns:
        None (pickles the list).
    """
    if data_dictl is None:
        print(f'Creating PyG dataloaders from \'{args.RAW_DATA_FILENAME}\'')
    
        # open pickled data dict list
        pyg_data_list_filepath = f'{args.DATA_DIR}/{args.RAW_DATA_FILENAME}'
        with open(pyg_data_list_filepath, "rb") as f:
            data_dictl = pickle.load(f)
    
    # create master list of torch_geometric.data.Data objects
    data_list = [None] * len(data_dictl)
    for i, data_dict in enumerate(data_dictl):

        # additional attribute(s) for Data objects
        extra_attribs_kwargs = {}

        # 'x': input signal data tensor
        # shape (n_nodes, n_features/channels)
        Pnfs = data_dict['Pnfs']
        if isinstance(Pnfs, spmatrix): # sp.coo_matrix
            # if not a scipy 'coo_matrix' but another sparse type,
            # first convert to 'coo_matrix', then to 'sparse_coo_tensor'
            if not isinstance(Pnfs, sp.coo_matrix):
                Pnfs = Pnfs.tocoo()
            indices = torch.tensor(
                np.stack((Pnfs.row, Pnfs.col)), 
                dtype=torch.long
            )
            values = torch.tensor(Pnfs.data, dtype=torch.float)
            x = torch.sparse_coo_tensor(
                indices, 
                values, 
                Pnfs.shape
            )
        else:
            x = torch.tensor(
                Pnfs, 
                dtype=args.FEAT_TENSOR_DTYPE
            )
        
        edge_index = torch.tensor( # shape (2, n_edges)
            np.stack(data_dict['W'].nonzero()), 
            dtype=torch.long
        )

        # print(data_dict['W'])
        if args.ETA_TYPE != 'indicator':
            edge_weight = torch.tensor(
                data_dict['W'].data,
                dtype=torch.float
            )
            extra_attribs_kwargs['edge_weight'] = edge_weight
        
        y = torch.tensor(
            data_dict[args.TARGET_NAME],
            dtype=args.TARGET_TENSOR_DTYPE
        )
        
        # optional: log-transform main target
        if args.LOG_TRANSFORM_TARGETS:
            y = torch.log(y)
        
        # graph spectral attributes    
        if 'L_eigenvecs' in data_dict:
            L_eigenvecs = torch.tensor(
                data_dict['L_eigenvecs'], 
                requires_grad=False,
                dtype=args.FLOAT_TENSOR_DTYPE
            )
            extra_attribs_kwargs['L_eigenvecs'] = L_eigenvecs

        if 'L_eigenvals' in data_dict:
            L_eigenvals = torch.tensor(
                data_dict['L_eigenvals'], 
                requires_grad=False,
                dtype=args.FLOAT_TENSOR_DTYPE
            )
            extra_attribs_kwargs['L_eigenvals'] = L_eigenvals
            
        # finally, create and add Data object to master list
        data = WaveletPyGData(
            x=x, 
            y=y,
            edge_index=edge_index,
            **extra_attribs_kwargs
        )
        data_list[i] = data

    save_path = manual_save_path \
        if (manual_save_path is not None) \
        else f'{args.DATA_DIR}/{args.PYG_DATA_LIST_FILENAME}'
    if save_path[-4:] != '.pkl':
        save_path += '.pkl'
    # save_path = f'{args.DATA_DIR}/{save_filename}'
    with open(save_path, "wb") as f:
        pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)  
    print(f'PyG Data list saved to \'{save_path}\'.')



class PyG_DataContainer_Creator:
    """
    Splits and loads a manifold wavelet dataset
    into a dict of train/valid/test (PyTorch Geometric) 
    DataLoaders, or a single Data object (with train/valid/
    test mask attributes, in the case of node classification 
    for a single graph).

    NOTE: ubclass this class, overriding the `_load_data_list` 
    method if custom data loading is needed for a PyG
    dataset.

    NOTE: if dealing with a one-graph (node-level) dataset,
    this method will attempt to move the (lone) Data object 
    onto the device specified in 'args'.
    
    This class can also return a positive class rebalancing
    weight, if doing binary classification (done in 
    `get_pos_class_bal_wt`, and in this class, since this 
    is where the train set is split, and this weight should
    be calculated only on the train set).

    NOTE: if dataset is from oversampling,
    samples from same subject should be in
    the same train/valid/test set, and those
    split indexes should be generated upstream
    and fed to this function in 'set_idxs_dict'.

    Args:
        args: ArgsTemplate subclass, holding
            experiment arguments.
        pyg_data_list_filepath: manual override of filepath
            to the PyG Data list dataset pickle.
        set_idxs_dict: optional dictionary of
            index arrays keyed by set name
            ('train'/'valid'/'test'). If 'None',
            set indexes are generated here, using
            the args. 
        verbosity: controls volume of print output
            as function executes.
    Can return:
        Tuple of: (1) PyG Data object (single graph,
        node-level task) with train/valid/test masks as
        attributes; or a dictionary of PyGDataLoaders
        (multiple graphs, graph-level task) keyed by set 
        name, and (2) positive class rebalancing weight 
        (float) if doing binary classification.
    """
    def __init__(
        self,
        args,
        pyg_data_list_filepath: Optional[str] = None,
        set_idxs_dict: Optional[Dict[str, int]] = None,
        verbosity: int = 0
    ) -> None:
        self.args = args
        self.task = args.TASK.lower()
        self.pyg_data_list_filepath = pyg_data_list_filepath
        # [if needed] generate new train/valid/test set split idxs
        if set_idxs_dict is None:
            self.set_idxs_dict = du.get_train_valid_test_idxs(
                seed=args.TRAIN_VALID_TEST_SPLIT_SEED,
                n=len(data_list),
                train_prop=args.TRAIN_PROP,
                valid_prop=args.VALID_PROP
            )
        else:
            self.set_idxs_dict = set_idxs_dict
        self.verbosity = verbosity

        
    def _load_data_list(self) -> None:
        if self.pyg_data_list_filepath is None:
            self.pyg_data_list_filepath = f'{self.args.DATA_DIR}/' \
                + f'{self.args.PYG_DATA_LIST_FILENAME}'
        # open pickled PyG Data object list
        with open(self.pyg_data_list_filepath, "rb") as f:
            self.data_list = pickle.load(f)

        # if self.args.EXCLUDED_FEAT_IDX is not None:
        #     self.data_list = [
        #     ]
    
    
    def get_pos_class_bal_wt(self) -> float:
        """
        [optional] extract positive class balance weight
        from train set, for use in reweighting the loss
        calculation in classification tasks.
        """
        if ('bin' in self.task) and ('class' in self.task):
            train_idx = self.set_idxs_dict['train']
            train_set = [self.data_list[i] for i in train_idx]
            train_targets = [data.y for data in train_set]
            rebalance_pos_wt = du.get_binary_class_rebalance_weight(
                train_targets
            )
            p = ct_1s / n
            perc_1s = p * 100
            mcc_acc = (100 - perc_1s) if (perc_1s < 50) else perc_1s
            mcc_f1 = (2 * p) / (p + 1)
    
            if self.verbosity > 0:
                print(f'binary target % of 1s: {perc_1s:.1f}%')
                print(f'\t-> balanced positive class weight: {rebalance_pos_wt:.2f}')
                print(f'\t-> majority-class classifier accuracy: {mcc_acc:.1f}%')
                print(
                    f'\t-> majority-class classifier F1: '
                    f'{mcc_f1:.2f} (if maj. class is pos. class)'
                )
            return rebalance_pos_wt
        else:
            warnings.warn(
                f"Calculating rebalancing weights for tasks other than" 
                f" binary classification has not been implemented!"
            )
            return None


    def _get_node_task_data(self) -> Data:
        # [node-level task] if there's only one graph in the dataset
        if len(self.data_list) == 1:
            data = self.data_list[0]

            # assign 'set_idxs_dict' idxs to the 'mask' attributes of the 
            # graph's Data object
            for i, (set_key, idx) in enumerate(self.set_idxs_dict.items()):
                set_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                set_mask[idx] = True
                
                if 'train' in set_key.lower():
                    data.train_mask = set_mask
                elif 'val' in set_key.lower():
                    data.val_mask = set_mask
                elif 'test' in set_key.lower():
                    data.test_mask = set_mask

            # move the one-graph dataset onto device
            data.to(args.DEVICE)
        
            return data

        # [node-level task] if there are still multiple graphs in the 
        # dataset for some reason
        else:
            raise NotImplementedError(
                f"Creating Dataloaders for node-level tasks for multiple" 
                f" graphs not yet been implemented!"
            )


    def _get_graph_task_dataloader(self) -> Dict[str, DataLoader]:
        # create dict of dataloaders for data_container
        data_container = {}

        # populate dict with set dataloaders
        for i, (set_key, idx) in enumerate(self.set_idxs_dict.items()):
            dataset = [self.data_list[i] for i in idx]
            # print('dc.get_pyg_data_or_dataloaders: len(dataset)', len(dataset))
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.args.BATCH_SIZES[i],
                shuffle=('train' in set_key),
                drop_last=self.args.DATALOADER_DROP_LAST
            )
            data_container[set_key] = dataloader
        return data_container

    
    def get_data_container(self) -> Data | Dict[str, DataLoader]:
        """
        Loads the PyG data list, then calls proper internal 
        function to create Data or DataLoader as the data_container.
        """
        # open data list
        self._load_data_list()
        
        # node-level tasks
        if 'node' in self.args.TASK:
            data_container = self._get_node_task_data()
            
        # graph-level tasks
        elif 'graph' in self.args.TASK:
           data_container = self._get_graph_task_dataloader()
            
        # other tasks: not implemented
        else:
            raise NotImplementedError(
                    f"Creating Dataloaders for this task ({self.args.TASK})"
                    f" has not been implemented! Did you forget 'node' or"
                    f" 'graph' in args.TASK?"
                )
            
        return data_container
        


class PyG_Dataset_DataContainer_Creator(PyG_DataContainer_Creator):
    """
    This subclass of PyG_DataContainer_Creator overrides 
    `_load_data_list` to load and pre-process (e.g., make sparser) datasets
    in PyG's Datasets collection. Where the node feature (e.g., atom 
    1-hot encodings) matrices have lots of 0s; this class makes features
    matrix sparse.
    """
    # def __init__(
    #     self,
    #     pyg_dataset_type: str = 'TUDataset',
    #     args,
    #     pyg_data_list_filepath: Optional[str] = None,
    #     set_idxs_dict: Optional[Dict[str, int]] = None,
    #     verbosity: int = 0
    # ) -> None:
    #     super(TUDataset_PyG_DataContainer_Creator, self).__init__(
    #         args,
    #         pyg_data_list_filepath,
    #         set_idxs_dict,
    #         verbosity
    #     )
    #     self.pyg_dataset_type = pyg_dataset_type

    
    def _load_data_list(self) -> None:
        if self.args.PYG_DATASET_COLLECTION == 'TUDataset':
            from torch_geometric.datasets import TUDataset
            dataset_class = TUDataset
        elif self.args.PYG_DATASET_COLLECTION == 'MoleculeNet':
            from torch_geometric.datasets import MoleculeNet
            dataset_class = MoleculeNet
            
        # this will download the dataset if it doesn't already exist
        # in the 'root' arg directory
        data_list = dataset_class(
            root=self.pyg_data_list_filepath,
            name=self.args.PYG_DATASET_KEY
        )

        # patch: Tox21 has missing y labels (if compound not tested on
        # that particular target, I'm guessing) -> prevent NaNs!
        # also cuts other extra attributes we don't use (edge_attr, smiles)
        if self.args.PYG_DATASET_KEY == 'Tox21':
            # drop samples with NaN for the target
            data_list = [
                d for d in data_list \
                if not torch.isnan(d.y[:, self.args.TARGET_INDEX])
            ]
            data_list = [
                Data(
                    x=d.x, 
                    edge_index=d.edge_index, 
                    y=d.y.squeeze()[self.args.TARGET_INDEX]
                    # y=torch.where(torch.isnan(d.y), 0, d.y) \
                    #     .squeeze()[self.args.TARGET_INDEX]
                ) \
                for d in data_list
            ]

        # for each graph's features matrix (x): remove features if needed and sparsify
        self.data_list = pygu.exclude_features_sparsify_Data_list(
            data_list=data_list,
            exclude_feat_idx=self.args.EXCLUDED_FEAT_IDX,
            target_tensor_dtype=self.args.TARGET_TENSOR_DTYPE,
        )
        # else:
        #     self.data_list = [
        #         Data(
        #             x=g.x.to(self.args.TARGET_TENSOR_DTYPE).to_sparse(), 
        #             edge_index=g.edge_index, 
        #             y=g.y.to(self.args.TARGET_TENSOR_DTYPE)
        #         ) \
        #         for g in data_list
        #     ]


def get_peptides_func_dataloaders_dict(
    args,
) -> Tuple[Dict[str, DataLoader], float]:
    """
    Loads the 'peptides_func' dataset from the LRGBDataset
    PyG collection, into a dictionary of PyG DataLoaders
    keyed by split ('train', 'valid', or 'test'). Also
    computes the class rebalancing weight (for binary
    classification, if one target has been selected, and
    if loss will be rebalanced during training).

    Returns:
        A 2-tuple of (1) the dictionary of DataLoaders by
        split, and (2) the class rebalancing weight (float)
        for the positive class in a binary classification.
    """
    from torch_geometric.datasets import LRGBDataset
    # from infogain_testing import peptides_func_args as a
    data_container = {}
    
    for split_i, split in enumerate(('train', 'val', 'test')):
    
        # extract dataset split
        split_list = LRGBDataset(
            root=args.DATA_DIR,
            name="Peptides-func", 
            split=split
        )
    
        # if a binary classification problem and weighting loss
        # to balance classes, calculate positive class weight
        if ('bin' in args.TASK) and ('class' in args.TASK) \
        and (args.REWEIGHT_LOSS_FOR_IMBALANCED_CLASSES) \
        and (split == 'train'):
            train_targets = [d.y for d in split_list]
            pos_weight = du.get_binary_class_rebalance_weight(
                train_targets
            )
        else:
            pos_weight = None
    
        # process
        # convert to floats and grab only 1 (binary) target
        split_list = [
            Data(
                x=d.x.to(torch.float), 
                edge_index=d.edge_index, 
                y=d.y.to(torch.float).squeeze()[args.TARGET_INDEX]
            ) \
            for d in split_list
        ]
    
        # exclude features (if applicable) and optionally sparsify
        # feature matrices
        split_list = pygu.exclude_features_sparsify_Data_list(
            data_list=split_list,
            target_tensor_dtype=args.TARGET_TENSOR_DTYPE,
            exclude_feat_idx=args.EXCLUDED_FEAT_IDX, # ok if None
            sparsify_feats=False
        )
    
        # put split in a dataloader
        dataloader = DataLoader(
            dataset=split_list,
            batch_size=args.BATCH_SIZES[split_i],
            shuffle=(split == 'train'),
            num_workers=args.DATALOADER_N_WORKERS,
            drop_last=args.DATALOADER_DROP_LAST
        )
    
        # contain dataloaders in a dict keyed by split
        full_split_key = 'valid' if (split == 'val') else split
        data_container[full_split_key] = dataloader

    return data_container, pos_weight


