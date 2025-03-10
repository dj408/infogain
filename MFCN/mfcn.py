"""
Module class for a Manifold Filter-Combine 
Network (MFCN), and other helper classes
(WaveletMFCNDataset and KDropData, subclasses
of pytorch-geometric Dataset and Data classes, 
respectively).
"""
import sys
sys.path.insert(0, '../')
import base_module as bm
import vanilla_nn as vnn
import data_utilities as du
import wavelets as w
import infogain
import nn_utilities as nnu
import pyg_utilities as pygu
from utilities import generate_random_integers

from numpy import (
    nanmax,
    linspace,
    log2
)
from numpy.random import RandomState
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Dataset
)
from torch import linalg as LA
# from torch_geometric.utils import to_torch_coo_tensor
from torch_geometric.loader import DataLoader
from torch_geometric.data import (
    Data,
    Dataset,
    InMemoryDataset,
    Batch
)
import os
import pickle
import warnings
from itertools import accumulate
from typing import (
    Tuple,
    List,
    Dict,
    Any,
    Optional,
    Iterable,
    Callable
)
import matplotlib.pyplot as plt

# for parallel processing in `infogain.calc_custom_P_wavelet_scales`
# import copyreg
# import types
# import multiprocessing as mp
# import torch.multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor

# def _pickle_method(m):
#     if m.im_self is None:
#         return getattr, (m.im_class, m.im_func.func_name)
#     else:
#         return getattr, (m.im_self, m.im_func.func_name)

# copyreg.pickle(types.MethodType, _pickle_method)






class MFCN_Module(bm.BaseModule):
    r"""
    For the following args definitions:
    Let 'n' be the number of samples / points on 
    a manifold, and 'k' be the number of eigenpairs 
    evaluated from the graph Laplacian of each manifold.

    __init__ args:
        task: string key containing 'graph' or 'node' substring
            for model task type.
        n_channels: number of signal input channels
            present in the model input ('x') tensor.
        wavelet_type: 'P' or 'spectral', indicating
            the type of wavelet filter operator used (defaults to
            'P' lazy random walk wavelets).
        non_wavelet_filter_type: 'P' or 'spectral', indicating
            the type of non-wavelet filter operator used, e.g.
            in a simpler MCN model. Leave None to default to wavelet
            filters instead.
        filter_c: optional float constant to apply to a 
            spectral filter, e.g. $e^{-\lambda} \rightarrow
            e^{-c\lambda}$ for lowpass filters. Also applies to
            dyadic wavelets.
        p_wavelet_scales: 'dyadic' or 'custom' wavelet scale
            type used.
        num_nodes_one_graph: number of nodes on the one graph,
            for node-level tasks.
        J: max index of wavelet filters. In a dyadic scale, $J = 4$ corresponds
            to $2^4 = 16$, the maximum polynomial order for spectral wavelets,
            or the maximum diffusion step for diffusion wavelets.
        P_wavelets_channels_t_is: pre-computed output tensor from 
            `infogain.calc_custom_P_wavelet_scales`, containing the non-dyadic 
            indices/powers of t (diffusion step) marking the custom
            wavelet boundaries; shape (n_channels, n_ts).
        cross_Wj_ns_combos_out_per_chan: m-tuple holding the desired
            number of new filtered signal combinations to
            create per channel, at each cross-filter convolution step in
            each cycle, where m is the number of cycles. Note 
            that the values here can be decreasing (from n_filters)
            to help mitigate a combinatorial explosion.
        within_Wj_ns_chan_out_per_filter: m-tuple holding the desired
            number of new cross-channel, within-filter combinations of
            input signals at each 'combine' step.
        channel_pool_moments: [graph-level tasks] if not None and 'channel_pool_key'
            is 'moments', this option computes moments of each channel of signal
            within each graph, of orders specified in its tuple (e.g., 1...4).
            This is a form of pooling across nodes (within channels), before 
            concatenating and feeding to an MLP head.
        channel_pool_key: see 'channel_pool_moments' argument description.
        node_pooling_key: 'mean', 'max', or 'linear' (for a learned linear
            combination) for pooling across final channels (within nodes).
            For node-level tasks, the output of this final pooling is the
            model output/predictions; for graph-level tasks, the output is
            fed to an MLP head.
        node_pool_linear_out_channels: number of final channels per node 
            after linear layer node pooling) to output when no fully-connected 
            head is used. Similar to 'out_channels' in pytorch geometric's
            GCN, etc., model classes.
        use_skip_connections: whether to add a skip connection concatenating (the
            pooled outputs, such as scattering moments of) the original input 
            (before MFCN cycles) and that of each MFCN layer, into the 
            fully-connected network. (This way, one can use 'zeroth-', 'first-', 
            and 'second-order' scattering moments.)
        use_input_recombine_layer: whether to use a linear layer for recombining 
            the original input channels (if 'use_skip_connections', will also 
            serve as an additional skip connection feature.
        input_recombine_layer_kwargs: kwargs for nn.Linear used
            if 'use_input_recombine_layer' is True (excluding 'in_features', which
            gets set to 'n_channels').
        use_k_drop: bool whether, when using 'P' (diffusion) wavelets, to
            randomly drop 'k_drop' edges from each node while training before
            calculating the P operator. Note that in order to use this step,
            the py-g Batch must have been constructed from KDropData objects,
            (a subclass of py-g's Data object).
        k_drop_kwargs: if 'use_k_drop', kwargs to feed the 'batch_drop_rand_k_edges'
            method.
        mfcn_nonlin_fn: nonlinear activation function within the MFCN
            model.
        mfcn_nonlin_fn_kwargs: kwargs for 'mfcn_nonlin_fn'.
        mfcn_wts_init_fn: the torch.nn parameter initialization
            function desired to initialize within-Wj and cross-Wj trainable
            parameter weights.
        mfcn_wts_init_kwargs: kwargs to pass to the 'mfcn_wts_init_fn'.
        base_module_kwargs: kwargs to pass to the BaseModule super class.
        fc_kwargs: kwargs to pass to a fully-connected layer.
        verbosity: integer controlling print output volume during
            methods executions.

    """
    def __init__(
        self,
        in_channels: int,
        wavelet_type: Optional[str] = 'p',
        non_wavelet_filter_type: Optional[str] = None, # defaults to wavelet filters
        filter_c: Optional[float] = None,
        p_wavelet_scales: str = 'dyadic',
        num_nodes_one_graph: Optional[int] = None,
        J: int = 4,
        P_wavelets_channels_t_is: Optional[torch.Tensor] = None,
        only_use_avg_P_wavelets_channels_t_is: bool = False,
        max_kappa: Optional[int] = None,
        include_lowpass_wavelet: bool = True,
        within_Wj_ns_chan_out_per_filter: Optional[Tuple] = (8, 4),
        cross_Wj_ns_combos_out_per_chan: Tuple = (8, 4),
        channel_pool_moments: Optional[Tuple[int]] = (1, 2, 3, 4, float('inf')),
        channel_pool_key: Optional[str] = None, # 'moments', 'max', 'mean'
        node_pooling_key: Optional[str] = None, # 'linear', 'max', 'mean'
        node_pool_linear_out_channels: int = 1,
        use_skip_connections: bool = True,
        use_input_recombine_layer: bool = False,
        input_recombine_layer_kwargs: Dict[str, Any] = {},
        use_k_drop: bool = False,
        k_drop_kwargs: Dict[str, Any] = {},
        mfcn_nonlin_fn = F.relu, # F.leaky_relu,
        mfcn_nonlin_fn_kwargs: dict = {}, # {'negative_slope': 0.01},
        mfcn_wts_init_kwargs: dict = {}, # {'nonlinearity': 'leaky_relu', 'a': 0.01},
        mfcn_wts_init_fn = nn.init.kaiming_uniform_,
        base_module_kwargs: Dict[str, Any] = {},
        fc_kwargs: Dict[str, Any] = {},
        verbosity: int = 0
    ):
        """
        attributes
        """
        if (within_Wj_ns_chan_out_per_filter is not None) \
        and (cross_Wj_ns_combos_out_per_chan is not None):
            if len(within_Wj_ns_chan_out_per_filter) \
            != len(cross_Wj_ns_combos_out_per_chan):
                raise Exception(
                    '''
                    WaveletMFCN requires the same number of within- and 
                    cross-filter steps (if both are not None).
                    '''
                )
                return None
                
        super(MFCN_Module, self).__init__(**base_module_kwargs)

        self.task = self.task.lower()
        self.in_channels = in_channels
        self.wavelet_type = wavelet_type.lower() if (wavelet_type is not None) else None
        # want to default to wavelets, and error if both types are None
        self.non_wavelet_filter_type = non_wavelet_filter_type.lower() \
            if (wavelet_type is None) else None
        self.filter_c = filter_c
        self.p_wavelet_scales = p_wavelet_scales
        self.max_kappa = max_kappa
        self.Wjs_spectral = None
        self.V_sparse = None
        self.J = J
        self.P_wavelets_channels_t_is = P_wavelets_channels_t_is
        # check that the number of input channels specified matches the
        # number of channels of custom wavelet scales
        if (self.P_wavelets_channels_t_is is not None) \
        and (self.in_channels != len(self.P_wavelets_channels_t_is)):
            raise Exception(
                f"self.in_channels != len(self.P_wavelets_channels_t_is)"
            )
        self.avg_P_wavelets_channels_t_is = infogain.get_avg_P_wavelet_scales(
            self.P_wavelets_channels_t_is
        ) \
            if ((self.P_wavelets_channels_t_is is not None) \
                and (P_wavelets_channels_t_is.dim() == 2)) \
            else None
        # if (verbosity > 1) and (self.avg_P_wavelets_channels_t_is is not None):
        #     print(
        #         'self.avg_P_wavelets_channels_t_is.shape:', 
        #         self.avg_P_wavelets_channels_t_is.shape
        #     )
        if only_use_avg_P_wavelets_channels_t_is:
            self.P_wavelets_channels_t_is = self.avg_P_wavelets_channels_t_is
        self.cross_Wj_ns_chan_in_accum = None # set later if needed
        self.include_lowpass_wavelet = include_lowpass_wavelet
        self.within_Wj_ns_chan_out_per_filter = within_Wj_ns_chan_out_per_filter
        self.cross_Wj_ns_combos_out_per_chan = cross_Wj_ns_combos_out_per_chan
        self.use_skip_connections = use_skip_connections
        self.use_input_recombine_layer = use_input_recombine_layer
        self.input_recombine_layer_kwargs = input_recombine_layer_kwargs
        self.use_k_drop = use_k_drop
        self.k_drop_kwargs = k_drop_kwargs
        
        self.mfcn_nonlin_fn = mfcn_nonlin_fn
        self.mfcn_nonlin_fn_kwargs = mfcn_nonlin_fn_kwargs
        self.channel_pool_moments = channel_pool_moments
        self.node_pooling_key = node_pooling_key.lower() \
            if (node_pooling_key is not None) else None
        self.channel_pool_key = channel_pool_key.lower() \
            if (channel_pool_key is not None) else None
        self.verbosity = verbosity
        
        self.n_cycles = len(cross_Wj_ns_combos_out_per_chan) \
            if (cross_Wj_ns_combos_out_per_chan is not None) \
            else len(within_Wj_ns_chan_out_per_filter)
        if (wavelet_type is not None):
            n_filters = (J + 2) if include_lowpass_wavelet else (J + 1)
        else:
             n_filters = 1
        if P_wavelets_channels_t_is is None:
            self.n_filters_by_cycle = [n_filters] * self.n_cycles
        else:
            # first cycle's P-wavelet scales are passed in 'P_wavelets_channels_t_is';
            # there are s - 1 filters where s is its length;
            custom_P_n_filters = len(P_wavelets_channels_t_is[0]) - 1 \
                if P_wavelets_channels_t_is.dim() == 2 \
                else len(P_wavelets_channels_t_is) - 1
            if include_lowpass_wavelet:
                custom_P_n_filters += 1
            # self.n_filters_by_cycle = [custom_P_n_filters] + ([n_filters] * (self.n_cycles - 1))
            self.n_filters_by_cycle = [custom_P_n_filters] * self.n_cycles
        # print('n_filters_by_cycle:', self.n_filters_by_cycle)
        if wavelet_type is not None:
            self.Wjs_key = 'Wjs_P' if ('p' in self.wavelet_type.lower()) else 'Wjs_spectral'

        """
        [optional] initialize linear layer to recombine channels
        of original input, before first filter step (also saved
        as a skip connection)
        """
        if self.use_input_recombine_layer:
            self.recombined_input_layer = nn.Linear(
                in_features=self.in_channels,
                **self.input_recombine_layer_kwargs
            )
            nn.init.kaiming_uniform_(
                self.recombined_input_layer.weight,
                nonlinearity='relu'
            )
            if self.recombined_input_layer.bias is not None:
                nn.init.zeros_(self.recombined_input_layer.bias)

            self.recomb_input_layer_batch_norm = nn.BatchNorm1d(
                num_features=self.input_recombine_layer_kwargs['out_features']
            )

        """
        [for certain tasks] initialize fully-connected network
        """
        if fc_kwargs is not None:
            # pre-calculate the final output size = input size to the 
            # fully-connected network head (if graph-level task)
            input_size_to_fc = self._calc_final_output_size(
                num_nodes_one_graph
            )
            if self.verbosity > 1:
                print(f'input_size_to_fc: {input_size_to_fc}\n')
                
            self.fc = vnn.VanillaNN(
                input_dim=input_size_to_fc,
                **fc_kwargs
            )
        else:
            self.fc = None

        """
        initialize MFCN learnable parameters
        """
        self._init_learnable_parameters(
            mfcn_wts_init_fn,
            mfcn_wts_init_kwargs
        )
        

    def forward(
        self, 
        batch: Batch | Data
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: pytorch geometric Batch object (holding
                info for batched graphs as one disjoint graph
                in COO form, with other collated attributes)
                or Data object (holding a single graph).
        Returns:
            Dictionary of model output, with model predictions
            keyed by 'preds'.
        """
        # extract batch graph objects
        # x shape: (batch_total_nodes, n_channels) = 'Nc'
        x, edge_index, batch_index = (
            batch.x, 
            batch.edge_index, 
            batch.batch
        )
        edge_weight = batch.edge_weight \
        if hasattr(batch, 'edge_weight') else None
        # print('x.shape:', x.shape)

        # 'batch' might actually be a single PyG Data object
        # (e.g., using a single graph, doing node regression)
        # note: unbatched 'Data' object doesn't have attribute 'num_graphs'
        num_graphs = 1 if (batch_index is None) else batch.num_graphs

        # init container for skip connections + x outputs from all cycles
        num_concat_x = (self.n_cycles + 1) 
        if self.use_input_recombine_layer:
            num_concat_x += 1
        xs = [None] * num_concat_x if self.use_skip_connections else [None]
            
        """
        MFCN cycles
        - loop through filter-combine steps, the same number of times
            as there are cross-filter convolution modules
        """
        # get number of cycles as the length of one of the combination steps' tuple
        # of output channels
        if self.cross_Wj_ns_combos_out_per_chan is not None:
            self.n_cycles = len(self.cross_Wj_ns_combos_out_per_chan)
        else:
            self.n_cycles = len(self.within_Wj_ns_chan_out_per_filter)


        # loop through cycles
        for i in range(self.n_cycles):   
            
            """
            [optional] skip connections and input recombination linear 
            layer
            """
            if self.use_skip_connections:
                # apply pooling to x and save in 'skip_xs' container
                # for concatenation (need dense matrix for pooling tensor methods)
                x0 = x.clone().to_dense() if x.is_sparse else x.clone()
                x0_pooled = self._all_poolings_node_first(x0, num_graphs, batch_index)
                # if recombined input channels are also used as a skip connection,
                # they will be inserted into xs[1]; -> need to increment i by 1 for 
                # saving cycle outputs as skip cxns for i > 0
                xs_i = (i + 1) if ((i > 0) and self.use_input_recombine_layer) else i
                xs[xs_i] = x0_pooled

            if (i == 0) and self.use_input_recombine_layer:
                x = self.recombined_input_layer(x)
                x = self.mfcn_nonlin_fn(x, **self.mfcn_nonlin_fn_kwargs)
                x = self.recomb_input_layer_batch_norm(x)
                if self.use_skip_connections:
                    x1 = x.clone().to_dense() if x.is_sparse else x.clone()
                    x1_pooled = self._all_poolings_node_first(x1, num_graphs, batch_index)
                    xs[1] = x1_pooled

                if self.verbosity > 1:
                    print(f'(input recomb.) x.shape: {tuple(x.shape)}')

            """
            (i) filter
            """
            # using 'P' (lazy random walk)-based filter(s), wavelet or not
            if (
                (self.wavelet_type is None) \
                and (self.non_wavelet_filter_type == 'p')
            ) \
            or (self.wavelet_type == 'p'):

                # if using k_drop (only when training)
                if self.use_k_drop and self.training:
                    edge_index, edge_weight = batch_drop_rand_k_edges(
                        batch=batch,
                        **self.k_drop_kwargs
                    )
                
                # use full custom wavelet scales in first cycle; then avg of
                # custom scales (same for all channels) in subsequent cycles
                # (this is a compromise, to prevent cost of re-calculating
                # custom scales on hidden features every epoch as model learns)
                # print('x.shape:', x.shape)
                if (self.p_wavelet_scales == 'custom'):

                    # 1 set of shared custom scales for all channels
                    if self.P_wavelets_channels_t_is.dim() == 1:
                        channels_t_is = self.P_wavelets_channels_t_is

                    # unique custom scales by channel: use in first layer,
                    # then 1 set of median-average scales for all channels in 
                    # next layers
                    else:
                        channels_t_is = self.P_wavelets_channels_t_is \
                            if (i == 0) \
                            else self.avg_P_wavelets_channels_t_is
                    # print('channels_t_is.shape:', channels_t_is.shape)
                else:
                    # None -> _P_filtration uses dyadic scales by default
                    channels_t_is = None 
                
                x = self._P_filtration(
                    x,
                    edge_index,
                    edge_weight,
                    channels_t_is
                )
        
            # using spectral-based filter(s), wavelet or not
            elif (
                (self.wavelet_type is None) \
                  and ('spect' in self.non_wavelet_filter_type)
            ) \
            or ('spect' in self.wavelet_type):
                x = self._spectral_filtration(
                    x,
                    num_graphs,
                    batch_index,
                    batch_L_eigenvals
                )

            # if x is a sparse tensor after filtering, convert to dense,
            # since torch.einsum doesn't handle sparse tensors (yet)
            if x.is_sparse:
                x = x.to_dense()
            
            # x shape = (total_n_nodes, n_channels, n_filters) = 'Ncj'
            if self.verbosity > 1:
                print(f'(filter) x.shape: {tuple(x.shape)}')

            """
            (ii) [optional] combine channels 
                - learn weights for recombining (fixed) Wj's across 
                    C channels -> C' channels
                - 'within_Wjs_wts' shape = (n_in_chan, n_filters, n_out_chan) 
                    = (J, C, C') = 'jcC'
                - keep output of combine step here in 'NCj' order, in case
                    this step is skipped, and x still has shape 'Ncj'
                - note: it's okay to work with xs stacked in first dimension
                    (size N=sum(n_i)) because channel and filter orders
                    are preserved, and the channel-combining coeffs do not
                    vary by graph (only by filter and channel)
            """
            if self.within_Wj_ns_chan_out_per_filter is not None:
                within_Wjs_wts = self.within_Wj_params[i]
                # print('within_Wjs_wts.shape:', within_Wjs_wts.shape)
                x = torch.einsum('Ncj,jcC->NCj', x, within_Wjs_wts)
                
                # x shape = (total_n_nodes, n_filters, n_channels_2) = 'NjC'
                if self.verbosity > 1:
                    print(f'(combine) x.shape: {tuple(x.shape)}')

            """
            (iii) cross-filter combinations, within C' channels
                - learn alphas/weights for J' new linear combos of the 
                    J filtered channel signal vectors
                - cross_Wjs_wts shape = (n_channels, n_filtrations_in, n_combos_out)
                    = (C', J, J') = 'CjJ' 
                - ('...jn,...jJ->...Jn') creates J' new linear combos of 
                    vectors/rows of the 'jn' matrix, which are Wjfs (of length n)
                - note: it's okay to work with xs stacked in first dimension
                    (size N=sum(n_i)) because channel and filter orders
                    are preserved, and the filter-combining coeffs do not
                    vary by graph (only by channel and filter)
            """
            if self.cross_Wj_ns_combos_out_per_chan is not None:
                cross_Wjs_wts = self.cross_Wj_params[i]
                x = torch.einsum('NCj,CjJ->NJC', x, cross_Wjs_wts)
                
                # x shape = (total_n_nodes, n_filters_2, n_channels_2) = 'NJC'
                if self.verbosity > 1:
                    print(f'(cross-Wj conv.) x.shape: {tuple(x.shape)}')

            """
            (iv) nonlinear activation
            """
            x = self.mfcn_nonlin_fn(x, **self.mfcn_nonlin_fn_kwargs)

            """
            (v) reshape (effectively, to have C'*J' channels)
                - want x shape (batch_size, new_n_channels, n), so we can 
                    filter-combine again
                - note: 'reshape' is only safe when axes are in the right 
                    'ravel' order
            """
            shape = x.shape # 'NJC'
            x = x.reshape(shape[0], -1)
            # x shape = (total_n_nodes, n_channels_3) = 'Nd'
            if self.verbosity > 1:
                print(f'(reshape) x.shape: {tuple(x.shape)}')
                print()

        """
        after MFCN cycles:
        - node/channel pooling
        - feed to regressor/classifier head
        """ 
        x = self._all_poolings_node_first(
            x,
            num_graphs,
            batch_index
        )

        # graph-level tasks
        if 'graph' in self.task: 
            # insert final x into xs list
            xs[-1] = x
            # concenate skip connections and last x
            if len(xs) > 1:
                x = torch.concatenate(xs, dim=-1)
            else:
                x = xs[0]

            # flatten graphs' xs into 2 dim each (req'd by MLP head)
            # -> x shape = (n_graphs, n_moments * [final_]n_channels_3)
            if x.dim() == 3:
                x = x.reshape(x.shape[0], -1)
    
            # feed (maybe concatenated) x to MLP head
            if self.fc is not None:
                if self.verbosity > 1:
                    print(f'(fc input) x.shape: {tuple(x.shape)}\n')
                model_output_dict = self.fc.forward(x)
                return model_output_dict
                
        # node-level tasks
        # final pooling produces the model preds/outputs
        elif 'node' in self.task:
            if self.use_skip_connections:
                raise NotImplementedError(
                    "Using skip connections is not implemented"
                    " in MFCN for node-level tasks."
                )
            else:
                # note at this point, x is any real value
                # -> need to map to prob then to class for
                # node-level classification tasks (in loss)
                model_output_dict = {'preds': x}
                return model_output_dict

        # not graph- or node-level tasks (probably incorrect task arg)
        else:
            raise NotImplementedError(
                "Pooling method not implemented in MFCN."
                " Did you forget 'node' or 'graph' in the"
                " 'task' arg?"
            )

        
    def _set_comb_steps_n_channels_in(self) -> None:
        """
        Pre-computes `within_Wj_ns_chan_in` and `cross_Wj_ns_chan_in`, the number
        of channels (C') fed into each cross-channel and cross-filter comb. steps;
        these are used in sizing the `within_Wj_params` and `cross_Wj_params` in
        `_init_learnable_parameters`.
        """ 
        first_chans_in = self.input_recombine_layer_kwargs['out_features'] \
            if self.use_input_recombine_layer else self.in_channels
        
        if (self.within_Wj_ns_chan_out_per_filter is not None):
            if (self.cross_Wj_ns_combos_out_per_chan is not None):
                # when using both cross-channel and cross-filter combination steps
                # -> in channels are multiplicative from both steps' out channels in a cycle
                self.within_Wj_ns_chan_in = [first_chans_in] + \
                    [
                        within_Wj_n_chan_out * cross_Wj_n_combos_out \
                        for within_Wj_n_chan_out, cross_Wj_n_combos_out \
                        in zip(
                            self.within_Wj_ns_chan_out_per_filter, 
                            self.cross_Wj_ns_combos_out_per_chan
                        )
                    ]
            else: 
                # only cross-channel combinations; no cross-filter combinations
                # -> in channels = out channels from last cross-channel combination step
                self.within_Wj_ns_chan_in = [first_chans_in] + \
                    list(self.within_Wj_ns_chan_out_per_filter)
                
            self.cross_Wj_ns_chan_in = self.within_Wj_ns_chan_out_per_filter
        else: 
            # no cross-channel combinations, only cross-filter combinations
            if self.cross_Wj_ns_chan_in_accum is None:
                self._set_cross_Wj_ns_chan_in_accum()
            self.cross_Wj_ns_chan_in = self.cross_Wj_ns_chan_in_accum[:-1]
            # print('cross_Wj_ns_chan_in:', self.cross_Wj_ns_chan_in)

    
    def _set_cross_Wj_ns_chan_in_accum(self) -> None:
        """

        """
        first_chans_in = self.input_recombine_layer_kwargs['out_features'] \
            if self.use_input_recombine_layer else self.in_channels
        self.cross_Wj_ns_chan_in_accum = [
                *accumulate(
                    tuple([first_chans_in]) + self.cross_Wj_ns_combos_out_per_chan, 
                    lambda a, b: a * b
                )
            ]
        
    
    def _init_learnable_parameters(
        self, 
        mfcn_wts_init_fn,
        mfcn_wts_init_kwargs
    ) -> None:
        """
        Initializes torch.nn.Parameter objects for the cross-
        channel and cross-filter learnable combination steps.
        """
        # compute and set `self.within_Wj_ns_chan_in` and `self.cross_Wj_ns_chan_in`
        self._set_comb_steps_n_channels_in()
        
        # for within-filter, cross-channel combinations/convolutions [step (ii)]
        # each Parameter size = (n_filters, n_channels_in, new_n_channels) = 'jcC'
        if self.within_Wj_ns_chan_out_per_filter is not None:
            self.within_Wj_params = nn.ParameterList([
                torch.nn.Parameter(
                    torch.zeros(
                        n_filters,
                        n_channels_in,
                        n_channels_out
                    ),
                    requires_grad=True
                ) for n_filters, n_channels_in, n_channels_out \
                  in zip(
                      self.n_filters_by_cycle,
                      self.within_Wj_ns_chan_in, 
                      self.within_Wj_ns_chan_out_per_filter
                  )
            ])
            # randomly initialize parameter weights
            for within_Wj in self.within_Wj_params:
                mfcn_wts_init_fn(within_Wj, **mfcn_wts_init_kwargs)

        # for cross-filter combinations/convolutions [step (iii)]
        # (used to learn new channel signal vectors, as linear combos of 
        # the filtered channel signal vectors (within channels, across filters), 
        # without a bias term)
        # each Parameter size = (n_channels, n_filtrations_in, n_filter_combos_out) = 'CjJ'
        if self.cross_Wj_ns_combos_out_per_chan is not None:
            self.cross_Wj_params = nn.ParameterList([
                torch.nn.Parameter(
                    torch.zeros(
                        n_chan,
                        n_filters,
                        n_filter_combos_out, 
                    ),
                    requires_grad=True
                ) for n_chan, n_filters, n_filter_combos_out in zip(
                    self.cross_Wj_ns_chan_in,
                    self.n_filters_by_cycle,
                    self.cross_Wj_ns_combos_out_per_chan
                )
            ])
            # randomly initialize parameter weights
            for cross_Wj in self.cross_Wj_params:
                mfcn_wts_init_fn(cross_Wj, **mfcn_wts_init_kwargs)

        # for node-level tasks: option to pool nodes' final channels into 1 
        # with learned weights (linear layer)
        if (self.node_pooling_key is not None) and ('linear' in self.node_pooling_key):
            if self.within_Wj_ns_chan_out_per_filter is not None:
                if (self.cross_Wj_ns_combos_out_per_chan is not None):
                    # both cross-channel combinations and cross-filter combinations
                    node_pool_mult = self.cross_Wj_ns_combos_out_per_chan[-1]
                else:
                    # only cross-channel combinations; no cross-filter combinations
                    node_pool_mult = 1
                final_reshape_chan_dim = node_pool_mult * \
                    self.within_Wj_ns_chan_out_per_filter[-1]
            else:
                # no cross-channel combinations; only cross-filter combinations
                final_reshape_chan_dim = self.cross_Wj_ns_chan_in_accum[-1]

            # init. weights to 1. (equal between channels) and bias to 0.
            self.node_pool_wts = torch.nn.Parameter(
                torch.ones((final_reshape_chan_dim, node_pool_linear_out_channels))
            )
            self.node_pool_bias = torch.nn.Parameter(torch.zeros(1))


    def _get_pooling_multiple(
        self,
        num_nodes_one_graph: Optional[int] = None
    ) -> int:
        """

        """
        # graph-level task
        if 'graph' in self.task:
            if ('mom' in self.channel_pool_key) and (self.channel_pool_moments is not None):
                # the input size to the first fully-connected layer is the (concatenated)
                # final num_channels * num_filters * n_moments: C'*(J'*n_moments)
                pool_mult = len(self.channel_pool_moments) # typically = 4
            elif ('mean' in self.channel_pool_key) and ('max' in self.channel_pool_key):
                pool_mult = 2
            else: # graph task, mean OR max channel (not both, nor moments) pooling
                pool_mult = 1
                
        # node-level task, on 1 graph
        else: 
            # without channel pooling, we get C'*(J'*num_nodes_one_graph)
            # this var not used in this case, but here for completeness/future
            pool_mult = num_nodes_one_graph
        
        return pool_mult


    
    def _calc_final_output_size(
        self,
        num_nodes_one_graph: Optional[int] = None
    ) -> int:
        """

        """
        pool_mult = self._get_pooling_multiple()

        if self.use_skip_connections:
            
            if self.within_Wj_ns_chan_out_per_filter is None \
            and self.cross_Wj_ns_combos_out_per_chan is None:
                raise Exception(
                    "Need one of 'within_Wj_ns_chan_out_per_filter'"
                    " or 'cross_Wj_ns_combos_out_per_chan' to not be None."
                )
            if (self.within_Wj_ns_chan_out_per_filter is not None) \
            and (self.cross_Wj_ns_combos_out_per_chan is not None) \
            and (len(self.within_Wj_ns_chan_out_per_filter) != \
            len(self.cross_Wj_ns_combos_out_per_chan)):
                raise Exception(
                    "Need tuples 'within_Wj_ns_chan_out_per_filter' and"
                    " 'cross_Wj_ns_combos_out_per_chan' to have equal"
                    " length, i.e. a value for each cycle, if not None."
                )

            num_concat_xs = (self.n_cycles + 2) \
                if self.use_input_recombine_layer else (self.n_cycles + 1)
            out_size_by_cycle = [None] * num_concat_xs

            # populate first 1 or two concatenated x lengths
            out_size_by_cycle[0] = self.in_channels
            if self.use_input_recombine_layer:
                out_size_by_cycle[1] = self.input_recombine_layer_kwargs['out_features']
            
            for cycle_i in range(1, self.n_cycles + 1):
                # if input recombined layer was used as add'l skip connection,
                # need to increment cycle_i for queries and insertions with
                # 'out_size_by_cycle'
                cycle_i_adj = (cycle_i + 1) if self.use_input_recombine_layer else cycle_i
                C = self.within_Wj_ns_chan_out_per_filter[cycle_i - 1] \
                    if (self.within_Wj_ns_chan_out_per_filter is not None) \
                    else out_size_by_cycle[cycle_i_adj - 1]
                J = self.cross_Wj_ns_combos_out_per_chan[cycle_i - 1] \
                    if (self.cross_Wj_ns_combos_out_per_chan is not None) \
                    else out_size_by_cycle[cycle_i_adj - 1]
                # if recombined input is also used as a skip connection, need to insert
                # into list at index (cycle_i + 1)
                out_size_by_cycle[cycle_i_adj] = (C * J)
                # print(out_size_by_cycle)
                
            out_size_by_cycle = [s * pool_mult for s in out_size_by_cycle]
            final_output_size = sum(out_size_by_cycle)
            
        else: # no skip connections
    
            # if using cross-channel combinations
            if self.within_Wj_ns_chan_out_per_filter is not None:
                last_within_Wj_mult = self.within_Wj_ns_chan_out_per_filter[-1]
                if self.cross_Wj_ns_combos_out_per_chan is not None:
                    # both cross-channel and cross-filter combination steps
                    last_cross_Wj_mult = self.cross_Wj_ns_combos_out_per_chan[-1]
                else:
                    # only cross-channel combinations; no cross-filter combinations
                    last_cross_Wj_mult = 1
                final_output_size = int(last_within_Wj_mult * last_cross_Wj_mult * pool_mult)
    
            # no cross-channel combinations, only cross-filter combinations
            else:
                if self.cross_Wj_ns_chan_in_accum is None:
                    self._set_cross_Wj_ns_chan_in_accum()
                final_output_size = int(self.cross_Wj_ns_chan_in_accum[-1] * pool_mult)
                
        return final_output_size
    
        
    def _P_filtration(
        self,
        x,
        edge_index,
        edge_weight: Optional[torch.Tensor] = None,
        channels_t_is: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        """
        # compute P (sparse) for all graphs (as one disconnected graph)
        P_sparse = pygu.get_Batch_P_sparse(
            edge_index=edge_index,
            edge_weight=edge_weight,
            n_nodes=x.shape[0],
            device=self.device
        )
        
        # single P^1 (non-wavelet) filter
        if (self.wavelet_type is None) \
        and (self.non_wavelet_filter_type == 'p'): 
            get_Batch_P_Wjxs_kwargs = {
                'x': x,
                'P_sparse': P_sparse,
                'scales_type': None,
                'channels_t_is': None,
                # 'custom_scales_max_t': 1, # deprecated
                # 'J': 1, # doesn't get used
                'include_lowpass': False
            }
            
        # 'P' wavelets filter bank
        else:
            # if channels_t_is is not None, we have computed the KL divergence-based
            # 'custom' P wavelets scales -> only apply to first MFCN cycle
            scales_type = 'custom' if (channels_t_is is not None) else 'dyadic'
            get_Batch_P_Wjxs_kwargs = {
                'x': x,
                'P_sparse': P_sparse,
                'scales_type': scales_type,
                'channels_t_is': channels_t_is,
                # 'custom_scales_max_t': 32, # deprecated
                'J': self.J, # 
                'include_lowpass': self.include_lowpass_wavelet
            }

        # use P_sparse and MFCN vs. MCN kwargs to get filtrations of x
        x = get_Batch_P_Wjxs(**get_Batch_P_Wjxs_kwargs)
        return x

    
    def _spectral_filtration(
        self,
        x,
        num_graphs,
        batch_index,
        batch_L_eigenvals
    ) -> torch.Tensor:
        """

        """
        # calc self.Wjs_spectral, of shape (num_graphs, k, num_filters)
        # only calc self.Wjs_spectral if None or batched multi-graph dataset 
        if (num_graphs > 1) or (self.Wjs_spectral is None):
            batch_L_eigenvals = batch.L_eigenvals # shape (num_graphs, k) or (k, )
            if batch_L_eigenvals.dim() == 1: # have 1 graph in batch
                # need 2 dims for iterating (num_graphs=1, k)
                batch_L_eigenvals = batch_L_eigenvals.unsqueeze(dim=0)

            # using spectral wavelets filter bank
            if (self.wavelet_type is not None):
                self.Wjs_spectral = torch.stack([
                    w.spectral_wavelets(
                        eigenvals=L_eigenvals, 
                        J=self.J,
                        include_low_pass=self.include_lowpass_wavelet,
                        spectral_c=self.filter_c,
                        device=self.device
                    ) \
                    for L_eigenvals in batch_L_eigenvals
                ], dim=0)
                
            # using non-wavelet lowpass spectral filter
            elif ('spect' in self.non_wavelet_filter_type):
                self.Wjs_spectral = torch.stack([
                        w.spectral_lowpass_filter(
                        eigenvals=L_eigenvals,
                        c=self.filter_c,
                        device=self.device
                    ) \
                    for L_eigenvals in batch_L_eigenvals
                    ], dim=0) 
            else:
                # other non-wavelet spectral filter(s): not implemented
                raise NotImplementedError(
                    f"Non-wavelet spectral filter type {self.non_wavelet_filter_type}" 
                    f" not implemented."
                )

        # L_eigenvecs shape: (total_n_nodes, n_eigenvectors) = 'Nk'
        # where N = sum(n_i) and k = n_eigenvectors
        # if num_graphs == 1 and self.V_sparse is not None, don't need
        # to recalculate it
        if (num_graphs > 1) or (self.V_sparse is None):
            self.V_sparse = get_Batch_V_sparse(
                num_graphs,
                batch_index,
                batch.L_eigenvecs,
                self.max_kappa,
                device=self.device
            )
            # V_sparse shape: (num_graphs * n_eigenvectors, total_n_nodes) = 'KN'
        
        x = get_Batch_spectral_Wjxs(
            num_graphs,
            x,
            batch_index, # None for 1-graph batches
            self.V_sparse,
            self.Wjs_spectral,
            batch.L_eigenvecs,
            self.max_kappa
        )
        return x

    
    def _all_poolings_node_first(
        self,
        x: torch.Tensor,
        num_graphs: int,
        batch_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Note: in (rare) cases where both node and
        channel poolings are applied, this applies
        node pooling first.
        """
        if self.node_pooling_key is not None:
            # pool across channels, within nodes
            # can be used in 'node' or 'graph' tasks,
            # though in 'graph' tasks, will have only
            # 1 channel of signal per node
            x = pygu.node_pool(
                x,
                node_pooling_key=self.node_pooling_key,
                node_pool_wts=self.node_pool_wts,
                node_pool_bias=self.node_pool_bias,
                pool_dim=1
            )

        # elif 'linear' in self.node_pooling_key:
        #     # channels are linearly combined
        #     # using learnable w and b in x' = wx + b
        #     x = torch.einsum(
        #         'Nd,d->N',
        #         x,
        #         self.node_pool_wts
        #     ) + self.node_pool_bias

        if 'graph' in self.task:
            """
            for graph-level tasks
                - pool channel vectors by taking max/mean/moments 
                    of each channel, for each graph
                - note we can't do a 'linear' channel pooling, since
                    we may have graphs of different numbers of nodes,
                    hence would need variable-sized 'w' in x' = wx + b,
                    where x is a channel (of size n_nodes)
                - feed pooled, concatenated features to MLP head
            """
            if ('max' in self.channel_pool_key) \
            or ('mean' in self.channel_pool_key):
                x = pygu.channel_pool(
                    self.channel_pool_key, 
                    x, 
                    num_graphs,
                    batch_index
                )
                        
                if self.verbosity > 1:
                    print(
                        f'({self.channel_pool_key} pooling) x.shape:'
                        f' {tuple(x.shape)}'
                    )
                    
            elif 'moment' in self.channel_pool_key \
            and (self.channel_pool_moments is not None):
                x = pygu.moments_channel_pool(
                    x,
                    batch_index,
                    num_graphs,
                    self.channel_pool_moments
                )
                
                if self.verbosity > 1:
                    print(f'(moment pooling) x.shape: {tuple(x.shape)}')
    
        else:
            raise NotImplementedError(
                "Pooling method not implemented in MFCN."
                " Did you forget 'node' or 'graph' in the"
                " 'task' arg?"
            )
        return x

    
    





class WaveletMFCNDataset(Dataset):
    """
    Subclass of `torch.utils.data.Dataset` that
    contains inputs and targets in dictionaries,
    for abstraction that allows for a generic PyTorch
    training function.

    Note that the MFCN model requires the graphs' Laplacian eigenvectors
    in its filtering steps, since it filters signals spectrally in 
    this Fourier domain (as a linear combination of Fourier coefficients
    of L's eigenvectors, using pre-made spectral filters based on L's
    eigenvalues).

    __init__ args:
        x: tensor for x/input, first dimension
            of which indexes into one sample/input:
            hence shape (n_samples, [n_channels], n_pts_per_sample)
        Ps: (sparse coo) tensor for 'P' (the lazy random walk matrix)
            for one manifold; shape (n_samples, n_samples).
        Wjs_spectral: tensor for spectral wavelet filters, 
            shape (n_samples, [n_channels], k, n_pts_per_sample).
        L_eigenvecs: tensor for each graph's Laplacian k
            eigenvectors, of shape (n_samples, [n_channels], k, n_pts_per_sample).
        targets_dictl: list of dictionaries holding
            target(s') keys and values.

    __getitem__ returns:
        A dictionary of containing one sample's x tensor, P or spectral
        filter tensors, and a sub-dictionary of its training target(s).
    """
    def __init__(self, 
                 wavelet_type: str,
                 x: torch.Tensor,
                 targets_dictl: List[Dict[str, Any]],
                 Ps: torch.Tensor = None, # could be sparse
                 Wjs_spectral: torch.Tensor = None,
                 L_eigenvecs: torch.Tensor = None,
                ) -> None:
        super(WaveletMFCNDataset, self).__init__()
        self.wavelet_type = wavelet_type
        self.x = x
        print(f'WaveletMFCNDataset: x.shape = {self.x.shape}')
        self.Ps = Ps
        self.Wjs_spectral = Wjs_spectral
        self.L_eigenvecs = L_eigenvecs
        self.targets_dictl = targets_dictl

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        data_obj_dict = {
            'x': self.x[idx],
            'target': self.targets_dictl[idx]
        }
        if self.wavelet_type == 'P':
            data_obj_dict = data_obj_dict | {
                'P': self.Ps[idx]
            }
        elif self.wavelet_type == 'spectral':
            data_obj_dict = data_obj_dict | {
                'Wjs_spectral': self.Wjs_spectral[idx],
                'L_eigenvecs': self.L_eigenvecs[idx]
            }
        return data_obj_dict



def split_and_pickle_WaveletMFCNDataset_dict(
    args,
    x: torch.Tensor,
    target_dictl: List[dict],
    Ps: torch.sparse_coo_tensor = None,
    Wjs_spectral: torch.Tensor = None,
    L_eigenvecs: torch.Tensor = None,
    set_idxs_dict: Dict[str, List[int]] = None
) -> None:
    """
    Creates 'WaveletMFCNDataset' objects, splits into
    train/valid/test sets, and saves/pickles as a 
    dictionary.

    Args:
        args: ArgsTemplate subclass with experiment
            parameters.
        x: master input/signal/function values, where 
            the first axis indexes into one sample's 
            input tensor; shape (n_samples, n_input_vals).
        target_dictl: training targets dictionaries.
        Ps: master tensor of P (lazy random walk) matrices,
            where the first axis indexes into one sample's 
            P tensor; shape (n_samples, n_input_vals, 
            n_input_vals).
        Wjs_spectral: master tensor of spectral filters,
            where the first axis indexes into one sample's 
            spectral filters tensor; (n_samples, [n_channels], 
            n_eigenpairs, n_pts_per_sample).
        L_eigenvecs: master tensor of graph Laplacian eigen-
            vectors, where the first axis indexes into one 
            sample's spectral eigenvectors tensor; 
            shape (n_samples, n_input_vals, n_eigenvectors).
        set_idxs_dict: optional dictionary of index lists 
            for train/valid/test sets. If 'None', new
            index lists will be calculated.
    Returns:
        None (pickles dataset dict).
    """
    # get train/valid/test split idxs
    if set_idxs_dict is None:
        set_idxs_dict = du.get_train_valid_test_idxs(
            seed=args.TRAIN_VALID_TEST_SPLIT_SEED,
            n=x.shape[0],
            train_prop=args.TRAIN_PROP,
            valid_prop=args.VALID_PROP
        )

    if args.WAVELET_TYPE.lower() == 'p':
        datasets_dict = {
            set: WaveletMFCNDataset(
                wavelet_type=args.WAVELET_TYPE,
                x=x[idx],
                # 'index_select' works with torch.sparse
                Ps=torch.index_select(
                    input=Ps, 
                    dim=0, 
                    index=torch.tensor(idx, dtype=torch.long)
                ),
                # Ps=Ps[idx],
                targets_dictl=[target_dictl[i] for i in idx]
            ) \
            for set, idx in set_idxs_dict.items()
        }
    elif args.WAVELET_TYPE.lower() == 'spectral':
        datasets_dict = {
            set: WaveletMFCNDataset(
                wavelet_type=args.WAVELET_TYPE,
                x=x[idx],
                Wjs_spectral=Wjs_spectral[idx],
                L_eigenvecs=L_eigenvecs[idx],
                targets_dictl=[target_dictl[i] for i in idx]
            ) \
            for set, idx in set_idxs_dict.items()
        }
    
    # pickle the dataset dict
    save_path = f'{args.DATA_DIR}/{args.DATASETS_DICT_FILENAME}'
    with open(save_path, "wb") as f:
        pickle.dump(datasets_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Datasets saved (\'{args.DATASETS_DICT_FILENAME}\').\n')


class KDropData(Data):
    """
    Subclass of PyG 'Data' class that collates
    wavelet filter attributes in new (first) dimension,
    instead of concatenating in dim 0 (default).
    
    Reference:
    https://pytorch-geometric.readthedocs.io/en/2.5.0/advanced/batching.html
    """
    def __cat_dim__(self, key, value, *args, **kwargs):
        if (key == 'node_count') \
        or (key == 'node_ks') \
        or (key == 'edge_index_offsets'):
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)
        

def batch_drop_rand_k_edges(
    batch, # type: PyG Batch,
    k_drop: int,
    rs_or_generator: Optional[RandomState | torch.Generator] = None,
    use_edge_wt_probs: bool = False
) -> Tuple[torch.Tensor]:
    """
    For a pytorch-geometric batch of graphs, this function
    computes a new 'edge_index' tensor (and 'edge_weight', if
    present) with 'k_drop' fewer edges dropped randomly from
    each node of each graph in the batch. Note that the batch
    must have been constructed from KDropData objects (a subclass
    of pyg's Data class), so that the required persistent
    attributes to assist this method are present.
    
    Args:
        batch: a pytorch-geometric Batch object of
            batched KDropData objects as one disconnected
            graph.
        k_drop: the number of edges to drop from the node
            of each graph in the batch.
        rs_or_generator: a numpy.random.RandomState or 
            torch.Generator instance for reproducibility.
            The latter is required when using weighted
            drop probabilities (i.e. 'use_edge_wt_probs'
            is True).
        use_edge_wt_probs: bool whether to use
            the batch's edge_weight tensor to weight the
            probabilities of dropping edges. Note this
            requires use of numpy methods, and will be 
            slower.
    Returns:
        If the batch has 'edge_weight', returns a
        2-tuple of the new 'edge_index' and new
        'edge_weight'. Else, returns 1-tuple of the
        new 'edge_index' only.
    """
    if use_edge_wt_probs \
    and (type(rs_or_generator) is torch._C.Generator):
        raise Exception(
            f"Must use numpy.random.RandomState for reproducibility"
            f"with weighted edge drop probabilities."
        )
    # place tensors created here on same device as the batch
    device = batch.x.device
    
    # since graphs are batched, need to additionally offset k_drop indexes
    # by each graph's number of nodes (not just number of k per node)
    graph_index_offsets = torch.cat((
        torch.tensor([0], device=device),
        torch.cumsum(batch.node_count, dim=0)[:-1]
    ))
    
    # create one tensor of all edge indexes to randomly drop, offsetting by
    # 'n_nodes_in_graph' and 'n_edges_for_node' index counts

    # loop through all graphs in batch
    batch_drop_k_is = [None] * batch.num_graphs
    for graph_i, (node_ks, edge_index_offsets, graph_index_offset) \
    in enumerate(zip(batch.node_ks, batch.edge_index_offsets, graph_index_offsets)):

        # loop through all nodes in each graph
        graph_drop_ks = [None] * node_ks.shape[0]
        for node_j, (node_k, edge_index_offset) \
        in enumerate(zip(node_ks, edge_index_offsets)):
            offset = (graph_index_offset + edge_index_offset).to(torch.long)
            offset_end = (offset + node_k).to(torch.long)
            if use_edge_wt_probs:
                # if using sampling probabilities (from edge weights),
                # need to use numpy 'choice' with 'p' arg
                # patch: RandomState.choice runs a check to see if p sums to 1.
                # precision can be poor enough to trigger this error,
                # (known numpy issue: https://github.com/numpy/numpy/pull/6131)
                # so we convert and re-normalize p as a numpy array here (slow...?)
                p = batch.edge_weight[offset:offset_end].cpu().numpy()
                p /= p.sum()
                # insert node_j's k_drops into the graph_i list
                graph_drop_ks[node_j] = torch.tensor(
                    rs_or_generator.choice(
                        a=range(node_k), 
                        size=k_drop, 
                        replace=False,
                        p=p
                    ), 
                    dtype=torch.long,
                    device=device
                ) + offset
            else:
                # this sampling strategy first shuffles edge
                # indexes for the node, then selects first 'k_drop'
                # edges; it does NOT support weighted probabilities
                graph_drop_ks[node_j] = torch.randperm(
                    n=node_k,
                    generator=rs_or_generator,
                    device=device
                )[:k_drop] + offset

        # concat all of graph_i's nodes' k_drops
        batch_drop_k_is[graph_i] = torch.cat(graph_drop_ks)

    # concat all graphs' k_drops tensors
    batch_drop_k_is = torch.cat(batch_drop_k_is) # .to(torch.long)

    # check shape: should be (k_drop * total_N_nodes_all_graphs_comb)
    # print(f"batch_drop_k_is.shape: {batch_drop_k_is.shape}")

    # drop the random edges from batch's 'edge_index' and 'edge_weight' (if exists)
    mask = torch.ones(batch.edge_index.shape[1], dtype=torch.bool) # , device=tensor.device)
    mask[batch_drop_k_is] = False  # set to False at edge indexes to remove
    new_edge_index = batch.edge_index[:, mask]
    if hasattr(batch, 'edge_weight'):
        new_edge_weight = batch.edge_weight[mask]
        return (new_edge_index, new_edge_weight)
    else:
        return (new_edge_index, )


def get_Batch_P_Wjxs(
    x: Batch,
    P_sparse: torch.Tensor,
    scales_type: str = 'dyadic',
    channels_t_is: Optional[torch.Tensor] = None,
    # custom_scales_max_t: int = 32, # deprecated
    J: int = 5,
    include_lowpass: bool = True,
    filter_stack_dim: int = -1
) -> torch.Tensor:
    r"""
    Computes P (diffusion) wavelet filtrations
    on a disconnected graph, using recursive
    sparse matrix multiplication. That is,
    skips computing increasingly dense powers of P, 
    by these steps:
    
    1. Compute $y_t = P^t x$ recursively via $y_t = P y_{t-1}$,
       (only using P, and not its powers, which grow denser).
    2. Subtract $y_{2^{j-1}} - y_{2^{j}}$ [dyadic scales]. 
        The result is $W_j x = (P^{2^{j-1}} - P^{2^j}) x$.
        (Thus, we never form the matrices P^t, t > 1, which get 
        denser with as the power increases.)
    
    Args:
        x: stacked node-by-channel (N, c) data matrix for a 
            disconnected  batch graph of a pytorch geometric 
            Batch object. 
        P_sparse: sparse diffusion operator matrix 
            for disconnected batch graph of a pytorch
            geometric Batch object (output of 'pygu.get_Batch_P_sparse').
        scales_type: 'dyadic' or 'custom' or None for fixed P^1.
        channels_t_is: tensor of shape (n_channels, n_scale_split_ts)
            for calculating 'custom' wavelet scales, containing the 
            indices of ts 0...max($t$). Scales are constructed uniquely
            for each channel of x from $t$s with adjacent indices in rows 
            of this tensor. If None, this function defaults to dyadic 
            scales.
        custom_scales_max_t: [deprecated; now calculated as 2 ** J] 
            the maximum power of $P$ to compute in $P^t$, for manual
            scales.
        J: max wavelet filter order, for dyadic scales. For example,
            $J = 4$ will give $T = 2^4 = 16$ max diffusion step.
        include_lowpass: whether to include the 
            'lowpass' filtration, $P^{2^J} x$.
        filter_stack_dim: new dimension in which to 
           stack Wjx (filtration) tensors.
    Returns:
        Dense tensor of shape (batch_total_nodes, n_channels,
        n_filtrations) = 'Ncj'.
    """

    # print('x.device:', x.device)
    # print('Ptx.device:', Ptx.device)
    # print('P_sparse.device:', P_sparse.device)

    # 1 set of scales shared by all channels: dyadic or custom
    shared_powers_to_save = None
    if (scales_type == 'dyadic'):
        shared_powers_to_save = 2 ** torch.arange(J + 1)
        range_upper_lim = J + 2
    elif (channels_t_is is not None) \
    and channels_t_is.dim() == 1:
        shared_powers_to_save = channels_t_is
        range_upper_lim = channels_t_is.shape[0]
        # print('shared_powers_to_save:', shared_powers_to_save)
        # print('range_upper_lim:', range_upper_lim)
        
    if shared_powers_to_save is not None:
        Ptxs = [x]
        Ptx = x.detach().clone()
        
        # calc P^t x for t \in 1...2^J, saving only needed P^txs
        # print('P_sparse.shape', P_sparse.shape)
        # print('Ptx.shape', Ptx.shape)
        for j in range(1, shared_powers_to_save[-1] + 1):
            try:
                Ptx = torch.sparse.mm(P_sparse, Ptx)
                if j in shared_powers_to_save:
                    # print(f"j={j}")
                    # it's possible the same power is in a
                    # custom 'channels_t_is' more than once
                    if channels_t_is is not None:
                        j_ct = (shared_powers_to_save == j).sum().item()
                        for _ in range(j_ct):
                            Ptxs.append(Ptx)
                    else:
                        Ptxs.append(Ptx)
            except Exception as e:
                print(f"j={j}")
                raise e

        # print('len(Ptxs):', len(Ptxs))
        Wjxs = [Ptxs[j - 1] - Ptxs[j] for j in range(1, range_upper_lim)] # J + 2)]
        if include_lowpass:
            Wjxs.append(Ptxs[-1])
        Wjxs = torch.stack(Wjxs, dim=filter_stack_dim)

    # custom unique scales for each channel
    elif (scales_type == 'custom') \
    and (channels_t_is.dim() == 2):
        Ptxs = [x.to_dense()]
        Ptx = x.detach().clone()
        
        # calc P^t x for t \in 1...T, saving all powers of t
        custom_scales_max_t = int(2 ** J) # e.g. J = 5 -> 32
        for j in range(1, custom_scales_max_t + 1):
            Ptx = torch.sparse.mm(P_sparse, Ptx)
            # print('Ptx.device:', Ptx.device)
            Ptxs.append(Ptx.to_dense())

        device = Ptxs[0].device

        # compute filtrations ('Wjxs')
        # note that filter (P^u - P^v)x = (P^u x) - (P^v x)
        # here indexes for (P^u x) and (P^v x) within 'Ptxs' for each
        # channel are adjacent entries in each channel's 't_is'
        Wjxs = torch.stack([
            torch.stack([
                # as of Nov 2024, bracket slicing doesn't work with sparse tensors
                # patch: entries of 'Ptxs' made dense above, when added to Ptxs
                Ptxs[t_is[t_i - 1]][:, c_i] - Ptxs[t_is[t_i]][:, c_i] \
                for t_i in range(1, len(t_is))
            ], dim=-1) \
            for c_i, t_is in enumerate(channels_t_is)
        ], dim=1) 
        
        '''
        Wjxs = [None] * channels_t_is.shape[0]
        for c_i, t_is in enumerate(channels_t_is):
            channel_Wjxs = [None] * (len(t_is) - 1)
            c_i_tensor = torch.tensor([c_i]).to(device)
            
            for t_i in range(1, len(t_is)):
                Pu = torch.index_select(Ptxs[t_is[t_i - 1]], 1, c_i_tensor)
                Pv = torch.index_select(Ptxs[t_is[t_i]], 1, c_i_tensor)
                channel_Wjxs[t_i - 1] = (Pu - Pv).squeeze()
                # print('channel_Wjxs.shape:', channel_Wjxs[t_i - 1].shape)
                
            channel_Wjxs = torch.stack(channel_Wjxs, dim=-1)
            Wjxs[c_i] = channel_Wjxs
        Wjxs = torch.stack(Wjxs, dim=1)
        print('Wjxs.shape:', Wjxs.shape)
        '''
        
        # lowpass = P^T x, for all channels
        if include_lowpass:
            # print('Ptxs[-1].shape:', Ptxs[-1].shape)
            Wjxs = torch.concatenate(
                (Wjxs, Ptxs[-1].unsqueeze(dim=-1)), 
                dim=-1
            )
        # Wjxs shape (N, n_channels, n_filters)
        # print('Wjxs.shape:', Wjxs.shape)

    elif scales_type is None:
        Ptx = x.detach().clone()
        Ptx = torch.sparse.mm(P_sparse, Ptx)
        Wjxs = Ptx.unsqueeze(dim=-1)
    else:
        raise NotImplementedError(f"No method implemented for scales_type={scales_type}")
        
    return Wjxs



def get_Batch_V_sparse(
    batch_size: int,
    batch_index: torch.Tensor,
    L_eigenvecs: torch.Tensor,
    max_kappa: Optional[int] = None,
    device: Optional[str] = None
) -> torch.Tensor:
    r"""
    Computes 

    Args:
        batch_size: number of graphs in the
            batch.
        batch_index: batch_index (e.g., from a
            pytorch_geometric Batch object). Can be
            None for a batch with 1 graph.
        L_eigenvecs: 2-d tensor holding stacked
            eigenvectors of the graph Laplacians 
            for each graph x_i; shape (N, k) =
            (total_n_nodes, n_eigenvectors).
        max_kappa: maximum number of eigenvectors to
            utilize, up to the number stored in 
            'L_eigenvecs'.
        device: string device key (e.g., 'cpu', 'cuda', 
            'mps') for placing the output tensor; if
            None, will check for cuda, else assign to cpu.
    Returns:
        Sparse block-diagonal matrix V, where blocks
        hold eigenvectors for each graph in the batch, hence
        shape (bk, N), where b is the batch size (number of
        graphs), k is the number of eigenvectors, and N is
        the total number of nodes across all graphs.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    values = []
    n_ctr = 0
    k_ctr = 0
    n_indices = []
    k_indices = []

    for i in range(batch_size):
        if batch_size > 1:
            x_i_mask = (batch_index == i)
            v = L_eigenvecs[x_i_mask] # shape (n_i, k)
        else:
            v = L_eigenvecs
            
        if max_kappa is not None:
            v = v[:, :max_kappa]
        # need to ravel individual v, not entire 'L_eigenvecs'
        values.extend(v.ravel())
        v_indices = v.nonzero() # shape (n_i * k, 2)
        
        n_indices.extend(v_indices[:, 0] + n_ctr)
        k_indices.extend(v_indices[:, 1] + k_ctr)
        
        n_ctr += v.shape[0]
        k_ctr += v.shape[1]

    # generate indices such that shape is (bk, N)
    indices = torch.stack((
        torch.tensor(k_indices),
        torch.tensor(n_indices)
    ))
    
    size = (
        k_ctr, # bk
        L_eigenvecs.shape[0] # N
    )
    
    V_sparse = torch.sparse_coo_tensor(
        indices=indices,
        values=torch.tensor(values),
        size=size,
        dtype=torch.float
    ).to(device)
    
    return V_sparse



def get_Batch_spectral_Wjxs(
    num_graphs: int,
    x: torch.Tensor,
    batch_index: torch.Tensor,
    V_sparse: torch.Tensor,
    Wjs_spectral: torch.Tensor,
    L_eigenvecs: torch.Tensor,
    max_kappa: Optional[int] = None
) -> torch.Tensor:
    r"""
    Computes spectral filtration values for a pytorch
    geometric batch of graphs with multiple channels,
    for multiple (precomputed scalar) filters in the 
    Fourier domain, defined by:
    $\sum_{i=1}^{k} w_j(\lambda_i) \hat{f}(i) \psi_i$
    for $1 \leq j \leq J$. For details, see the MFCN
    manuscript. Note $w_j(\lambda_i)$ and $\hat{f}(i)$
    are scalars.

    Args:
        num_graphs: number of graphs in the pytorch geometric 
            Batch. 
        x: stacked node-by-channel (N, c) data matrix for a 
            disconnected  batch graph of a pytorch geometric 
            Batch object. 
        batch_index: 'batch' (index set) from the pytorch 
            geometric Batch. Optional for batches with 1 graph.
        V_sparse: sparse block-diagonal 2d tensor where blocks are
            (k, n_i) matrices of each graph's eigenvectors; the
            output of 'get_Batch_V_sparse()'.
        Wjs_spectral: tensor of pre-computed spectral filter scalars;
            shape (kappa, n_filters) [1 graph] or 
            (n_graphs, kappa, n_filters) [1 or batched graphs].
        L_eigenvecs: tensor of pre-computed eigenvalues for all graphs
            in batch stacked node-wise -> shape 
            (total_n_nodes, n_eigenvectors) = (N, k).
        max_kappa: maximum number of eigenvectors to
            utilize, up to the number stored in 
            'L_eigenvecs'.
    Returns:
        Tensor of spectrally filtered signal values, of 
        shape (total_n_nodes, n_channels, n_filters) = (N, c, j).
        Note FWV[0:n_0, 0, 0] selects the spectral convolution for
        x_0, channel_0, filter_0, which has length n_0 (n_nodes in 
        first graph).
    """
    
    # compute Fourier coefficients of each channel signal
    # on each graph, wrt k graph-Laplacian eigenvectors
    # (bk=K, N) @ (N, c) -> (K, c)
    # -> F shape: (num_graphs * n_eigenvectors, n_channels) = 'Kc'
    # note this gets rid of 'N' and standardizes all graphs of 
    # possibly different sizes/num_nodes into the same Fourier space size
    F = torch.sparse.mm(V_sparse, x)
    # if x is also sparse, F will be sparse

    # PATCH: as of Oct 2024, torch sparse tensors can't use 'tensor_split'
    F = F.to_dense()

    # split and stack F into shape 'bkc' -> inner matrix
    # cols are a graph's k Fourier coeffs for each channel
    F = torch.stack(
        torch.tensor_split(F, num_graphs, dim=0),
        dim=0
    )
    
    # 'Wjs_spectral' (stored in DataBatch object) needs shape
    # 'bkj' -> inner row is a graph's eigenpair's W(lambda_i)
    # j filter scalar values
    # -> this einsum produces FW, a 4-d tensor where each
    # inner row is a graph's channel's j filters, times 
    # each of the k Fourier coeffs -> size 'bcjk'
    # (that is, FW holds all of the $w_j(\lambda_i) \cdot \hat{f}(i)$
    # scalar combinations)

    # Wjs_spectral.shape (kappa, n_filters) [1 graph] or 
    # (n_graphs, kappa, n_filters) [batched graphs]
    # L_eigenvecs.shape (total_n_nodes, kappa) [1 graph or (stacked) batched graphs]
    if Wjs_spectral.dim() == 2:
        # have 1 graph in batch; unsqueeze to have size 1 at dim 0
        Wjs_spectral = Wjs_spectral.unsqueeze(dim=0)
        # L_eigenvecs = L_eigenvecs.unsqueeze(dim=0)
    # print('Wjs_spectral.shape:', Wjs_spectral.shape) 
    # print('L_eigenvecs.shape:', L_eigenvecs.shape) 

    # trim spectral objects to use max_kappa eigenpairs
    if max_kappa is not None:
        Wjs_spectral = Wjs_spectral[:, :max_kappa, :]
        L_eigenvecs = L_eigenvecs[:, :max_kappa]

    # calc FW -> each row_j has k Four. coeffs for filter w_j
    # (for each graph in the batch)
    FW = torch.einsum('bkc,bkj->bcjk', F, Wjs_spectral)
    # print('FW.shape:', FW.shape) 

    # 'L_eigenvecs' (stored in DataBatch object) has shape
    # (sum(n_i), k): eigenvectors of all graphs are stacked
    # vertically in blocks (like batch.x data: in case graphs
    # have different n_i, and indexable by batch.batch)
    fwvs = [None] * num_graphs
    for i in range(num_graphs):

        if num_graphs > 1:
            mask = (batch_index == i)
            # subset one graph's eigenvectors -> shape (k, n_i) (after 
            # the transpose); k rows are e-vecs of length n_i
            v = L_eigenvecs[mask].T
        else:
            v = L_eigenvecs.T

        # subset one graph's fw -> shape (c, j, k), where each 
        # row_j has k Four. coeffs for filter w_j
        fw = FW[i]
        
        # this einsum calculates the spectral convolutions for one graph
        # by multiplying e-vec $\psi_i$s by the 
        # $w_j(\lambda_i) \cdot \hat{f}(i)$ scalar product, and summing 
        # e-vecs 1...k
            # if '->cjn': each row_j is a channel's convolution using 
            # filter W_j (linear combo of eigenvectors of length n_i), 
            # all for one graph
            # if '->ncj': each element is one node's convolution value
            # for one channel and one filter, all for one graph
        fwv = torch.einsum('cjk,kn->ncj', fw, v)
        fwvs[i] = fwv

    # concat. all graphs' fwvs along first (node) axis 
    # -> shape (N, c, j), where N = sum(n_i)
    FWV = torch.concatenate(fwvs)
    return FWV


def set_mfcn_model_kwargs(
    args,
    model_key: str,
    in_channels: int,
    num_nodes_one_graph: Optional[int] = None, 
    node_pool_out_channels: int = 1,
    k_drop_kwargs: Dict[str, Any] = {},
    k_drop_generator = None,
) -> Dict[str, Any]:
    """
    Creates a dictionary of kwargs for a MFCN module.

    Args:
        args: ArgsTemplate instance for the dataset
            in use.
        model_key: string key for MFCN model, e.g. 
            'mfcn_p' or 'mcn'.
        in_channels: number of input signal channels.
        num_nodes_one_graph: optional number of nodes
            if the dataset consists of one graph.
        node_pool_out_channels: number of final channels per node 
            (after linear layer node pooling) to output when no fully-
            connected head is used. Similar to 'out_channels' in pytorch-
            geometric's GCN, etc., model classes. Set to 1 if not used.
        k_drop_kwargs: if 'use_k_drop', kwargs to feed the 
            'batch_drop_rand_k_edges' method.
        k_drop_generator: if 'use_k_drop', the torch.Generator 
            object used to generate random numbers.
    Returns:
        Dictionary/kwargs for initializing an MFCN_Module
        instance.
    """
    filter_type = model_key.split("_")[-1] # 'p' or 'spectral'
    # MCN models have no wavelets
    args.WAVELET_TYPE = filter_type if ('mfcn' in model_key) else None
    args.NON_WAVELET_FILTER_TYPE = filter_type if ('mcn' in model_key) else None
    # MCN models have no cross-filter combinations
    cross_Wj_ns_combos_out_per_chan = None \
        if ('mcn' in model_key) else args.MFCN_CROSS_FILTER_COMBOS_OUT
    # set cross-channel combinations depending on 'mcn' vs. 'mfcn' model
    within_Wj_ns_chan_out_per_filter = args.MCN_WITHIN_FILTER_CHAN_OUT \
        if ('mcn' in model_key) else args.MFCN_WITHIN_FILTER_CHAN_OUT

    if args.USE_K_DROP:
        k_drop_kwargs = {
            'k_drop': args.K_DROP, 
            'rs_or_generator': k_drop_generator,
            'use_edge_wt_probs': args.K_DROP_USE_EDGE_WT_PROBS,
        }
    else:
       k_drop_kwargs = {} 
    
    model_kwargs = {
        'wavelet_type': args.WAVELET_TYPE,
        'non_wavelet_filter_type': args.NON_WAVELET_FILTER_TYPE,
        'filter_c': args.SPECTRAL_C,
        'p_wavelet_scales': args.P_WAVELET_SCALES,
        'P_wavelets_channels_t_is': torch.tensor(args.P_WAVELET_SCALES_PRECALC) \
            if args.P_WAVELET_SCALES == 'custom' else None,
        'only_use_avg_P_wavelets_channels_t_is': args.CUSTOM_P_AVG_SCALES_ONLY,
        'in_channels': in_channels,
        'use_skip_connections': args.MFCN_USE_SKIP_CONNECTIONS,
        'use_input_recombine_layer': args.MFCN_USE_INPUT_RECOMBINE_LAYER,
        'input_recombine_layer_kwargs': {
            'out_features': args.MFCN_INPUT_RECOMBINE_LAYER_OUT_CHANNELS
        },
        'use_k_drop': args.USE_K_DROP,
        'k_drop_kwargs': k_drop_kwargs,
        'num_nodes_one_graph': num_nodes_one_graph,
        'J': args.J,
        'include_lowpass_wavelet': args.INCLUDE_LOWPASS_WAVELET,
        'within_Wj_ns_chan_out_per_filter': within_Wj_ns_chan_out_per_filter,
        'cross_Wj_ns_combos_out_per_chan': cross_Wj_ns_combos_out_per_chan,
        'max_kappa': args.MFCN_MAX_KAPPA,
        'channel_pool_key': args.MFCN_FINAL_CHANNEL_POOLING,
        'node_pooling_key': args.MFCN_FINAL_NODE_POOLING,
        'node_pool_linear_out_channels': node_pool_out_channels,
    }
    return model_kwargs
