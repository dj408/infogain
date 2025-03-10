"""
LEGS module and accessory methods
Original author: Alex Tong
Edited by: David Johnson [davejohnson408@u.boisestate.edu]

DJ key changes:
1. Generalized to to use 32 (or any dyadic) number of diffusion steps.
2. Made final 1st- and 2nd-order scattering features optionally absolute value.
(This loses information where -x is conflated with x).
3. Swapped normalized/statistical moments for unnormalized moments in pooling
step. Added further pooling options (mean, max).
4. Consolidated tensor multiplications and reshaping into torch.einsum operations;
replaced list appends with insertions; replaced tensor concatenations in new
dimension with torch.stack operations.
5. Added method to save state of best selector matrix tensor, for inspection
after model training.
6. Integrated with the rest of DJ's codebase with the 'LEGS_MLP' class (for
combining the LEGS module with a regressor/classifier head).

TO DO
[ ] implement sparse P in recursive P^t x diffusion calculations?

LEGS reference paper: "Data-Driven Learning of Geometric Scattering Networks"
IEEE Machine Learning for Signal Processing Workshop 2021

Original LEGS code repo:
https://github.com/KrishnaswamyLab/LearnableScattering/blob/main/models/LEGS_module.py
"""
import numpy as np
import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add # scatter_mean

"""
DJ add'l imports
"""
# import pickle
from pathlib import Path
from typing import (
    Tuple,
    List,
    Dict,
    Any,
    Optional,
    Iterable,
    Callable
)
from pyg_utilities import (
    channel_pool,
    moments_channel_pool
)
import base_module as bm
import vanilla_nn as vnn


class LazyLayer(torch.nn.Module):
    r""" 
    AT: Currently a single elementwise multiplication with one laziness parameter per
    channel. This is run through a softmax so that this is a real laziness parameter.

    DJ: This optional layer creates a trainable weight in an alternative construction of 
    the diffusion operator $P$, $P_{\alpha}$. See "Relaxed geometric scattering" on
    p. 2 of https://ml4molecules.github.io/papers2020/ML4Molecules_2020_paper_63.pdf
    (The 'Lazy' refers to modification of the lazy random walk matrix)
    """

    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(2, n))

    def forward(self, x, propogated):
        inp = torch.stack((x, propogated), dim=1)
        s_weights = torch.nn.functional.softmax(self.weights, dim=0)
        return torch.sum(inp * s_weights, dim=-2)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)
    

def gcn_norm(
    edge_index,
    edge_weight=None,
    num_nodes=None,
    add_self_loops=False,
    dtype=None
):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1), ), 
            dtype=dtype,
            device=edge_index.device
        )

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    
    return edge_index, deg_inv_sqrt[row] * edge_weight


class Diffuse(MessagePassing):
    """ 
    Implements low pass walk with optional weights.
    """
    def __init__(
        self, 
        in_channels,
        out_channels, 
        trainable_laziness=False,
        fixed_weights=True
    ):
        super().__init__(aggr="add", node_dim=-3)  # "Add" aggregation.
        assert in_channels == out_channels
        self.trainable_laziness = trainable_laziness
        self.fixed_weights = fixed_weights
        if trainable_laziness:
            self.lazy_layer = LazyLayer(in_channels)
        if not self.fixed_weights:
            self.lin = torch.nn.Linear(in_channels, out_channels)


    def forward(self, x, edge_index, edge_weight=None):

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 2: Linearly transform node feature matrix.
        # turn off this step for simplicity
        if not self.fixed_weights:
            x = self.lin(x)

        # Step 3: Compute normalization
        edge_index, edge_weight = gcn_norm(
            edge_index, 
            edge_weight, 
            num_nodes=x.size(self.node_dim), 
            dtype=x.dtype
        )

        # Step 4-6: Start propagating messages.
        propogated = self.propagate(
            x=x,
            edge_index=edge_index, 
            edge_weight=edge_weight,
            size=None,
        )
        if not self.trainable_laziness:
            return 0.5 * (x + propogated)

        return self.lazy_layer(x, propogated)


    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return edge_weight.view(-1, 1, 1) * x_j


    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)


    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out


class Scatter(torch.nn.Module):
    """
    DJ: added 'J' arg, to allow dyadic scales
    different than J = 4 (previously hardcoded).
    """
    def __init__(
        self,
        in_channels: int,
        channel_pool_key: str = 'moments',
        J: int = 4,
        n_moments: Optional[int] = 4,
        use_mod_scat_features: bool = True, # AT's default
        trainable_laziness: bool = False,
        save_best_selector_matrix: bool = True,
        selector_matrix_save_path: Optional[str] = None,
        verbosity: int = 0
    ):
        super().__init__()
        self.save_best_selector_matrix = save_best_selector_matrix
        # self.selector_matrix_save_path = selector_matrix_save_path
        self.verbosity = verbosity
        self.in_channels = in_channels
        """
        DJ: add other pooling options besides scattering
        moments
        """
        self.channel_pool_key = channel_pool_key.lower()
        self.n_moments = n_moments
        if self.channel_pool_key == 'moments':
            self.out_channel_mult = self.n_moments
        elif (self.channel_pool_key == 'max') \
        or self.channel_pool_key == 'mean':
            self.out_channel_mult = 1
        elif ('max' in self.channel_pool_key) \
        and ('mean' in self.channel_pool_key):
            self.out_channel_mult = 2
        else:
            raise NotImplementedError(
                f"Channel pooling '{self.out_channel_mult}' not implemented."
            )
        self.J = J
        self.use_mod_scat_features = use_mod_scat_features
        self.trainable_laziness = trainable_laziness
        self.diffusion_layer1 = Diffuse(
            in_channels,
            in_channels,
            trainable_laziness
        )
        self.diffusion_layer2 = Diffuse(
            self.out_channel_mult * in_channels, 
            self.out_channel_mult * in_channels, 
            trainable_laziness
        )
        self.wavelet_constructor = torch.nn.Parameter(
            self._get_dyadic_scales_matrix_tensor(self.J).clone().detach(),    
        )
        self.n_wavelets = self.wavelet_constructor.shape[0] # 4 / 5
        self.n_diffusion_steps = self.wavelet_constructor.shape[1] # 17 / 33
        self.feng_filters_2nd_order = self._feng_filters_2nd_order()
        

    def on_best_model(self, save_path: str, fold_i: Optional[int]) -> None:
        """
        Save the state of the wavelet scales selector
        matrix of the best model (as a numpy array).
        If saved during k-folds cross validation, save
        best for each fold by overwriting npy file with
        the filename index for the fold in its name.

        Note that if no matrix is saved, the model
        failed to learn / improved beyond initial weights,
        meaning the dyadic initial matrix was 'best'.
        """
        if self.save_best_selector_matrix:
            # to save folds' matrix 'npy' files in a subdirectory:
            # save_path = f"{save_path}/selector_matrix" \
            #     if (fold_i is not None) else f"{save_path}"
            # Path(save_path).mkdir(exist_ok=True)
            filepath = f"{save_path}/best_selector_matrix"
            if (fold_i is not None):
                filepath += f"_{fold_i}"
            m = self.wavelet_constructor.cpu().detach().numpy()
            np.save(filepath, m)
            

    def out_shape(self):
        # x output for each channel is a 1-d tensor of concatenated 0th-, 1st-, and 2nd-order 
        # pooled features
        # length = [n filters] * [n pools: e.g. 1 max/mean/sum, or 4 moments] * [in_channels]
        # n_filters_all_orders = 11 for J = 4; 16 for J = 5
        n_filters_all_orders = 1 \
            + self.n_wavelets \
            + int(self.n_wavelets * (self.n_wavelets - 1) / 2) # 2nd-order
        return n_filters_all_orders * self.out_channel_mult * self.in_channels

    
    def _get_dyadic_scales_matrix_tensor(self, J: int) -> torch.Tensor:
        """
        DJ: added this method instead of hardcoded selector 
        matrix dyadic initialization.
        - If J = 4, get T = 16
        - If J = 5, get T = 32
        """
        m = torch.zeros((J, 2 ** J + 1))
        # in the original LEGS code, -1s were left of (smaller index than)
        # +1s in hardcoded scales selector matrix: I think this is 
        # backwards (we want to subtract greater powers)
        # cf. Eq 4 in LEGS paper
        # I've corrected things here, so -1s are at greater indices than +1s
        ones_idx = [2 ** i for i in range(0, J)]
        neg_ones_idx = [2 ** i for i in range(1, J + 1)]
        
        for j in range(J):
            m[j, ones_idx[j]] = 1.0
            m[j, neg_ones_idx[j]] = -1.0
        return m.to(torch.float)

    
    def _feng_filters_2nd_order(self):
        """
        DJ: these Feng filters (index subset) are to only
        include 2nd-order filters where a higher-power 
        (lower-frequency) wavelet is applied to all lower-power
        (higher frequency) 1st order filtrations; that is, 
        ensure j < j'. (Also generalized this method for 
        arbitrary J.)
        """
        idx = [self.J]
        for j in range(2, self.J):
            for jprime in range(0, j):
                idx.append(self.J * j + jprime)
        # example: idx = [4, 8, 9, 12, 13, 14] if self.J = 4
        return idx


    def forward(self, data):
        """
        einsums key:
            j: number of wavelet filters
            p: number of wavelet filters, repeated for outer product
            t: max diffusion step + 1 (= num. cols in selector matrix)
            n: total number of nodes in batch
            c: number of (node signal) channels
        """
        x, edge_index = data.x, data.edge_index
        if x.is_sparse:
            x = x.to_dense()

        
        # 0th-order scattering (don't take modulus of x)
        s0 = x.detach().clone().unsqueeze(dim=1) # shape 'n1c'

        
        # 1st-order scattering: |Wjx| or Wjx for 1 <= j <= J
        avgs = [None] * self.n_diffusion_steps
        # P^0 x = x
        avgs[0] = s0
        for i in range(self.n_diffusion_steps - 1):
            # recursive diffusion (powers of P^t @ x) starting at t = 1
            avgs[i+1] = self.diffusion_layer1(avgs[i], edge_index)
        diffusion_levels = torch.stack(avgs) # shape 'tnc1'

        s1 = torch.einsum(
            'jt,tnc->njc',
            self.wavelet_constructor,
            diffusion_levels.squeeze() 
        )
        # optional: take modulus of s1 (loses information: -x is conflated with x)
        # to use as the final s1 output features
        if self.use_mod_scat_features:
            s1 = torch.abs(s1)

        
        # 2nd-order scattering: |Wj'|Wjx|| or Wj'|Wjx| for 1 <= j < j' <= J
        avgs = [None] * self.n_diffusion_steps
        # take modulus of s1 if not taken already before applying Wj'
        avgs[0] = s1 if self.use_mod_scat_features else torch.abs(s1)
        for i in range(self.n_diffusion_steps - 1):
            avgs[i+1] = self.diffusion_layer2(avgs[i] , edge_index)
        # take modulus of 1st-order filtrations
        diffusion_levels_2 = torch.stack(avgs)

        # note: pj dimension is squared/outer product of j filters: each filter 
        # gets applied to every other in AT's approach, and then subsetted out 
        # to keep only where j > j'
        s2 = torch.einsum(
            'pt,tncj->npjc', 
            self.wavelet_constructor,
            diffusion_levels_2 
        )
        # flatten pj into one dimension ('njc') and subset to where Wj'|Wjx| has 
        # j < j' only
        s2 = s2.reshape(s2.shape[0], -1, self.in_channels)
        s2 = s2[:, self.feng_filters_2nd_order, :]
        if self.use_mod_scat_features:
            s2 = torch.abs(s2)

        # concatenate 0th-, 1st-, and (feng-filtered) 2nd-order scattering coeffs
        # in the 'j' dim (all should have shape 'njc')
        x = torch.cat((s0, s1, s2), dim=1)

        if self.verbosity > 0:
            print('x.shape (before pooling):', x.shape)

        # channel pooling
        batch_index = data.batch if hasattr(data, 'batch') else None
        num_graphs = 1 if (batch_index is None) else data.num_graphs
        if self.channel_pool_key == 'moments':
            x = moments_channel_pool(x, batch_index, num_graphs)
        else: # max, mean, etc. pooling
            x = channel_pool(
                self.channel_pool_key,
                x,
                num_graphs,
                batch_index
            )

        # flatten graphs' xs into 2 dim each (req'd by MLP head)
        # -> x shape = (n_graphs, poolings_per_channel * n_channels)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
            
        if self.verbosity > 0:
            print('x.shape (after pooling):', x.shape)

        return x



class LEGS_MLP(bm.BaseModule):
    """
    DJ: this class replaces Alex's TSNet class, to
    be compatible with rest of DJ's code.
    """
    def __init__(
        self,
        legs_module_kwargs: Dict[str, Any] = {},
        base_module_kwargs: Dict[str, Any] = {},
        fc_kwargs: Dict[str, Any] = {},
        verbosity: int = 0
    ):
        super(LEGS_MLP, self).__init__(**base_module_kwargs)
        self.verbosity = verbosity
        self.scatter = Scatter(**legs_module_kwargs)
        self.post_scatter_act = torch.nn.LeakyReLU() # AT's original choice
        self.fc = vnn.VanillaNN(
            input_dim=self.scatter.out_shape(),
            **fc_kwargs
        )

    def on_best_model(self, kwargs: Dict[str, Any] = {}) -> None:
        self.scatter.on_best_model(**kwargs)

    def forward(self, x):
        x = self.scatter(x)
        x = self.post_scatter_act(x)
        model_output_dict = self.fc.forward(x)
        return model_output_dict
        






'''
Note: `moments_channel_pool` from DJ's
`pyg_utilities` now used instead

def scatter_moments(
    graph: torch.Tensor,
    batch_indices: Optional[torch.Tensor], 
    moments_returned: int = 4
) -> torch.Tensor: 
    """ 
    Compute specified statistical coefficients for each feature of each graph passed. 
    The graphs expected are disjoint subgraphs within a single graph, whose feature 
    tensor is passed as argument "graph."
    "batch_indices" connects each feature tensor to its home graph. [DJ: if None, 
    we use a tensor with all one int value (likely got a 1-graph batch)]
    "Moments_returned" specifies the number of statistical measurements to compute. 
    If 1, only the mean is returned. If 2, the mean and variance. If 3, the mean, 
    variance, and skew. If 4, the mean, variance, skew, and kurtosis.
    The output is a dictionary. You can obtain the mean by calling output["mean"] 
    or output["skew"], etc.
    """

    if batch_indices is None:
        batch_indices = torch.zeros(data.x.shape[0], dtype=torch.long)

    # Step 1: Aggregate the features of each mini-batch graph into its own tensor
    graph_features = [torch.zeros(0).to(graph) for i in range(torch.max(batch_indices) + 1)]

    for i, node_features in enumerate(graph):

        # Sort the graph features by graph, according to batch_indices. For each graph, 
        # create a tensor whose first row is the first element of each feature, etc.
        # print("node features are", node_features)
        
        if (len(graph_features[batch_indices[i]]) == 0):  
            # If this is the first feature added to this graph, fill it in with the features.
            # .view(-1,1,1) changes [1,2,3] to [[1],[2],[3]], so that we can add each column 
            # to the respective row.
            graph_features[batch_indices[i]] = node_features.view(-1, 1, 1)
        else:
            graph_features[batch_indices[i]] = torch.cat(
                (graph_features[batch_indices[i]], 
                 node_features.view(-1, 1, 1)), 
                dim=1
            ) # concatenates along columns

    statistical_moments = {"mean": torch.zeros(0).to(graph)}

    if moments_returned >= 2:
        statistical_moments["variance"] = torch.zeros(0).to(graph)
    if moments_returned >= 3:
        statistical_moments["skew"] = torch.zeros(0).to(graph)
    if moments_returned >= 4:
        statistical_moments["kurtosis"] = torch.zeros(0).to(graph)

    for data in graph_features:
        data = data.squeeze()
        def m(i):  # ith moment, computed with derivation data
            return torch.mean(deviation_data ** i, axis=1)
        mean = torch.mean(data, dim=1, keepdim=True)
        
        if moments_returned >= 1:
            statistical_moments["mean"] = torch.cat(
                (statistical_moments["mean"], mean.T), 
                dim=0
            )

        # produce matrix whose every row is data row - mean of data row
        deviation_data = data - mean
        
        # variance: difference of u and u mean, squared element wise, summed and 
        # divided by n-1
        variance = m(2)
        
        if moments_returned >= 2:
            statistical_moments["variance"] = torch.cat(
                (statistical_moments["variance"], variance[None, ...]), 
                dim=0
            )

        # skew: 3rd moment divided by cubed standard deviation (sd = sqrt variance), 
        # with correction for division by zero (inf -> 0)
        skew = m(3) / (variance ** (3 / 2)) 
        skew[skew > 1000000000000000] = 0  # multivalued tensor division by zero produces inf
        skew[skew != skew] = 0  # single valued division by 0 produces nan. In both cases we replace with 0.
        if moments_returned >= 3:
            statistical_moments["skew"] = torch.cat(
                (statistical_moments["skew"], skew[None, ...]), 
                dim=0
            )

        # kurtosis: fourth moment, divided by variance squared. Using Fischer's definition 
        # to subtract 3 (default in scipy)
        kurtosis = m(4) / (variance ** 2) - 3 
        kurtosis[kurtosis > 1000000000000000] = -3
        kurtosis[kurtosis != kurtosis] = -3
        if moments_returned >= 4:
            statistical_moments["kurtosis"] = torch.cat(
                (statistical_moments["kurtosis"], kurtosis[None, ...]), 
                dim=0
            )
    
    # Concatenate into one tensor (alex)
    statistical_moments = torch.cat([v for k,v in statistical_moments.items()], axis=1)
    return statistical_moments
'''


'''
first draft at editing forward method (no einsums)

    def forward(self, data):
        """
        debugging print output:
        
        J = 4
        s0.shape: torch.Size([9515, 29, 1])
        diffusion_levels.shape: torch.Size([17, 9515, 29, 1]) # tnc1
        self.wavelet_constructor.shape: torch.Size([4, 17])
        diff_levels_view.shape: torch.Size([17, 275935])
        subtracted.shape 1: torch.Size([4, 275935])
        subtracted.shape 2: torch.Size([4, 9515, 29])
        s1.shape 1: torch.Size([9515, 29, 4])
        diffusion_levels_2.shape: torch.Size([17, 9515, 29, 4])
        diff_levels2_view.shape: torch.Size([17, 1103740])
        subtracted2.shape 1 torch.Size([4, 9515, 29, 4])
        subtracted2.shape 2 torch.Size([9515, 4, 29, 4])
        subtracted2.shape 3 torch.Size([38060, 29, 4])
        subtracted2.shape 4 torch.Size([38060, 4, 29])
        s2_swapped.shape torch.Size([9515, 16, 29])
        s2.shape (feng-filtered) torch.Size([9515, 6, 29])
        x.shape 1 torch.Size([9515, 29, 5])
        x.shape 2 torch.Size([9515, 5, 29])
        x.shape 3 torch.Size([9515, 11, 29])
        """
        
        x, edge_index = data.x, data.edge_index
        """
        DJ patch: original code doesn't do sparse tensors
        (torch.sparse prob. didn't exist)
        """
        if x.is_sparse:
            x = x.to_dense()
            
        """
        j: number of wavelet filters
        t: max diffusion step + 1
        n: total number of nodes in batch
        c: number of node signal channels
        
        s1 = torch.einsum(
            'jt,tnc->ncj',
            self.wavelet_constructor,
            diffusion_levels.squeeze()
        )

        s2 = torch.einsum(
            # j gets squared (each filter gets applied to every 
            # other in AT's approach)
            'jt,tncj->ncj', 
            self.wavelet_constructor,
            diffusion_levels_2
        )


        """
        0th-order scattering (s0)
        """
        s0 = x.detach().clone().unsqueeze(dim=2)
        print('s0.shape:', s0.shape)

        """
        1st-order scattering (s1)
        """
        # DJ: replaced AT's list append with list population, here and below
        avgs = [None] * self.n_diffusion_steps
        avgs[0] = s0
        for i in range(self.n_diffusion_steps - 1):
            # recursive diffusion starting at index 1
            avgs[i+1] = self.diffusion_layer1(avgs[i], edge_index)
        
        # Combine the diffusion levels into a single tensor. 
        # DJ: replaced torch.cat with extra dim -> stack, here and below
        diffusion_levels = torch.stack(avgs)
        print('diffusion_levels.shape:', diffusion_levels.shape)
        
        # Reshape the 3d tensor into a 2d tensor and multiply with the wavelet_constructor matrix
        # This simulates the below subtraction:
        # filter1 = avgs[1] - avgs[2]
        # filter2 = avgs[2] - avgs[4]
        # filter3 = avgs[4] - avgs[8]
        # filter4 = avgs[8] - avgs[16]
        diff_levels_view = diffusion_levels.view(self.n_diffusion_steps, -1)
        print('self.wavelet_constructor.shape:', self.wavelet_constructor.shape)
        print('diff_levels_view.shape:', diff_levels_view.shape) # [33, n, 29, 1]
        subtracted = torch.matmul(
            self.wavelet_constructor, 
            diff_levels_view
        )
        print('subtracted.shape 1:', subtracted.shape)
        # reshape into given input shape
        subtracted = subtracted.view(self.n_wavelets, x.shape[0], x.shape[1]) 
        print('subtracted.shape 2:', subtracted.shape)
        # transpose the dimensions to match previous
        s1 = torch.abs(
            torch.transpose(torch.transpose(subtracted, 0, 1), 1, 2)
        )
        print('s1.shape 1:', s1.shape)

        
        """
        2nd-order scattering (s2)
        """
        # perform a second wave of diffusing, on the recently diffused.
        avgs = [None] * self.n_diffusion_steps
        avgs[0] = s1
        for i in range(self.n_diffusion_steps - 1): # diffuse over diffusions
            avgs[i+1] = self.diffusion_layer2(avgs[i], edge_index)
        diffusion_levels_2 = torch.stack(avgs)
        print('diffusion_levels_2.shape:', diffusion_levels_2.shape)
        
        # having now generated the diffusion levels, we can combine them as before
        diff_levels2_view = diffusion_levels_2.view(self.n_diffusion_steps, -1)
        print('diff_levels2_view.shape:', diff_levels2_view.shape)
        subtracted2 = torch.matmul(
            self.wavelet_constructor, 
            diff_levels2_view
        )
        # reshape into given input shape
        """
        DJ: fixing AT's hardcoded view reshaping (for 4 filters)
        """
        sub2_new_size = torch.Size([self.n_wavelets] + list(s1.shape))
        subtracted2 = subtracted2.view(sub2_new_size)
        # subtracted2 = subtracted2.view(
        #     self.n_wavelets, 
        #     s1.shape[0],
        #     s1.shape[1], 
        #     s1.shape[2]
        # )  
        print('subtracted2.shape 1', subtracted2.shape)
        # subtracted2 = torch.transpose(subtracted2, 0, 1)
        subtracted2 = subtracted2.transpose(0, 1)
        print('subtracted2.shape 2', subtracted2.shape)
        subtracted2 = torch.abs( # this abs seems misplaced?
            subtracted2.reshape(-1, self.in_channels, self.n_wavelets)
        )
        print('subtracted2.shape 3', subtracted2.shape)
        # subtracted2 = torch.transpose(subtracted2, 1, 2)
        subtracted2 = subtracted2.transpose(1, 2)
        print('subtracted2.shape 4', subtracted2.shape)
        
        # s2_swapped_dim0_size = int(subtracted2.shape[0] / self.J)
        s2_swapped = subtracted2.reshape(
            (-1, self.n_wavelets ** 2, self.in_channels) # -1 -> s2_swapped_dim0_size
        )
        print('s2_swapped.shape', s2_swapped.shape)
        s2 = s2_swapped[:, self.feng_filters_2nd_order, :]
        print('s2.shape (feng-filtered)', s2.shape)

        """
        concatenate 0th-, 1st-, and (feng-filtered) 2nd-order scattering coeffs
        """
        x = torch.cat([s0, s1], dim=2)
        print('x.shape 1', x.shape)
        x = torch.transpose(x, 1, 2)
        print('x.shape 2', x.shape)
        x = torch.cat([x, s2], dim=1)
        print('x.shape 3', x.shape, '\n')

        if self.verbosity > 0:
            print('(after cat all scat. orders) x.shape:', x.shape)

        # channel pooling
        batch_index = data.batch if hasattr(data, 'batch') else None
        num_graphs = 1 if (batch_index is None) else data.num_graphs
        if self.channel_pool_key == 'moments':
            # x = scatter_moments(x, batch_index, self.n_moments) # AT method
            x = moments_channel_pool(x, batch_index, num_graphs) # DJ method
        else: # max, mean, etc. pooling
            x = channel_pool(
                self.channel_pool_key,
                x,
                num_graphs,
                batch_index
            )

        # flatten graphs' xs into 2 dim each (req'd by MLP head)
        # -> x shape = (n_graphs, n_moments * [final_]n_channels_3)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
            
        if self.verbosity > 0:
            print('(final scattering output) x.shape:', x.shape)
            # [batch_size, 11, in_channels]
        
        return x, self.wavelet_constructor
    '''

