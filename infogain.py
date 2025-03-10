"""
Functions for 'InfoGain', an unsupervised method
for selecting diffusion (lazy random walk) based
wavelet filter scales.

Author: Dave Johnson [davejohnson408@u.boisestate.edu]

TODO [ideas]
    [ ] parallelize by channel where graphs are sliced out
    [ ] ensure no two t cutoffs are the same index? right now it
        can happen, which effectively drops a wavelet: (P^3 - P^3)x = 0,
        but this means features extracted by wavelets reflect the same
        infogain quantile, where some channels have more than that quantile's
        worth of infogain (the t diffusion steps aren't fine enough)
    [ ] outlier control in KLD loss calcs?
    [ ] some form of regression target 'imbalance' correction?
"""
import sys
sys.path.insert(0, '../')
import data_utilities as du
import wavelets as w
import nn_utilities as nnu
import pyg_utilities as pygu
from utilities import generate_random_integers

from numpy import (
    nanmax,
    linspace,
    log2,
    log,
)
from numpy.random import RandomState
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Dataset
)
# from torch import linalg as LA
from torch_geometric.utils import to_torch_coo_tensor
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
# from itertools import accumulate
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


def calc_custom_P_wavelet_scales(
    pyg_train_set: Dict | Data,
    task: str,
    device: str,
    n_streams: int = 1,
    T: int = 32,
    cmltv_kld_quantiles: Iterable[float] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875),
    include_lowpass: bool = True,
    start_from_t2: bool = True,
    fixed_above_zero_floor: Optional[float] = None,
    failure_value: int | float = -1,
    reweight_klds: bool = True,
    data_subsample_prop: Optional[int] = None,
    data_subsample_random_state: Optional[RandomState] = None,
    auto_process_uninformative_channels: bool = True,
    uninformative_channel_strategy: str = 'drop',
    savepath_kld_by_channel_plot: Optional[str] = None,
    plot_dyadic_kld_curve: bool = False,
    kld_by_channel_plot_name: str = "custom_P_wav_scales_plot",
    verbosity: int = 0
) -> Tuple[torch.Tensor]:
    r"""
    Calculates the scales for 'custom' wavelets (non-dyadic,
    with unique scales for each channel of signal on graphs); the
    output 'channels_t_is' (calculated once over all training data) 
    can then be used in 'get_Batch_P_Wjxs'.

    Give a diffusion operator $P$ and powers $P^t$ for $t \in 0...T$,
    and a graph channel signal $x$, we take normalized $P^T x$ as our 
    reference distribution, and calculate relative entropy (KL divergence/
    information gain) of each $P^t x$ versus this reference distribution.
    We then select P-wavelet scales based on $t$ cutoffs uniquely for
    each channel, based on which powers of $t$ cross the cumulative
    KL divergence thresholds passed in the 'cmltv_kld_quantiles' argument. 
    (Here, we know that relative entropies relative to $P^T x$ decrease
    with increasing powers of $t$, so each channel has a slowing cumulative
    sum; if 'start_from_t2' is true, we also automatically keep t <= 2 as
    scale cutoffs, as the greatest values of relative entropy are expected
    in these lowest powers, and corresponding wavelets should be kept by 
    default).

    Thus, instead of dyadic-scale wavelets (where $W_j x = P^{2^{j-1}} 
    - P^{2^j}) x$), we obtain, for example, wavelets unique to each channel,
    such as (P^3 - P^5)x in one, but (P^4 - P^7)x in another, with both
    capturing the same volume change in relative entropy against their 
    channel's steady state diffusion (P^T x) at the wavelet index.

    Notes on KL divergence / relative entropy: 
        - If any entry in a channel is <=0, a NaN will be created
        by log, and that NaN is then part of a sum -> sum = NaN
        - Thus we normalize each P^t x into a probability vector
        first (i.e. with range 0-1 and sum = 1), and this prevents
        zeros, since KLD can't handle them.
        - We also prevent skewing relative KLDs by replacing zeros 
        with too tiny of a value by replacing zeros with the value
        halfway between the (pre-normalized) minimum channel value
        and second-lowest channel value
    
    Args:
        pyg_train_set: either dictionary with 'train' keying a 
            pytorch geometric DataLoader object 
            containing the test set graphs in batches 
            (multiple graphs); or single-graph pytorch
            geometric Data object with a 'train' mask.
        task: string description of the modeling task, e.g.,
            'binary_classification'.
        device: string key for the torch device, e.g. 'cuda'.
        n_streams: number of CUDA streams to use in processing
            (if using CUDA).
        T: max power of $P$, for $P^t$ where $t \in 1...T$.
        cmltv_kld_quantiles: iterable of cumulative KLD 
            quantiles/percentile cutoffs, which powers of P
            must reach to be a wavelet scale boundary $P^t$.
        include_lowpass: boolean, whether to include the 
            lowpass wavelet $P^T$.
        start_from_t2: boolean whether to keep filters with 
            $P^1$ and $P^2$, and choose subsequent scales
            (and ignore their contribution to cumulative KLD;
            calc from $P^3...P^T$ instead). This is useful since
            these lowest powers of $t$ generally cover the largest
            steps in KLD, and perhaps should be included scale
            steps in all channels by default.
        fixed_above_zero_floor: optional fixed float value to
            replace zeros in original features with. If None,
            the linear midpoint between 0 and the next lowest value
            is used. Note zeros must be replaced since KLD uses
            logarithms.
        failure_value: value to return in final indices tensor
            in place of NaN, etc., in case computation fails (e.g.
            in the case of features/channels without sufficient
            information change over diffusion steps.
        reweight_klds: boolean whether to re-weight each graph's
            contribution to a channels' total (sum) KLD loss, e.g. 
            to rebalance KLD for unbalanced target classes.
        data_subsample_prop: if not None, a random sample of size
            int(data_subsample_prop * num_graphs) will be taken of 
            the graphs (in a multi-graph training dataset) and 
            used to calculate the wavelet scales, instead of the 
            full train set.
        data_subsample_random_state: optional np.RandomState generating
            data_subsample_n when fitting infogain on a subset of the 
            training data.
        auto_process_uninformative_channels: whether to automatically
            process uninformative channels by the strategy in 
            'uninformative_channel_strategy'.
        uninformative_channel_strategy: 'drop' (to remove channels) 
            or 'average' to replace channels with the median scales from
            informative channels.
        savepath_kld_by_channel_plot: optional save path for a plot
            of cumulative KLDs by channel. Set to None to skip creation
            of the plot.
        kld_by_channel_plot_name: filename ('.png' added automatically)
            for the optional KLD by channel plot.
        plot_dyadic_kld_curve: whether to plot a line on the 'KLD by channel
            plot' showing the dyadic scale KLD curve. [CURRENTLY BROKEN.]
        verbosity: integer controlling volume of print output as
            the function runs.
    Returns:
        2-tuple of torch tensors: (1) optional tensor of the indices of 
        uninformative channels found (if autoprocessing them here); and
        (2) tensor containing indices of wavelet scale $t$s (which also 
        happen to be their values in $P^t x, t \in 0...T$) for each channel 
        in the graph dataset; shape (n_channels, n_ts).
    """        
    # loop through batches in train set again, collecting KL divergence stats
    # for the entire train set
    klds_by_x_t_chan = []
    targets_by_xi = []
    Ptx_i_start = 2 if start_from_t2 else 0
    
    if isinstance(pyg_train_set, dict):
        pyg_train_set = pyg_train_set['train']
        multiple_graphs = True
    elif isinstance(pyg_train_set, DataLoader):
        multiple_graphs = True
    elif isinstance(pyg_train_set, Data):
        # make iterable of 1 batch of 1 graph
        pyg_train_set = (pyg_train_set, )
        multiple_graphs = False
    
    
    for batch_i, batch in enumerate(pyg_train_set):
        batch = batch.to(device)
        x = batch.x #.to_dense()
        if x.ndim == 1:
            x = x.unsqueeze(dim=-1)
        print(f"\tbatch {batch_i + 1} (shape={x.shape})")
        n_channels = x.shape[1]
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight \
            if hasattr(batch, 'edge_weight') \
            else None
        
        if multiple_graphs:
            num_graphs = batch.num_graphs
            # use 'extend' since we are populating by batch
            targets_by_xi.extend(batch.y)
            # batch_index = batch.batch
        else: # 1 graph in single Data object
            # note that in a node-level graph task,
            # we don't mask any node signals for train vs. valid
            # set until loss calculation / evaluation time
            num_graphs = 1
            # use 'extend' since we are populating by batch
            targets_by_xi.extend(batch.y[batch.train_mask])

        if (num_graphs > 1) and (data_subsample_prop is not None):
            graphs_loop_idx = generate_random_integers(
                n=int(data_subsample_prop * num_graphs),
                max_val=num_graphs,
                random_state=data_subsample_random_state
            )
        else:
            graphs_loop_idx = range(num_graphs)
            
        # get P_sparse
        P_sparse = pygu.get_Batch_P_sparse(
            edge_index=edge_index, 
            edge_weight=edge_weight,
            n_nodes=x.shape[0],
            device=device
        )
        
        # calc P^t x for t \in 1...T
        # make each Ptx in list dense here so tensor slicing below works
        Ptx = x.detach().clone()
        Ptxs = [x.to_dense()] # densify AFTER copying sparse x for recursive mm
        for j in range(1, T + 1):
            Ptx = torch.sparse.mm(P_sparse, Ptx)
            Ptxs.append(Ptx.to_dense())

        # 'load balancing' across streams
        if n_streams > 1:
            streams = [torch.cuda.Stream() for _ in range(n_streams)]
            stream_indices = [i % n_streams for i in range(len(graphs_loop_idx))]
            stream_lookup = {g_i: stream_indices[i] for i, g_i in enumerate(graphs_loop_idx)}

        # loop through all graphs in batch
        # to calc KL divergence of x and each P^t x versus baseline P^T x 
        # (lowpass/smoothest), uniquely for each x_i (graph) and channel
        for g_i in graphs_loop_idx:
                      
            if multiple_graphs:
                # subset out ith graph
                x_i_mask = (batch.batch == g_i)
                # grab Ptx_i for each t power, for graph_i's xs
                Ptx_is = [ptx[x_i_mask] for ptx in Ptxs] # each Ptx_i has shape (n_i, c)
            else: # one graph
                Ptx_is = Ptxs

            if verbosity > 0:
                print('len(Ptx_is):', len(Ptx_is))
                print('Ptx_is[0].shape:', Ptx_is[0].shape)
                print('Ptx_is[0].device:', Ptx_is[0].device)

            # parallelize graphs across cuda streams
            if ('cuda' in device) and (n_streams > 1):
                # channel kld results container
                streams_channel_klds = []
                s = stream_lookup[g_i]
                with torch.cuda.stream(streams[s]):
                    channel_klds = [None] * n_channels
                    for c in range(n_channels):
                        _, chan_klds = process_channel(
                            c, 
                            Ptx_is,
                            Ptx_i_start,
                            fixed_above_zero_floor,
                        )
                        channel_klds[c] = chan_klds
                        
                    # after processing KLDs for all ts and all channels for one graph, 
                    # stack into tensor of shape (num_t_powers, n_channels) and add
                    # to list collecting streams' single-graph results
                    channel_klds = torch.stack(channel_klds, dim=-1)
                    streams_channel_klds.append(channel_klds)
    
                # after syncing streams...
                torch.cuda.synchronize()
                
                # ...append 'kld_all_ts_by_chan' for each graph in each batch to 'klds_by_x_t_chan',
                # a growing list of eventual length n_graphs, where each list element is a tensor 
                # of shape (num_t_powers, n_channels)
                for channel_klds in streams_channel_klds:
                    klds_by_x_t_chan.append(channel_klds)
            
            else: # cpu or 1 cuda stream
                channel_klds = [None] * n_channels
                for c in range(n_channels):
                    _, chan_klds = process_channel(
                        c, 
                        Ptx_is,
                        Ptx_i_start,
                        fixed_above_zero_floor,
                    )
                    channel_klds[c] = chan_klds
                    
                channel_klds = torch.stack(channel_klds, dim=-1) 
                klds_by_x_t_chan.append(channel_klds)
        
    # after all graphs in all batches:
    # stack KLD values for all graphs x t powers x channels into a tensor
    # print('klds_by_x_t_chan length:', len(klds_by_x_t_chan))
    klds_by_x_t_chan = torch.stack(klds_by_x_t_chan) # shape (n_graphs, num_t_powers, n_channels)
    if verbosity > 0:
        print('klds_by_x_t_chan.shape:', klds_by_x_t_chan.shape)
        print('klds_by_x_t_chan:\n', klds_by_x_t_chan)
    targets_by_xi = torch.stack(targets_by_xi) # shape (n_graphs, )

    '''
    # I THINK THIS IS NO LONGER NECESSARY
    # replace any remaining NaNs (likely from 0s in logs in KLD) with (max KLD value w/in same channel)
    kld_chan_maxs = nanmax(
        nanmax(klds_by_x_t_chan.numpy(), axis=1),
        axis=0
    )
    # kld_chan_maxs.shape # (n_channel, )
    for x_klds in klds_by_x_t_chan:
        for t_klds in x_klds:
            nan_mask = torch.isnan(t_klds)
            t_klds[nan_mask] = torch.tensor(kld_chan_maxs[nan_mask])
    '''


    # quantify relative KLD over all graphs, by t and channel
    '''
    TODO
    - if taking sum over all xs, beware of outliers contributing
        a huge amount of KLD: only consider 10th-90th percentiles?
    - define regression target imbalance and reweight for that?
    '''
    # re-weight klds for target class balance
    if reweight_klds:
        # 0s get weight 1, 1s get relative weight 'pos_class_wt'
        if 'bin' in task.lower() and 'class' in task.lower():
            ct_1s = targets_by_xi.sum()
            pos_class_wt = (targets_by_xi.shape[0] - ct_1s) / ct_1s
            kld_weights = torch.ones(len(targets_by_xi))
            kld_weights[targets_by_xi == 1] = pos_class_wt
        else:
            # raise NotImplementedError()
            warnings.warn(f"Reweighting KLDs not implemented for task='{task}'.")
            kld_weights = None
    
        # reweight KLDs
        if kld_weights is not None:
            klds_by_x_t_chan = torch.einsum(
                'bTc,b->bTc',
                klds_by_x_t_chan,
                kld_weights
            )
        
    # sum (reweighted) KLD across graphs, by t and channel
    klds_by_t_chan = torch.sum(klds_by_x_t_chan, dim=0) # shape (T, n_channels)
    
    # get cumulative sums as t increases
    klds_cum_by_t_chan = torch.cumsum(klds_by_t_chan, dim=0) # shape (T, n_channels)
    
    # minmax scale channel cumulative KLDs, for cross-channel comparison (i.e.
    # so all channels' cumulative KLDs range from 0 to 1)
    for c in range(n_channels):
        # klds_cum_by_t_chan[:, c] = (klds_cum_by_t_chan[:, c] - min_kld) / (max_kld - min_kld)
        chan_rescaled_cum_kld = nnu.minmax_scale_1d_tensor(
            v=klds_cum_by_t_chan[:, c], 
            min_v=klds_cum_by_t_chan[:, c][0], # min cmltv kld is at start
            max_v=klds_cum_by_t_chan[:, c][-1] # max cmltv kld is at end
        )   
        # if min-max scaling doesn't work (likely no variance in channel), None is returned
        # for the 'chan_rescaled_cum_kld' vector -> insert tensor of -1s instead into
        # 'klds_cum_by_t_chan', this will lead to 'failure_value' in 'channels_t_is' below
        if chan_rescaled_cum_kld is None:
            klds_cum_by_t_chan[:, c] = -torch.ones(klds_cum_by_t_chan.shape[0])
        else:
            klds_cum_by_t_chan[:, c] = chan_rescaled_cum_kld

    # optional check: plot 'klds_cum_by_t_chan'
    if savepath_kld_by_channel_plot is not None:

        # plot curves for each channel
        for c in range(n_channels):
            chan_vals = klds_cum_by_t_chan[:, c].numpy()
            if chan_vals[0] > -1: # don't plot uninformative channels
                plt.plot(range(Ptx_i_start, T), chan_vals)

        # optional: plot dyadic curve
        if plot_dyadic_kld_curve and ((T == 16) or (T == 32)):
            pass
            # def log_func(x, a, b):
            #     return 
            # xx = linspace(2, T, 100)
            # if T == 16:
            #     yy = 0.5 * log(xx) - 0.3333
            # elif T == 32:
            #     yy = 0.3607 * log(xx) - 0.25
            # plt.plot(xx, yy, color='black', linestyle='--', linewidth=2)
            

        # plot attributes
        plt.title(
            f"Normalized cumulative (all-node) sums of KL divergences of $P^t x$"
            f"\nfor $t \in ({Ptx_i_start},\ldots,(T-1))$ from $P^T x$, by channel"
        )
        plt.xlabel('$t$ in $P^t x$')
        # assuming T is a power of 2, make x ticks powers of 2
        plt.xticks([0] + [2 ** p for p in range(0, int(log2(T)) + 1)])
        # plt.xticks(range(0, T + 1, 4))
        plt.yticks(linspace(0, 1, 11))
        plt.ylabel('cmltv KLD from $P^T x$')
        plt.grid()
        # plt.show()
        os.makedirs(savepath_kld_by_channel_plot, exist_ok=True) 
        plt.savefig(f"{savepath_kld_by_channel_plot}/{kld_by_channel_plot_name}.png")
        plt.clf()
        
    # for each channel, find (indexes of) t-integer scale cutoffs, 
    # following the quantiles of cmltv KLD in 'cmltv_kld_quantiles'
    # print('klds_cum_by_t_chan', klds_cum_by_t_chan)
    # replace NaNs with 0.
    # klds_cum_by_t_chan = torch.nan_to_num(klds_cum_by_t_chan)
    channels_t_is = torch.stack([
        torch.stack([
            # adjust t indexes returned for the 'Ptx_i_start' index
            # also make sure an index is found; else return failure value
            y[0].squeeze() + Ptx_i_start \
            if ((y := torch.argwhere(klds_cum_by_t_chan[:, c] >= q)).numel() > 0) \
            else torch.tensor(failure_value) \
            for q in cmltv_kld_quantiles
        ]) \
        for c in range(n_channels)
    ]) # shape (n_channels, n_quantiles)
    
    # calc wavelet filters, by (P^t - P^u)x = (P^t)x - (P^u)x
    # uniquely for each channel, following 'channels_t_is'
    # (all P^ts and P^us calc'd above for all xs and channels:
    # just need to subtract using specific ts and us for each channel)
    if start_from_t2:
       channels_t_is = torch.concatenate((
            torch.stack((
                torch.zeros(n_channels), 
                torch.ones(n_channels), 
                torch.ones(n_channels) * 2
            ), dim=-1),
            channels_t_is,
            (torch.ones(n_channels) * T).unsqueeze(dim=1)
        ), dim=-1).to(torch.long)

    if verbosity > 0:
        print('channels_t_is.shape:', channels_t_is.shape) # shape (n_channels, n_quantiles)
        print('channels_t_is\n', channels_t_is)

    # find rows where channels_t_is contains failure_value
    mask = (channels_t_is == failure_value)        
    failure_row_indices = torch.unique(torch.argwhere(mask)[:, 0])

    # if failures are found without autoprocess strategy, raise warning
    if (not auto_process_uninformative_channels) \
    and (len(failure_row_indices) > 0):
        fail_idx_l = [i.item() for i in failure_row_indices]
        warnings.warn(
            f"The channels/features at indexes {fail_idx_l}"
            f" failed to generate valid diffusion scales. Consider"
            f" dropping these features?"
        )

    # check if all channels' scales (rows) are the same
    # if so, return 1-d tensor (more efficient wavelet filtrations
    # can be done during training)
    if (channels_t_is[0] == channels_t_is).all():
        warnings.warn(
            f"All channels' custom scales were found to be equal:"
            f" returning 1-d tensor of scales instead."
        )
        
        channels_t_is = channels_t_is[0]

    #
    if auto_process_uninformative_channels:
        uninform_chan_is, channels_t_is = process_uninformative_channels(
            channels_t_is=channels_t_is,
            strategy=uninformative_channel_strategy,
            T=T
        )
        return uninform_chan_is, channels_t_is
    else:
        return None, channels_t_is


def process_uninformative_channels(
    channels_t_is: torch.Tensor,
    strategy: str = "drop",
    T: int = 32
) -> Tuple[torch.Tensor]:
    """
    If uninformative channels are found, rows in
    'channels_t_is' will have -1 values. This method
    processes 'channels_t_is' according to the
    desired strategy. 
    
    Args:
        channels_t_is: tensor of integer indices
            for wavelet scale boundaries. Shape:
            (n_channels, n_ts).
        strategy: how the uninformative channels are 
            to be handled: 'drop' (remove them), 
            'average' (replace with the median scales
            from informative channels), or 'dyadic'
            (replace with zero-padded dyadic scales).
        T: max diffusion step (e.g. 16 or 32).
    Returns:
        2-tuple of tensors: (1) indices of uninformative
        channels found; (2) wavelet scale t cutoffs by
        channel with uninformative channels removed.
    """
    # find where the tensor equals -1
    mask = (channels_t_is == -1)
    # use argwhere to get row indices where any column contains -1
    uninform_channels_is = torch.unique(torch.argwhere(mask)[:, 0])
    # print('len(uninform_channels_is):', len(uninform_channels_is))
    # print('uninform_channels_is:\n', uninform_channels_is)

    if strategy == 'drop':
        incl_mask = torch.ones(len(channels_t_is), dtype=torch.bool)
        incl_mask[uninform_channels_is] = False
        channels_t_is = channels_t_is[incl_mask]
        # print('channels_t_is_proc.shape:', channels_t_is_proc.shape)
        print(
            f"\tDropped {len(uninform_channels_is)} uninformative channels:"
            f" {uninform_channels_is}"
        )
    elif strategy == 'average' or strategy == 'avg':
        median_scales = get_avg_P_wavelet_scales(channels_t_is)
        channels_t_is[uninform_channels_is] = median_scales
    elif strategy == 'dyadic':
        T_to_J_lookup = {8: 3, 16: 4, 32: 5, 64: 6} # ugly patch
        n_ts = channels_t_is.shape[1]
        dyadic_scales = torch.cat((
            torch.tensor([0]),
            2 ** torch.arange(T_to_J_lookup[T] + 1)
        ))
        if n_ts > len(dyadic_scales):
            # left-pad with 0s if there are more ts than in a true 
            # dyadic scale
            pad_zeros = torch.zeros(n_ts - len(dyadic_scales))
            dyadic_scales = torch.cat((pad_zeros, dyadic_scales))
            channels_t_is[uninform_channels_is] = dyadic_scales
        
    return uninform_channels_is, channels_t_is
    


def get_avg_P_wavelet_scales(
    channels_t_is: torch.Tensor,
    average_method: str = 'median'
) -> torch.Tensor:
    r"""
    Averages each custom P-wavelet scale indices across
    channels. Useful for MFCN networks with more than one
    filter cycle; can use the average custom scales found
    here for all new (recombined feature) channels, instead
    of recomputing custom scales in each second or further
    cycle of each training epoch (as the model learns new
    channel-filter combinations).

    Args:
        channels_t_is: Torch tensor containing indices of 
            $t$s (which also happen to be their values in
            $P^t x, t \in 0...T$) for each channel in the
            graph dataset; shape (n_channels, n_ts).
        average_method: string key for the averaging method 
            used, e.g. 'median' (which uses the integer floor
            in the case of #.5s).
    Returns:
        Tensor of channel's averaged $t$ indices; shape
        (n_ts, ).
    """
    if average_method == 'median':
        return torch.median(channels_t_is, dim=0).values.to(torch.long)
    else:
        raise NotImplementedError(
            f"Averaging method '{average_method}' not"
            f" implemented."
        )



def process_channel(
    c: int, # channel index
    Ptx_is: torch.Tensor, 
    Ptx_i_start: int, 
    above_zero_floor: Optional[float] = None, 
) -> Tuple[int, torch.Tensor]:
    """
    Processes channels (optionally in parallel) within 
    `calc_custom_P_wavelet_scales`.

    Args:
        c: index of the channel to process.
        Ptx_is: tensor of diffused signals (P operations
            done on a channel of the node signals of
            the ith graph).
        Ptx_i_start: index (power of t) at which
            to start including KL divergences.
        above_zero_floor:
    Returns:
        2-tuple of c (the index channel) and a tensor
        of the KL divergence values for the channel.
    """
    # calc KLD(P^t x, P^T x) for t \in 1...(T-1), by channel
    # (T-1) -> excludes KLD(PTx_i, PTx_i)
    PTx_i = Ptx_is[-1]
    PTx_ic = PTx_i[:, c]
    kld_all_ts_one_chan = [None] * (len(Ptx_is) - Ptx_i_start - 1)
    T_above_zero_floor = above_zero_floor \
        if above_zero_floor is not None \
        else nnu.get_mid_btw_min_and_2nd_low_vector_vals(PTx_ic)
    if (T_above_zero_floor is not None) \
    and (not isinstance(T_above_zero_floor, str)) \
    and (T_above_zero_floor <= 0):
        T_above_zero_floor = -T_above_zero_floor

    # for each P^t x_i within the same channel, the reference 'stationary'
    # (i.e. max diffusion step) distribution P^T x_i is the same 
    # -> calc it once per channel and normalize
    # print('here 1')
    PTx_ic = nnu.norm_1d_tensor_to_prob_mass(
        PTx_ic, 
        above_zero_floor=T_above_zero_floor
    )
    # print('here 2')use
    

    # calc all lesser diffusion steps within channel c and normalize
    for t, Ptx_i in enumerate(Ptx_is[Ptx_i_start:-1]):
        Ptx_ic = Ptx_i[:, c]
        ts_above_zero_floor = above_zero_floor \
            if above_zero_floor is not None \
            else nnu.get_mid_btw_min_and_2nd_low_vector_vals(Ptx_ic)
        if (ts_above_zero_floor is not None) \
        and (not isinstance(ts_above_zero_floor, str)) \
        and (ts_above_zero_floor <= 0):
            ts_above_zero_floor = -ts_above_zero_floor
            
        Ptx_ic = nnu.norm_1d_tensor_to_prob_mass(
            Ptx_ic, 
            above_zero_floor=ts_above_zero_floor
        )
        # in edge cases / bad data, Ptx_ic may be an empty tensor
        # this propagates through above nnu methods as None
        kld = (Ptx_ic * (Ptx_ic / PTx_ic).log()).sum() \
            if (Ptx_ic is not None) else 0.
        kld_all_ts_one_chan[t] = kld
        
    return c, torch.tensor(kld_all_ts_one_chan)

