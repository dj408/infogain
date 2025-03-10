"""
Utility classes and functions for 
pytorch neural networks.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset, 
    DataLoader
)
# from torcheval.metrics import (
#     MeanSquaredError,
#     R2Score
# )
from torchmetrics.regression import (
    MeanSquaredError,
    R2Score
)

from typing import (
    Optional,
    Tuple,
    List,
    Dict,
    Any
)

def minmax_scale_1d_tensor(
    v: torch.Tensor,
    min_v: Optional[torch.Tensor | float] = None, 
    max_v: Optional[torch.Tensor | float] = None,
    above_zero_floor: Optional[float] = None, 
    div_0_thresh: float = 1e-10,
    verbosity: int = 0
) -> Optional[torch.Tensor]:
    """
    Min-max scales a 1-d tensor onto the interval
    [0, 1] or [above_zero_floor, 1].
    
    Allows min and max values to be passed
    as optional arguments, e.g. in case they
    come from a sorted array (and are simply at
    indices 0 and -1).

    Args:
        v: 1-d tensor.
        min_v: optional minimum value in v, if already
            known upstream, to save computation.
        max_v: optional maximum value in v, if already
            known upstream, to save computation.
        above_zero_floor: optional value > 0,
            to map v onto interval [above_zero_floor, 1]
            instead of [0, 1], e.g. if log is going
            to be taken of output.
        div_0_thresh: if the range between max and min
            values is less than this positive threshold,
            we assume they are the same, and v cannot be 
            rescaled (because there's no variance in its
            elements).
        verbosity: integer value controlling the volume
            of print output as this function runs.
    Returns:
        1-d tensor of rescaled values from v; or None.
    """
    if v is None:
        return None
    else:
        try:
            if min_v is None:
                min_v = torch.min(v)
            if max_v is None:
                max_v = torch.max(v)
                
            if above_zero_floor is not None:
                if isinstance(above_zero_floor, str): 
                    # above_zero_floor is an error string: an algorithm used
                    # to compute a floor likely found an edge case,
                    # such as no variance in v
                    if above_zero_floor == 'error':
                        return None
                    else:
                        raise Exception(
                            f"Unrecognized error key for 'above_zero_floor'."
                        )
                elif above_zero_floor <= 0:
                    raise Exception(f"above_zero_floor = {above_zero_floor:.4f} <= 0")
                
                # adjust min_v such that the new floor will be above zero
                min_v -= above_zero_floor

            # calc the range
            vals_range = (max_v - min_v)

            # make sure the range is > 0
            if torch.abs(vals_range) < div_0_thresh:
                return None
            # if the range is > 0, min-max scale onto [above_zero_floor, 1] interval
            else:
                return (v - min_v) / vals_range
        except:
            print('Error in `nnu.minmax_scale_1d_tensor`.')
            print(f"\tv = {v}, min_v = {min_v}, max_v = {max_v}")
            return None


def norm_1d_tensor_to_prob_mass(
    v: torch.Tensor,
    min_v: Optional[torch.Tensor | float] = None, 
    max_v: Optional[torch.Tensor | float] = None,
    above_zero_floor: Optional[float] = None
) -> Optional[torch.Tensor]:
    r"""
    Uses 'minmax_scale_1d_tensor' to obtain v', then 
    divides v' by the sum of all elements (i.e. 
    $\ell^1$-normalization) to obtain a probability 
    vector (a vector on [0, 1] where the sum of all
    elements is 1.0).
    
    Args:
        v: 1-d tensor.
        min_v: optional minimum value in v, if already
            known upstream, to save computation.
        max_v: optional maximum value in v, if already
            known upstream, to save computation.
        above_zero_floor: optional value > 0,
            to map v onto interval [above_zero_floor, 1]
            instead of [0, 1], e.g. if log is going
            to be taken of output.
    Returns:
        1-d tensor of v transformed into a probability 
        vector; or None.
    """
    if v is None:
        return None
    else:
        try:
            # first map all values onto [0, 1]
            v = minmax_scale_1d_tensor(v, min_v, max_v, above_zero_floor)
            if v is not None:
                # lastly, rescale all values so their sum = 1
                sum_v = v.sum()
                return v / sum_v
            else:
                return None
        except:
            print('Error in `nnu.norm_1d_tensor_to_prob_mass`.')
            print(f"\tv = {v}")
            return None


def get_mid_btw_min_and_2nd_low_vector_vals(
    v: torch.Tensor,
    no_var_thresh: float = 1e-10,
) -> Optional[float] | str:
    r"""
    Computes the linear midpoint between the 
    minimum and second-lowest values in a vector
    (1-d tensor) v.

    Args:
        v: 1-d tensor.
        no_var_thresh: if the difference between
        the max and min values in v is less than 
        this positive threshold, then all values
        in v appear to be equal, and no 2nd lowest
        value can be found; None will be returned
        instead.
    Returns:
        The float value of $\frac{1}{2}(min(v) +
        \text{2nd-lowest}(v))$, None, or string
        holding error key.
    """
    if v is not None:
        i_2nd_low = None
        try:
            v = v.sort().values
            min_v = v[0]
            max_v = v[-1]
            # if the min and max values in v are the same, 
            # return None
            if (max_v - min_v) < no_var_thresh:
                return 'error'
            else:
                # there could be ties for lowest value: get
                # first index of second-lowest value in v
                i_2nd_low = torch.argwhere(v > min_v)[0][0]
                return (min_v + v[i_2nd_low]) / 2.
        except:
            print('Error in `nnu.get_mid_btw_min_and_2nd_low_vector_vals`.')
            print(f"\tv = {v}; i_2nd_low = {i_2nd_low}")
            return None
    else:
        return None


def get_inv_class_wts(
    train_labels: List[Any]
) -> List[float]:
    """
    Calculates inverse class weights from a set of training
    labels. Can be used with torch.nn.CrossEntropyLoss(weight=.),
    for example, to help with training a class-imbalanced
    dataset.

    Args:
        train_labels: a list of labels (ints, floats, strings,
            etc.) for a training set.
    Returns:
        List of float class weights for 'balanced' 
        re-weighting use in, e.g., torch.nn.CrossEntropyLoss.
    """
    from collections import Counter
    cts = Counter(sorted(train_labels))
    inv_wts = [
        1 - (ct / len(train_labels)) \
        for (label, ct) in cts.items()
    ]
    return inv_wts
    

def log_parameter_grads_weights(
    args,
    model: torch.nn.Module, 
    grad_track_param_names: Tuple[str],
    epoch_i: int, 
    batch_i: int,
    save_grads: bool = True,
    save_weights: bool = True,
    verbosity: int = 0
) -> None:
    """
    Appends the flattened gradient values for 
    parameters of interest in a model to rows
    of corresponding CSVs (saved in the model save 
    directory), with the first 2 columns saving
    the epoch and batch numbers.

    Args:
        args: an ArgsTemplate class or subclass
            instance.
        model: a torch.nn.Module.
        epoch_i: index of the epoch for which
            gradients and weights are being logged.
        batch_i: index of the batch for which
            gradients and weights are being logged.
        save_grads: boolean whether to log gradients.
        save_weights: boolean whether to log weights.
        verbosity: integer value controlling the
            volume of console print output as this
            function runs.
    Returns:
        None; saves csv file(s).
    """
    path = args.MODEL_SAVE_DIR
    for name, param in model.named_parameters():
        for tracked_param_name in grad_track_param_names:
            if tracked_param_name in name:
                if save_grads:
                    # note pytorch appends '.1' etc. to repeated parameter names
                    filename = f'grads_{name}'.replace('.', '_') + '.csv'
                    if param.grad is not None:
                        grads = torch.flatten(param.grad).tolist()
                        out = [epoch_i, batch_i] + grads
                        with open(f'{path}/{filename}', 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(out)
                    else:
                        if verbosity > 0:
                            print(
                                f'Warning: {name} grad was None;',
                                f'is_leaf = {param.is_leaf}'
                            )
                if save_weights:
                    filename = f'weights_{name}'.replace('.', '_') + '.csv'
                    weights = torch.flatten(param).tolist()
                    out = [epoch_i, batch_i] + weights
                    with open(f'{path}/{filename}', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(out)


def get_trained_model_preds(
    trained_model: torch.nn.Module,
    dataloaders_dict: Dict[str, dict],
    set: str = 'test',
    return_on_cpu: bool = True,
    verbosity: int = 0
) -> torch.Tensor:
    """
    Runs trained_model.forward on batches of 
    a set (train/valid/test), and collects
    model predictions on the set into one tensor,
    optionally moved to cpu.

    Args:
        trained_model: a torch.nn.Module with
            trained weights.
        dataloaders_dict: a dictionary holding
            torch.utils.data.DataLoader objects
            which themselves contain model inputs
            in dictionaries, keyed by set ('train', 
            'valid', 'test').
        set: the string key for the set to get
            model predictions on.
        return_on_cpu: boolean whether to return
            model predictions on the CPU.
        verbosity: integer value controlling the
            volume of console print output as this
            function runs.
    Returns:
        A tensor of model predictions for the 
        specified set.
    """
    test_preds = []
    device = next(trained_model.parameters()).device
    trained_model.eval()
    with torch.no_grad():
        for input_dict in dataloaders_dict[set]:
            # print('input_dict[\'x\'].shape', input_dict['x'].shape)
            # for i, param in enumerate(trained_model.parameters()):
            #     print(f'layer {i + 1} param shape: {param.data.shape}')
            # print(input_dict['x'], '\n')
            # batch_x = input_dict['x'].to(device)
            # print(f'batch_x.shape: {batch_x.shape}')
            trained_model_output_dict = trained_model(input_dict)
            batch_preds = trained_model_output_dict['preds']
            test_preds.extend(batch_preds)
    test_preds_tensor = torch.stack(test_preds).squeeze()
    if return_on_cpu:
        test_preds_tensor = test_preds_tensor.cpu()

    # check
    if verbosity > 0:
        print('test_preds.shape', test_preds.shape)
        print(test_preds, '\n')
    return test_preds_tensor


def get_target_tensors(
    trained_model: torch.nn.Module,
    datasets_dict: Dict[str, dict],
    sets: Tuple[str] = ('test', ),
    target_name: str = 'y'
) -> Tuple[torch.Tensor]:
    """
    Extracts targets from datasets_dict for
    train/valid/test sets, and if more
    than one set was requested, collects into
    a tuple of 1d/vector tensors.

    Args:
        trained_model: a torch.nn.Module with
            trained weights.
        datasets_dict: a dictionary of 
            torch.utils.data.Dataset (or subclass)
            objects, which themselves contain model inputs
            in dictionaries, keyed by set ('train', 'valid',
            'test')
        sets: tuple of string keys for which to
            obtain targets collected in tensors.
        target_name: string key value for the target,
            as keyed in the dictionaries in datasets_dict.
    Returns:
        Tuple of tensors containing set targets.
    """
    target_tensors = [
        torch.stack([
            dict['target'][target_name] \
            for dict in datasets_dict[set]
        ]) for set in sets
    ]
    if len(sets) > 1:
        target_tensors = tuple(target_tensors)
    return target_tensors


def regressor_preds_plots(
    trained_model: torch.nn.Module,
    datasets_dict: Dict[str, dict],
    dataloaders_dict: Dict[str, DataLoader],
    target_name: str = 'y',
    # device: str = 'mps:0',
    train_targets_bins: Optional[int] = None,
    test_preds_bins: Optional[int] = None,
    fig_size: Tuple[float, float] = (6., 4.)
) -> None:
    """
    Prints useful analytic plots for the predictions
    of a regressor model versus a mean model.

    Args:
        trained_model: a torch.nn.Module with
            trained weights.
        datasets_dict: a dictionary of 
            torch.utils.data.Dataset (or subclass)
            objects, which themselves contain model inputs
            in dictionaries, keyed by set ('train', 'valid',
            'test')
        dataloaders_dict: a dictionary holding
            torch.utils.data.DataLoader objects
            which themselves contain model inputs
            in dictionaries, keyed by set ('train', 
            'valid', 'test').
        target_name: string key value for the target,
            as keyed in the dictionaries in datasets_dict.
        train_targets_bins: optional int arg to pass 
            to 'bins' arg in plt.hist, when making a
            histogram of train set targets.
        test_preds_bins: optional int arg to pass 
            to 'bins' arg in plt.hist, when making a
            histogram of test set predictions.
        fig_size: 2-tuple of floats to pass to plt
            to set output figure size.
    Returns:
        None; prints plots instead (e.g. in a Jupyter 
        notebook).
    """
    # extract train and test targets from datasets_dict
    # train_targets = torch.stack([
    #     dict['target'][target_name] \
    #     for dict in datasets_dict['train']
    # ])
    
    # test_targets = torch.stack([
    #     dict['target'][target_name] \
    #     for dict in datasets_dict['test']
    # ])

    train_targets, test_targets = get_target_tensors(
        trained_model,
        datasets_dict,
        ('train', 'test'),
        target_name
    )
    # print('test_targets.shape', test_targets.shape)
    # print(test_targets, '\n')

    # get trained model's test set predictions
    test_preds = get_trained_model_preds(
        trained_model,
        dataloaders_dict
    )

    # compute mean model's regression metrics
    train_mean = torch.mean(train_targets)
    mean_model_preds = train_mean.repeat(test_targets.shape[0])
    # print('mean_model_preds.shape', mean_model_preds.shape)
    # print(mean_model_preds, '\n')
    mse = MeanSquaredError()
    R2 = R2Score()
    mse.update(mean_model_preds, test_targets)
    R2.update(mean_model_preds, test_targets)
    mean_model_mse = mse.compute()
    mean_model_R2 = R2.compute()
    mse.reset()
    R2.reset()
    
    # trained model's test set regression metrics
    MSE_test = MeanSquaredError()
    MSE_test.update(test_preds, test_targets)
    trained_model_mse = MSE_test.compute()
    MSE_test.reset()
    
    R2_test = R2Score()
    R2_test.update(test_preds, test_targets)
    trained_model_R2 = R2_test.compute()
    R2_test.reset()

    # set plot size
    plt.rcParams["figure.figsize"] = fig_size
    
    # histogram of test set targets
    plt.hist(test_targets, bins=train_targets_bins)
    plt.ylabel('count')
    plt.xlabel(f'{target_name}')
    plt.title(f'Histogram of \'{target_name}\' test set target values')
    test_xlim = plt.gca().get_xlim()
    plt.show()

    # histogram of preds
    plt.title(
        f'distribution of model predictions'
        f' for \'{target_name}\'\n'
        f'MSE = {trained_model_mse:.4f} '
        f'(mean model: {mean_model_mse:.4f})'
    )
    plt.hist(test_preds, bins=test_preds_bins)
    # set preds xlim to that of test targets
    plt.xlim(test_xlim)
    plt.xlabel('model prediction')
    plt.ylabel('count')
    plt.show()
    
    # scatterplot of preds vs. targets
    plt.title(
        f'model predictions vs. test-set targets'
        f' for \'{target_name}\'\n'
        rf'$R^2$ = {trained_model_R2:.4f} '
        f'(mean model: {mean_model_R2:.4f})'
    )
    plt.scatter(
        test_targets, 
        test_preds, 
        color='C0',
        alpha=0.5,
        zorder=np.inf
    )

    # simple y = x line (since preds = targets in perfect model)
    plt.plot(
        np.unique(test_targets), 
        np.poly1d(np.polyfit(
            test_targets, test_preds, 1)
        )(np.unique(test_targets)),
        c='C0'
    )
    # plt.axis('equal')
    plt.axvline(train_mean, linestyle='--', c='gray', zorder=0)
    plt.axhline(train_mean, linestyle='--', c='gray', zorder=0)
    lims = [
        np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # min of both axes
        np.max([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # max of both axes
    ]
    plt.text(
        lims[0],
        train_mean + 0.1, 
        f'train set mean = {train_mean:.2f}', 
        rotation=0.,
        c='gray'
    )
    # now plot both limits against eachother
    plt.plot(lims, lims, c='gray', zorder=0)
    plt.xlabel('test set target')
    plt.ylabel('model prediction')
    plt.ylim(plt.gca().get_xlim())
    plt.show()



def build_ffnn(
    input_dim: int, 
    output_dim: int, 
    hidden_dims_list: List[int], # e.g. [1024, 256, 64, 16], 
    bias_in_hidden_layers: bool,
    nonlin_fn: torch.nn.Module, # a torch.nn activation fn
    nonlin_fn_kwargs: Dict[str, Any]
) -> Tuple[nn.ModuleList, nn.Module, nn.Module]:
    """
    Builds a simple feed-forward, fully-connected neural 
    network (aka multilayer perceptron, or MLP) programatically.
    
    Note: returns pieces that model's 'forward()' must iterate
    through, e.g.:
    def forward(self, x):
        for i in range(len(self.lin_fns)):
            x = self.lin_fns[i](x)
            x = self.nonlin_fns[i](x)
            if self.use_dropout:
                x = nn.Dropout(self.dropout_p)
        x = self.lin_out(x)

    Args:
        input_dim: int value of network's input dimension.
        output_dim: int value of network's final output
            dimension.
        hidden_dims_list: list of int values of dimension
            for each hidden linear layer.
        bias_in_hidden_layers: bool whether to include a
            bias term in the hidden layers.
        nonlin_fn: the torch.nn activation function to 
            apply to each linear layer.
        nonlin_fn_kwargs: any kwargs to pass to nonlin_fn.
    Returns:
        3-tuple of the linear layers (as nn.ModuleList), 
        the activation function (as nn.Module), and final 
        the linear layer (as a single nn.Module, with no 
        activation function).
    """
    lin_fns = [None] * len(hidden_dims_list)
    next_input_dim = input_dim
    for i, hidden_dim in enumerate(hidden_dims_list):
        lin_fns[i] = nn.Linear(
            next_input_dim, 
            hidden_dim, 
            bias=bias_in_hidden_layers
        )
        next_input_dim = hidden_dim
    lin_fns = nn.ModuleList(lin_fns)
    nonlin_fns = nonlin_fn(**nonlin_fn_kwargs)
    final_lin = nn.Linear(
        next_input_dim, 
        output_dim, 
        bias=True
    )
    return (lin_fns, nonlin_fns, final_lin)


def get_model_settings_str(args) -> str:
    """
    Gathers key settings from an args file
    into lines to print.
    """
    out = f'{args.MODEL_NAME_TIMESTAMP}'
    out += f'\n\nParameters:\n'
    out += f'MODEL_NAME = {args.MODEL_NAME}\n'
    out += f'TASK = {args.TASK}\n'
    out += f'TARGET_NAME = {args.TARGET_NAME}\n'
    out += f'N_EPOCHS = {args.N_EPOCHS}\n'
    out += f'BURNIN_N_EPOCHS = {args.BURNIN_N_EPOCHS}\n'
    out += f'LEARN_RATE = {args.LEARN_RATE}\n'
    out += f'STOP_RULE = {args.STOP_RULE}\n'
    out += f'PATIENCE = {args.NO_VALID_LOSS_IMPROVE_PATIENCE}\n'

    out += f'\nMLP head parameters (if applicable):\n'
    out += f'NN_HIDDEN_DIMS = {args.NN_HIDDEN_DIMS}\n'
    out += f'USE_BATCH_NORMALIZATION = {args.USE_BATCH_NORMALIZATION}\n'
    out += f'MLP_USE_DROPOUT = {args.MLP_USE_DROPOUT} (p = {args.MLP_DROPOUT_P})\n'
    out += f'BATCH_SIZES (train/valid/test) = {args.BATCH_SIZES}\n'

    if 'mfcn' in args.MODEL_NAME.lower():
        out += f'\nMFCN-wavelet parameters:\n'
        out += f'WAVELET_TYPE = {args.WAVELET_TYPE}\n'
        out += f'INCLUDE_LOWPASS_WAVELET = {args.INCLUDE_LOWPASS_WAVELET}\n'
        if args.WAVELET_TYPE.lower() == 'p':
            out += f'P_WAVELET_SCALES = {args.P_WAVELET_SCALES}\n'
            if 'custom' in args.P_WAVELET_SCALES.lower():
                out += (f"CUSTOM_P_CMLTV_KLD_QUANTILES = " \
                    + f"{args.CUSTOM_P_CMLTV_KLD_QUANTILES}\n")
        elif 'spect' in args.WAVELET_TYPE.lower():
            out += f'SPECTRAL_C = {args.SPECTRAL_C}\n'
            out += f'MFCN_MAX_KAPPA = {args.MFCN_MAX_KAPPA}\n'
        out += f'MFCN_WITHIN_FILTER_CHAN_OUT = {args.MFCN_WITHIN_FILTER_CHAN_OUT}\n'
        out += f'MFCN_CROSS_FILTER_COMBOS_OUT = {args.MFCN_CROSS_FILTER_COMBOS_OUT}\n'
        out += f'MFCN_FINAL_CHANNEL_POOLING = {args.MFCN_FINAL_CHANNEL_POOLING}\n'
        out += f'MFCN_FINAL_NODE_POOLING = {args.MFCN_FINAL_NODE_POOLING}\n'
    
    elif 'mcn' in args.MODEL_NAME.lower():
        out += f'\nMFCN-lowpass parameters:\n'
        out += f'NON_WAVELET_FILTER_TYPE = {args.NON_WAVELET_FILTER_TYPE}\n'
        if 'spect' in args.NON_WAVELET_FILTER_TYPE.lower():
            out += f'SPECTRAL_C = {args.SPECTRAL_C}\n'
            out += f'MFCN_MAX_KAPPA = {args.MFCN_MAX_KAPPA}\n'

    return out



class EpochCounter:
    """
    Class for counting epochs and best metrics achieved at which
    epoch, with state_dict implementations, for use in saving and 
    loading model states for continuing training, etc. Useful with 
    the 'accelerate' library, which requires state_dict methods
    for saving/reloading object states.

    Note that validation loss is tracked by default, but an 
    additional metric of interest can also be tracked, when
    its string key is passed to 'metric_name' in __init__.
    """
    def __init__(
        self,
        n: int = 0,
        metric_name: str = 'loss_valid'
    ):
        """
        Args:
            n: int starting epoch.
            metric_name: string key value of the primary 
                metric of interest, for tracking by epoch.
        """
        self.n = n
        self.best = {
            metric_name: {
                'epoch': 0,
                'score': 0.0
            },
            # track valid loss separately, even if 
            # it's the metric of interest
            '_valid_loss': {
                'epoch': 0,
                'score': float('inf')
            }
        }

    def __iadd__(self, m: int):
        self.n = self.n + m
        return self

    def state_dict(self) -> dict:
        return self.__dict__

    def load_state_dict(self, state_dict: dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def set_best(
        self, 
        metric: str, 
        epoch: int, 
        score: float
    ) -> None:
        if metric in self.best.keys():
            self.best[metric]['epoch'] = epoch
            self.best[metric]['score'] = score
        else:
            self.best[metric] = {
                'epoch': epoch,
                'score': score
            }

    def __str__(self) -> str:
        return str(self.n)


class Class1PredsCounter:
    """
    For binary classification tasks, this
    object keeps track of class 1 logit prediction
    counts, in order to print the proportion
    of class 1 predictions within each epoch 
    and each train and valid phase).
    """
    def __init__(self):
        self.reset()
    
    def update(self, output_dict, phase):
        preds = output_dict['preds'].squeeze()
        # print("preds.shape:", preds.shape)
        # note: a (logit > 0) = (p > 0.5) = class 1 pred
        self.ctr[phase]['class1'] += torch.sum(preds > 0.)
        self.ctr[phase]['total'] += preds.shape[0]

    def print_preds_counts(self):
        print(f'class 1 predictions:')
        for phase in ('train', 'valid'):
            preds_d = self.ctr[phase]
            class1_preds_ct, all_preds_ct = preds_d['class1'], preds_d['total']
            perc = 100 * (class1_preds_ct / all_preds_ct)
            print(f'\t{phase}: {class1_preds_ct} / {all_preds_ct} ({perc:.1f}%)')

    def reset(self):
        self.ctr = {
            'train': {'class1': 0, 'total': 0},
            'valid': {'class1': 0, 'total': 0}
        }

