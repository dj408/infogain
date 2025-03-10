"""
DD graph classification task 
experiment arguments.
"""
import sys
sys.path.insert(0, '../')
from args_template import ArgsTemplate
from dataclasses import dataclass, field
import os
import time
import datetime
from torch import float32, long
from typing import (
    Tuple,
    Any,
    Optional,
    Iterable,
    Callable
)


@dataclass
class Args(ArgsTemplate):

    # seeds
    POINT_SAMPLING_SEED: int = 156238
    FN_ON_MANIFOLD_SEED: int = 358446
    CV_SPLIT_SEED: int = 162769
    VALID_SET_SHUFFLE_SEED: int = 473895
    DATALOADERS_SEED: int = 240056
    TRAIN_VALID_TEST_SPLIT_SEED: int = 253279
    # TORCH_MANUAL_SEED: int = 635585 # cv.py.151
    TORCH_MANUAL_SEEDS: Tuple[int] = (
        665877,
        515435,
        453669,
        190358,
        908537,
        578348,
        599959,
        898580,
        232237,
        349476
    )
    
    # data creation params
    SAVE_SPECTRAL_OBJS: bool = True
    SAVE_SCAT_MOMENTS_OBJS: bool = False
    SAVE_FN_VALS: bool = True # needed for torch_geometric.data.Data objects
    SAVE_GRAPH: bool = True # needed for torch_geometric.data.Data objects
    
    # which axes should serve as function-on-manifold values?
    # 'all', or if == (2, ) -> fn on manifolds are 'z' coord. values
    COORD_FN_AXES: str | Tuple[int] = 'all' 
    STACK_CHANNEL_INPUTS: bool = False
    D_MANIFOLD: int = 12
    MANIFOLD_N_AXES: int = 35
    AMBIENT_DIM: int = 89
    N_OVERSAMPLES: Optional[int] = None
    N_MANIFOLDS: int = 1178 
    MANIFOLD_KEY: str = 'atom_type' # old: 'manifold'
    # N_PTS_ON_MANIFOLD: int = 400 # pts have diff. numbers of cell samples!
    GRAPH_TYPE: str = 'knn'
    K_OR_EPS: str | int = 'auto'
    
    # filters params
    NON_WAVELET_FILTER_TYPE: Optional[str] = 'p' # 'spectral', 'p'
    SPECTRAL_C: Optional[float] = 0.5
    WAVELET_TYPE: Optional[str] = 'p' # 'spectral', 'p'
    KAPPA: int = 20 # max num. eigenpairs calc'd
    MFCN_MAX_KAPPA: Optional[int] = 20 # max num. eigenpairs used in MFCN-spectral model: leave None to use all saved
    J: int = 4
    Q: int = 4 # which scat. moments to calc
    # Q_IDX = which scat. moments to use
        # (1, ) for q=2 moments only
        # (0, 1, 2, 3) for q=1...4
    Q_IDX: Tuple[int] = (0, 1, 2, 3)
    INCLUDE_LOWPASS_WAVELET: bool = True

    """
    Custom P wavelet scales
    """
    P_WAVELET_SCALES: str = 'custom' # 'dyadic'
    CUSTOM_P_AVG_SCALES_ONLY: bool = False
    N_NODE_FEATURES: int = 89 # 35 # 38 # 56
    CUSTOM_P_AUTOPROCESS_UNINFORM_CHAN: bool = True
    # CUSTOM_P_BATCH_SIZE: int = 512
    CUSTOM_P_AUTOPROCESS_UNINFORM_CHAN_STRATEGY: str = 'drop'
    CUSTOM_P_REWEIGHT_KLDS_FOR_CLASS_IMBALANCE: bool = True
    CUSTOM_P_CMLTV_KLD_QUANTILES: Iterable[float] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
    EXCLUDED_FEAT_IDX: Optional[list] = None
    P_WAVELET_SCALES_PRECALC: Optional[list] = None
    CUSTOM_P_ZERO_FLOOR: Optional[float] = 1e-2
    CUSTOM_P_SUBSAMPLE_PROP: Optional[int] = None # 0.5
    CUSTOM_P_SUBSAMPLE_SEED: int = 578439
    
    EXCLUDED_FEAT_IDX: Optional[list] = field(default_factory=lambda: \
        # typically dropped in InfoGain CV folds
        [21, 27, 34, 35, 41, 47, 51, 52, 56, 59, 60, 61, 63, 66, 68, 72, 76, 78, 86,
         22, 25, 26, 31, 39, 50, 62, 65, 74, 80, 82]
    )
    
        # 50% of data; J = 4 / T = 16 -> N_NODE_FEATURES = 35
        # [20, 21, 22, 23, 24, 25, 26, 27, 31, 32, 35, 36, 37, 38, 39, 40, 41, 42,
        # 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 59, 60, 61, 62, 63, 65,
        # 66, 67, 68, 69, 71, 72, 73, 74, 75, 78, 79, 80, 81, 82, 83, 84, 86, 87]
                                              
        # 50% of data; J = 5 / T = 32 -> N_NODE_FEATURES = 38
        # [20, 21, 22, 25, 26, 27, 30, 31, 32, 35, 36, 37, 41, 42, 43, 44, 45, 46,
        #  47, 48, 49, 50, 51, 52, 54, 55, 56, 59, 60, 61, 62, 63, 65, 66, 67, 68,
        #  69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 82, 83, 86, 87]
                                              
        # 100% of data J = 5 / T = 32 -> N_NODE_FEATURES = 56
        # [21, 25, 27, 31, 35, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55,
        #  59, 60, 61, 63, 65, 66, 67, 71, 72, 74, 78, 79, 83, 86, 87]
    # )
    
    # T = 16 (J = 4), 6 wavelets (incl. lowpass)
    # CUSTOM_P_CMLTV_KLD_QUANTILES: Iterable[float] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)  # (0.34, 0.66)
    # P_WAVELET_SCALES_PRECALC: Optional[list] # field(default_factory=lambda: \
        # median ts of 38 (T=32) or 35 (T=16) channels
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32] # T = 32
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16] # T = 16

        # all 35 channels; T = 16
        # [[ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  7, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  7, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  5,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  7, 16],
        # [ 0,  1,  2,  3,  3,  4,  5,  6,  7,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  7, 16],
        # [ 0,  1,  2,  3,  3,  4,  5,  5,  7,  8, 16],
        # [ 0,  1,  2,  3,  3,  4,  4,  5,  6,  8, 16]]
                                                     
        # all 38 channels; T = 32
        # [[ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7, 10, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7, 10, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 12, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7, 10, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7, 10, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  4,  5,  7,  8, 12, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  4,  5,  7,  9, 12, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 12, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  6,  7,  9, 11, 15, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  7,  8, 11, 15, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  7, 10, 13, 32],
        # [ 0,  1,  2,  3,  4,  5,  7,  9, 11, 14, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 14, 32],
        # [ 0,  1,  2,  3,  4,  5,  7,  8, 10, 14, 32],
        # [ 0,  1,  2,  3,  4,  5,  7,  8, 11, 14, 32],
        # [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 14, 32],
        # [ 0,  1,  2,  3,  4,  5,  7,  8, 10, 14, 32],
        # [ 0,  1,  2,  3,  4,  6,  7,  9, 11, 15, 32]]
    # )
    

    # T = 32, 11 wavelets (incl. lowpass)
    # CUSTOM_P_CMLTV_KLD_QUANTILES: Iterable[float] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875) 
    # P_WAVELET_SCALES_PRECALC: Optional[list] = field(default_factory=lambda: \
#        [[ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7, 10, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7, 10, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 12, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7, 10, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 14, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 14, 32],
#         [ 0,  1,  2,  3,  4,  4,  5,  7,  8, 12, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 13, 32],
#         [ 0,  1,  2,  3,  4,  4,  5,  7,  9, 12, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 12, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7, 10, 13, 32],
#         [ 0,  1,  2,  3,  4,  6,  7,  9, 11, 15, 32],
#         [ 0,  1,  2,  3,  4,  6,  7,  9, 12, 15, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 14, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  7,  8, 11, 15, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 14, 32],
#         [ 0,  1,  2,  3,  4,  5,  7,  8, 11, 14, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 12, 32],
#         [ 0,  1,  2,  3,  4,  4,  5,  6,  8, 11, 32],
#         [ 0,  1,  2,  3,  4,  5,  7,  8, 10, 14, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 14, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 12, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7, 10, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  7,  9, 11, 15, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  7,  9, 13, 32],
#         [ 0,  1,  2,  3,  4,  5,  6,  8, 10, 14, 32],
#         [ 0,  1,  2,  3,  4,  5,  7,  8, 11, 14, 32],
#         [ 0,  1,  2,  3,  4,  6,  7,  9, 11, 15, 32]]
#     )
        
    # data params
    # RAW_DATA_FILENAME: str = ''
    PYG_DATA_LIST_FILENAME: str = 'proteins_pyg_data_list.pkl'
    FEATURE_SCALING_TYPE: str = None # 'minmax'; None for no rescaling
    DROP_ZERO_VAR_FEATURES: bool = True
    LOG_TRANSFORM_TARGETS: bool = False
    DATALOADER_N_WORKERS: int = 0
    DATALOADER_DROP_LAST: bool = False

    # model params
    MODEL_NAME: str = 'model'
    GNN_TYPES: Tuple[str] = ('gcn', 'sage', 'gin', 'gat')
    TASK: str = 'binary_graph_classification'
    TARGET_NAME: str = 'y' # response
    # note `F.binary_cross_entropy_with_logits` needs targets as floats
    TARGET_TENSOR_DTYPE = float32 
    STRATIF_SAMPLING_KEYS: Tuple[str] = None
    MFCN_USE_SKIP_CONNECTIONS: bool = True
    MFCN_USE_INPUT_RECOMBINE_LAYER: bool = False
    MFCN_INPUT_RECOMBINE_LAYER_OUT_CHANNELS: int = 16
    MCN_WITHIN_FILTER_CHAN_OUT: Tuple[int] = (32, 16)
    MFCN_WITHIN_FILTER_CHAN_OUT: Tuple[int] = None # (8, 4)
    MFCN_CROSS_FILTER_COMBOS_OUT: Tuple[int] = (8, ) # (8, 4)
    # MFCN final channel pool options: 'max+mean', 'moments', 'max', 'mean'
    MFCN_FINAL_CHANNEL_POOLING: Optional[str] = 'moments'
    MFCN_FINAL_NODE_POOLING: Optional[str] = None # 'linear', 'max', 'mean'
    
    GNN_FINAL_CHANNEL_POOLING: Optional[str] = 'mean'
    GNN_FINAL_NODE_POOLING: Optional[str] = None
    NN_HIDDEN_DIMS: Tuple[int] = (128, 64, 32, 16) # (512, 64, 16, 4)
    OUTPUT_DIM: int = 1
    REWEIGHT_LOSS_FOR_IMBALANCED_CLASSES: bool = True
    USE_BATCH_NORMALIZATION: bool = True
    MLP_USE_DROPOUT: bool = False
    MLP_DROPOUT_P: float = 0.5 # probability of being 'zeroed'
    GNN_USE_DROPOUT: bool = False
    GNN_DROPOUT_P: float = 0.5

    # training params
    N_FOLDS: int = 10
    # if using test set in CV, make sure folds are the same size! 
    # e.g. 80/10/10 split for 10-fold CV
    USE_CV_TEST_SET: bool = True
    # TRAIN_PROP: float = 0.8
    # note: for ex, 0.2 (vs 0.21) may underpopulate the valid set
    # VALID_PROP: float = 0.1
    # change batch size depending on memory avail; use powers of 2
    BATCH_SIZES: Tuple[int] = (128, 1024, 1024) # (128, 64, 64)
    
    N_EPOCHS: int = 256
    # burn-in num of epochs prevents early stopping before it's reached
    BURNIN_N_EPOCHS: int = 100
    LEARN_RATE: float = 0.01
    SAVE_STATES: bool = False
    STOP_RULE: str = 'no_improvement'
    NO_VALID_LOSS_IMPROVE_PATIENCE: int = 50
    SAVE_FINAL_MODEL_STATE: bool = False
    TRAIN_HIST_PREFIX: Optional[str] = None
    
    ADAM_BETAS: Tuple[float] = (0.9, 0.999)
    ADAM_WEIGHT_DECAY: float = 1e-2 # default 1e-2 in AdamW
    LRELU_NSLOPE: float = 0.01
    # args for saving 'best' model during training, by a 
    # validation metric
    MAIN_METRIC: str = 'loss_valid'
    MAIN_METRIC_IS_BETTER: str = 'lower' # or: 'higher'
    # MAIN_METRIC_INIT_VAL: float = 1.0
    # prevents new best model saves until a relative improvement
    # in main metric is reached (not arbitrarily small)
    MAIN_METRIC_REL_IMPROV_THRESH: Optional[float] = None # 0.999

    # paths vars (full paths set in super().__post_init__)
    ROOT: str = None
    DATA_DIR: str = None
    MODEL_SAVE_DIR: str = None
    PRINT_DIR: str = None
    TRAIN_LOGS_SAVE_DIR: str = None
    DATA_SUBDIR: str = 'infogain'
    PYG_DATASET_COLLECTION: str = 'TUDataset'
    PYG_DATASET_KEY: str = 'DD'
    MODEL_SAVE_SUBDIR: str = 'DD'
    RESULTS_SAVE_SUBDIR: str = 'DD'


    def set_raw_data_filename(self):
        pass


