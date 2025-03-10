"""
Class to hold model training arguments,
of python's `dataclass` type.

Notes:
- these args can be overwritten by subclasses and
command line args in a script (if coded as such).
- there are `__post_init__` and other methods in
this class for dynamic setting of variables and
directories (at bottom).
- conforms to accelerate's `state_dict` and
`load_state_dict` protocols. Other args
file inherit this class (and its default
values).

`@dataclass` general reference:
https://peps.python.org/pep-0557/
"""

from dataclasses import dataclass
import os
import time
import datetime
from torch import float32
from typing import (
    Tuple,
    Any,
    Optional,
    Iterable,
    Callable
)


@dataclass
class ArgsTemplate:

    # timestamping variable, used in 'set_model_name_timestamp'
    ts = None
    
    # hardware: changed if nec. in __post_init__() below
    MACHINE: str = 'desktop'
    # note that MPS (Apple silicon GPU) and some tensor ops don't support float64
    FEAT_TENSOR_DTYPE = float32
    TARGET_TENSOR_DTYPE = float32
    FLOAT_TENSOR_DTYPE = float32
    VERBOSITY: int = 0

    # seeds
    POINT_SAMPLING_SEED: int = 956438
    ABC_SEED: int = 274851
    FN_ON_MANIFOLD_SEED: int = 758976
    DATALOADERS_SEED: int = 549256
    TRAIN_VALID_TEST_SPLIT_SEED: int = 245879
    
    # data creation params
    SAVE_MFCN_SPECTRAL_OBJS: bool = False
    SAVE_MFCN_P_OBJS: bool = False
    SAVE_SCAT_MOMENTS_OBJS: bool = False
    SAVE_FN_VALS: bool = True
    SAVE_GRAPH: bool = True # needed for torch_geometric.data.Data objects
    
    # which axes should serve as function-on-manifold values?
    # 'all', or if == (2, ) -> fn on manifolds are 'z' coord. values
    COORD_FN_AXES: str | Tuple[int] = 'all' 
    STACK_CHANNEL_INPUTS: bool = False
    D_MANIFOLD: int = 2
    MANIFOLD_N_AXES: int = 3
    AMBIENT_DIM: int = 3
    N_NODE_FEATURES: Optional[int] = None
    N_MANIFOLDS: int = 512
    N_PTS_ON_MANIFOLD: int = 128
    GRAPH_TYPE: str = 'knn'
    ETA_TYPE: str = 'indicator'
    K_OR_EPS: str | int = 'auto'
    USE_K_DROP: bool = False

    # geometric scattering params
    WAVELET_TYPE: str = 'P' # 'spectral'
    KAPPA: int = 64 # max num. eigenpairs calc'd
    J: int = 4
    Q: int = 4 # which scat. moments to calc?
    # Q_IDX = which scat. moments to use? 
        # (1, ) for q=2 moments only
        # (0, 1, 2, 3) for q=1...4
    Q_IDX = (0, 1, 2, 3)
    INCLUDE_LOWPASS_WAVELET: bool = False

    # training data params
    RAW_DATA_FILENAME: str = 'raw_data.pkl'
    DATASETS_DICT_FILENAME: str = 'datasets.pkl'
    FEATURE_SCALING_TYPE: str = None # 'minmax' # None for no rescaling
    DROP_ZERO_VAR_FEATURES: bool = True
    LOG_TRANSFORM_TARGETS: bool = True
    DATALOADER_N_WORKERS: int = 0
    DATALOADER_DROP_LAST: bool = False

    TRAIN_PROP: float = 0.8
    # note: for ex, 0.2 (vs 0.21) may underpopulate the valid set
    VALID_PROP: float = 0.1
    # change batch size depending on memory avail; use powers of 2
    BATCH_SIZES: Tuple[int] = (1024, 1024, 1024)
    # TRAIN_BATCH_SIZE: int = 256
    # VALID_BATCH_SIZE: int = 256
    # TEST_BATCH_SIZE: int = 256

    # model params
    MODEL_NAME: str = 'model'
    TASK: str = 'regression' # should have 'graph' or 'node' in the TASK str!
    TARGET_NAME: str = 'target'
    NN_HIDDEN_DIMS: Tuple[int] = (512, 128, 32, 8) # (1024, 256, 64, 16)
    # USE_DROPOUT: bool = True
    # DROPOUT_P: float = 0.5 # probability of being 'zeroed'

    # model training params
    SAVE_STATES: bool = False
    STOP_RULE: str = 'no_improvement'
    VALIDATE_EVERY_N_EPOCHS: int = 1
    NO_VALID_LOSS_IMPROVE_PATIENCE: int = 32
    SAVE_FINAL_MODEL: bool = True
    TRAIN_HIST_PREFIX: Optional[str] = None

    # P wavelets
    USE_PRECALC_P_SCALES: bool = False
    
    # burn-in num of epochs prevents early stopping before it's reached
    BURNIN_N_EPOCHS: int = 5
    N_EPOCHS: int = 5
    LEARN_RATE: float = 0.003
    ADAM_BETAS: Tuple[float] = (0.9, 0.999)
    ADAM_WEIGHT_DECAY: float = None
    LRELU_NSLOPE: float = 0.2
    # args for saving 'best' model during training, by a 
    # validation metric
    MAIN_METRIC: str = 'loss_valid'
    MAIN_METRIC_IS_BETTER: str = 'lower' # or: 'higher'
    # MAIN_METRIC_INIT_VAL: float = 1.0
    # prevents new best model saves until a relative improvement
    # in main metric is reached (not arbitrarily small)
    MAIN_METRIC_REL_IMPROV_THRESH: Optional[float] = None # 0.999

    # paths inits (actually set in __post_init__ below)
    ROOT: str = None
    DATA_DIR: str = None
    MODEL_SAVE_DIR: str = None
    PRINT_DIR: str = None
    TRAIN_LOGS_SAVE_DIR: str = None
    DATA_SUBDIR: str = 'ellipsoids'
    MODEL_SAVE_SUBDIR: str = 'ellipsoids'
    RESULTS_SAVE_SUBDIR: str = 'ellipsoids'


    def set_model_name_timestamp(
        self, 
        new_timestamp: bool = True
    ) -> None:
        """
        Generates and sets the MODEL_NAME_TIMESTAMP arg,
        used as the directory name where the model's 
        training objects are saved.
        
        Note: use GM time for MODEL_NAME_TIMESTAMPing: otherwise,
        running on machines in different time zones will cause 
        inconsistent times!
        """
        if new_timestamp or self.ts is None:
            ts = time.gmtime()
            self.ts = time.strftime('%Y-%m-%d-%H%M%S', ts)
        self.MODEL_NAME_TIMESTAMP = f"{self.MODEL_NAME}_{self.ts}"


    def set_model_save_dirs(
        self, 
        new_subdir: Optional[str] = None,
        train_hist_prefix: Optional[str] = None,
        make_dirs: bool = False
    ) -> None:
        """
        Sets directory args that depend on 
        MODEL_NAME_TIMESTAMP; creates leading
        directories if 'make_dirs' is True.
        """
        self.MODEL_SAVE_DIR = f"{self.ROOT}/models"
        if new_subdir is not None:
            self.MODEL_SAVE_SUBDIR += f"/{new_subdir}"
            
        if train_hist_prefix is not None:
            self.TRAIN_HIST_PREFIX = train_hist_prefix
        
        if self.MODEL_SAVE_SUBDIR is not None:
            self.MODEL_SAVE_DIR += f"/{self.MODEL_SAVE_SUBDIR}/{self.MODEL_NAME_TIMESTAMP}"
        else: 
            self.MODEL_SAVE_DIR += f"/{self.MODEL_NAME_TIMESTAMP}"
            
        # optional: create dirs if they don't exist
        if make_dirs:
            os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)

        # reset dir to save print output and train history files (w/ model)
        self.PRINT_DIR = f"{self.MODEL_SAVE_DIR}/out.txt" 
        if self.TRAIN_HIST_PREFIX is not None:
            self.TRAIN_LOGS_FILENAME = f"{self.MODEL_SAVE_DIR}/{self.TRAIN_HIST_PREFIX}_train_history.pkl"
        else:
            self.TRAIN_LOGS_FILENAME = f"{self.MODEL_SAVE_DIR}/train_history.pkl"

    
    def set_raw_data_filename(self) -> None:
        """
        Sets RAW_DATA_FILENAME, based on a bunch
        of other args, to help better identify
        what's in the dataset by filename.
        """
        self.RAW_DATA_FILENAME = f'{self.DATA_SUBDIR}_{self.N_MANIFOLDS}' \
            + f'_{self.N_PTS_ON_MANIFOLD}_{self.GRAPH_TYPE}-{self.K_OR_EPS}'


    def set_batch_sizes(
        self,
        train_size: int = 1024,
        valid_size: int = 1024,
        test_size: int = 1024
    ) -> None:
        """
        Sets 3-tuple of train/valid/test batch
        sizes in the BATCH_SIZES arg.
        """
        self.BATCH_SIZES = (train_size, valid_size, test_size)

        
    def __post_init__(self):
        """
        Set args that depend on other args, such as
        on the machine in use.
        """
        self.set_model_name_timestamp()
        self.set_raw_data_filename()
        
        if 'laptop' in self.MACHINE.lower():
            self.ON_CPU = True
            self.DEVICE = 'cpu'
            # self.ON_CPU = False
            # self.DEVICE = 'mps'
            self.ROOT = "SET/LAPTOP/ROOT/DIR"
        elif 'desktop' in self.MACHINE.lower():
            self.ON_CPU = False
            self.DEVICE = 'cuda'
            self.ROOT = "SET/CUDA/DESKTOP/ROOT/DIR"

        self.set_model_save_dirs()
        self.DATA_DIR = f"{self.ROOT}/data/{self.DATA_SUBDIR}"
        self.DATA_TRANSFORM_INFO_DIR = f"{self.DATA_DIR}"
        self.RESULTS_SAVE_DIR = f"{self.ROOT}/results"
        if self.RESULTS_SAVE_SUBDIR is not None:
            self.RESULTS_SAVE_DIR += f"/{self.RESULTS_SAVE_SUBDIR}"
        # self.VALID_PREDS_FILENAME = f"{self.MODEL_SAVE_DIR}/{self.MODEL_NAME_TIMESTAMP}_class1probs.dat"
        # self.VALID_SET_REF_FILENAME = f"{self.MODEL_SAVE_DIR}/{self.MODEL_NAME_TIMESTAMP}_validsetref.pkl"
        
    
    def state_dict(self) -> dict:
        """
        Returns args in this class as a dictionary. Required for
        this class to be saved by the 'accelerate' package along
        with a model.
        """
        # from dataclasses import asdict
        # return {k: v for k, v in asdict(self).items()}
        return self.__dict__


    def load_state_dict(self, state_dict: dict) -> None:
        """
        Loads class args from a dictionary. Required for
        this class to be loaded by the 'accelerate' package, along
        with a model.
        """
        for k, v in state_dict.items():
            setattr(self, k, v)

