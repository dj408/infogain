"""
Utility classes and functions.
"""
import os
import time
import numpy as np
from numpy.random import RandomState
from typing import (
    Tuple,
    List,
    Any,
    Optional
)

def generate_random_integers(
    n: int, 
    min_val: int = 0, 
    max_val: int = 100, 
    replace: bool = False,
    random_state: Optional[RandomState] = None, 
    seed: Optional[int] = None
) -> List[int]:
    """
    Generates a list of random integers.

    Args:
        n: number of integers to generate.
        min_val: minimum integer value.
        max_val: maximum integer value.
        replace: whether to sample with
            replacement.
        random_state: numpy RandomState
            instance. If None, a new one
            will be instantiated.
        seed: seed for random_state.
    Returns:
        List of random integers of length n.
    """
    if n > (max_val - min_val + 1):
        raise ValueError(
            f"Cannot sample {n} unique integers from " 
            f"the range {min_val} to {max_val}."
        )

    # create new RandomState if needed (i.e. if not passed
    # a RandomState instance, or passed a new seed)
    if (random_state is None) or (seed is not None):
        random_state = RandomState(seed)
    sample = random_state.choice(
        np.arange(min_val, max_val + 1), 
        size=n, 
        replace=replace
    ).tolist()
    
    return sample
    

def flatten_list(
    list_of_lists: List[List[Any]],
    method: str = 'comprehension' # concatenation
) -> List[Any]:
    """
    Flattens a list of lists via list
    comprehension or concatenation.

    Args:
        list_of_lists: a list of lists.
    Returns:
        A flattened list.
    """
    if method == 'comprehension':
        flat_list = [x for xs in list_of_lists for x in xs]
    elif method == 'concatenation':
        flat_list = []
        for l in list_of_lists:
            flat_list += l
    return flat_list


def clean_column_name(col_name: str) -> str:
    """
    Cleans a string to make it more column name
    friendly.
    """
    col_name = col_name.lower()
    col_name = col_name.replace(" ", "_")
    col_name = col_name.replace(",", "")
    col_name = col_name.replace("-", "_")
    col_name = col_name.replace(".", "_")
    col_name = col_name.replace("(", "")
    col_name = col_name.replace(")", "")
    col_name = col_name.replace("/", "_")
    col_name = col_name.replace(":", "_")
    col_name = col_name.replace("%", "_percent")
    return col_name



def central_moving_average(
    arr: np.ndarray, 
    n_trail_lead: int,
    boundary_strategy: str = 'look_inward',
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes a same-size central moving average
    for a given vector, by shrinking the averaging
    window where there are not enough trailing/leading
    entries (i.e. at the array ends). Any given entry
    is replaced by the mean of the most possible adjacent
    entries, up to 'n_trail_lead' on both sides. Thus,
    the tails are noisier / less smoothed, but as the same
    size, the smoothed array can be easily plotted against 
    the original.

    Args:
        arr: vector of values.
        n_trail_lead: number of trailing and leading
            adjacent entries to average. Example: 
            n_trail_lead=50 means replacing a value
            with the mean of 101 values, where possible.
        boundary_strategy: parameter to deal with the
            boundary issue for central averaging: that vector
            elements near the head and tail cannot average the
            same leading or trailing number of entries. One of:
            (1) 'look_inward', to preserve n_trail_lead
            averaging in the direction possible and maximize in the 
            other direction; 
            (2) 'shrink_n', to always average the same
            number of entries to the left and right of entries
            near the boundaries, by shrinking 'n_trail_lead';
            (3) 'wrap', to include elements from the opposite
            end of the array in order to satisfy 'n_trail_lead'
            on 'both sides' of each entry (this is applicable
            to periodic functions on an interval satisfied by
            'arr').
        weights: optional array of weights for weighted-average 
            smoothing. If not None, must be of length 
            (2 * n_trail_lead + 1).
    Returns:
        Vector array of centrally-averaged values.
    """
    # check that the array is long enough for the given
    # central averaging period
    if 2 * n_trail_lead >= len(arr):
        print(
            f'Array of length {len(arr)} is too short for a '
            f'trailing/leading period of {n_trail_lead}! '
            f'Exiting.'
        )
        return None

    # weighted averaging is only implemented for 'wrap'
    if (weights is not None):
        if (boundary_strategy != 'wrap'): 
            print(
                'Weighted central averaging not implemented for '
                f'{boundary_strategy}! Exiting.'
            )
            return None
        # weights array must be appropriate length
        if weights.shape[0] != (2 * n_trail_lead + 1):
            print(
                'Weights array must be of length (2 * n_trail_lead + 1).'
                'Exiting'
            )
            return None

    # define inner fn to take symmetric moving central averages
    def ctr_avg(
        a: np.ndarray, 
        m: int, 
        w: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        For a vector array, computes symmetric moving
        central averages.
        
        Args:
            a: numpy array (vector) of values.
            m: number of trailing/leading values
                to include in average.
            w: (optional) array of weights for
                a weighted average.
        Returns:
            Vector array of moving central averages.
        """
        # print(f'len(a) ={len(a)}')
        out = [
            np.average(
                a=a[(i - m):(i + m + 1)],
                weights=w
            ) \
            for i in np.arange(m, len(a) - m)
        ]
        return out

    # init empty array of same size as input
    avg_arr = [None] * len(arr)
    
    # calc averages with enough trail and lead to take full average
    avg_arr[n_trail_lead:-n_trail_lead] = ctr_avg(
        arr, 
        n_trail_lead,
        weights
    )

    # entries without enough trail or lead (near array boundaries):
    if boundary_strategy == 'look_inward':
        for j in range(0, n_trail_lead):
            inner_stop = j + n_trail_lead + 1

            # left side
            left = arr[:inner_stop]
            avg_arr[j] = np.mean(left)

            # right side
            right = arr[-inner_stop:]
            avg_arr[-(j + 1)] = np.mean(right)
            
    elif boundary_strategy == 'shrink_n':
        # first and last entries are same as 'arr'
        avg_arr[0], avg_arr[-1] = arr[0], arr[-1]
    
        # entries from 1:n_trail_lead and -(n_trail_lead + 1):-1
        # use next largest possible averaging windows
        for j in range(1, n_trail_lead):
            # inner_stop = {5, 7, 9, ...} for n_trail_lead = {2, 3, 4, ...}
            # note 1 is added to stop index in ctr_avg()
            inner_stop = 2 * j + 1 

            # left side
            avg_arr[j] = ctr_avg(arr[:inner_stop], j)[0]

            # right side
            avg_arr[-(j + 1)] = ctr_avg(arr[-inner_stop:], j)[0]

    elif boundary_strategy == 'wrap':
        for j in range(0, n_trail_lead):
            inner_stop = j + n_trail_lead + 1
            wrap_stop = n_trail_lead - j

            # left side: note wrapped chunk from right
            # side concats in front, so central entry
            # stays in center of new array
            left = np.concatenate(
                (arr[-wrap_stop:], arr[:inner_stop])
            )
            # print(arr[-wrap_stop:], arr[:inner_stop])
            avg_arr[j] = np.average(left, weights=weights)

            # right side
            right = np.concatenate(
                (arr[-inner_stop:], arr[:wrap_stop])
            )
            # print(arr[-inner_stop:], arr[:wrap_stop])
            avg_arr[-(j + 1)] = np.average(right, weights=weights)
    
    return avg_arr
    

def trailing_average(
    arr: np.ndarray, 
    n: int = 3,
    ramp_up: bool = True
) -> np.ndarray:
    """
    Calculates the n-moving average array
    for a given array. 

    Args:
        arr: vector of values.
        n: period of moving average.
        ramp_up: if true, preserves original
            length of array by taking shorter
            1-, 2-, ..., (n-1)-moving averages 
            prior to the nth entry.
    Returns:
        Array of moving average values, of length
        len(arr) if ramp_up, or len(array) - (n - 1).

    Adapted from:
    https://stackoverflow.com/a/14314054
    """
    ret = np.cumsum(arr, dtype=float)
    if ramp_up:
        ramp_mvg_avg = np.concatenate((
            np.array([ret[0]]),
            ret[1:(n - 1)] / (np.arange(1, (n - 1)) + 1)
        ))
        # print('ramp_mvg_avg:', ramp_mvg_avg)
    ret[n:] = ret[n:] - ret[:-n]
    mvg_avg = ret[(n - 1):] / n
    if ramp_up:
        mvg_avg = np.concatenate((
            ramp_mvg_avg, mvg_avg
        ))
    return mvg_avg
    

def get_time_min_sec(
    t_1: float, 
    t_0: float = None
) -> Tuple[float, float]:
    """
    Calculates minutes and seconds 
    elapsed between 2 timepoints, or
    for one length of time.

    Args:
        t_1: 'end' timepoint or full
            time.
        t_0: 'start' timepoint. Leave
            'None' to calculate for a
            t_1 length of time.
    Returns:
        2-tuple of floats: min and sec
        elapsed.
    """
    if t_0 is None:
        t = t_1
    else:
        t = t_1 - t_0
    t_min, t_sec = t // 60, t % 60
    return t_min, t_sec


def pickle_obj(
    path: str, 
    obj: Any, 
    overwrite: bool = False
) -> None:
    """
    Robust pickling function.

    Args:
        path: full filepath to which to save
            pickle.
        obj: object to be pickled.
    Returns:
        None (pickles object to file).
    """
    import pickle
    
    if (path is not None) and (path != ""):
        try:
            file_exists = os.path.isfile(path)
            if (not overwrite) and (file_exists):
            # if file already exists, and we don't want to overwrite it,
            # append, e.g., '_1' to filename before .filetype
                while file_exists:
                    dir = os.path.dirname(os.path.realpath(path))
                    filename = os.path.basename(path)
                    filename_pref_suf = filename.split('.')
                    filename_suffix = filename_pref_suf[-1]
                    filename_pref = filename_pref_suf[0]
                    filename_pref_parts = filename_pref.split('_')
                    filename_pref_no_last_thing = '_'.join(filename_pref_parts[:-1])
                    last_filename_thing = filename_pref_parts[-1]
                    
                    if last_filename_thing.isdigit():
                        new_version = str(int(last_filename_thing) + 1)
                        new_filename = filename_pref_no_last_thing + f'_{new_version}'
                    else:
                        new_filename = filename_pref + '_1'
                        
                    new_filename += f'.{filename_suffix}'
                    # print(new_filename)
                    path = os.path.join(dir, new_filename)
                    file_exists = os.path.isfile(path)
                
            with open(path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        except pickle.UnpicklingError as e:
            # normal, somewhat expected
            pass
        except (AttributeError,  EOFError, ImportError, IndexError) as e:
            # secondary errors
            print(e)
        except FileNotFoundError as e:
            print(e)
            print(f"FileNotFoundError: attempted path: {path}")
            print("File not saved!")
            return
        except Exception as e:
            # everything else, possibly fatal
            print(e)
            print(f"Attempted path: {path}")
            print("File not saved!")
            return
    else:
        print("No save path given; file not saved!")


def get_newest_dir(dir: str) -> str:
    """
    Returns folder path with latest 'time
    modified' attribute.

    Args:
        dir: path to parent directory.
    Returns:
        Path to last-modified subdirectory.
    """
    import glob
    return max(
        glob.glob(os.path.join(dir, '*/')), 
        key=os.path.getmtime
    )


# def number_filename_if_exists(
#     dir: str, 
#     filename_stem: str,
#     filename_suffix: str
# ) -> str:
#     """
#     Appends, e.g., '_1' (then '_2' if it already ends in
#     '_1') to filename before .filetype. Useful to avoid
#     overwriting an existing file, BUT THE FIRST FILE WRITTEN
#     SHOULDN'T END IN SOMETHING LIKE '_456', OR THE SECOND
#     WILL BE '_457' NOT '_456_1'!

#     Args:
#         dir: 
#         filename_stem:

#     Returns:
#         Revised filename.
#     """
#     matches_found = 0
#     versions = []
#     if '.' in filename_suffix:
#         filename_suffix = filename_suffix.replace(".", "")
    
#     for f in os.listdir(dir):
#         if filename_stem in f:
#             matches_found += 1
#             name_underscore_parts = filename_stem.split('_')
#             name_last_char = name_underscore_parts[-1]
#             if name_last_char.isdigit():
#                 versions.append(name_last_char)
#     if matches_found == 0:
#         new_name = filename_stem + '.' + filename_suffix
#     elif matches_found == 1:
#         # print(f'only matching 1 file found')
#         new_name = filename_stem + '_1.' + filename_suffix
#     elif matches_found > 1:
#         max_version = max(versions)
#         # print(f'current max_version: {max_version}')
#         new_num_str = str(int(max_version) + 1)
#         new_name = filename_stem + '_' + new_num_str + '.' + filename_suffix
#     return new_name
    
