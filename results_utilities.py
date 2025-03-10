"""
Useful functions for calculating/summarizing/
tabulating experimental results.
"""
from args_template import ArgsTemplate
import utilities as u
import os
import warnings
import pickle
import numpy as np
import pandas as pd
from typing import (
    Tuple,
    List,
    Dict,
    Optional,
    Callable,
    Iterable,
    Any
)
PUB_MODELNAMES_DICT: Dict[str, str] = {
    'mfcn_spectral': 'MFCN-wavelet-spectral-1.0',
    'mfcn_spectral-0.5': 'MFCN-wavelet-spectral-0.5',
    'mfcn_spectral-1.0': 'MFCN-wavelet-spectral-1.0',
    'mfcn_p': 'MFCN-wavelet-approx',
    'mfcn_p-infogain': 'MFCN-wavelet-approx-Infogain',
    'mfcn_p-dyadic': 'MFCN-wavelet-approx-dyadic',
    'mcn_p': 'MFCN-low-pass-approx',
    'mcn_spectral': 'MFCN-low-pass-spectral-1.0', 
    'mcn_spectral-1.0': 'MFCN-low-pass-spectral-1.0', 
    'mcn_spectral-0.1': 'MFCN-low-pass-spectral-0.1', 
    'mcn_spectral-0.5': 'MFCN-low-pass-spectral-0.5', 
    'sage': 'GraphSAGE',
    'gat': 'GAT',
    'gcn': 'GCN',
    'gin': 'GIN',
    'legs': 'LEGS'
}
METRIC_MULTIPLIERS_DICT: Dict[str, float] = {
    ('acc', 'mean'): 100.,
    ('acc', 'std'): 100.,
    ('bal_acc', 'mean'): 100.,
    ('bal_acc', 'std'): 100.,
}
METRIC_BEST_BOLD_FNS_DICT: Dict[str, Callable] = {
    'acc': np.argmax,
    'bal_acc': np.argmax,
    'f1': np.argmax,
    'f1_neg': np.argmax,
    'specificity': np.argmax,
    'auroc': np.argmax,
    'R2': np.argmax,
    'mse': np.argmin
}


def generate_cv_results_df(
    task: str,
    metrics_records: List[Dict[str, Any]],
    timing_dict: Optional[Dict[str, Any]],
    validate_every: int = 1, 
    model_suffix: Optional[str] = None,
    filter_tuple: Optional[Tuple[str, Any]] = None,
    decimal_round: Optional[int] = None,
    regress_metrics: List[str] = ['R2', 'mse'],
    bin_classif_metrics: List[str] = ['acc', 'bal_acc', 'f1', 'f1_neg', 'specificity'],
    multiclass_classif_metrics: List[str] = ['acc']
) -> pd.DataFrame:
    """
    Combines 'metrics_records' and 'timing_dict'
    (created and pickled by 'gnns_cv.py' and 
    'mfcn_cv.py' scripts) into a summary pandas
    dataframe table.

    Args:
        task: string key for model task, e.g.
            'regression' or 'binary classification'.
        metrics_records: records object containing
            metrics by CV fold.
        timing_dict: dictionary containing seconds
            elapsed per epoch (fold-agnostic). If None,
            timing stats are excluded from the returned
            results dataframe. Note that `cv.run_cv' filters
            this dict down to only include times up through
            the epoch deemed the final/best/trained model
            (e.g., excludes 'patience' epochs where validation
            loss was not improving).
        validate_every: if validation did not occur every
            epoch, adjust epoch counts by this multiple. Note
            that overall times will be approximate when this
            is the case (epoch seconds elapsed are only recorded
            with validation metrics, as it stands).
        model_suffix_dict: optional string suffix value, 
            to append to the model name in the 'model' 
            column of the returned results dataframe.
        filter_tuple: optional tuple of column name and 
            filtering value, to subset metrics_records 
            with.
        decimal_round: final rounding of floats in
            the output dataframe.
        regress_metrics: list of string keys for
            regression metrics to include in the results
            dataframe.
        bin_classif_metrics: list of string keys for
            binary classification metrics to include in the 
            results dataframe.
    Returns:
        Multi-level dataframe showing mean and standard
        deviations of metrics and times elapsed across
        k-fold cross validation.
    """
    # print('len(metrics_records):', len(metrics_records))
    task = task.lower()
    # select metrics depending on task
    # see 'base_module.test_nn' to check which metrics are calculated
    if 'reg' in task:
        metric_keys = regress_metrics
    elif 'class' in task and 'bin' in task:
        metric_keys = bin_classif_metrics
    elif 'class' in task and 'multi' in task:
        metric_keys = multiclass_classif_metrics

    # dataframe 1: metrics (un-summarized across folds)
    results_df = pd.DataFrame.from_records(metrics_records)
    # print(results_df)
    
    if filter_tuple is not None:
        k, v = filter_tuple
        
        # mask for filtering one model's results
        # assumes all models have the same number of values
        # in the column being filtered!
        model_row_cts = results_df['model'].value_counts()
        one_model_row_ct = model_row_cts.iloc[0]
        one_model_mask = results_df \
            .iloc[:one_model_row_ct] \
            [results_df.iloc[:one_model_row_ct][k] == v] \
            .index.to_numpy()
        # print('one_model_mask', one_model_mask)

        # after generating mask for individual models, filter
        # entire results_df
        results_df_mask = (results_df[k] == v)
        results_df = results_df[results_df_mask]
        # print(results_df)
        
        
    # create group-aggregated metrics dataframe 1
    results_df = results_df \
        .groupby('model') \
        [metric_keys] \
        .agg(['mean', 'std']) \
        .sort_values(by=[(metric_keys[0], 'mean')], ascending=False)

    if model_suffix is not None:
        results_df.index = results_df.index + '-' + model_suffix
    
    # dataframe records containers
    total_min_elaps_records = []
    sec_elaps_records = []
    n_epochs_records = []

    if timing_dict is not None:
        for i, (model_key, sec_per_epoch_ll) in enumerate(timing_dict.items()):

            # optional: append suffix to model name in dataframe
            if model_suffix is not None:
                model_key = model_key + "-" + model_suffix
                
            # print('len(sec_per_epoch_ll)', len(sec_per_epoch_ll))
            if filter_tuple is not None:
                # sec_per_epoch_ll = sec_per_epoch_ll[one_model_mask]
                sec_per_epoch_ll = [sec_per_epoch_ll[i] for i in one_model_mask]

            # dataframe 2 records: total training time elapsed
            total_train_times_by_fold = [
                validate_every * np.sum(sec_per_epoch_l) \
                for sec_per_epoch_l in sec_per_epoch_ll
            ]
            total_min_elaps_record = {
                'model': model_key,
                'mean': np.mean(total_train_times_by_fold) / 60,
                'std': np.std(total_train_times_by_fold) / 60
            }
            total_min_elaps_records.append(total_min_elaps_record)
            

            # flatten ll (list of folds' lists of sec per epoch)
            # -> we are averaging across all folds and epochs (left after filter)
            sec_per_epoch_l = u.flatten_list(sec_per_epoch_ll)
        
            # calculate mean and st dev. across all recorded epochs from all runs 
            sec_mean, sec_std = np.mean(sec_per_epoch_l), np.std(sec_per_epoch_l)
        
            # dataframe 3 records
            sec_elaps_record = {
                'model': model_key,
                'mean': sec_mean,
                'std': sec_std
            }
            sec_elaps_records.append(sec_elaps_record)
    
            # dataframe 4 records: number of epochs per fold
            n_epochs_l = np.array([
                validate_every * len(l) for l in sec_per_epoch_ll
            ])
            n_epochs_mean, n_epochs_std = np.mean(n_epochs_l), np.std(n_epochs_l)
            n_epochs_record = {
                'model': model_key,
                'mean': n_epochs_mean,
                'std': n_epochs_std
            }
            n_epochs_records.append(n_epochs_record)

        # construct dataframe 2
        total_elaps_df = pd.DataFrame.from_records(total_min_elaps_records) \
            .set_index('model')
        total_elaps_df.columns = pd.MultiIndex.from_product(
            [['train_min_per_fold'], total_elaps_df.columns]
        )
        
        # construct dataframe 3
        sec_elaps_df = pd.DataFrame.from_records(sec_elaps_records) \
            .set_index('model')
        sec_elaps_df.columns = pd.MultiIndex.from_product(
            [['sec_per_epoch'], sec_elaps_df.columns]
        )
    
        # construct dataframe 4
        n_epochs_elaps_df = pd.DataFrame.from_records(n_epochs_records) \
            .set_index('model')
        n_epochs_elaps_df.columns = pd.MultiIndex.from_product(
            [['n_epochs'], n_epochs_elaps_df.columns]
        )
    
        # combine (multilevel) dataframes and round entries
        dfs_l = [results_df, sec_elaps_df, n_epochs_elaps_df, total_elaps_df]
        results_df = pd.concat(dfs_l, axis=1)
        
    if decimal_round is not None:
        results_df = results_df.round(decimal_round)

    # print(results_df)
        
    return results_df


def get_cv_results_df(
    args: ArgsTemplate,
    model_dir: str | Iterable[str],
    include_times: bool = True,
    validate_every: int = 1,
    model_suffix_dict: Optional[Dict[str, str]] = None,
    filter_tuple: Optional[Tuple[str, Any]] = None,
    decimal_round: Optional[int] = None
) -> pd.DataFrame:
    """
    Opens pickled metrics and times record files
    and calls 'generate_cv_results_df'.

    Args:
        args: ArgsTemplate object holding the
            proper RESULTS_SAVE_DIR and TASK
            for the results being opened.
        model_dir: string(s) of the directory(ies)
            for the model(s) whose results are 
            being opened.
        include_times: boolean whether to include
            'seconds per epoch' and 'number of epochs'
            metrics in the results dataframe.
        validate_every: if validation did not occur every
            epoch, adjust epoch counts by this multiple.
        model_suffix_dict: dictionary with model keys
            and string suffix values, to append to
            the model name in the 'model' column of the
            returned results dataframe.
        decimal_round: number of decimals to
            round the entries of the final
            dataframe to.
        filter_tuple: optional tuple of column name and 
            filtering value, to subset metrics_records 
            with.
    Returns:
        Pandas dataframe of combined results.
    """
    if isinstance(model_dir, str):
        model_dirs = [model_dir]
    else:
        model_dirs = model_dir

    # print('model_dirs:', model_dirs)
    dfs = []

    for i, model_dir in enumerate(model_dirs):
    #     metrics_records_path = f"{args.RESULTS_SAVE_DIR}/{model_dir}/cv_metrics.pkl"
    #     timing_dict_path = f"{args.RESULTS_SAVE_DIR}/{model_dir}/cv_times.pkl"
        dir_path = f"{args.RESULTS_SAVE_DIR}/{model_dir}"
        dir_files = [f for f in os.listdir(dir_path)]
        metrics_records_paths = [os.path.join(dir_path, f) for f in dir_files if 'metrics' in f]
        # print('metrics_records_paths found:\n', len(metrics_records_paths))
        timing_dict_paths = [os.path.join(dir_path, f) for f in dir_files if 'times' in f]
        # print('timing_dict_paths found:\n', len(timing_dict_paths))
        

        for metrics_records_path, timing_dict_path \
        in zip(metrics_records_paths, timing_dict_paths):
            # print('metrics_records_path:', metrics_records_path)
            
            with open(metrics_records_path, "rb") as f:
                metrics_records = pickle.load(f)
        
            if include_times:
                with open(timing_dict_path, "rb") as f:
                    timing_dict = pickle.load(f)
            else:
                timing_dict = None

            # optional suffix to append to model name in results df
            if model_suffix_dict is not None:
                if model_dir in model_suffix_dict.keys():
                    model_suffix = model_suffix_dict[model_dir]
                else:
                    model_suffix = None
            else:
                model_suffix = None
                    
            df = generate_cv_results_df(
                args.TASK,
                metrics_records,
                timing_dict,
                validate_every,
                model_suffix,
                filter_tuple,
                decimal_round
            )
            dfs.append(df)

    # if len(dfs) > 1:
    results_df = pd.concat(dfs)
    # else:
    #     results_df = dfs[0]
    
    return results_df


def avg_avg_results_df(
    df: pd.DataFrame,
    mean_key: str = 'mean',
    st_dev_key: str = 'std',
    sort_col_tuple: Optional[Tuple[str] | List[Tuple[str]]] \
        = [('acc', 'mean', 'nanmean'), ('acc', 'std', 'mean_std')],
    sort_col_ascends: Optional[bool | List[bool]] = [False, True]
) -> pd.DataFrame:
    r"""
    For a dataframe of multiple models' mean $\pm$ standard
    deviation results, where each model has multiple entries,
    this function creates a 'mean of means' $\pm$ mean standard
    deviation dataframe to summarize the multi-run results.

    Args:
        df: DataFrame with multiple runs of same model (names
            at index) with mean and standard deviation columns
            (at index 1 of multi-index column) for performance metrics.
        mean_key: string key for mean columns at second column
            index level.
        st_dev_key: string key for standard deviation columns at 
            second column index level.
        sort_col_tuple: tuple of strings or list of such tuples, 
            holding column level names for the metric column(s) to 
            sort rows (models) by. Provide more than one for 
            tie-breaking (two-strata) sort. Note that defaults are 
            for accuracy (classification tasks).
        sort_col_ascends: attendant bool or list of bools for the 
            columns in the 'sort_col_tuple' arg, whether to sort 
            ascending.
    Returns:
        DataFrame of summarized multi-run results.
    """
    # assume 2-layer cols
    mean_cols = [col for col in list(df) if mean_key in col[1]]
    st_dev_cols = [col for col in list(df) if st_dev_key in col[1]]
    df1 = df[mean_cols] \
        .groupby(df.index) \
        .agg([np.nanmean])
        # .agg(['mean'])
    df2 = df[st_dev_cols] \
        .groupby(df.index) \
        .agg([mean_std])

    # df_out = df1.join(df2, axis=1)
    df_out = pd.concat([df1, df2], axis=1)
    # this organizes combined dfs with mult-index cols
    sorted_cols = sorted([col[0] for col in list(df1)]) # list(df1) prevents duplicates
    # print(sorted_cols)
    df_out = df_out[sorted_cols]
    if sort_col_tuple is not None:
        df_out.sort_values(
            sort_col_tuple, 
            ascending=sort_col_ascends, 
            inplace=True
        )
    
    return df_out


def mean_std(st_devs: Iterable[float]) -> float:
    """
    Computes mean standard deviation of a list/iterable
    of standard deviations. This mean is the square root
    of (sum of variances / number of variances).

    Args:
        st_devs: iterable of standard deviations.
    Returns:
        Scalar of the mean standard deviation.
    """
    return np.sqrt(
        np.nansum(np.array(st_devs) ** 2) / len(st_devs)
    ).item()
    

def get_mean_pm_std_df(
    df: pd.DataFrame,
    mean_std_colnames: Tuple[str],
    col_rounds: Dict[Tuple[str], int],
    preserve_colnames: Optional[Tuple[str]] = None,
    final_colnames: Optional[Dict[str, str]] = None,
    metric_multipliers_dict: Optional[Dict[str, float]] = METRIC_MULTIPLIERS_DICT,
    best_bolding_fns_dict: Optional[Dict[str, Callable]] = METRIC_BEST_BOLD_FNS_DICT,
    final_modelnames: Dict[str, str] = None, # PUB_MODELNAMES_DICT,
    mean_subcol_tuple: Tuple[str] = ('mean', ),
    std_subcol_tuple: Tuple[str] = ('std', ),
    add_midrule_models_seperator: bool = True,
    sort_col_tuple: Optional[Tuple[str] | List[Tuple[str]]] \
        = [('acc', 'mean'), ('acc', 'std')],
    sort_col_ascends: Optional[bool | List[bool]] = [False, True]
) -> Tuple[pd.DataFrame, str]:
    r"""
    Converts a dataframe of metrics with paired mean and standard
    deviation columns into a formatted dataframe where metrics
    columns are strings of rounded means $\pm$ rounded standard
    deviations. Returns both a pandas DataFrame its a LaTeX table
    string.

    Args:
        df: pd.DataFrame of results with mean and standard deviation
            columns for metrics.
        mean_std_colnames: tuple containing column names of metrics.
        col_rounds: dictionary with column names as keys and integer
            rounding values for those columns. For multilevel columns,
            column names are tuples of strings, e.g. ('R2', 'mean').
        preserve_colnames: tuple with names of any columns (that are not
            metrics with means and standard deviations) to include in
            the output dataframe.
        final_colnames: dictionary map for renaming columns in final
            output dataframe and LaTeX table.
        metric_multipliers_dict: dictionary lookup for multipliers for
            metric score and st. dev. values, e.g., to convert some
            to percentages by multiplying by 100.
        best_bolding_fns_dict: dictionary lookup for bolding the best 
            result in a column (whose names are keys); values are the
            numpy 'argmax' or 'argmin' function that determines the 
            index of the best results. Defaults to METRIC_BEST_BOLD_FNS_DICT,
            found at top of this file.
        final_modelnames: dictionary map for renaming models in the final
            output dataframe and LaTeX table; defaults to None; but
            PUB_MODELNAMES_DICT (found at the top of this file) holds
            some useful mappings for the MFCN project.
        mean_subcol_tuple: tuple identifying the mean subcolumn, e.g. 
            ('mean', ); used to construct the full column identifier tuple
            in a multicolumn pandas environment: e.g., (col, 'mean') or
            (col, 'mean', 'nanmean').
        std_subcol_tuple: equivalent to 'mean_subcol_tuple' except
            identifying the standard deviation column.
        add_midrule_models_seperator: bool whether to include \\midrule
            row separator between model groups in the LaTeX table.
        sort_col_tuple: tuple of strings or list of such tuples, 
            holding column level names for the metric column(s) to 
            sort rows (models) by. Provide more than one for 
            tie-breaking (two-strata) sort. Note that defaults are 
            for accuracy (classification tasks).
        sort_col_ascends: attendant bool or list of bools for the 
            columns in the 'sort_col_tuple' arg, whether to sort 
            ascending.
    Returns:
        2-tuple of (1) formatted pandas DataFrame, and (2) its LaTeX 
        table string.
    """
    # optional: bold best results
    if best_bolding_fns_dict is not None:
        best_bold_idxs = {
            col: fn(df[col]) \
            for col, fn in best_bolding_fns_dict.items() \
            if (
                (col in mean_std_colnames) 
                and (col in best_bolding_fns_dict.keys())
            )
        }
    # optional: sort models by scores
    if sort_col_tuple is not None:
        df.sort_values(
            sort_col_tuple, 
            ascending=sort_col_ascends, 
            inplace=True
        )
    # optional: multiply scores by multiples (e.g. convert to %)
    if metric_multipliers_dict is not None:
        for k, v in metric_multipliers_dict.items():
            df[k] = df[k] * v
        
    # create $[mean] \pm [std]$ cols, rounded to desired decimals
    series_l = [df[col] for col in preserve_colnames] \
        if (preserve_colnames is not None) else []
    for col in mean_std_colnames:
        mean_key = (col, ) + mean_subcol_tuple
        mean_rd = col_rounds[mean_key]
        std_key = (col, ) + std_subcol_tuple
        std_rd = col_rounds[std_key]
        
        series = r'$' + df[mean_key].apply(lambda x: f"{x:.{mean_rd}f}")
        # only append '\pm [std]' if std is a valid value
        if (not np.isnan(df[std_key]).any() and (df[std_key] > 0.).all()):
            series += r' \pm ' + df[std_key].apply(lambda x: f"{x:.{std_rd}f}") + r'$'
        else:
            series += r'$'
        if best_bolding_fns_dict is not None:
            if col in best_bolding_fns_dict.keys():
                best_idx = best_bolding_fns_dict[col]
                series.iloc[best_idx] = r'$\mathbf{' + series.iloc[best_idx][1:-1] + r'}$'
            
        series.name = col
        series_l.append(series)
    df = pd.concat(series_l, axis=1)
    
    # change col and model names to 'official' names
    if final_modelnames is not None:
        df.index = df.index.map(final_modelnames)
        if df.index.isna().any():
            warnings.warn(
                "Missing value(s) in model names (dataframe index)"
                " may be due to missing items in `final_modelnames`"
                " argument for `ru.get_mean_pm_std_df`."
            )
        df.rename(columns=final_colnames, inplace=True)

    # generate latex string
    df_latex_str = df.to_latex()

    if add_midrule_models_seperator:
        # lines 0:4 are latex table header
        df_latex_byline = df_latex_str.split('\n')
        table_header = df_latex_byline[:5]
        df_latex_byline = df_latex_byline[5:]
    
        n_models = len(df.index.unique())
        i = n_models
        while i < (len(df_latex_byline) - n_models):
            df_latex_byline.insert(i, '\\midrule')
            i += (n_models + 1)
        df_latex_str = "\n".join(table_header + df_latex_byline)
    
    return df, df_latex_str

