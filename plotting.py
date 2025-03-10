"""
Plotting functions.
"""
from utilities import central_moving_average

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from typing import (
    Tuple,
    List,
    Dict,
    Any,
    Optional,
    Iterable,
    Callable
)

"""
Misc. plotting
"""
def jitter(
    arr: np.ndarray, 
    spread: float = 0.01
) -> np.ndarray:
    """
    Adds jitter (random noise) to an array.
    Useful for plots where many points are
    on top of one another.

    Args:
        arr: array of values.
        spread: coefficient controlling the
            st. deviation of the jitter noise.
        
    Returns:
        Array of jittered values.
        
    Adapted from: https://stackoverflow.com/a/21276920
    """
    if spread > 0.:
        stdev = spread * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev
    else:
        return arr


"""
2-d plotting
"""
def show_train_plot(
    history_df, # pandas dataframe
    metrics_l: List[float], 
    burnin_n_epochs: int = 0,
    smooth_window: int = None,
    vline_x: float = None,
    hline_y: float = None,
    title: str = None,
    legend_loc: str ='upper right',
    fig_size: Tuple[float, float] = (10., 5.),
    y_lim: Tuple[float, float] = (-0.05, 1.05),
    match_ylims: bool = False,
    y_scale_log: bool = False,
    grid_step_x: str = 'auto',
    line_colors: Tuple[str] = ('tab:blue', 'tab:red')
) -> None:
    """
    Constructs a (multi-)line plot for the metrics in 
    `history_df`, showing their values by training epoch.

    Optionally plots a smoothed line over the raw metric
    trends, using 'utilities.central_moving_avg'.
    """
    # figure setup
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams['grid.color'] = 'lightgray'

    # x-axis
    if grid_step_x == 'auto':
        grid_step_x_used = len(history_df) // 16
    else:
        grid_step_x_used = grid_step_x
    grid_step_x_used = max([2, grid_step_x_used])
    x_ticks = range(burnin_n_epochs + 1, 
                    history_df.epoch.max() + 1, 
                    grid_step_x_used)
    
    # if smoothing, plot more opaquely on top
    if smooth_window is not None:
        alpha_orig_data = 0.2
        metrics_smooth = [
            central_moving_average(
                history_df[metric], 
                smooth_window
            ) \
            for metric in metrics_l
        ]
    else:
        alpha_orig_data = 1.0

    fig, ax1 = plt.subplots()

    # metric 1 plotting
    ax1.set_xlabel('epoch')
    ax1.set_ylabel(metrics_l[0], color=line_colors[0])
    ax1.plot(
        history_df['epoch'][burnin_n_epochs:], 
        history_df[metrics_l[0]][burnin_n_epochs:], 
        color=line_colors[0],
        alpha=alpha_orig_data
    )
    ax1.tick_params(axis='y', labelcolor=line_colors[0])
    ax1.set_ylim(y_lim)
    if y_scale_log:
        ax1.set_yscale('log')
    if smooth_window is not None:
        ax1.plot(
            history_df['epoch'][burnin_n_epochs:], 
            metrics_smooth[0][burnin_n_epochs:], 
            color=line_colors[0]
        )

    # metric 2 plotting
    # if there are 2 metrics, give them separate y axes
    if len(metrics_l) == 2:
        # metric 2
        ax2 = ax1.twinx()
        ax2.set_ylabel(metrics_l[1], color=line_colors[1])
        ax2.plot(
            history_df['epoch'][burnin_n_epochs:], 
            history_df[metrics_l[1]][burnin_n_epochs:], 
            color=line_colors[1],
            alpha=alpha_orig_data
        )
        ax2.tick_params(axis='y', labelcolor=line_colors[1])
        if match_ylims:
            ax2.set_ylim(ax1.get_ylim())
        else:
            ax2.set_ylim(y_lim)
        if y_scale_log:
            ax2.set_yscale('log')
        if smooth_window is not None:
            ax2.plot(
                history_df['epoch'][burnin_n_epochs:], 
                metrics_smooth[1][burnin_n_epochs:], 
                color=line_colors[1]
            )

    # hline and vline options
    if vline_x is not None:
        ax1.axvline(x=vline_x, color='gray', ls='--')
    if hline_y is not None:
        ax1.axhline(y=hline_y, color='gray', ls='--')

    # legend and title options
    if legend_loc is not None:
        plt.legend(framealpha=1.0, loc=legend_loc)
    if title is not None:
        plt.title(title)

    plt.xticks(x_ticks)
    plt.grid()
    plt.show()


def make_sinusoidal(
    x: np.ndarray,
    y: np.ndarray,
    trend_check_n: int = 50,
    decreasing_check_prop: float = 0.8,
    zero_threshold: float = 1.0e-5,
    verbosity: int = 0
) -> np.ndarray:
    """
    Shift and/or reflect a circular periodic
    vector so that it starts at 0. and is 
    increasing.

    Args:
        x: vector of support values, e.g., on [0, 2pi].
        y: a vector of values, assumed to
            be (1) centered at 0 and (b) sorted over 
            an 'x' support interval.
        trend_check_n: number of adjacent vector
            entries to check for 'increasing' or
            'decreasing' trend.
        decreasing_check_prop: proportion (0 to 1)
            of vector sample neighbors that must be
            show '>=' relation to establish that the
            vector is decreasing over the sample.
        zero_threshold: the 'zero equivalent' float
            value threshold.

    Returns:
        Vector of values shifted along [0, 2pi]
        such that the vector is sinusoidal.
    
    """
    # if vector starts near 0...
    if (np.abs(y[0]) < zero_threshold):
        # and is decreasing -> reflection makes sinusoidal
        y_check = y[1:(trend_check_n + 2)]
        # prop_decreasing = np.sum(y_check[:-1] >= y_check[1:]) / (y_check.shape[0] - 1)
        prop_decreasing = np.sum(y_check < 0.) / (y_check.shape[0])
        if prop_decreasing > decreasing_check_prop:
            if verbosity > 0:
                print('decreasing!')
            return x, -y
        # but is increasing: already sinusoidal
        else:
            if verbosity > 0:
                print('already sinusoidal!')
            return x, y
    else:
        if verbosity > 0:
            print('shifting')
        # vector needs phase-shifting
        # find index of next 0 after min (implies increasing)
        min_i = np.argmin(y)
        # print(min_i)
        zeros_is = np.squeeze(np.argwhere(np.abs(y) < zero_threshold))
        # print(zeros_is)
        zeros_after_min = zeros_is[(zeros_is > min_i)]
        if zeros_after_min.shape[0] > 0:
            zero_i = zeros_after_min[0]
        else:
            zero_i = zeros_is[0]
            
        # split and re-concatenate vectors at this index
        # note that x needs to be shifted too, since xs may 
        # not be uniformly spaced, which would distort the plot
        # if shifted ys were plotted over unshifted xs
        x_shift = x[zero_i]
        if verbosity > 0:
            print(f'{x_shift / np.pi:.4f}pi')
        shifted_x = np.concatenate((
            x[zero_i:] - x_shift, 
            x[:zero_i] + (np.max(x) - x_shift)
        ))
        shifted_y = np.concatenate((y[zero_i:], y[:zero_i]))
        return shifted_x, shifted_y


def circle_scatterplot(
    fig,
    ax,
    c: np.ndarray,
    xy: np.ndarray,
    title: str = None,
    show_colorbar: bool = True,
    show_circle_axes: bool = False,
    cmap: str = 'gnuplot',
    s: float = 30.,
    jitter_spread: float = 0.
) -> None:
    """
    Plots a circular scatterplot to the passed 
    ax and fig, with optional colorbar.

    Args:
        fig: plt 'fig' object.
        ax: plt 'ax' object associated with 'fig'.
        c: vector of values for scatter points' color.
        xy: array of x and y coordinates of scatter
            points.
        show_colorbar: boolean to show/hide colorbar.
        show_circle_axes: boolean to show/hide x and y
            axis ticks and labels.
        cmap: plt colormap to use.
        s: scatter point size.
        jitter_spread: coefficient controlling amount
            of jittering, to prevent overplotting of
            the scatter points.

    Returns:
        None; modifies the passed 'fig' and 'ax' objects
        in place.
    """
    ax.set_title(title)
    ax.scatter(
        x=jitter(xy[:, 0], jitter_spread), 
        y=jitter(xy[:, 1], jitter_spread),
        c=c, 
        cmap=cmap, 
        s=s
    )
    
    if show_circle_axes == False:
        # ax.axes.get_xaxis().set_visible(show_circle_axes)
        # ax.axes.get_yaxis().set_visible(show_circle_axes)
        
        ax.tick_params(
            axis='both',       # changes apply to the x and y axes
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False
        )
    
    if show_colorbar:
        norm = plt.Normalize(min(c), max(c))
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)


"""
3-d plotting
"""
def plot_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str = '',
    fig = None,
    ax = None, # matplotlib.axes.Axes type
    xyz_lim: Optional[Tuple[Tuple[float, float]]] = None,
    xyz_labels: Tuple[str] = ('x', 'y', 'z'),
    label: str = None,
    marker: str = 'o',
    s: float = 10.0,
    c: np.ndarray = None,
    alpha: Optional[float] = None,
    show_colorbar: bool = True,
    cmap: str = 'bwr',
    colorbar_num_format: str = '%.4E',
    equal_axes_scale: bool = True,
    show_tick_labels: bool = True
) -> None:
    """
    Produces a 3-d scatterplot using
    matplotlib, with optional colormapping
    and colorbar.

    Adds plot to existing ax if passed;
    otherwise, creates new fig with single
    ax and plots automatically.

    Ref:
    - matplotlib colorscale options:
        https://matplotlib.org/stable/users/explain/colors/colormaps.html
   
    Args:
        title: string for plot title.
        x: array of (ordered) x coords.
        y: array of (ordered) y coords.
        z: array of (ordered) z coords.
        fig:
        ax: matplotlib.axes.Axes object, if
            the 3-d plot produced here is to
            be assigned to an existing one.
        xyz_lim:
        xyz_labels: 3-tuple of labels for the
            x, y, and z axes.
        label: string key that will show in
            (optional) plot legend.
        marker: marker type/shape.
        s: marker size.
        c: array of colormap values.
        alpha: transparency level (0 to 1), or if
            None, in 3d scatterplot, more distant
            points will automatically be more
            transparent.
        show_colorbar: boolean whether to display
            colorbar.
        cmap: string key for matplotlib colormap.
        colorbar_num_format: formatting string for
            colorbar numbers.
        equal_axes_scale:
        show_tick_labels:

    Returns:
        An 'ax' (matplotlib.axes.Axes type) to None
        (calls plt.show() instead, if no 'ax' is passed).
    """
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(projection='3d')

    # axes settings
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    if xyz_lim is not None:
        ax.set_xlim(xyz_lim[0])
        ax.set_ylim(xyz_lim[1])
        ax.set_zlim(xyz_lim[2])
    elif equal_axes_scale:
        # plt.axis('equal')
        ax.set_aspect('equal')

    ax.set_title(title)
    if xyz_labels is not None:
        ax.set_xlabel(xyz_labels[0], labelpad=0.01)
        ax.set_ylabel(xyz_labels[1], labelpad=0.01)
        ax.set_zlabel(xyz_labels[2], labelpad=0.01)
   
    if not show_tick_labels:
        # for k, v in ax.spines.items(): # left, right, bottom, top
        #     ax.spines[k].set_visible(False)
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.set_tick_params(length=0, width=0, which='minor')
        ax.yaxis.set_tick_params(length=0, width=0, which='minor')
        ax.zaxis.set_tick_params(length=0, width=0, which='minor')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

    # if c is passed, map colors and add colorbar
    if (c is not None):
        im = ax.scatter(
            x, y, z,
            label=label,
            marker=marker,
            s=s,
            c=c,
            alpha=alpha
       )
        im.set_cmap(cmap)
        if show_colorbar:
            fig.colorbar(
                im,
                ax=ax,
                fraction=0.03,
                pad=0.06,
                format=colorbar_num_format
            )
    else:
        ax.scatter(
            x, y, z,
            label=label,
            marker=marker,
            s=s,
            alpha=alpha
        )
   
    # ax.plot_trisurf(x, y, z, antialiased=False, color='orange')

    if ax is None:
        plt.show()
    else:
        return ax


def plot_Fourier_coeffs_convergence(
    records: List, 
    sampling_case: str,
    n_coeffs: int = 5,
    verbosity: int = 0
) -> None:
    """
    Creates plot of 'n_coeffs' Fourier coefficients
    over Ns (increasing sample sizes).
    Note that coefficients are rescaled
    to be relative to those of the largest
    (last) N, and also as if these last reference
    coefficients had values n_coeffs...1.
    
    Args:
        records: dictionary of experimental results, 
            in which 'fourier_coeffs' are a key.
        sampling_case: 'uniform' or 'nonuniform' key.
        n_coeffs: how many Fourier coefficients to plot.
        verbosity: controls print output while function
            runs.

    Returns:
        None (prints plot).
        
    """
    # grab first 'n_coeffs' Fourier coeffs as n increases
    # NOTE: we take abs. value, since eigendecomp. can 
    # vary by factor of +/- 1
    ns = [record['n'] for record in records]
    first_fcs = [None] * len(ns)
    for n_i, n in enumerate(ns):
        fcs = np.abs(records[n_i]['fourier_coeffs'][:n_coeffs])
        first_fcs[n_i] = fcs
    first_fcs = np.vstack(first_fcs)
    if verbosity > 0:
        print('first_fcs\n', first_fcs)
    
    # rescale Fourier coeffs relative to the largest n,
    # so they can all be shown on same plot
    ref_fc = np.abs(records[-1]['fourier_coeffs'][0])
    mults = first_fcs[-1] / ref_fc
    first_fcs_scaled = np.einsum(
        'i,ji->ji',
        1. / mults,
        first_fcs
    ) / ref_fc
    
    # rescale again, such that Fourier coeffs
    # have integer descending values
    first_fcs_scaled = np.einsum(
        'i,ji->ji',
        np.flip(np.arange(1, n_coeffs + 1)),
        first_fcs
    )
    if verbosity > 0:
        print('first_fcs_scaled\n', first_fcs_scaled)
    
    # plot
    for i in range(n_coeffs):
        label = f'fc_{i}'
        plt.plot(ns, first_fcs_scaled[:, i], label=label)
    plt.xticks(ns)
    plt.xlabel('n')
    plt.yscale('log')
    plt.ylabel(f'relative, rescaled Fourier coeffs')
    plt.legend()
    plt.title(f'Relative, rescaled Fourier coefficients by sample size ({sampling_case} sampling).'
              f'\nLevel lines indicate convergence.')
    plt.show()


