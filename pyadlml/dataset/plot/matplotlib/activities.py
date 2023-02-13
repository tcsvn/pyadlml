import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from pyadlml.constants import ACTIVITY, TIME, START_TIME, END_TIME
from pyadlml.dataset.stats.activities import activities_duration_dist, activity_duration, \
    activities_transitions, activities_count, activity_duration, activities_dist, activity_order_by_duration
from pyadlml.dataset._core.activities import add_other_activity
from .util import func_formatter_seconds2time_log, ridgeline, \
    func_formatter_seconds2time, heatmap, annotate_heatmap, heatmap_square, savefig, \
    _num_bars_2_figsize, _num_boxes_2_figsize, \
    _num_items_2_heatmap_square_figsize, _num_items_2_ridge_figsize, \
    _num_items_2_ridge_ylimit, xaxis_format_one_day, save_fig, get_qualitative_cmap, plot_grid, xaxis_format_time2, \
    map_time_to_numeric
from pyadlml.util import get_sequential_color, get_secondary_color, get_primary_color, get_diverging_color
from pyadlml.dataset.util import activity_order_by

@save_fig
def count(df_acts=None, df_ac=None, scale="linear", other=False,
          figsize=None, color=None, file_path=None):
    """
    Plot a bar chart displaying how often activities are occurring.

    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    other : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    scale : {"log", "linear"}, default: linear
        The axis scale type to apply.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    color : str, optional
        sets the color of the plot. When not set, the primary theming color is used.
        Learn more about theming in the :ref:`user guide <theming>`
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_activities_count
    >>> from pyadlml.dataset import fetch_amsterdam
    >>> data = fetch_amsterdam()
    >>> plot_activities_count(data.df_activities, idle=True)

    .. image:: ../_static/images/plots/act_bar_cnt.png
       :height: 300px
       :width: 500 px
       :scale: 90 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing 
    """
    assert not (df_acts is None and df_ac is None)
    assert scale in ['linear', 'log']

    title ='Activity occurrences'
    col_label = 'occurrence'
    x_label = 'counts'
    color = (get_primary_color() if color is None else color)
        
    # create statistics if the don't exists
    if df_ac is None:
        df_acts = df_acts.copy()
        if other:
            df_acts = add_other_activity(df_acts)
        df = activities_count(df_acts)
    else:
        df = df_ac
    
    # prepare dataframe for plotting
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by=[col_label], axis=0)
    
    # define plot modalities
    num_act = len(df)
    figsize = (_num_bars_2_figsize(num_act) if figsize is None else figsize)
    
    # create plot
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    plt.xlabel(x_label)
    ax.barh(df[ACTIVITY], df[col_label], color=color)
    
    if scale == 'log':
        ax.set_xscale('log')

    return fig


@save_fig
def boxplot(df_acts, scale='linear', other=False, figsize=None,
            order='alphabetical', file_path=None):
    """
    Plot a boxplot for activity durations.

    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    scale : {"log", "linear"}, default: None
        The axis scale type to apply.
    other : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *other* or be ignored.
    order : str, default='alphabetical'
        The order in which the activities are displayed
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_activity_boxplot
    >>> plot_activity_boxplot(data.df_activities)

    .. image:: ../_static/images/plots/act_bp.png
       :height: 300px
       :width: 500 px
       :scale: 90 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing
    """
    assert scale in ['linear', 'log']
    
    title = 'Activity durations'
    xlabel = 'Seconds'

    if other:
        df_acts = add_other_activity(df_acts)

    act_order = np.flip(activity_order_by(df_acts, rule=order))
    df = activities_duration_dist(df_acts)

    # Select data for each device
    df['seconds'] = df['minutes']*60

    num_act = len(act_order)
    figsize = (_num_bars_2_figsize(num_act) if figsize is None else figsize)

    dat = []
    for activity in act_order:
        df_activity = df[df[ACTIVITY] == activity]
        dat.append(df_activity['seconds'])
    
    # Plot boxsplot
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(dat, vert=False)
    ax.set_title(title)
    ax.set_yticklabels(act_order, ha='right')
    ax.set_xlabel(xlabel)

    if scale == 'log':
        ax.set_xscale('log')

    # Create secondary axis with time format 1s, 1m, 1d
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_seconds2time))

    return fig


@save_fig
def duration(df_acts=None, df_dur=None, scale='linear', idle=False,
             figsize=None, color=None, file_path=None):
    """
    Plots the cumulative duration for each activity in a bar plot.

    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : lst of str, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    scale : {"log", "linear"}, default: None
        The axis scale type to apply.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    color : str, optional
        sets the color of the plot. When not set, the primary theming color is used.
        Learn more about theming in the :ref:`user guide <theming>`
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import duration
    >>> duration(data.df_activities)

    .. image:: ../_static/images/plots/act_bar_dur.png
       :height: 300px
       :width: 500 px
       :scale: 90 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing
    """
    assert scale in ['linear', 'log']
    assert not (df_acts is None and df_dur is None)

    title = 'Cummulative activity durations'
    xlabel = 'seconds'
    freq = 'seconds'
    color = (get_primary_color() if color is None else color)

    if df_dur is None:
        if idle:
            df_acts = add_other_activity(df_acts.copy())
        df = activity_duration(df_acts, time_unit=freq)
    else:
        df = df_dur
    df = df.sort_values(by=[freq], axis=0)

    num_act = len(df)
    figsize = (_num_bars_2_figsize(num_act) if figsize is None else figsize)
    
    # plot
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    ax.barh(df[ACTIVITY], df['seconds'], color=color)
    if scale == 'log':
        ax.set_xscale('log')
        
        
    # create secondary axis with time format 1s, 1m, 1d
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax_top.set_xlabel('time')
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_seconds2time))

    return fig


@save_fig
def transitions(df_acts=None, df_trans=None, scale="linear",
                figsize=None, idle=False, numbers=True, grid=True,
                cmap=None, file_path=None):
    """
    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    df_trans : pd.DataFrame
        A precomputed transition table. If the *df_trans* parameter is given, parameters
        *df_acts* and *lst_acts* are ignored. The transition table can be computed
        in :ref:`stats <stats_acts_trans>`.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    scale : {"log", "linear"}, default: None
        The axis scale type to apply.
    numbers : bool, default: True
        Whether to display numbers inside the heatmaps fields or not.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    cmap : str or Colormap, optional
        The Colormap instance or registered colormap name used to map scalar
        data to colors. This parameter is ignored for RGB(A) data.
        Defaults 'viridis'.
    grid : bool, default: True
        determines whether to display a white grid, seperating the fields or not.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import transitions
    >>> transitions(data.df_activities)

    .. image:: ../_static/images/plots/act_hm_trans.png
       :height: 300px
       :width: 500 px
       :scale: 90 %
       :alt: alternate text
       :align: center


    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing.
    """
    assert scale in ['linear', 'log'], 'scale has to be either of type None or log'
    assert not (df_acts is None and df_trans is None)

    title = 'Activity transitions'
    z_label = 'count'

    if df_trans is None:
        df_acts = add_other_activity(df_acts) if idle else df_acts
        df = activities_transitions(df_acts)
    else:
        df = df_trans

    # get the list of cross tabulations per t_window
    act_lst = list(df.columns)

    num_act = len(act_lst)
    figsize = (_num_items_2_heatmap_square_figsize(num_act) if figsize is None else figsize)
    cmap = (get_sequential_color() if cmap is None else cmap)

    x_labels = act_lst
    y_labels = act_lst
    values = df.values
    

    log = True if scale == 'log' else False
    valfmt = '{x:.0f}'
        
     # begin plotting
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap_square(values, y_labels, x_labels, log=log, cmap=cmap, ax=ax, cbarlabel=z_label, grid=grid)
    if numbers:
        texts = annotate_heatmap(im, textcolors=("white", "black"),log=log, valfmt=valfmt)
    ax.set_title(title)
    
    return fig


@save_fig
def density(df_acts=None, df_act_dist=None, idle=False, dt=None,
            n=1000, ylim_upper=None, color=None, figsize=None, file_path=None):
    """
    Plots the activity density distribution over one day.

    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : lst of str, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    df_act_dist : pd.DataFrame, optional
        A precomputed activity density distribution. If the *df_trans* parameter is given, parameters
        *df_acts* and *lst_acts* are ignored. The transition table can be computed
        in :ref:`stats <stats_acts_trans>`.
    n : int, default=1000
        The number of monte-carlo samples to draw.
    ylim_upper: float, optional
        The offset from the top of the plot to the first ridge_line. Set this if
        the automatically determined value is not satisfying.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    color : str, optional
        sets the color of the plot. When not set, the primary theming color is used.
        Learn more about theming in the :ref:`user guide <theming>`
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_activity_density
    >>> plot_activity_density(data.df_activities)

    .. image:: ../_static/images/plots/act_ridge_line.png
       :height: 300px
       :width: 500 px
       :scale: 90 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified otherwise nothing.
    """
    assert not (df_acts is None and df_act_dist is None)

    title = 'Activity density over one day'
    xlabel = 'Time'
    color = (get_primary_color() if color is None else color)


    if df_act_dist is None:
        if idle:
            df_acts = add_other_activity(df_acts)
        df = activities_dist(df_acts.copy(), n=n, dt=dt, relative=True)
        if df.empty:
            raise ValueError("no activity was recorded and no activity list was given.")
    else:
        df = df_act_dist

    if dt is None:
        # plot ridgeline
        grouped = [(activity, df.loc[df[ACTIVITY] == activity, TIME].sort_values().values)
                   for activity in df[ACTIVITY].unique()]
        acts, data = zip(*grouped)
        num_act = len(list(acts))

        # infer visual properties
        figsize = (_num_items_2_ridge_figsize(num_act) if figsize is None else figsize)
        ylim_upper = (_num_items_2_ridge_ylimit(num_act) if ylim_upper is None else ylim_upper)

        # plot the ridgeline
        fig, ax = plt.subplots(figsize=figsize)
        ridgeline(data, labels=acts, overlap=.85, fill=color, n_points=100, dist_scale=0.13)
        plt.title(title)

        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.ylim((0, ylim_upper))
        plt.xlabel(xlabel)

        xaxis_format_one_day(ax, 0, 60*60*24)
        fig.autofmt_xdate(rotation=45)

        plt.grid(zorder=0)
    else:
        # plot heatmap
        x_labels = list(df[TIME])
        activities = df.columns[1:]
        dat = df.iloc[:, 1:].values.T

        cmap = get_sequential_color()
        cbarlabel = 'counts'

        # begin plotting
        fig, ax = plt.subplots(figsize=figsize)
        im, cbar = heatmap(dat, activities, x_labels, ax=ax, cmap=cmap, cbarlabel=cbarlabel)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        xaxis_format_one_day(ax, *ax.get_xlim())

        ax.set_aspect(aspect='auto')

    return fig


@save_fig
def correction(df_act_pre, df_act_post, grid=True, figsize=(10, 5), file_path=None):
    """ Plots the activities before and after the automatic correction. Refer to the user guide: TODO


    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    df_act_dist : pd.DataFrame, optional
        A precomputed activity density distribution. If the *df_trans* parameter is given, parameters
        *df_acts* and *lst_acts* are ignored. The transition table can be computed
        in :ref:`stats <stats_acts_trans>`.

    color : str, optional
        sets the color of the plot. When not set, the primary theming color is used.
        Learn more about theming in the :ref:`user guide <theming>`
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.dataset import fetch_amsterdam
    >>> from pyadlml.plot import plot_activity_correction

    >>> corr = fetch_amsterdam().correction_activities
    >>> plot_activity_correction(corr[0][0], corr[0][1])

    .. image:: ../_static/images/plots/act_ridge_line.png
       :height: 300px
       :width: 500 px
       :scale: 90 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified otherwise nothing.

    """
    title = 'Activity corrections'
    activity_y_label = 'Activities'
    right_title = 'Post-correction'
    left_title = 'Prae-correction'

    df_act_post = df_act_post.sort_values(by=START_TIME)
    df_act_pre = df_act_pre.sort_values(by=START_TIME)

    start_time = min(df_act_pre[START_TIME].iloc[0], df_act_post[START_TIME].iloc[0])
    end_time = max(df_act_pre[END_TIME].iloc[-1], df_act_post[END_TIME].iloc[-1])

    df_act_post = df_act_post.sort_values(by=START_TIME, ascending=False).reset_index(drop=True)
    df_act_pre = df_act_pre.sort_values(by=START_TIME, ascending=False).reset_index(drop=True)

    acts = activity_order_by_duration(df_act_pre)

    df_act_pre['num_st'], _, _ = map_time_to_numeric(df_act_pre[START_TIME], start_time, end_time)
    df_act_pre['num_et'], _, _ = map_time_to_numeric(df_act_pre[END_TIME], start_time, end_time)
    df_act_pre['diff'] = df_act_pre['num_et'] - df_act_pre['num_st']

    df_act_post['num_st'], _, _ = map_time_to_numeric(df_act_post[START_TIME], start_time, end_time)
    df_act_post['num_et'], _, _ = map_time_to_numeric(df_act_post[END_TIME], start_time, end_time)
    df_act_post['diff'] = df_act_post['num_et'] - df_act_post['num_st']


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    tab = get_qualitative_cmap(len(acts))

    if grid:
        plot_grid(fig, ax1, start_time, end_time)
        plot_grid(fig, ax2, start_time, end_time)

    # create a column with the color for each department
    def color(row):
        return tab(np.where(np.array(acts) == row[ACTIVITY])[0][0])

    df_act_pre['color'] = df_act_pre.apply(color, axis=1)
    df_act_post['color'] = df_act_post.apply(color, axis=1)

    ax1.barh(df_act_pre.index, df_act_pre['diff'], left=df_act_pre['num_st'],
             color=df_act_pre['color'], label=df_act_pre[ACTIVITY])

    ax2.barh(df_act_post.index, df_act_post['diff'], left=df_act_post['num_st'],
             color=df_act_post['color'], label=df_act_post[ACTIVITY])

    ax1.set_xlim(-0.05, 1.05)
    ax2.set_xlim(-0.05, 1.05)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tab(i), label=act) for i, act in enumerate(acts)]

    ax1.set_title(left_title)
    ax2.set_title(right_title)
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc="upper left")
    xaxis_format_time2(fig, ax1, start_time, end_time)
    xaxis_format_time2(fig, ax2, start_time, end_time)

    return fig