import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from pyadlml.dataset import ACTIVITY
from  pyadlml.dataset.stats.activities import activities_duration_dist, activity_durations,\
    activities_transitions, activities_count, activity_durations, activities_dist
from pyadlml.dataset._core.activities import add_idle
from pyadlml.dataset.plot.util import func_formatter_seconds2time_log, ridgeline, \
    func_formatter_seconds2time, heatmap, annotate_heatmap, heatmap_square, savefig, \
    _num_bars_2_figsize, _num_boxes_2_figsize, \
    _num_items_2_heatmap_square_figsize, _num_items_2_ridge_figsize,\
    _num_items_2_ridge_ylimit
from pyadlml.util import get_sequential_color, get_secondary_color, get_primary_color, get_diverging_color


def hist_counts(df_acts=None, lst_acts=None, df_ac=None, y_scale="linear", idle=False,
                figsize=None, color=None, file_path=None):
    """
    Plot a bar chart displaying how often activities are occurring.

    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : lst of str, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    y_scale : {"log", "linear"}, default: linear
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
    >>> from pyadlml.plot import plot_activity_bar_count
    >>> plot_activity_bar_count(data.df_activities, idle=True)

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
    assert y_scale in ['linear', 'log']

    title ='Activity occurrences'
    col_label = 'occurrence'
    x_label = 'counts'
    color = (get_primary_color() if color is None else color)
        
    # create statistics if the don't exists
    if df_ac is None:
        df_acts = df_acts.copy()
        if idle:
            df_acts = add_idle(df_acts)
        df = activities_count(df_acts, lst_acts=lst_acts)
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
    ax.barh(df['activity'], df[col_label], color=color)
    
    if y_scale == 'log':
        ax.set_xscale('log')

    # save or return fig
    if file_path is not None:
        savefig(fig, file_path)
        return None
    else:
        return fig


def boxplot_duration(df_acts, lst_acts=None, y_scale='linear', idle=False,
                     figsize=None, file_path=None):
    """
    Plot a boxplot for activity durations.

    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : lst of str, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    y_scale : {"log", "linear"}, default: None
        The axis scale type to apply.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_activity_bp_duration
    >>> plot_activity_bp_duration(data.df_activities)

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
    assert y_scale in ['linear', 'log']
    
    title = 'Activity durations'
    xlabel = 'seconds'

    if idle:
        df_acts = add_idle(df_acts)

    df = activities_duration_dist(df_acts, lst_acts=lst_acts)
    # select data for each device
    activities = df[ACTIVITY].unique()
    df['seconds'] = df['minutes']*60     

    num_act = len(activities)
    figsize = (_num_bars_2_figsize(num_act) if figsize is None else figsize)

    dat = []
    for activity in activities:
        df_activity = df[df[ACTIVITY] == activity]
        #tmp = np.log(df_device['td'].dt.total_seconds())
        dat.append(df_activity['seconds']) 
    
    # plot boxsplot
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(dat, vert=False)
    ax.set_title(title)
    ax.set_yticklabels(activities, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_xscale('log')

    # create secondary axis with time format 1s, 1m, 1d
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    #ax_top.set_xlabel('time')
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_seconds2time))

    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig

def hist_cum_duration(df_acts=None, lst_acts=None, df_dur=None, y_scale='linear', idle=False,
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
    y_scale : {"log", "linear"}, default: None
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
    >>> from pyadlml.plot import plot_activity_bar_duration
    >>> plot_activity_bar_duration(data.df_activities)

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
    assert y_scale in ['linear', 'log']
    assert not (df_acts is None and df_dur is None)

    title = 'Cummulative activity durations'
    xlabel = 'seconds'
    freq = 'seconds'
    color = (get_primary_color() if color is None else color)

    if df_dur is None:
        if idle:
            df_acts = add_idle(df_acts.copy())
        df = activity_durations(df_acts, lst_acts=lst_acts, time_unit=freq)
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
    if y_scale == 'log':
        ax.set_xscale('log')
        
        
    # create secondary axis with time format 1s, 1m, 1d
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax_top.set_xlabel('time')
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_seconds2time))

    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig

def heatmap_transitions(df_acts=None, lst_acts=None, df_trans=None, z_scale="linear",
                        figsize=None, idle=False, numbers=True, grid=True,
                        cmap=None, file_path=None):
    """
    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : lst of str, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    df_trans : pd.DataFrame
        A precomputed transition table. If the *df_trans* parameter is given, parameters
        *df_acts* and *lst_acts* are ignored. The transition table can be computed
        in :ref:`stats <stats_acts_trans>`.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    z_scale : {"log", "linear"}, default: None
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
    >>> from pyadlml.plot import plot_activity_hm_transition
    >>> plot_activity_hm_transition(data.df_activities)

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
    assert z_scale in ['linear', 'log'], 'z-scale has to be either of type None or log'
    assert not (df_acts is None and df_trans is None)

    title = 'Activity transitions'
    z_label = 'count'

    if df_trans is None:
        df_acts = add_idle(df_acts) if idle else df_acts
        df = activities_transitions(df_acts, lst_acts=lst_acts)
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
    

    log = True if z_scale == 'log' else False
    valfmt = '{x:.0f}'
        
     # begin plotting
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap_square(values, y_labels, x_labels, log=log, cmap=cmap, ax=ax, cbarlabel=z_label, grid=grid)
    if numbers:
        texts = annotate_heatmap(im, textcolors=("white", "black"),log=log, valfmt=valfmt)
    ax.set_title(title)
    
    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig

def ridge_line(df_acts=None, lst_acts=None, df_act_dist=None, idle=False,
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
    >>> from pyadlml.plot import plot_activity_rl_daily_density
    >>> plot_activity_rl_daily_density(data.df_activities)

    .. image:: ../_static/images/plots/act_ridge_line.png
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
    assert not (df_acts is None and df_act_dist is None)

    title = 'Activity distribution over one day'
    xlabel = 'day'
    color = (get_primary_color() if color is None else color)


    if df_act_dist is None:
        if idle:
            df_acts = add_idle(df_acts)
        df = activities_dist(df_acts.copy(), lst_acts=lst_acts, n=n)
        if df.empty:
            raise ValueError("no activity was recorded and no activity list was given.")
    else:
        df = df_act_dist

    def date_2_second(date):
        """ maps time onto seconds of a day 
        Parameters
        ----------
        date : np.datetime64
            all the dates are on the day 1990-01-01

        Returns
        -------

        """
        if pd.isnull(date):
            return -1
        val = (date - np.datetime64('1990-01-01')) / np.timedelta64(1, 's')
        total_seconds = 60*60*24
        assert val <= total_seconds and val >= 0
        return int(val)

    
    df = df.apply(np.vectorize(date_2_second))
    # sort every columns values ascending
    for col in df.columns:
        df[col] = df[col].sort_values()

    grouped = [(col, df[col].values) for col in df.columns]
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
    
    # set xaxis labels
    def func(x,p):
        #x = x + 0.5
        #if x == 0.0 or str(x)[-1:] == '5':
        #    return ''
        #else:
        if True:
            if np.ceil(x/k) < 10:
                return '0{}:00'.format(int(x/k)+1)
            else:
                return '{}:00'.format(int(x/k)+1)
    a = 0
    b = 60*60*24
    k = (b-a)/24
    
    plt.xlim((a,b))
    tcks_pos = np.arange(0,23)*k + (-0.5 + k)
    
    x_locator = ticker.FixedLocator(tcks_pos)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(func))
    ax.xaxis.set_major_locator(x_locator)
    fig.autofmt_xdate(rotation=45)
    
    plt.grid(zorder=0)

    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig