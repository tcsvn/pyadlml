import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyadlml.constants import ACTIVITY, DEVICE, VALUE, START_TIME, END_TIME, TIME, BOOL, CAT, NUM
from pyadlml.dataset._core.devices import contains_non_binary, split_devices_binary, device_events_to_states
from pyadlml.dataset._core.activities import add_other_activity
from pyadlml.dataset.stats import contingency_states as stat_cont_st, contingency_events as stat_cont_ev, cross_correlogram as stat_cc
from .devices import _plot_device_states
from .util import func_formatter_seconds2time, save_fig, \
    heatmap_square, heatmap, annotate_heatmap, heatmap_contingency, map_time_to_numeric, \
    get_qualitative_cmap, plot_cc, create_todo, xaxis_format_time, xaxis_format_time2, \
    dev_raster_data_gen,plot_grid
from pyadlml.dataset.plot.plotly.util import format_device_labels 
from pyadlml.dataset.util import select_timespan, infer_dtypes, str_to_timestamp

from pyadlml.dataset.stats.activities import activity_order_by_duration

DEV_CON_HM = {(8, 12): (8, 7), (8, 14): (8, 8), (8, 3): (12, 10), (26, 20): (10, 10), (11, 17): (10, 10)}
DEV_CON_01_HM = {(8, 14): (12, 10), (11, 24): (12, 6), (11, 34): (16, 12)}
DEV_CON_01_HM_WT = {(10, 24): (14, 7), (7, 28): (16, 10)}


@save_fig
def cross_correlogram(df_devices, df_actvities, corr_data=[None, None, None, None], binsize='1s', maxlag='2m',
                      axis='off', figsize=(5, 5), file_path=""):
    #Plot cross-correlograms of all pairs.
    #   plotCCG(ccg,bins) plots a matrix of cross(auto)-correlograms for
    #   all pairs of clusters. Inputs are:
    #       ccg     array of cross correlograms           #bins x #clusters x #clusters
    #       bins    array with bin timings                #nbins x 0

    title = 'Cross-correlogram'

    if corr_data[0] is None or corr_data[1] is None:
        ccg, bins = stat_cc(df_devices, df_actvities, binsize=binsize, maxlag=maxlag)
        devices = df_devices[DEVICE].unique()
        activities = df_actvities[ACTIVITY].unique()
    else:
        ccg = corr_data[0]
        bins = corr_data[1]
        devices = corr_data[2]
        activities = corr_data[3]

    return plot_cc(ccg, bins, title=title, x_label=activities, y_label=devices, axis=axis, figsize=figsize)


@save_fig
def contingency_events(df_devs=None, df_acts=None, df_con_tab=None, other=False, per_state=False, \
                       scale=None, numbers=True, figsize=None, file_path="", n_jobs=1):
    """ Computes a table where the device triggers are counted per activity

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`. If the parameter *df_devs* is not set,
        the parameter *df_con_tab* has to be set.
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`. If the parameter *df_acts* is not set,
        the parameter *df_con_tab* has to be set.
    df_con_tab : pd.DataFrame, optional
        A precomputed contingency table. If the *df_con_tab* parameter is given, parameters
        *df_acts* and *df_devs* are ignored. The contingency table can be computed
        in :ref:`stats <stats_dna_con_dur>`.
    per_state : bool, default=False
        If set events are
    figsize : (float, float), optional
        width, height in inches. If not provided, the figsize is inferred by automatically.
    scale : {"log", "linear"}, default: 'log'
        The axis scale type to apply.
    numbers : bool, default: True
        Whether to display numbers inside the heatmaps fields or not.
    other : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *other* or be ignored.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.
    n_jobs : int, default=1
        The number of parallel threads to use for computing the statistics.

    Examples
    --------
    >>> from pyadlml.plot import plot_contingency_events
    >>> plot_contingency_events(data.df_activities, data.df_activities)

    .. image:: ../_static/images/plots/cont_hm_trigger.png
       :height: 300px
       :width: 500 px
       :scale: 100 %
       :alt: alternate text
       :align: center

    >>> contingency_events(data.df_devices, data.df_activities, per_state=True)

    .. image:: ../_static/images/plots/cont_hm_trigger.png
       :height: 300px
       :width: 500 px
       :scale: 100 %
       :alt: alternate text
       :align: center

    Returns
    -------
    fig : Figure or None
        If the parameter file_path is specified, the method return None rather than a matplotlib figure.

    """
    assert (df_devs is not None and df_acts is not None) or df_con_tab is not None

    title = 'Events'
    cbarlabel = 'counts'
    textcolors = ("white", "black")
    # if log than let automatically infer else
    log = (scale == 'log')
    valfmt = (None if log else "{x:.0f}")

    if df_con_tab is None:
        ct = stat_cont_ev(df_devs, df_acts, idle=other, per_state=per_state, n_jobs=n_jobs)
    else:
        ct = df_con_tab.copy()

    # First column are devices
    vals = ct.iloc[:, 1:].values.T
    acts = ct.columns[1:].values
    devs = ct[DEVICE].values

    # format labels
    dtypes = infer_dtypes(df_devs)
    if per_state:
        new_devs, new_order = format_device_labels(devs, dtypes, boolean_state=True, categorical_state=True)
    else:
        new_devs, new_order = format_device_labels(devs, dtypes)

    devs = list(new_devs)
    vals = vals[:, new_order]


    fig = heatmap_contingency(acts, devs, vals, title, cbarlabel, valfmt=valfmt,
                              textcolors=textcolors, z_scale=scale, numbers=numbers, figsize=figsize)

    return fig



@save_fig
def plot_contingency_states(df_devs: pd.DataFrame = None, df_acts: pd.DataFrame = None, df_con_tab: pd.DataFrame = None,
                            figsize: tuple = None, z_scale: str = 'log', idle: bool = False, numbers: bool = True,
                            file_path: str = "") -> object:
    """
    Plots a heatmap the device on and off intervals are measured against
    the activities

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`. If the parameter *df_devs* is not set,
        the parameter *df_con_tab* has to be set.
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`. If the parameter *df_acts* is not set,
        the parameter *df_con_tab* has to be set.
    df_con_tab : pd.DataFrame, optional
        A precomputed contingency table. If the *df_con_tab* parameter is given, parameters
        *df_acts* and *df_devs* are ignored. The contingency table can be computed
        in :ref:`stats <stats_dna_con_dur>`.
    figsize : (float, float), optional
        width, height in inches. If not provided, the figsize is inferred by automatically.
    z_scale : {"log", "linear"}, default: 'log'
        The axis scale type to apply.
    numbers : bool, default: True
        Whether to display numbers inside the heatmaps fields or not.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.dataset import fetch_amsterdam
    >>> from pyadlml.plot import contingency_states
    >>> data = fetch_amsterdam()
    >>> fig = contingency_states(data.df_devices, data.df_activities)
    >>> fig.show()

    .. image:: ../_static/images/plots/cont_hm_duration.png
       :height: 300px
       :width: 800 px
       :scale: 90 %
       :alt: alternate text
       :align: center

    Returns
    -------
    fig : Figure or None
        If the parameter file_path is specified, the method returns None rather than a matplotlib figure.

    """

    assert (df_devs is not None and df_acts is not None) or df_con_tab is not None

    title = 'Activity vs. device states'
    cbarlabel = 'time in seconds'

    if df_con_tab is None:
        if idle:
            df_acts = add_other_activity(df_acts.copy())
        df_con = stat_cont_st(df_devs, df_acts)
    else:
        df_con = df_con_tab

    # convert time (ns) to seconds
    df_con = df_con.astype(int)/1000000000

    devs = df_con.index.values
    vals = df_con.values.T
    acts = df_con.columns.values

    # sort values and format device state labels
    if df_devs is not None:
        dtypes = infer_dtypes(df_devs)
    else:
        dtypes = {BOOL: [d.split(':')[0] for d in devs if (':on' in d) or (':off' in d)],
                  CAT: [d.split(':')[0] for d in devs if not (':on' in d) and not (':off' in d)],
                  NUM: []}
    dtypes.pop(NUM, None)
    new_devs, new_order = format_device_labels(devs, dtypes, boolean_state=True, categorical_state=True)
    devs = list(new_devs)
    vals = vals[:, new_order]


    valfmt = matplotlib.ticker.FuncFormatter(lambda x, p: func_formatter_seconds2time(x, p))
    
    return heatmap_contingency(acts, devs, vals, title, cbarlabel,
             valfmt=valfmt, figsize=figsize, z_scale=z_scale,
             numbers=numbers)


@save_fig
def activities_and_device_events(df_devices, df_activities, start_time=None, end_time=None,
                                 figsize=(20, 8), grid=False, file_path=""):
    """ Plots the devices events as raster combined with the activites over a timespan

    Parameters
    ----------
    df_devices : pd.DataFrame
        A device dataframe
    df_activities : pd.DataFrame
        An activity dataframe
    start_time : str or None, default=None
        When set, a string representing the time where the plots starts
    end_time : str or None, default=None
        When set, a string representing the time where the plots ends
    figsize : tuple, default=(20,8)
        The plots figure given in (width, height)
    file_path : str, default=""
        When set the file

    Returns
    ------
    fig or None
        Either a figure or nothing depending on if file_path is set.
    """
    title='Activities vs. device events'
    y_label = 'Devices'
    event_color = 'grey'

    activity_y_label = 'Activities'
    act_offset_space = 0.25
    act_bar_height = 1


    df_acts = df_activities.copy()
    df_devs = df_devices.copy()\
                        .sort_values(by=TIME)\
                        .reset_index(drop=True)

    devs = df_devs[DEVICE].unique()
    acts = activity_order_by_duration(df_acts)

    start_time = str_to_timestamp(start_time) if (start_time is not None) else None
    end_time = str_to_timestamp(end_time) if (end_time is not None) else None
    df_devs, df_acts = select_timespan(df_devs, df_acts, start_time, end_time, clip_activities=True)
    if start_time is None:
        start_time = min(df_devs[TIME].iloc[0], df_acts[START_TIME].iloc[0])
    if end_time is None:
        end_time = max(df_devs[TIME].iloc[-1], df_acts[END_TIME].iloc[-1])


    df_devs['num_time'], start_time, end_time = map_time_to_numeric(df_devs[TIME], start_time, end_time)
    data_lst = dev_raster_data_gen(df_devs, devs)

    fig, ax = plt.subplots(figsize=figsize)

    tab = get_qualitative_cmap(len(acts))
    plot_activity_bar(ax, df_acts, acts, start_time, end_time, act_bar_height,
                      len(devs)+act_offset_space, tab(np.arange(0, len(acts))))

    ax.eventplot(data_lst, linelengths=[0.2]*len(devs), colors=event_color)


    ax.set_title(title)
    yticks = np.arange(len(devs)+1, dtype=np.float32)
    yticks[-1] += act_offset_space + act_bar_height/2    # correct offset for activities
    ax.set_yticks(yticks)
    ax.set_yticklabels(devs.tolist() + [activity_y_label])
    ax.set_ylabel(y_label)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_xlim([0, 1])

    xaxis_format_time2(fig, ax, start_time, end_time)

    if grid:
        plot_grid(fig, ax, start_time, end_time)

    return fig





@save_fig
def activities_and_device_states(df_devices: pd.DataFrame, df_activities: pd.DataFrame,
                                 start_time: str = None, end_time: str = None,
                                 figsize=(20, 8), grid=False, file_path=""):
    """ Plots devices events as raster combined with activities over time

    Parameters
    ----------
    df_devices : pd.DataFrame
        A device dataframe
    df_activities : pd.DataFrame
        An activity dataframe
    start_time : str or None, default=None
        When set, a string representing the time where the plots starts
    end_time : str or None, default=None
        When set, a string representing the time where the plots ends
    figsize : tuple, default=(20,8)
        The plots figure given in (width, height)
    file_path : str, default=""
        When set the file

    Returns
    ------
    fig or None
        Either a figure or nothing depending on if file_path is set.
    """

    ylabel = 'Devices'
    title = 'Device states'
    binary_on_label = 'on'
    binary_off_label = 'off'
    activity_y_label = 'Activities'

    color_dev_on = 'seagreen'
    color_dev_off = 'lightgrey'
    color_dev_num = 'blue'

    act_offset_space = 0.25
    act_bar_height = 1

    df_acts = df_activities.copy()
    df_devs = df_devices.copy()\
                        .sort_values(by=TIME)\
                        .reset_index(drop=True)


    start_time = str_to_timestamp(start_time) if (start_time is not None) else None
    end_time = str_to_timestamp(end_time) if (end_time is not None) else None
    df_devs, df_acts = select_timespan(df_devs=df_devs, df_acts=df_acts,
                                       start_time=start_time, end_time=end_time,
                                       clip_activities=True)


    if start_time is None:
        start_time = min(df_devs[TIME].iloc[0], df_acts[START_TIME].iloc[0])
    if end_time is None:
        end_time = max(df_devs[TIME].iloc[-1], df_acts[END_TIME].iloc[-1])

    devs = df_devs[DEVICE].unique()
    acts = activity_order_by_duration(df_acts)

    fig, ax = plt.subplots(figsize=figsize)
    tab = get_qualitative_cmap(len(acts))

    if grid:
        plot_grid(fig, ax, start_time, end_time)

    j = _plot_device_states(ax, df_devs, devs, start_time, end_time, color_dev_on, color_dev_off,
                           color_dev_num, binary_off_label, binary_on_label, return_nr_categories_used=True)

    plot_activity_bar(ax, df_acts, acts, start_time, end_time, act_bar_height,
                      len(devs)+act_offset_space,
                      tab(np.arange(j, len(acts)+j))
    )

    ax.set_title(title)
    yticks = np.arange(len(devs)+1, dtype=np.float32)
    yticks[-1] += act_offset_space + act_bar_height/2    # correct offset for activities

    ax.set_yticks(yticks)
    ax.set_yticklabels(devs.tolist() + [activity_y_label])
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, 1])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left")
    xaxis_format_time2(fig, ax, start_time, end_time)

    return fig

def plot_activity_bar(ax, df_acts: pd.DataFrame, acts: list, start_time, end_time, act_bar_height, y_pos, c_map):
    """ Plot a horizontal activity event representation.
    Parameters
    ----------
    c_map : np.ndarray
        Array of colors where each color corresponds to one activity
    """
    df_acts['num_st'], _,_ = map_time_to_numeric(df_acts[START_TIME], start_time, end_time)
    df_acts['num_et'], _,_ = map_time_to_numeric(df_acts[END_TIME], start_time, end_time)
    df_acts['diff'] = df_acts['num_et'] - df_acts['num_st']

    # create list of lists, where each list corresponds to an activity with tuples of start_time and time_length
    xranges = []
    for i, act in enumerate(acts):
        xranges.append(df_acts.loc[df_acts[ACTIVITY] == act, ['num_st', 'diff']].values.tolist())

    for i in range(len(acts)):
        ax.broken_barh(xranges[i], (y_pos, act_bar_height), facecolors=c_map[i], label=acts[i])



def plot_events_over_act_hist(df_acts: pd.DataFrame, df_devs: pd.DataFrame, activity: str, device: str, dt: str = '1s') -> plt.Figure:
    """ Plot a histogram when events happen centered around an activity

    Parameters
    ----------
    df_acts : pd.DataFrame
        An activity dataframe with columns: 'start_time', 'end_time' and 'activity'


    Returns
    -------
    Figure or None

    """
    pass