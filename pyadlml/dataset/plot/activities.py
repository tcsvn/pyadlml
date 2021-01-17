import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from  pyadlml.dataset.stats.activities import activities_duration_dist, activity_durations,\
    activities_transitions, activities_count, activity_durations, activities_dist
from pyadlml.dataset.activities import add_idle 
from pyadlml.dataset.plot.util import func_formatter_seconds2time_log, ridgeline, \
    func_formatter_seconds2time, heatmap, annotate_heatmap, heatmap_square, savefig, \
    _num_bars_2_figsize, _num_boxes_2_figsize, \
    _num_items_2_heatmap_square_figsize, _num_items_2_ridge_figsize,\
    _num_items_2_ridge_ylimit
from pyadlml.util import get_sequential_color, get_secondary_color, get_primary_color, get_diverging_color




def hist_counts(df_act=None, df_ac=None, lst_act=None, y_scale=None, idle=False, figsize=None, color=None, file_path=None):
    """ bar chart displaying how often activities are occuring
    Parameters
    ----------
    df_act : pd.DataFrame or None
        Dataframe of all recorded activities
    df_ac : pd.DataFrame or None
        Statistic of activities
    y_scale : str or None
        If it is 'log' then scale y appropriately
    idle : bool
        indicates if the activity 'idle' should be inserted everywhere
        where there is no 
    file_path : String
        path where the image will be stored

    Returns
    -------
        Either a figure if file_path is not specified or nothing 
    """
    assert not (df_act is None and df_ac is None)
    assert y_scale in [None, 'log']

    title ='Activity occurrences'
    col_label = 'occurrence'
    xlabel = 'counts'
    color = (get_primary_color() if color is None else color)
        
    # create statistics if the don't exists
    if df_ac is None:
        df_act = df_act.copy()
        if idle:
            df_act = add_idle(df_act)
        df = activities_count(df_act, lst_activities=lst_act)
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
    plt.xlabel(xlabel)
    ax.barh(df['activity'], df[col_label], color=color)
    
    if y_scale == 'log':
        ax.set_xscale('log')

    # save or return fig
    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig


def boxplot_duration(df_act, lst_act=None, y_scale=None, idle=False, figsize=None, file_path=None):
    """ boxplot of activity durations (mean) max min
    Parameters
    ----------
    df_act : pd.DataFrame or None
        Dataframe of all recorded activities
    y_scale : str or None
        If it is 'log' then scale y appropriately
    idle : bool
        indicates if the activity 'idle' should be inserted everywhere
        where there is no other activity present
    file_path : String or None
        path where the image will be stored

    Returns
    -------
        Either a figu
    """
    assert y_scale in [None, 'log']
    
    title = 'Activity durations'
    xlabel = 'seconds'

    if idle:
        df_act = add_idle(df_act)

    df = activities_duration_dist(df_act, list_activities=lst_act)
    # select data for each device
    activities = df['activity'].unique()
    df['seconds'] = df['minutes']*60     

    num_act = len(activities)
    figsize = (_num_bars_2_figsize(num_act) if figsize is None else figsize)

    dat = []
    for activity in activities:
        df_activity = df[df['activity'] == activity]
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

def hist_cum_duration(df_act=None, act_lst=None, df_dur=None, y_scale=None, idle=False, figsize=None, color=None, file_path=None):
    """ plots the cummulated duration for each activity in a bar plot

    Parameters
    ----------
    df_act : pd.DataFrame or None
        Dataframe of all recorded activities
    df_dur : pd.DataFrame or None
        Dataframe of statistic
    y_scale : str or None
        If it is 'log' then scale y appropriately
    idle : bool
        indicates if the activity 'idle' should be inserted everywhere
        where there is no other activity present
    figsize : tuple (w,h)
        size of the figure
    file_path : String or None
        path where the image will be stored
    Returns
    -------
        Either a figu
    """
    assert y_scale in [None, 'log']
    assert not (df_act is None and df_dur is None)

    title = 'Cummulative activity durations'
    xlabel = 'seconds'
    freq = 'seconds'
    color = (get_primary_color() if color is None else color)

    if df_dur is None:
        if idle:
            df_act = add_idle(df_act.copy())
        df = activity_durations(df_act, list_activities=act_lst, freq=freq)
    else:
        df = df_dur
    df = df.sort_values(by=[freq], axis=0)

    num_act = len(df)
    figsize = (_num_bars_2_figsize(num_act) if figsize is None else figsize)
    
    # plot
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    ax.barh(df['activity'], df['seconds'], color=color)
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

def heatmap_transitions(df_act=None, lst_act=None, df_trans=None, z_scale=None, figsize=None, \
    idle=False, numbers=True, grid=True, cmap=None, file_path=None):
    """    """
    assert z_scale in [None, 'log'], 'z-scale has to be either of type None or log'
    assert not (df_act is None and df_trans is None)

    title = 'Activity transitions'
    z_label = 'count'

    if df_trans is None:
        df_act = add_idle(df_act) if idle else df_act
        df = activities_transitions(df_act, lst_act=lst_act)
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

def ridge_line(df_act=None, lst_act=None, act_dist=None, t_range='day', idle=False, \
        n=1000, ylim_upper=None, color=None, figsize=None, file_path=None):
    """
    Parameters
    ----------
    ylim_upper: float
        height that determines how many ridgelines are displayed. Adjust value to fit all 
        the ridgelines into the plot
    dist_scale: float
        the scale of the distributions of a ridgeline. 
    """
    assert not (df_act is None and act_dist is None)

    title = 'Activity distribution over one day'
    xlabel = 'day'
    color = (get_primary_color() if color is None else color)


    if act_dist is None:
        if idle:
            df_act = add_idle(df_act)
        df = activities_dist(df_act.copy(), lst_act=lst_act, t_range=t_range, n=n)
        if df.empty:
            raise ValueError("no activity was recorded and no activity list was given.")
    else:
        df = act_dist

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