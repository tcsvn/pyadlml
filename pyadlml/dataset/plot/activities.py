from pyadlml.dataset.stats.activities import activities_count
import matplotlib.pyplot as plt
from  pyadlml.dataset.stats.activities import activities_duration_dist, activities_durations
import matplotlib.ticker as ticker
from pyadlml.dataset.plot.util import func_formatter_log, func_formatter_min
from pyadlml.dataset.stats.activities import activities_durations
from pyadlml.dataset.plot.util import func_formatter_sec, ridgeline
from pyadlml.dataset.stats.activities import activities_transitions
from pyadlml.dataset.plot import _heatmap, _annotate_heatmap
from pyadlml.dataset.stats.activities import activities_dist
from pyadlml.dataset.activities import add_idle 
import numpy as np


def hist_counts(df_act, y_scale=None, idle=False):
    """ plots the activities durations against each other
    """
    assert y_scale in [None, 'log']
    df_act = df_act.copy()

    col_label = 'occurence'
    title ='Activity occurrences'
    xlabel = 'counts'

    if idle:
        df_act = add_idle(df_act)
    df = activities_count(df_act)
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by=['occurence'], axis=0)
    
    # plot
    fig, ax = plt.subplots(figsize=(9,3))
    plt.title(title)
    plt.xlabel(xlabel)
    ax.barh(df['activity'], df['occurence'])
    if y_scale == 'log':
        ax.set_xscale('log')
    return fig


def boxplot_duration(df_act, y_scale='norm', idle=False):
    """
        plot a boxplot of activity durations (mean) max min 
    """
    assert y_scale in ['norm', 'log']

    if idle:
        df_act = add_idle(df_act)
    df = activities_duration_dist(df_act)  
    
    # select data for each device
    activities = df['activity'].unique()
    df['seconds'] = df['minutes']*60 

    dat = []
    for activity in activities:
        df_activity = df[df['activity'] == activity]
        #tmp = np.log(df_device['td'].dt.total_seconds())
        dat.append(df_activity['seconds']) 
    
    # plot boxsplot
    fig, ax = plt.subplots(figsize=(10,8))
    ax.boxplot(dat, vert=False)
    ax.set_title('Activity durations')
    ax.set_yticklabels(activities, ha='right')
    ax.set_xlabel('log seconds')
    ax.set_xscale('log')

    # create secondary axis with 

    # create secondary axis with time format 1s, 1m, 1d
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    #ax_top.set_xlabel('time')
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_sec))
    return fig


def hist_cum_duration(df_act, y_scale=None, idle=False):
    """ plots the cummulated activities durations in a histogram for each activity 
    """
    assert y_scale in [None, 'log']

    title = 'Cummulative activity durations'
    if y_scale == 'log':
        xlabel = 'log seconds'
    else: 
        xlabel = 'seconds'
    if idle:
        df_act = add_idle(df_act)

    act_dur = activities_durations(df_act)
    df = act_dur[['minutes']]
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by=['minutes'], axis=0)
    # TODO change in activities duration to return time in seconds
    df['seconds'] = df['minutes']*60 
    
    # plot
    fig, ax = plt.subplots(figsize=(9,3))
    plt.title(title)
    plt.xlabel(xlabel)
    ax.barh(df['activity'], df['seconds'])
    if y_scale == 'log':
        ax.set_xscale('log')
        
        
    # create secondary axis with time format 1s, 1m, 1d
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax_top.set_xlabel('time')
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_sec))
    return fig


def heatmap_transitions(df_act, z_scale=None, figsize=(8,6), idle=False):
    """    """
    assert z_scale in [None, 'log'], 'z-scale has to be either of type None or log'

    title = 'Activity transitions'
    if idle:
        df_act = add_idle(df_act)
   
    # get the list of cross tabulations per t_window
    df = activities_transitions(df_act)
    act_lst = list(df.columns)
    x_labels = act_lst
    y_labels = act_lst
    values = df.values
    

    if z_scale == 'log':
        zlabel = 'log count'
        values = np.log(values)
        valfmt = '{x:.2f}'
    else:
        zlabel = 'count'
        valfmt = '{x}'
        
     # begin plotting
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = _heatmap(values, y_labels, x_labels, ax=ax, cmap='viridis', cbarlabel=zlabel)
    texts = _annotate_heatmap(im, textcolors=("white", "black"), valfmt=valfmt)
    ax.set_title(title)

    
    return fig



def ridge_line(df_act, t_range='day', idle=False, n=1000, dist_scale=0.05, ylim_upper=1.1):
    """
    Parameters
    ----------
    ylim_upper: float
        height that determines how many ridgelines are displayed. Adjust value to fit all 
        the ridgelines into the plot
    dist_scale: float
        the scale of the distributions of a ridgeline. 
    """
    if idle:
        df_act = add_idle(df_act)
 

    def date_2_second(date):
        """ maps time onto seconds of a day 
        Parameters
        ----------
        date : np.datetime64
            all the dates are on the day 1990-01-01

        Returns
        -------

        """
        val = (date - np.datetime64('1990-01-01')) / np.timedelta64(1, 's')
        total_seconds = 60*60*24
        assert val <= total_seconds and val >= 0
        return int(val)
    title = 'Activity distribution over one day'
    
    df = activities_dist(df_act.copy(), t_range, n)
    df = df.apply(np.vectorize(date_2_second))
    # sort every columns values ascending
    for col in df.columns:
        df[col] = df[col].sort_values()

    grouped = [(col, df[col].values) for col in df.columns]

    fig, ax = plt.subplots(figsize=(10, 8))
    acts, data = zip(*grouped)
    ridgeline(data, labels=acts, overlap=.85, fill='tab:blue', n_points=1000, dist_scale=dist_scale)
    plt.title(title)

    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylim((0, 1.1))
    plt.xlabel('day')
    
    # set xaxis labels
    def func(x,p):
        #x = x + 0.5
        #if x == 0.0 or str(x)[-1:] == '5':
        #    return ''
        #else:
        if True:
            if int(x/k) < 10:
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
    plt.show()