import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from  pyadlml.dataset.stats.activities import activities_duration_dist, activities_durations,\
    activities_transitions, activities_count, activities_durations, activities_dist
from pyadlml.dataset.activities import add_idle 
from pyadlml.dataset.plot.util import func_formatter_log, func_formatter_min, ridgeline, \
    func_formatter_sec, heatmap, annotate_heatmap, heatmap_square, hm_key_NN

ACT_BAR_HM = {  7:(8,4), 9:(9,5), 10:(9,5), 12:(7,5), 22:(10,10), 23:(10,9), 
                26:(10,11)}
ACT_BP_HM = {   7:(8,4), 9:(9,5), 11:(10,6), 22:(10,10), 23: (10,10), 26:(10,11)}
ACT_HM_HM = {   9:(5,5), 10:(6,6), 11:(8,6), 22:(10,10), 26:(11,11)}
ACT_RDG_HM = {  8:(10,8), 11:(10,11), 22:(10,12), 24:(10,12)}

ACT_RDG_YL = {  8:1.1, 11:1.75, 22:3.5, 24:(3.7)}
ACT_RDG_SC = {  8:0.1, 11:0.12, 22:0.16, 24:0.15}


def num_activity_2_figsize(hm, figsize, df):
    if figsize is not None:
        return figsize
    if isinstance(df, pd.DataFrame):
        num_act = len(df['activity'].unique())
    elif isinstance(df, list):
        num_act = len(df)
    else:
        raise ValueError
    return hm_key_NN(hm, num_act)

def hist_counts(df_act=None, df_ac=None, y_scale=None, idle=False, figsize=None):
    """ plots the activities durations against each other
    """
    assert not (df_act is None and df_ac is None)
    assert y_scale in [None, 'log']

    title ='Activity occurrences'
    col_label = 'occurence'
    xlabel = 'counts'
    if df_ac is None:
        df_act = df_act.copy()
        if idle:
            df_act = add_idle(df_act)
        df = activities_count(df_act)
        figsize = num_activity_2_figsize(ACT_BAR_HM, figsize, df_act)
    else: 
        df = df_ac
        figsize = num_activity_2_figsize(ACT_BAR_HM, figsize, list(df['activity']))

    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by=['occurence'], axis=0)
    
    # plot
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    ax.barh(df['activity'], df['occurence'])
    if y_scale == 'log':
        ax.set_xscale('log')
    return fig


def boxplot_duration(df_act, y_scale='norm', idle=False, figsize=None):
    """
        plot a boxplot of activity durations (mean) max min 
    """
    assert y_scale in ['norm', 'log']
    title = 'Activity durations'
    xlabel = 'log seconds'

    if idle:
        df_act = add_idle(df_act)

    figsize = num_activity_2_figsize(ACT_BP_HM, figsize, df_act)
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
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(dat, vert=False)
    ax.set_title(title)
    ax.set_yticklabels(activities, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_xscale('log')

    # create secondary axis with 

    # create secondary axis with time format 1s, 1m, 1d
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    #ax_top.set_xlabel('time')
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_sec))
    return fig

def hist_cum_duration(df_act=None, df_dur=None, y_scale=None, idle=False, figsize=None):
    """ plots the cummulated activities durations in a histogram for each activity 
    """
    assert y_scale in [None, 'log']
    assert not (df_act is None and df_dur is None)

    title = 'Cummulative activity durations'

    if y_scale == 'log':
        xlabel = 'log seconds'
    else: 
        xlabel = 'seconds'

    if df_dur is None:
        if idle:
            df_act = add_idle(df_act.copy())
        act_dur = activities_durations(df_act)
        figsize = num_activity_2_figsize(ACT_BAR_HM, figsize, df_act)
    else:
        act_dur = df_dur
        figsize = num_activity_2_figsize(ACT_BAR_HM, figsize, list(act_dur['activity']))
    df = act_dur[['minutes']]
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by=['minutes'], axis=0)
    # TODO change in activities duration to return time in seconds
    df['seconds'] = df['minutes']*60 
    
    # plot
    fig, ax = plt.subplots(figsize=figsize)
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


def heatmap_transitions(df_act=None, df_trans=None, z_scale=None, figsize=None, \
    idle=False, numbers=True, grid=True):
    """    """
    assert z_scale in [None, 'log'], 'z-scale has to be either of type None or log'
    assert not (df_act is None and df_trans is None)

    title = 'Activity transitions'
    z_label = 'count'

    if df_trans is None:
        df_act = add_idle(df_act) if idle else df_act
        df = activities_transitions(df_act)
    else:
        df = df_trans

    # get the list of cross tabulations per t_window
    act_lst = list(df.columns)
    figsize = num_activity_2_figsize(ACT_HM_HM, figsize, act_lst)
    x_labels = act_lst
    y_labels = act_lst
    values = df.values
    

    log = True if z_scale == 'log' else False
    valfmt = '{x:.0f}'
        
     # begin plotting
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap_square(values, y_labels, x_labels, log=log, ax=ax, cbarlabel=z_label, grid=grid)
    if numbers:
        texts = annotate_heatmap(im, textcolors=("white", "black"),log=log, valfmt=valfmt)
    ax.set_title(title)
    
    return fig



def ridge_line(df_act=None, act_dist=None, t_range='day', idle=False, \
        n=1000, dist_scale=None, ylim_upper=None, figsize=None):
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
    if act_dist is None:
        if idle:
            df_act = add_idle(df_act)
        df = activities_dist(df_act.copy(), t_range, n)
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
    figsize = num_activity_2_figsize(ACT_RDG_HM, figsize, list(acts))
    if dist_scale is None:
        dist_scale = hm_key_NN(ACT_RDG_SC, len(acts))
    if ylim_upper is None:
        ylim_upper = hm_key_NN(ACT_RDG_YL, len(acts))

    fig, ax = plt.subplots(figsize=figsize)
    ridgeline(data, labels=acts, overlap=.85, fill='tab:blue', n_points=100, dist_scale=dist_scale)
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