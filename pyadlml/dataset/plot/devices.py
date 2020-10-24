import numpy as np
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors

from pyadlml.dataset.stats.devices import duration_correlation, \
    trigger_time_diff, device_tcorr, device_triggers_one_day, \
    devices_trigger_count, devices_on_off_stats

from pyadlml.dataset.plot.util import heatmap_square, hm_key_NN, \
    func_formatter_sec, heatmap, annotate_heatmap

DEV_BAR_HM = {10:(9,5), 12:(10,6), 14: (10,6), 20: (10,9), 22:(10,9),40:(10,15), 72:(10,21)}
DEV_BP_HM = {12:(9,6), 20:(10,9), 22:(10,8), 72:(11,22)}
DEV_HM_HM = {12:(8,8), 20:(9,9), 22:(13,13), 40:(15,15),72:(23,22)}
DEV_HM_TIME = {12:(10,6), 20:(10,8)}

def num_device_2_figsize(hm, figsize, df):
    if figsize is not None:
        return figsize
    if isinstance(df, pd.DataFrame):
        num_dev = len(df['device'].unique())
    elif isinstance(df, list):
        num_dev = len(df)
    else:
        raise ValueError
    return hm_key_NN(hm, num_dev)


def hist_trigger_time_diff(df_dev=None, x=None, n_bins=50, figsize=(10,6)):
    """
        plots
    """
    assert not (df_dev is None and x is None)
    title='Time difference between device triggers'
    log_sec_col = 'total_log_secs'
    sec_col = 'total_secs'

    if x is None:
        X = trigger_time_diff(df_dev.copy())
    else:
        X = x

    # make equal bin size from max to min
    bins = np.logspace(min(np.log10(X)), max(np.log10(X)), n_bins)

    # make data ready for hist
    hist, _ = np.histogram(X, bins=bins)
    cum_percentage = hist.cumsum()/hist.sum()
    cum_percentage = np.concatenate(([0], cum_percentage)) # let the array start with 0

    # plots
    fig,ax = plt.subplots(figsize=figsize)
    plt.xscale('log')
    #plt.yscale('log')
    
    ax.hist(X, bins=bins, label='amount of trigger with set difference')
    ax.set_ylabel('count')
    ax.set_xlabel('log seconds')
    
    # create axis for line
    ax2=ax.twinx()
    ax2.plot(bins, cum_percentage, 'r', label='% of data to the left')
    ax2.set_ylabel('%')
    ax2.set_xscale('log')
    
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_sec))
    
    # plot single legend for multiple axis
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='center right')
    
    plt.title(title, y=1.08)
    
    return fig

def boxsplot_on_duration(df_dev, figsize=None):
    """
    draws a boxsplot of all devices
    Parameters
    ----------
        df_dev: pd.DataFrame
            device dataframe in representation 1 
    """
    title = 'Devices on-duration'
    figsize = num_device_2_figsize(DEV_BP_HM, figsize, df_dev)    
    xlabel = 'log seconds'
    xlabel_top = 'time'


    # create duration differences
    df_dev = df_dev.copy()
    df_dev['td'] = df_dev['end_time'] - df_dev['start_time']

    # select data for each device
    devices = df_dev['device'].unique()
    dat = []
    for device in devices:
        df_device = df_dev[df_dev['device'] == device]
        #tmp = np.log(df_device['td'].dt.total_seconds())
        tmp = df_device['td'].dt.total_seconds()
        dat.append(tmp)
    
    #return df_dev #DEBUG
    # plot boxsplot
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(dat, vert=False)
    ax.set_title(title)
    ax.set_yticklabels(devices, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_xscale('log')

    # create secondary axis with 

    # create secondary axis with time format 1s, 1m, 1d
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax_top.set_xlabel(xlabel_top)
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_sec))
    return fig

def heatmap_trigger_one_day(df_dev=None, df_tod=None, t_res='1h', figsize=None):
    """
    computes the heatmap for one day where all the device triggers are showed
    """
    assert not (df_dev is None and df_tod is None)
    title = "Device triggers cummulative over one day"
    xlabel =  'time'
    figsize = num_device_2_figsize(DEV_HM_TIME, figsize, df_dev)

    if df_tod is None:
        df = device_triggers_one_day(df_dev.copy(), t_res)
    else:
        df = df_tod

    x_labels = list(df.index)
    y_labels = df.columns
    dat = df.values.T
    
    # begin plotting
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap(dat, y_labels, x_labels, ax=ax, cmap='viridis', cbarlabel='counts')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    
    # format the x-axis
    def func(x,p):
        if True:
            if int(x/k) < 10:
                return '0{}:00'.format(int(x/k)+1)
            else:
                return '{}:00'.format(int(x/k)+1)
    
    # calculate the tick positions 
    a,b = ax.get_xlim()
    k = (b-a)/24
    tcks_pos = np.arange(0,23)*k + (-0.5 + k)
    
    x_locator = ticker.FixedLocator(tcks_pos)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(func))
    ax.xaxis.set_major_locator(x_locator)
    ax.set_aspect(aspect='auto')
    return fig

def heatmap_trigger_time(df_dev=None, df_tcorr=None, t_window='5s', figsize=None, z_scale=None):
    """
    """
    assert not (df_dev is None and df_tcorr is None)
    title = "Triggercount with sliding window of " + t_window

    color = 'trigger count'
    cbarlabel = 'counts'
    figsize = num_device_2_figsize(DEV_HM_HM, figsize, df_dev)

    if df_tcorr is None:
        df = device_tcorr(df_dev, t_window)[0]
    else:
        df = df_tcorr

    # get the list of cross tabulations per t_window
    vals = df.astype(int).values.T
    devs = list(df.index)


    fig, ax = plt.subplots(figsize=figsize)

    log = True if z_scale == 'log' else False       
    valfmt = "{x:.0f}"
        
    im, cbar = heatmap_square(vals, devs, devs, ax=ax, #cmap='viridis', 
                        cbarlabel=cbarlabel, log=log)#, cbar_kw=cbar_kw)
    
    texts = annotate_heatmap(im, textcolors=("white", "black"), log=log, valfmt=valfmt)

    ax.set_title(title)
    fig.tight_layout()
    return fig

def heatmap_cross_correlation(df_dev=None, df_dur_corr=None, figsize=None, parallel=True, numbers=True):
    """ plots the cross correlation between the device signals
    Parameters
    ----------
    df_dev: pd.DataFrame 
        devices in representation 1
    """
    assert not (df_dev is None and df_dur_corr is None)

    title = 'Cross-correlation of signals'
    cmap = 'BrBG'
    cbarlabel = 'counts'
    
    if df_dur_corr is None:
        if parallel:
            from pyadlml.dataset.stats.devices import duration_correlation_parallel
            ct = duration_correlation_parallel(df_dev)
        else:
            ct = duration_correlation(df_dev)
    else:
        ct = df_dur_corr 
    vals = ct.values.T
    devs = list(ct.index)

    figsize = num_device_2_figsize(DEV_HM_HM, figsize, devs)
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap_square(vals, devs, devs, ax=ax, cmap=cmap, cbarlabel=cbarlabel,
                       vmin=-1, vmax=1)
    if numbers:
        texts = annotate_heatmap(im, textcolors=("black", "white"), 
                             threshold=0.5, valfmt="{x:.2f}")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def hist_on_off(df_dev=None, df_onoff=None, figsize=None):
    """ plots the percentage a device is on against the percentage it is off
    Parameters
    ----------
    df_dev: Dataframe
        device list in representation 1
    """
    assert not (df_dev is None and df_onoff is None)

    title = 'Devices fraction on/off'
    xlabel ='Percentage in binary states' 
    ylabel = 'Devices'

    figsize = num_device_2_figsize(DEV_BAR_HM, figsize, df_dev)

    if df_onoff is None:
        df = devices_on_off_stats(df_dev)
    else:
        df = df_onoff
    df = df.sort_values(by='frac_on', axis=0)
    dev_lst = list(df.index)
    # Figure Size 
    fig, ax = plt.subplots(figsize=figsize) 
    plt.barh(dev_lst, df['frac_off'].values, label='off')  

    # careful: notice "bottom" parameter became "left"
    plt.barh(dev_lst,  df['frac_on'].values, left=df['frac_off'], label='on')

    # we also need to switch the labels
    plt.title(title)
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)
    
    # set 
    widths = df['frac_off']
    xcenters = widths/2
    text_color='white'
    for y, (x, c) in enumerate(zip(xcenters, widths)):
        ax.text(x, y, '{:.4f}'.format(c), ha='center', va='center',
                color=text_color)
    
    ax.legend(ncol=2, bbox_to_anchor=(0, 1),
              loc='upper left', fontsize='small')
    
    # Remove axes splines 
    for s in ['top', 'right']: 
        ax.spines[s].set_visible(False) 

    plt.show()  

def hist_counts(df_dev=None, df_tc=None, figsize=None, y_scale=None):
    """
    plots the trigger count of each device 
    """
    assert not (df_dev is None and df_tc is None)
    title = 'Count of on/off activations per Device'
    col_label = 'trigger count'
    col_device = 'device'

    figsize = num_device_2_figsize(DEV_BAR_HM, figsize, df_dev)

    if df_tc is None:
        df = devices_trigger_count(df_dev.copy())
    else:
        df = df_tc

    df.reset_index(level=0, inplace=True)
    df.columns = ['device', col_label]

    df = df.sort_values(by=col_label, axis=0, ascending=True)
    
    # plot
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    plt.xlabel(col_label)
    ax.barh(df['device'], df['trigger count'])
    if y_scale == 'log':
        ax.set_xscale('log')
    return fig