import numpy as np
from pyadlml.dataset.util import print_df
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from pyadlml.dataset.stats.devices import duration_correlation, \
    devices_trigger_time_diff, device_tcorr, device_triggers_one_day

from pyadlml.dataset.stats.devices import devices_on_off_stats
# TODO move to plot.util
from pyadlml.dataset.plot import _heatmap, _annotate_heatmap
from pyadlml.dataset.plot.util import heatmap, annotate_heatmap
from pyadlml.dataset.stats.devices import devices_trigger_time_diff
from pyadlml.dataset.plot.util import func_formatter_sec
from datetime import timedelta
import matplotlib.ticker as ticker
from pyadlml.dataset.stats.devices import devices_trigger_count
from pyadlml.dataset.plot.util import heatmap_square


def hist_trigger_time_diff(df_dev, n_bins=50):
    """
        plots
    """
    title='Time difference between device triggers'
    log_sec_col = 'total_log_secs'
    sec_col = 'total_secs'
    df = devices_trigger_time_diff(df_dev.copy())
    
    # convert timedelta to total minutes
    df[sec_col] = df['row_duration']/timedelta(seconds=1)

    X = df[sec_col].values[:-1]
    
    # make equal bin size from max to min
    bins = np.logspace(min(np.log10(X)), max(np.log10(X)), n_bins)

    # make data ready for hist
    hist, _ = np.histogram(X, bins=bins)
    cum_percentage = hist.cumsum()/hist.sum()
    cum_percentage = np.concatenate(([0], cum_percentage)) # let the array start with 0

    # plots
    fig,ax = plt.subplots(figsize=(10,6))
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

def boxsplot_on_duration(df_dev):
    """
    draws a boxsplot of all devices
    Parameters
    ----------
        df_dev: pd.DataFrame
            device dataframe in representation 1 
    """
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

    # plot boxsplot
    fig, ax = plt.subplots(figsize=(10,8))
    ax.boxplot(dat, vert=False)
    ax.set_title('Devices "ON"-Duration')
    ax.set_yticklabels(devices, ha='right')
    ax.set_xlabel('log seconds')
    ax.set_xscale('log')

    # create secondary axis with 

    # create secondary axis with time format 1s, 1m, 1d
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax_top.set_xlabel('time')
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(_func_formatter))
    return fig

def _func_formatter(x, pos):
    if x-60 < 0:
        return "{:.0f}s".format(x)
    elif x-3600 < 0:
        return "{:.0f}m".format(x/60)
    elif x-86400 < 0:
        return "{:.0f}h".format(x/3600)
    else:
        return "{:.0f}t".format(x/86400)


def heatmap_trigger_one_day(df_dev, t_res='1h', figsize=(10,6)):
    """
    computes the heatmap for one day where all the device triggers are showed
    """
    df = device_triggers_one_day(df_dev.copy(), t_res)
    x_labels = list(df.index)
    y_labels = df.columns
    dat = df.values.T
    
    # begin plotting
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap(dat, y_labels, x_labels, ax=ax, cmap='viridis', cbarlabel='counts')
    ax.set_title("Device triggers cummulative over one day")
    ax.set_xlabel('time')
    
    # format the x-axis
    def func(x,p):
        if True:
            if int(x/k) < 10:
                return '0{}:00'.format(int(x/k)+1)
            else:
                return '{}:00'.format(int(x/k)+1)
    
    # calculate the tick positions 
    a,b = ax.get_xlim()
    k = (b-a)/23
    tcks_pos = np.arange(0,23)*k + (-0.5 + k)
    
    x_locator = ticker.FixedLocator(tcks_pos)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(func))
    ax.xaxis.set_major_locator(x_locator)
    ax.set_aspect(aspect='auto')
    return fig



def heatmap_trigger_time(df_dev, t_window = '5s', figsize=(8,8), z_scale=None):
    color = 'trigger count'
    cbarlabel = 'counts'
    
    # get the list of cross tabulations per t_window
    df = device_tcorr(df_dev, t_window)[0]
    vals = df.astype(int).values.T
    devs = list(df.index)


    fig, ax = plt.subplots(figsize=figsize)

    log = True if z_scale == 'log' else False       
    valfmt = "{x:.0f}"
        
    im, cbar = heatmap_square(vals, devs, devs, ax=ax, #cmap='viridis', 
                        cbarlabel=cbarlabel, log=log)#, cbar_kw=cbar_kw)
    
    texts = annotate_heatmap(im, textcolors=("white", "black"), log=log, valfmt=valfmt)


    ax.set_title("Triggercount with sliding window of " + t_window)
    fig.tight_layout()
    plt.show()

def heatmap_cross_correlation(df_dev, figsize=(10,8)):
    ct = duration_correlation(df_dev)
    vals = ct.values.T
    devs = list(ct.index)

    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = _heatmap(vals, devs, devs, ax=ax, cmap='PuOr', cbarlabel='counts',
                       vmin=-1, vmax=1)

    texts = _annotate_heatmap(im, textcolors=("black", "white"), valfmt="{x:.2f}")

    ax.set_title("Cross-correlation of signals")
    fig.tight_layout()
    plt.show()


def hist_on_off(df_dev):
    """ plots the percentage a device is on against the percentage it is off
    Parameters
    ----------
    df_dev: Dataframe
        device list in representation 1
    """
    df = devices_on_off_stats(df_dev)
    df = df.sort_values(by='frac_on', axis=0)
    dev_lst = list(df.index)
    title = 'Devices fraction on/off'

    # Figure Size 
    fig, ax = plt.subplots(figsize =(13, 9)) 
    plt.barh(dev_lst, df['frac_off'].values, label='off')  

    # careful: notice "bottom" parameter became "left"
    plt.barh(dev_lst,  df['frac_on'].values, left=df['frac_off'], label='on')

    # we also need to switch the labels
    plt.title(title)
    plt.xlabel('Percentage in binary states')  
    plt.ylabel('Devices')
    
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

def hist_counts(df_dev, figsize=(10,6), y_scale=None):
    """
    plots the trigger count of each device 
    """
    df = devices_trigger_count(df_dev.copy())
    df.reset_index(level=0, inplace=True)

    title = 'Count of on/off activations per Device'
    col_label = 'trigger count'
    col_device = 'device'
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