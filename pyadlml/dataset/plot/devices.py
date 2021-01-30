import numpy as np
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors

from pyadlml.dataset import DEVICE
from pyadlml.dataset.stats.devices import duration_correlation, \
    trigger_time_diff, device_tcorr, device_triggers_one_day, \
    devices_trigger_count, devices_on_off_stats

from pyadlml.dataset.plot.util import heatmap_square, func_formatter_seconds2time,\
    heatmap, annotate_heatmap, savefig, _num_bars_2_figsize, \
    _num_items_2_heatmap_square_figsize, _num_boxes_2_figsize, \
    _num_items_2_heatmap_one_day_figsize, _num_items_2_heatmap_square_figsize_ver2 

from pyadlml.dataset.devices import _is_dev_rep2, device_rep1_2_rep2
from pyadlml.util import get_sequential_color, get_secondary_color, get_primary_color, get_diverging_color

def hist_trigger_time_diff(df_devs=None, x=None, n_bins=50, figsize=(10, 6), color=None, file_path=None):
    """
    plots

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    x : ndarray, optional
        array of time deltas used to plot the histogram
    n_bins : int, default=50
        the number of bins for the histogram.
    color : str, optional
        sets the color of the plot. When not set, the primary theming color is used.
        Learn more about theming in the :ref:`user guide <theming>`
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_trigger_time_dev
    >>> plot_trigger_time_dev_todo(data.df_devices)

    .. image:: ../_static/images/plots/dev_hist_trigger_td.png
       :height: 300px
       :width: 500 px
       :scale: 100 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing.


    """
    assert not (df_devs is None and x is None)
    title='Time difference between succeeding device'
    log_sec_col = 'total_log_secs'
    sec_col = 'total_secs'
    ylabel='count'
    ax2label = 'cummulative percentage'
    ax1label = 'timedeltas count '
    xlabel = 'log seconds'
    color = (get_primary_color() if color is None else color)
    color2 = get_secondary_color()

    if x is None:
        X = trigger_time_diff(df_devs.copy())
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
    ax.hist(X, bins=bins, label=ax1label, color=color)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    # create axis for line
    ax2=ax.twinx()
    ax2.plot(bins, cum_percentage, 'r', label=ax2label, color=color2)
    ax2.set_ylabel('%')
    ax2.set_xscale('log')
    
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_seconds2time))
    
    # plot single legend for multiple axis
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='center right')
    
    plt.title(title, y=1.08)
    
    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig


def boxplot_on_duration(df_devs, lst_devs=None, order='mean', figsize=None, file_path=None):
    """
    generates a boxplot for all devices.

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    lst_devs : lst of str, optional
        A list of devices that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    order : {'mean', 'alphabetically', 'room'}, default='mean'
        determines the order in which the devices are listed.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plots import plot_device_bp_on_duration
    >>> plot_device_bp_on_duration(data.df_devices)

    .. image:: ../_static/images/plots/dev_bp_dur.png
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
    title = 'Devices on-duration'
    xlabel = 'log seconds'
    xlabel_top = 'time'
    from pyadlml.dataset.stats.devices import devices_td_on
    df_devs = devices_td_on(df_devs)

    # select data for each device
    devices = list(df_devs[DEVICE].unique())
    devices.sort(reverse=True)
    dat = []
    for device in devices:
        df_device = df_devs[df_devs[DEVICE] == device]
        tmp = df_device['td'].dt.total_seconds()
        dat.append(tmp)

    if lst_devs is not None:
        nan_devs = list(set(lst_devs).difference(set(list(devices))))
        nan_devs.sort(reverse=True)
        for dev in nan_devs:
            dat.append([])
        devices = devices + nan_devs

    num_dev = len(devices)
    figsize = (_num_boxes_2_figsize(num_dev) if figsize is None else figsize)

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(dat, vert=False)
    ax.set_title(title)
    ax.set_yticklabels(devices, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_xscale('log')

    # create secondary axis with time format 1s, 1m, 1d
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax_top.set_xlabel(xlabel_top)
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_seconds2time))

    # save or return figure
    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig

def heatmap_trigger_one_day(df_devs=None, lst_devs=None, df_tod=None, t_res='1h',
                            figsize=None, cmap=None, file_path=None):
    """
    Plots the heatmap for one day where all the device triggers are shown

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    lst_devs : lst of str, optional
        A list of devices that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    df_tod : pd.DataFrame
        A precomputed transition table. If the *df_trans* parameter is given, parameters
        *df_acts* and *lst_acts* are ignored. The transition table can be computed
        in :ref:`stats <stats_acts_trans>`.
    t_res : str of {'[1-12]h', default='1h'
        the resolution, time_bins in hours. The number are
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    cmap : str or Colormap, optional
        The Colormap instance or registered colormap name used to map scalar
        data to colors. This parameter is ignored for RGB(A) data.
        Defaults 'viridis'.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plots import plot_device_hm_time_trigger
    >>> plot_device_hm_time_trigger(data.df_devices, t_res='1h')

    .. image:: ../_static/images/plots/dev_hm_trigger_one_day.png
       :height: 300px
       :width: 500 px
       :scale: 100 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing.
    """
    assert not (df_devs is None and df_tod is None)
    title = "Device triggers cummulative over one day"
    xlabel =  'time'

    df = (device_triggers_one_day(df_devs.copy(), lst_devs=lst_devs, t_res=t_res) if df_tod is None else df_tod)
    num_dev = len(list(df.columns))
    figsize = (_num_items_2_heatmap_one_day_figsize(num_dev) if figsize is None else figsize)
    cmap = (get_sequential_color() if cmap is None else cmap)

    x_labels = list(df.index)
    y_labels = df.columns
    dat = df.values.T
    
    # begin plotting
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap(dat, y_labels, x_labels, ax=ax, cmap=cmap, cbarlabel='counts')
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

    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig

def heatmap_trigger_time(df_devs=None, lst_devs=None, df_tcorr=None, t_window='5s', figsize=None,
                         z_scale="linear", cmap=None, numbers=None, file_path=None):
    """
    Plot todo

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    lst_devs : lst of str, optional
        A list of devices that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    df_tcorr : pd.DataFrame
        A precomputed correlation table. If the *df_tcorr* parameter is given, parameters
        *df_devs* and *lst_devs* are ignored. The transition table can be computed
        in :ref:`stats <stats_devs_tcorr>`.
    t_window : str of {'[1-12]h', default='1h'
        the resolution, time_bins in hours. The number are
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    z_scale : {"log", "linear"}, default: None
        The axis scale type to apply.
    numbers : bool, default: True
        Whether to display numbers inside the heatmaps fields or not.
    cmap : str or Colormap, optional
        The Colormap instance or registered colormap name used to map scalar
        data to colors. This parameter is ignored for RGB(A) data.
        Defaults 'viridis'.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plots import plot_device_hm_time_trigger
    >>> plot_device_hm_time_trigger(data.df_devices, t_res='1h')

    .. image:: ../_static/images/plots/dev_hm_trigger_one_day.png
       :height: 300px
       :width: 500 px
       :scale: 100 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing.
    """
    assert not (df_devs is None and df_tcorr is None)
    title = "Triggercount with sliding window of " + t_window

    color = 'trigger count'
    cbarlabel = 'counts'

    if df_tcorr is None:
        df = device_tcorr(df_devs, lst_devs=lst_devs, t_window=t_window)
    else:
        df = df_tcorr

    # get the list of cross tabulations per t_window
    vals = df.astype(int).values.T
    devs = list(df.index)

    num_dev = len(devs)
    figsize = (_num_items_2_heatmap_square_figsize_ver2(num_dev) if figsize is None else figsize)
    cmap = (get_sequential_color() if cmap is None else cmap)

    fig, ax = plt.subplots(figsize=figsize)

    log = True if z_scale == 'log' else False       
    valfmt = "{x:.0f}"
        
    im, cbar = heatmap_square(vals, devs, devs, ax=ax, cmap=cmap,
                        cbarlabel=cbarlabel, log=log)#, cbar_kw=cbar_kw)
    
    # show numbers for small sizes
    if numbers is None: 
        if num_dev < 20:
            texts = annotate_heatmap(im, textcolors=("white", "black"), log=log, valfmt=valfmt)
    elif numbers:
        texts = annotate_heatmap(im, textcolors=("white", "black"), log=log, valfmt=valfmt)

    ax.set_title(title)
    fig.tight_layout()

    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig

def heatmap_cross_correlation(df_devs=None, lst_devs=None, df_dur_corr=None, figsize=None,
                              numbers=None, file_path=None):
    """
    Plots the cross correlation between the device signals

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    lst_devs : lst of str, optional
        A list of devices that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    df_tcorr : pd.DataFrame
        A precomputed correlation table. If the *df_tcorr* parameter is given, parameters
        *df_devs* and *lst_devs* are ignored. The transition table can be computed
        in :ref:`stats <stats_devs_tcorr>`.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    numbers : bool, default: True
        Whether to display numbers inside the heatmaps fields or not.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plots import plot_dev_hm_similarity
    >>> plot_dev_hm_similarity(data.df_devices)

    .. image:: ../_static/images/plots/dev_hm_dur_cor.png
       :height: 400px
       :width: 500 px
       :scale: 90 %
       :alt: alternate text
       :align: center




    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing.
    """
    assert not (df_devs is None and df_dur_corr is None)

    title = 'Devices cross-correlation'
    cmap = 'RdBu'
    cbarlabel = 'similarity'
    
    if df_dur_corr is None:
        ct = duration_correlation(df_devs, lst_devs=lst_devs)
    else:
        ct = df_dur_corr
    ct = ct.replace(pd.NA, np.inf)
    vals = ct.values.T
    devs = list(ct.index)

    num_dev = len(devs)
    figsize = (_num_items_2_heatmap_square_figsize_ver2(num_dev) if figsize is None else figsize)
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap_square(vals, devs, devs, ax=ax, cmap=cmap, cbarlabel=cbarlabel,
                       vmin=-1, vmax=1)
    if numbers is None:
        if num_dev < 15:
            valfmt = "{x:.2f}"
            texts = annotate_heatmap(im, textcolors=("black", "white"), 
                             threshold=0.5, valfmt=valfmt)
        elif num_dev < 30:
            valfmt = "{x:.1f}"
            texts = annotate_heatmap(im, textcolors=("black", "white"), 
                             threshold=0.5, valfmt=valfmt)
    if numbers:
        texts = annotate_heatmap(im, textcolors=("black", "white"), 
                             threshold=0.5, valfmt="{x:.2f}")
    ax.set_title(title)
    fig.tight_layout()
    if file_path is not None:
        savefig(fig, file_path)
        return
    else:
        return fig


def hist_on_off(df_devs=None, lst_devs=None, df_onoff=None, figsize=None,
                color=None, color_sec=None, order='frac_on', file_path=None):
    """
    Plot bars the on/off fraction of all devices

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    lst_devs : lst of str, optional
        A list of devices that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    df_onoff : pd.DataFrame
        A precomputed correlation table. If the *df_tcorr* parameter is given, parameters
        *df_devs* and *lst_devs* are ignored. The transition table can be computed
        in :ref:`stats <stats_devs_tcorr>`.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    color : str, optional
        sets the primary color of the plot. When not set, the primary theming color is used.
        Learn more about theming in the :ref:`user guide <theming>`
    color_sec : str, optional
        sets the secondary color of the plot. When not set, the secondary theming color is used.
        Learn more about theming in the :ref:`user guide <theming>`
    order : {'frac_on', 'alphabetically', 'room'}, default='frac_on'
        determines the order in which the devices are listed.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plots import plot_device_on_off
    >>> plot_device_on_off(data.df_devices)

    .. image:: ../_static/images/plots/dev_on_off.png
       :height: 300px
       :width: 500 px
       :scale: 100 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing.
    """
    assert not (df_devs is None and df_onoff is None)
    assert order in ['frac_on', 'name', 'area']

    title = 'Devices fraction on/off'
    xlabel ='Percentage in binary states' 
    ylabel = 'Devices'
    on_label = 'on'
    off_label = 'off'

    color = (get_primary_color() if color is None else color)
    color2 = (get_secondary_color()if color_sec is None else color_sec)

    if df_onoff is None:
        df = devices_on_off_stats(df_devs, lst_devs=lst_devs)
    else:
        df = df_onoff

    num_dev = len(df)
    figsize = (_num_bars_2_figsize(num_dev) if figsize is None else figsize)

    if order == 'frac_on':
        df = df.sort_values(by='frac_on', axis=0)
    elif order == 'name':
        df = df.sort_values(by=DEVICE, axis=0)
    else:
        raise NotImplementedError('room order will be implemented in the future')

    dev_lst = list(df[DEVICE])
    # Figure Size 
    fig, ax = plt.subplots(figsize=figsize)
    if lst_devs is not None:
        df['tmp'] = 0
        plt.barh(df[DEVICE], df['tmp'].values, alpha=0.0)
        plt.barh(df[DEVICE], df['frac_off'].values, label=off_label, color=color)
        plt.barh(df[DEVICE],  df['frac_on'].values, left=df['frac_off'], label=on_label, color=color2)
    else:
        plt.barh(dev_lst, df['frac_off'].values, label=off_label, color=color)
        # careful: notice "bottom" parameter became "left"
        plt.barh(dev_lst,  df['frac_on'].values, left=df['frac_off'], label=on_label, color=color2)

    # we also need to switch the labels
    plt.title(title)
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)
    
    # set the text centers to the middle for the greater fraction
    widths = df['frac_off'].apply(lambda x: x if x >= 0.5 else 1-x)
    xcenters = df['frac_off'].apply(lambda x: x/2 if x >= 0.5 else (1-x)/2 + x)
    first_number_left = True
    for y, c, w in zip(range(len(xcenters)), xcenters, widths):
        if y == len(xcenters)-1 and c < 0.5:
           first_number_left = False
        if c > 0.5:
            text_color='black'
        else:
            text_color='white'
        ax.text(c, y, '{:.4f}'.format(w), ha='center', va='center', color=text_color)

    if first_number_left:
        ax.legend(ncol=2, bbox_to_anchor=(0, 1),
              loc='upper left', fontsize='small')
    else:
         ax.legend(ncol=2, bbox_to_anchor=(1,1),
              loc='upper right', fontsize='small')


    # Remove axes splines 
    for s in ['top', 'right']: 
        ax.spines[s].set_visible(False)

    if file_path is not None:
        savefig(fig, file_path)
        return
    else:
        return fig

def hist_counts(df_devs=None, lst_devs=None, df_tc=None, figsize=None,
                y_scale=None, color=None, order='count', file_path=None):
    """
    bar chart displaying how often activities are occurring

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    lst_devs : lst of str, optional
        A list of devices that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    df_tc : pd.DataFrame
        A precomputed correlation table. If the *df_tcorr* parameter is given, parameters
        *df_devs* and *lst_devs* are ignored. The transition table can be computed
        in :ref:`stats <stats_devs_tcorr>`.
    y_scale : {"log", "linear"}, default: None
        The axis scale type to apply.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    color : str, optional
        sets the primary color of the plot. When not set, the primary theming color is used.
        Learn more about theming in the :ref:`user guide <theming>`
    order : {'count', 'alphabetically', 'room'}, default='count'
        determines the order in which the devices are listed.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plots import plot_device_bar_count
    >>> plot_device_bar_count(data.df_devices)

    .. image:: ../_static/images/plots/dev_bar_trigger.png
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
    assert not (df_devs is None and df_tc is None)
    assert y_scale in ['log', 'linear']
    assert order in ['alphabetic', 'count', 'room']
    
    title = 'Device triggers'
    x_label = 'count'
    df_col = 'trigger_count'

    df = (devices_trigger_count(df_devs.copy(), lst_devs=lst_devs) if df_tc is None else df_tc)
    num_dev = len(df)
    figsize = (_num_bars_2_figsize(num_dev) if figsize is None else figsize)
    color = (get_primary_color() if color is None else color)

    if order == 'alphabetic':
        df = df.sort_values(by=[DEVICE], ascending=True)
    elif order == 'count':
        df = df.sort_values(by=[df_col])
    else:
        raise NotImplemented('the room order is going to be implemented')

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    plt.xlabel(x_label)
    ax.barh(df[DEVICE], df[df_col], color=color)

    if y_scale == 'log':
        ax.set_xscale('log')

    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig