import numpy as np
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from pyadlml.stats import device_state_fractions
from pyadlml.constants import DEVICE, TIME, VALUE, BOOL, CAT, START_TIME, END_TIME, NUM
from pyadlml.dataset.stats.devices import state_cross_correlation as stat_scc, \
    inter_event_intervals as stat_iei, event_cross_correlogram as stat_event_cc, events_one_day, \
    event_count as stats_event_count, event_cross_correlogram2

from .util import heatmap_square, func_formatter_seconds2time, \
    heatmap, annotate_heatmap, savefig, _num_bars_2_figsize, save_fig, \
    _num_items_2_heatmap_square_figsize, _num_boxes_2_figsize, \
    _num_items_2_heatmap_one_day_figsize, _num_items_2_heatmap_square_figsize_ver2, LOG, xaxis_format_one_day, \
    sort_devices, add_top_axis_time_format, map_time_to_numeric, plot_cc, create_todo, xaxis_format_time, \
    dev_raster_data_gen, xaxis_format_time2, get_qualitative_cmap, plot_grid

from pyadlml.dataset.plotly.util import format_device_labels
from pyadlml.dataset._core.devices import _is_dev_rep2, device_events_to_states, split_devices_binary, contains_non_binary
from pyadlml.dataset.stats.util import comp_tds_sums, comp_tds_sums_mean, comp_tds_sums_median
from pyadlml.dataset.util import select_timespan, infer_dtypes, str_to_timestamp, device_order_by
from pyadlml.util import get_sequential_color, get_secondary_color, get_primary_color, get_diverging_color


@save_fig
def inter_event_intervals(df_devices=None, inter_event_intervals=None, scale='log',
                          nr_merged_events_at=[], n_bins=50, figsize=(10, 6),
                          color=None, file_path=None):
    """
    Plot a histogram of the differences between succeeding device triggers.

    Parameters
    ----------
    df_devices : pd.DataFrame, default=None
        Recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    inter_event_intervals : ndarray, default=None
        Array time deltas used to plot the histogram. Compute this array using (TODO REF TO STATS)
    nr_merged_events_at : list of strings, default=[]
        Timepoints for wich the fraction of events disregarded due to
        value manipulation by discretizing time-series data is plotted
    n_bins : int, default=50
        The number of histogram bins.
    color : str, default=None
        sets the color of the plot. When not set, the primary theming color is used.
        Learn more about theming in the :ref:`user guide <theming>`
    figsize : (float, float), default: None
        Width and height in inches. If not provided, the figsize is inferred automatically.
    file_path : str, default=None
        If set, saves the plot under the given file path and returns *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_device_inter_event_intervals
    >>> plot_device_inter_event_intervals(df_devices=data.df_devices)

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
    assert not (df_devices is None and inter_event_intervals is None)
    title='Devices Inter-event-intervals'
    log_sec_col = 'total_log_secs'
    sec_col = 'total_secs'
    ylabel='count'
    xlabel_top = 'time'
    ax1label = '$\Delta t$ count'
    ax3label = 'imputed events'
    xlabel = 'log seconds' if scale == 'log' else 'seconds'
    color = (get_primary_color() if color is None else color)
    color2 = get_secondary_color()

    if inter_event_intervals is None:
        X = stat_iei(df_devices.copy())
    else:
        X = inter_event_intervals


    if nr_merged_events_at:
        from pyadlml.dataset.stats.devices import resampling_imputation_loss
        imp_frac_y = np.zeros(len(nr_merged_events_at))
        for i, dt in enumerate(nr_merged_events_at):
            imp_frac_y[i] = resampling_imputation_loss(df_devices, dt, return_fraction=True)

        # compute corresponding time in seconds
        imp_frac_x = [pd.Timedelta(dt).seconds for i, dt in enumerate(nr_merged_events_at)]

    # make equal bin size from max to min
    bins = np.logspace(min(np.log10(X)), max(np.log10(X)), n_bins)

    # make data ready for hist
    hist, _ = np.histogram(X, bins=bins)

    # plots
    fig, ax = plt.subplots(figsize=figsize)
    if scale == 'log':
        plt.xscale(LOG)
    ax.hist(X, bins=bins, label=ax1label, color=color)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if nr_merged_events_at:
        # create axis for line
        ax2=ax.twinx()
        #ax2.plot(bins, cum_percentage, 'r', label=ax2label, color=color2)
        ax2.set_ylabel('%')
        ax2.set_xscale(LOG)
        ax2.plot(imp_frac_x, imp_frac_y, label=ax3label, color=color2, marker='.')
        ax2.set_ylim([0, 1])

    add_top_axis_time_format(ax)

    # plot single legend for multiple axis
    h1, l1 = ax.get_legend_handles_labels()

    if nr_merged_events_at:
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc='center right')
    else:
        ax.legend(h1, l1, loc='center right')
    plt.title(title, y=1.08)
    
    return fig


@save_fig
def state_boxplot(df_devs, binary_state='on', categories=False, order='mean',
                  scale='log', figsize=None, file_path=None):
    """
    generates a boxplot for all devices.

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    figsize : (int, int), default=None
        Width, height in inches. If not provided, the figsize is inferred automatically.
    binary_state : str one of {"on", "off"}, default='on'
        Determines which of the state is choosen for a boxplot for binary devices
    categories : bool, default=False
        If set a boxplot for each categorie of a categorical device is drawn
    order : {'mean', 'alphabetically', 'room'}, default='mean'
        determines the order in which the devices are listed.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.dataset import fetch_amsterdam
    >>> data = fetch_amsterdam()

    >>> from pyadlml.plot import plot_device_boxplot
    >>> plot_device_boxplot(data.df_devices)

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
    title = 'Device state distribution'
    xlabel = 'log seconds'
    xlabel_top = 'time'
    from pyadlml.dataset.stats.devices import state_times

    df = state_times(df_devs, binary_state=binary_state, categorical=categories)

    # select data for each device
    devices = list(df[DEVICE].unique())
    num_dev = len(devices)
    figsize = (_num_boxes_2_figsize(num_dev) if figsize is None else figsize)

    dat = np.zeros((num_dev,), dtype=object)
    for i, device in enumerate(devices):
        dat[i] = df.loc[df[DEVICE] == device, 'td'].dt.seconds.to_numpy()


    new_devices, new_order = format_device_labels(devices, infer_dtypes(df_devs), categorical_state=True)
    devices = np.flip(new_devices) # boxplots plot in reverse order
    dat = dat[np.flip(new_order)]

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(dat, vert=False)
    ax.set_title(title)
    ax.set_yticklabels(devices, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_xscale(LOG)

    # create secondary axis with time format 1s, 1m, 1d
    add_top_axis_time_format(ax, xlabel_top)

    return fig


@save_fig
def event_density_one_day(df_devices=None, df_tod=None, dt='1h',
                          figsize=None, cmap=None, file_path=None):
    """
    Plots the heatmap for one day where all the device triggers are shown

    Parameters
    ----------
    df_devices : pd.DataFrame, optional
        recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    df_tod : pd.DataFrame
        A precomputed transition table. If the *df_trans* parameter is given, parameters
        *df_acts* and *lst_acts* are ignored. The transition table can be computed
        in :ref:`stats <stats_acts_trans>`.
    dt : str of {'[1-12]h', default='1h'
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
    >>> from pyadlml.plot import plot_device_event_density
    >>> plot_device_event_density(data.df_devices, dt='1h')

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
    assert not (df_devices is None and df_tod is None)
    title = "Device event density"
    xlabel = 'time'
    cbarlabel = 'counts'

    if df_tod is None:
        df = events_one_day(df_devices.copy(), dt=dt)
    else:
        df = df_tod

    if dt is None:
        raise NotImplementedError
    else:
        num_dev = len(list(df.columns))
        figsize = (_num_items_2_heatmap_one_day_figsize(num_dev) if figsize is None else figsize)
        cmap = (get_sequential_color() if cmap is None else cmap)

        x_labels = list(df[TIME])
        devs = df.columns[1:]
        dat = df.iloc[:, 1:].values.T

        # reorder devices
        devs, new_order = format_device_labels(devs, infer_dtypes(df_devices))
        dat = dat[new_order, :]

        # begin plotting
        fig, ax = plt.subplots(figsize=figsize)
        im, cbar = heatmap(dat, devs, x_labels, ax=ax, cmap=cmap, cbarlabel=cbarlabel)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        xaxis_format_one_day(ax, *ax.get_xlim())

        ax.set_aspect(aspect='auto')

    return fig


@save_fig
def event_correlogram(df_devs=None, lst_devs=None, df_tcorr=None, t_window='5s', figsize=None,
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
    >>> from pyadlml.plot import plot_device_event_correlogram
    >>> plot_device_event_correlogram(data.df_devices, dt='1h')

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
    log = True if z_scale == 'log' else False
    valfmt = "{x:.0f}"


    if df_tcorr is None:
        df = event_cross_correlogram(df_devs, lst_devs=lst_devs, t_window=t_window)
    else:
        df = df_tcorr

    # get the list of cross tabulations per t_window
    vals = df.astype(int).values.T
    devs = list(df.index)

    num_dev = len(devs)
    figsize = (_num_items_2_heatmap_square_figsize_ver2(num_dev) if figsize is None else figsize)
    cmap = (get_sequential_color() if cmap is None else cmap)

    fig, ax = plt.subplots(figsize=figsize)
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

    return fig


@save_fig
def state_similarity(df_devs=None, df_state_sim=None, figsize=None, order='alphabetical',
                     numbers=None, file_path=None):
    """
    Plots the cross correlation between the device signals

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    df_state_sim : pd.DataFrame
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
    >>> from pyadlml.plot import plot_device_state_similarity
    >>> plot_device_state_similarity(data.df_devs)

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
    assert not (df_devs is None and df_state_sim is None)

    title = 'State similarity'
    cmap = 'RdBu'
    cbarlabel = 'similarity'
    
    if df_state_sim is None:
        ct = stat_scc(df_devs)
    else:
        ct = df_state_sim
    ct = ct.replace(pd.NA, np.inf)
    values = ct.values.T
    devs = ct.index

    devs, new_order = format_device_labels(devs, infer_dtypes(df_devs), categorical_state=True)
    values = values[:, new_order]
    values = values[new_order, :]

    num_dev = len(devs)
    figsize = (_num_items_2_heatmap_square_figsize_ver2(num_dev) if figsize is None else figsize)
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap_square(values, devs, devs, ax=ax, cmap=cmap, cbarlabel=cbarlabel,
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

    return fig

@save_fig
def state_fractions(df_devs=None, df_states=None, figsize=None,
                    color=None, color_sec=None, order='frac_on', file_path=None):
    """
    Plot bars the on/off fraction of all devices

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    df_states : pd.DataFrame
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
    order : {'frac_on', 'alphabetically', 'area'}, default='frac_on'
        determines the order in which the devices are listed.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_device_state_fractions
    >>> plot_device_state_fractions(data.df_devices)

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
    assert not (df_devs is None and df_states is None)
    assert order in ['frac_on', 'alphabetical', 'area']

    title = 'Device state fractions'
    xlabel ='fraction per state'
    ylabel = 'Devices'
    on_label = 'on'
    off_label = 'off'

    color = (get_primary_color() if color is None else color)
    color2 = (get_secondary_color()if color_sec is None else color_sec)

    if df_states is None:
        df = device_state_fractions(df_devs)
    else:
        df = df_states

    num_dev = len(df_devs[DEVICE].unique())
    figsize = (_num_bars_2_figsize(num_dev) if figsize is None else figsize)

    df = sort_devices(df, 'frac')

    from pyadlml.dataset.util import infer_dtypes
    dtypes = infer_dtypes(df_devs)


    # Figure Size 
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plot boolean devices
    df_bool = df[df[DEVICE].isin(dtypes[BOOL])].copy().sort_values(by=DEVICE)
    off_values = df_bool.loc[(df_bool[VALUE] == False), 'frac'].values
    plt.barh(dtypes[BOOL], off_values, label=off_label, color=color)
    # careful: notice "bottom" parameter became "left"
    plt.barh(dtypes[BOOL], (1-off_values), left=off_values, label=on_label, color=color2)

    # set the text centers for the boolean devices to the middle for the greater fraction
    first_number_left = True
    for i, dev in enumerate(dtypes[BOOL]):
        text_in_off = (off_values[i] >= 0.5)
        center = off_values[i]/2 if text_in_off else (1 - off_values[i])/2 + off_values[i]
        text = off_values[i] if text_in_off else (1 - off_values[i])
        if i == len(dtypes[BOOL])-1 and text_in_off:
           first_number_left = False
        text_color = 'white' if text_in_off else 'black'
        ax.text(center, i, '{:.4f}'.format(text), ha='center', va='center', color=text_color)

    # plot categorical devices
    cat_centers = []
    for j, dev in enumerate(dtypes[CAT]):
        categories = df.loc[(df[DEVICE] == dev), VALUE].copy().sort_values().values
        cum_sum = 0
        cat_centers.append([])
        for i, cat in enumerate(categories):
            value = df.loc[(df[DEVICE] == dev) & (df[VALUE] == cat), 'frac'].values[0]
            # save the center in bar plot as well as the value for setting
            # the annotations later
            cat_centers[j].append((cum_sum + value/2, value))
            plt.barh([dev], [value], label=dev + ' : ' + cat, left=[cum_sum])
            cum_sum += value


    # set the text centers for categorical devices into the middle for each category
    for c, y in zip(range(len(dtypes[CAT])), range(len(dtypes[BOOL]), num_dev-1)):
        for center, text in cat_centers[c]:
            ax.text(center, y, '{:.4f}'.format(text), ha='center', va='center', color='white')

    if dtypes[CAT]:
        ax.legend(bbox_to_anchor=(1, 1))
    else:
        if first_number_left:
            ax.legend(ncol=2, bbox_to_anchor=(0, 1),
                  loc='upper left', fontsize='small')
        else:
             ax.legend(ncol=2, bbox_to_anchor=(1,1),
                  loc='upper right', fontsize='small')


    # Remove axes splines 
    for s in ['top', 'right']: 
        ax.spines[s].set_visible(False)

    return fig


@save_fig
def event_count(df_devs=None, df_tc=None, figsize=None,
                scale='linear', color=None, order='count', file_path=None):
    """
    bar chart displaying how often activities are occurring

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. Fore more information refer to the
        :ref:`user guide<device_dataframe>`.
    df_tc : pd.DataFrame
        A precomputed correlation table. If the *df_tcorr* parameter is given, parameters
        *df_devs* and *lst_devs* are ignored. The transition table can be computed
        in :ref:`stats <stats_devs_tcorr>`.
    scale : {"log", "linear"}, default: None
        The axis scale type to apply.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    color : str, optional
        sets the primary color of the plot. When not set, the primary theming color is used.
        Learn more about theming in the :ref:`user guide <theming>`
    order : {'count', 'alphabetically', 'area'}, default='count'
        determines the order in which the devices are listed.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_device_event_count
    >>> plot_device_event_count(data.df_devices)

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
    assert scale in ['log', 'linear']
    assert order in ['alphabetic', 'count', 'room']
    
    title = 'Device events'
    x_label = 'count'
    df_col = 'event_count'

    df = (stats_event_count(df_devs.copy()) if df_tc is None else df_tc)
    num_dev = len(df)
    figsize = (_num_bars_2_figsize(num_dev) if figsize is None else figsize)
    color = (get_primary_color() if color is None else color)

    # sort devices according to strategy
    sort_strategy = df_col if (order == 'count') else order
    df = sort_devices(df, sort_strategy)

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    plt.xlabel(x_label)
    ax.barh(df[DEVICE], df[df_col], color=color)

    if scale == LOG:
        ax.set_xscale(LOG)

    return fig


@save_fig
def event_raster(df_devices, figsize=(10, 6), start_time=None, end_time=None,
                 order='alphabetical', file_path=None):
    """ Plots an event raster

    """
    event_color = 'grey'
    title = 'Event raster'
    y_label = 'Devices'

    df_devs = df_devices.copy()\
                        .sort_values(by=TIME)\
                        .reset_index(drop=True)

    start_time = str_to_timestamp(start_time) if (start_time is not None) else None
    end_time = str_to_timestamp(end_time) if (end_time is not None) else None
    df_devs = select_timespan(df_devs=df_devs, start_time=start_time, end_time=end_time)

    if start_time is None:
        start_time = df_devs[TIME].iloc[0]
    if end_time is None:
        end_time = df_devs[TIME].iloc[-1]

    devs = device_order_by(df_devs, rule=order)

    df_devs['num_time'], start_time, end_time = map_time_to_numeric(
        df_devs[TIME], start_time=start_time, end_time=end_time)

    data_lst = dev_raster_data_gen(df_devs, devs)

    # create list of lists, where each list corresponds to an activity with tuples of start_time and time_length
    fig, ax = plt.subplots(figsize=figsize)
    ax.eventplot(data_lst, linelengths=[0.2]*len(devs), colors=event_color)

    ax.set_title(title)
    ax.set_yticks(np.arange(len(devs)))
    ax.set_yticklabels(devs.tolist())
    ax.set_ylabel(y_label)
    ax.set_xlim([0, 1])

    xaxis_format_time2(fig, ax, start_time, end_time)

    return fig


@save_fig
def plot_time_spans_slicer(X, window_size, stride, unit='m', file_path=None):
    unit = 'h'
    title = 'Time-span per window'
    y_label = 'counts'

    sums = comp_tds_sums(X, window_size, stride)
    mean = comp_tds_sums_mean(X, window_size, stride)
    median = comp_tds_sums_median(X, window_size, stride)

    seconds = np.zeros(shape=sums.shape)
    for i in range(len(sums)):
        seconds[i] = sums[i].seconds

    if unit == 's':
        f = 1
        x_label = 'seconds'
    elif unit == 'm':
        f = (1/60)
        x_label = 'minutes'
    else:
        f = (1/3600)
        x_label = 'hours'

    mean_str = str(int(mean.seconds*f))
    median_str = str(int(mean.seconds*f))

    hist, bin_edges = np.histogram(seconds*f, bins=20)
    fig = plt.figure()
    plt.hist(seconds*f, bins=bin_edges, label='timelength one window')
    plt.plot([mean.seconds*f]*2, [0, hist.max()], label='mean: ' + mean_str + ' ' + x_label)
    plt.plot([median.seconds*f]*2, [0, hist.max()], label='median: ' + median_str + ' ' + x_label)
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return fig


@save_fig
def event_cross_correlogram(df_devices=None, corr_data=(None, None, None), bin_size='1s', max_lag='2m', axis='off',
                            figsize=(5, 5), file_path=None):
    #Plot cross-correlograms of all pairs.
    #   plotCCG(ccg,bins) plots a matrix of cross(auto)-correlograms for
    #   all pairs of clusters. Inputs are:
    #       ccg     array of cross correlograms           #bins x #clusters x #clusters
    #       bins    array with bin timings                #nbins x 0

    title = 'Event cross-correlogram'

    if corr_data[0] is None or corr_data[1] is None:
        ccg, bins = event_cross_correlogram2(df_devices, binsize=bin_size, maxlag=max_lag)
        devices = df_devices[DEVICE].unique()
    else:
        ccg = corr_data[0]
        bins = corr_data[1]
        devices = corr_data[2]

    fig = plot_cc(ccg, bins, title=title, y_label=devices, axis=axis, figsize=figsize)

    return fig


@save_fig
def states(df_devices, start_time=None,end_time=None, figsize=(20, 8),
           order='alphabetical', grid=False, file_path=""):
    """ Plots the devices events as raster combined with the activites over a timespan

    Parameters
    ----------
    """
    ylabel = 'Devices'
    title = 'Device states'
    binary_on_label = 'on'
    binary_off_label = 'off'

    color_dev_on = 'seagreen'
    color_dev_off = 'lightgrey'
    color_dev_num = 'blue'

    df_devs = df_devices.copy()\
                        .sort_values(by=TIME)\
                        .reset_index(drop=True)

    start_time = str_to_timestamp(start_time) if (start_time is not None) else None
    end_time = str_to_timestamp(end_time) if (end_time is not None) else None
    df_devs = select_timespan(df_devs=df_devs, start_time=start_time, end_time=end_time)

    if start_time is None:
        start_time = df_devs[TIME].iloc[0]
    if end_time is None:
        end_time = df_devs[TIME].iloc[-1]

    devs = df_devs[DEVICE].unique()

    fig, ax = plt.subplots(figsize=figsize)

    if grid:
        plot_grid(fig, ax, start_time, end_time)

    j = _plot_device_states(ax, df_devs, devs, start_time, end_time, color_dev_on, color_dev_off,
                           color_dev_num, binary_off_label, binary_on_label, return_nr_categories_used=True)

    ax.set_title(title)
    ax.set_yticks(np.arange(len(devs), dtype=np.float32))
    ax.set_yticklabels(devs.tolist())
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, 1])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left")
    xaxis_format_time2(fig, ax, start_time, end_time)

    return fig

def _plot_device_states(ax, df_devs: pd.DataFrame, devs: list, start_time, end_time, color_on='red',
    color_off='grey', color_num='blue', off_label='off', on_label='on', return_nr_categories_used=False):
    """ Plot vertical bar plots for device each state

    Parameters
    ----------

    Returns
    -------
    j : int
        The number of colors used up for different devices.
    """
    col_bar_start = 'num_st'
    col_bar_len = 'diff'

    # do preprocessing
    dtypes = infer_dtypes(df_devs)
    df_devs = device_events_to_states(df_devs, extrapolate_states=True, st=start_time, et=end_time)

    df_devs['num_st'], _, _ = map_time_to_numeric(df_devs[START_TIME], start_time, end_time)
    df_devs['num_et'], _, _ = map_time_to_numeric(df_devs[END_TIME], start_time, end_time)
    df_devs['diff'] = df_devs['num_et'] - df_devs['num_st']


    j = 0
    tab = get_qualitative_cmap(len(devs))
    set_on_off_label = True
    for i, dev in enumerate(devs):
        df = df_devs[df_devs[DEVICE] == dev].copy()
        if dev in dtypes[BOOL]:
            tuples_off = df.loc[(df[VALUE] == False), [col_bar_start, col_bar_len]].values.tolist()
            tuples_on = df.loc[(df[VALUE] == True), [col_bar_start, col_bar_len]].values.tolist()

            ax.broken_barh(tuples_off, (i-0.25, 0.5), facecolors=color_off, label=off_label)
            ax.broken_barh(tuples_on, (i-0.25, 0.5), facecolors=color_on, label=on_label)

            #  The on and off label appear only once in the legend
            if set_on_off_label:
                set_on_off_label = False
                off_label, on_label = None, None

        elif dev in dtypes[CAT]:
            categories = df.loc[df[DEVICE] == dev, VALUE].unique()
            for cat in categories:
                values = df.loc[(df[VALUE] == cat), [col_bar_start, col_bar_len]].values.tolist()
                ax.broken_barh(values, (i-0.25, 0.5), facecolors=tab(j), label=dev + ' - ' + cat)
                j += 1

        elif dev in dtypes[NUM]:
            values = pd.to_numeric(df[VALUE])
            values = (values-values.min())/(values.max() - values.min())*0.5
            values = values + i - 0.25
            ax.plot(df['num_st'], values, color=color_num, linestyle='--', marker='o')

    if return_nr_categories_used:
        return j

