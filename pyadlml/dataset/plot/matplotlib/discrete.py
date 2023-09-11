import functools
from pyadlml.constants import ACTIVITY, END_TIME, START_TIME
from pyadlml.dataset.plot.util import CatColMap
from pyadlml.dataset.stats.discrete import contingency_table_01 as cn01, cross_correlation as cc
from pyadlml.dataset.plot.matplotlib.util import heatmap_contingency as hm_cont, save_fig
from pyadlml.dataset.plot.matplotlib.util import annotate_heatmap, heatmap, heatmap_square, \
    _num_bars_2_figsize, func_formatter_log_1
from pyadlml.dataset.util import extract_kwargs
from pyadlml.util import get_primary_color, get_diverging_color, get_secondary_color
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from copy import copy



def ensure_yis_series(func):
    @extract_kwargs
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        y = kwargs.pop('y')
        if isinstance(y, pd.DataFrame) and len(y.columns) == 1:
            y = y.iloc[:,0]
        kwargs['y'] = y
        return func(*args, **kwargs)
    return wrapper


@save_fig
@ensure_yis_series
def contingency_table(X, y, rep='', z_scale=None, figsize=None, numbers=True, file_path=None):
    """ plots the contingency between features and labels of data
    Parameters
    ----------
    X: pd.DataFrame
        one of the representation raw, lastfired, changepoint
    y: pd.DataFrame
        a series of labels
    rep: string
        the name of the representation to add to the title
    """
    title = rep + " On/Off contingency"
    cbarlabel = 'counts'
    valfmt = ("{x:.0f}" if z_scale!='log' else func_formatter_log_1)

    df_con = cn01(X, y)
    vals = df_con.values.T
    acts = df_con.columns.values
    devs = list(df_con.index)
    
    # format labels by replacing every 'on'
    for i, dev in enumerate(devs):
        if 'On' in dev:
            devs[i] = 'On'
    fig =  hm_cont(acts, devs, vals, title, cbarlabel, z_scale=z_scale, figsize=figsize, valfmt=valfmt, numbers=numbers)
    plt.tight_layout()
    return fig

@save_fig
def device_fraction(X, figsize=(9,4), color=None, file_path=None):

    title = 'Fraction'
    x_label = 'count'

    color = (get_primary_color() if color is None else color)
    color2 = get_secondary_color()


    df = X.apply(lambda x: x.value_counts())
    off_values = df.loc[0,:]
    on_values = df.loc[1,:]


    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(x_label)


    # Plot 0os 
    ax.barh(df.columns, off_values, label='0', color=color)
    ax.barh(df.columns, on_values, left=off_values, label='1', color=color2)

    return fig

@save_fig
@ensure_yis_series
def activity_count(y, scale="linear", color=None, figsize=(9,3), file_path=None):
    """ Plot activity count

    Parameters
    ----------
    y: np array or pd.Series
        label of strings
    scale: None or log

    """
    assert scale in ['linear', 'log']

    title = 'Label frequency'
    xlabel = 'counts'
    color = (get_primary_color() if color is None else color)

    ser = pd.Series(data=y).value_counts()
    ser = ser.sort_values(ascending=True)

    figsize = (_num_bars_2_figsize(len(ser)) if figsize is None else figsize)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if scale == 'log': 
        plt.xscale('log')

    ax.barh(ser.index, ser.values, orientation='horizontal', color=color)
    ax.set_xlabel(xlabel)
    fig.suptitle(title)

    plt.tight_layout()
    return fig


@save_fig
def mutual_info(X, y, scale="linear", color=None, figsize=(9,3), file_path=None):
    """ Plot activity count

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        asdf
    y: np array or pd.Series
        label of strings
    X
    scale: None or log

    """
    assert scale in ['linear', 'log']

    title = 'Mutual Information: I(X,y) = H(y)-H(y|X)'
    xlabel = 'I(X,y)'
    color = (get_primary_color() if color is None else color)

    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X, y, discrete_features=True)


    figsize = (_num_bars_2_figsize(len(X.columns)) if figsize is None else figsize)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if scale == 'log': 
        plt.xscale('log')

    ax.barh(X.columns, mi, orientation='horizontal', color=color)
    ax.set_xlabel(xlabel)
    fig.suptitle(title)

    plt.tight_layout()
    return fig



def corr_devices_01(rep, figsize=(19,14)):
    """ correlation between the on and off states of every device.
    Parameters
    ----------
    rep: pd.DataFrame
        A dataframe where columns are devices and rows are a binary representation
        e.g raw, changepoint, lastfired representation
    
    """
    df = rep.iloc[:,:-1] # get raw without activities
    df = df.reset_index(drop=True)
    df = df.astype(int)

    for device in df.columns:
        mask1 = df[[device]] == 1
        col_off = device + ' Off'
        df[[col_off]] = df[[device]].astype(bool)
        df[[col_off]] = ~df[[col_off]]
        df[[col_off]] = df[[col_off]].astype(int)
        df = df.rename(columns={device: device + ' On'})
        
    dev_count = int(len(df.columns)/2)
    
    ct = df.corr()
    ct_on_on = ct.iloc[:dev_count, :dev_count]
    ct_on_on_values = ct_on_on.values.T
    ct_on_on_devs = list(ct_on_on.index)
    
    ct_on_off = ct.iloc[dev_count:, :dev_count]
    ct_on_off_values = ct_on_off.values.T
    ct_on_off_rows = list(ct_on_off.index)
    ct_on_off_cols = list(ct_on_off.columns)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
    
    # plot on-on corr
    im, cbar = heatmap(ct_on_on_values, ct_on_on_devs, ct_on_on_devs, 
                       ax=ax1, cmap='PuOr', cbarlabel='counts',
                       vmin=-1, vmax=1)

    texts = annotate_heatmap(im, textcolors=("black", "white"), valfmt="{x:.2f}")
    
    # plot on-off corr
    im, cbar = heatmap(ct_on_off_values, ct_on_off_rows, ct_on_off_cols, 
                       ax=ax2, cmap='PuOr', cbarlabel='counts',
                       vmin=-1, vmax=1)
    
    texts = annotate_heatmap(im, textcolors=("black", "white"), valfmt="{x:.2f}")
    
    ax1.set_title("Correlation of on-on signals")
    ax2.set_title("Correlation of on-off signals")
    fig.tight_layout()
    plt.show()



def heatmap_cross_correlation(df_dev, figsize=(10,8)):


    title = 'Devices cross-correlation'
    cmap = get_diverging_color()
    cbarlabel = 'similarity'

    ct = cc(df_dev)
    vals = ct.values.T.astype(np.float64)
    devs = list(ct.index)
    
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap_square(vals, devs, devs, ax=ax, cmap=cmap, cbarlabel=cbarlabel,
                       vmin=-1, vmax=1)
    annotate_heatmap(im, textcolors=("black", "white"), valfmt="{x:.2f}", threshold=0.6)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


    import matplotlib.pyplot as plt

from pyadlml.dataset._core.activities import is_activity_df
from pyadlml.dataset.plot.matplotlib.util import xaxis_format_time2
from pyadlml.dataset.plot.plotly.discrete import discreteRle2timeRle, rlencode
import warnings

def time2num(ser: np.ndarray, start_time: pd.Timedelta):
    """ Converts """
    td = (pd.to_datetime(ser) - start_time)
    if isinstance(td, pd.Series):
        td = td.dt
    x = td.total_seconds()
    return x

def _plot_confidences_into(fig, col, row, y_prob, act_order, cat_col_map, y_times, start_time, alpha=0.4):

    # Retrive correct axes
    ax = np.atleast_2d(fig.get_axes())[row-1, col-1]
    for act in act_order:
        cat_col_map.update(act, fig)

    x_values = time2num(y_times, start_time)
    ax.stackplot(
        x_values, 
        y_prob.T,
        labels=act_order, 
        colors =[cat_col_map[act] for act in act_order],
        alpha=alpha,
        step='post'
        )
    ax.set_ylim(0, 1)
    return fig

def _mp_get_axes(fig, row, col):
    ax = np.array(fig.get_axes())
    ax = ax if len(ax.shape) == 2 else ax[:, None]
    ax = ax[row-1, col-1]
    return ax


def _plot_activity_bar(fig, y: pd.DataFrame, activities: list, row, col, 
                       cat_col_map, time, act_bar_height, y_pos, y_label, 
                       start_time,
                       alpha=0.4):

    """ Plot a horizontal activity event representation.
    Parameters
    ----------
    c_map : np.ndarray
        Array of colors where each color corresponds to one activity
    """
    if is_activity_df(y):
        df = y.copy()
        df = df.rename(columns={START_TIME:'start', END_TIME:'end'})
        df['lengths'] = (df['end'] - df['start'])/pd.Timedelta('1ms')
        activities = df[ACTIVITY].unique()
    else:
        if activities is None:
            activities = np.unique(y)
        time = pd.Series(time) 

        df = pd.DataFrame(data=zip(*rlencode(y)), columns=['start', 'lengths', ACTIVITY])
        if time is not None:
            df = discreteRle2timeRle(df, time)
        # The last prediction has no length!!!
        df = df.loc[df.index[:-1]]
        df['end'] = df['start'] + df['lengths']
        if time is not None:
            df['lengths' ] = df['lengths'].astype("timedelta64[s]")

    ax = _mp_get_axes(fig, row, col)
    df['num_st'] = time2num(df['start'], start_time)
    df['num_et'] = time2num(df['end'], start_time)
    df['diff'] = df['num_et'] - df['num_st']


    # create list of lists, where each list corresponds to an activity with tuples of start_time and time_length
    for act in activities:
        x_range = df.loc[df[ACTIVITY] == act, ['num_st', 'diff']].values.tolist()
        ax.broken_barh(x_range, (y_pos, act_bar_height), linewidth=0,
                       color=cat_col_map[act], label=act, alpha=alpha)
    return fig


def plot_acts_and_probs(y_true=None, y_pred=None, y_prob=None, y_times=None, act_order=[]):
    import matplotlib.gridspec as gridspec

    if y_prob is None:
        cols, rows = 1,1
    else:
        cols, rows = 1,2
        assert len(y_prob.shape) == 2, "y_prob should be a 2D array"
        assert y_prob.shape[1] >= 2, "y_prob should have at least two columns"


    title = 'asdf'
    cat_col_map = CatColMap(plotly=False, theme='set')
    # Equation for position is ki-h/2. 
    # Gap between bars is e=k-h. Fix gap as 20% of bar length 
    # leads to k=e+h
    bar_height = 1
    e = 0.25
    pos_func = lambda i: i*(e+bar_height)
    pos_bar_func = lambda i: i*(e+bar_height) - bar_height/2
    #pos_bar_func = lambda i: i- bar_height/2
    alpha = 0.6
    start_time = y_times[0]
    end_time = y_times[-1]

    # Create a figure and a grid of subplots with 1 row and 3 columns
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 4),
                            gridspec_kw={'height_ratios':[6, 4]}
    )
    #gs = gridspec.GridSpec(1, 2, width_ratios=[8,2])

    # Show plot
    if y_prob is not None:
        fig = _plot_confidences_into(fig, 1, 1, y_prob, act_order, cat_col_map,  
                                     y_times, alpha=alpha, start_time=start_time
        )
        ax = np.atleast_2d(fig.get_axes())[0, 0]
        ax.set_ylabel('Probs')

    y_act_labels = []

    if y_true is not None:
        fig = _plot_activity_bar(fig, y_true, y_label='y_true', cat_col_map=cat_col_map, 
                                 row=rows, col=1, time=y_times, activities=act_order, 
                                 act_bar_height=bar_height, y_pos=pos_bar_func(0),
                                 alpha=alpha, start_time=start_time
            )
        y_act_labels.append('y_true')

    if y_pred is not None:
        fig = _plot_activity_bar(fig, y_pred, y_label='y_pred', cat_col_map=cat_col_map, 
                                 row=rows, col=1, time=y_times, activities=act_order, 
                                 act_bar_height=bar_height, y_pos=pos_bar_func(1),
                                 alpha=alpha, start_time=start_time
            )
        y_act_labels.append('y_pred')

    if y_pred is not None or y_true is not None:
        axes[1].set_yticks(np.vectorize(pos_func)(np.arange(len(y_act_labels))))
        axes[1].set_yticklabels(y_act_labels)


    handles = []
    import matplotlib.patches as mpatches

    # Plot data and create a Line2D object for each entry, and add them to handles
    for color, name in cat_col_map.items():
        # Create a Line2D object for the legend and add to handles
        handles.append(mpatches.Patch(color=color, label=name, alpha=alpha))

    # Create a legend with the handles and labels
    fig.legend(loc='upper left', bbox_to_anchor=(1.01, 1), 
               borderaxespad=0, handles=handles
    )
    # Format labels
    axes[0].set_xticks([])
    xaxis_format_time2(fig, axes[1], start_time, end_time)

    return fig



def _plot_devices_into(fig, X, time, row, col, cat_col_map, dev_order=None):
    ON = 1
    OFF = 0

    # Determine device order
    if dev_order is None:
        device_order = X.columns.to_list() 
    elif isinstance(dev_order, np.ndarray):
        device_order = device_order.tolist() 
    elif isinstance(dev_order, str) and dev_order == 'alphabetical':
        device_order = X.columns.to_list()
        device_order.sort(reverse=True)
    else:
        device_order = dev_order

    def _is_in_powerset(s, base_set):
        import itertools
        return any(set(combo) == s for i in range(len(base_set) + 1) 
                                   for combo in itertools.combinations(base_set, i))

    start_time = time[0]
    bar_height = 0.3
    devs = X.columns.to_list() if dev_order is None else copy(dev_order)
    devs = devs.tolist() if isinstance(devs, np.ndarray) else devs
    dev_pos = np.arange(len(devs))

    # Split in binary and numerical features
    binary_features = []
    numerical_features = []
    for dev in devs:
        vals = set(X[dev].unique())
        assert vals != {}, f'No values for feature {dev}'
        if _is_in_powerset(vals, {0, 1}) or _is_in_powerset(vals, {-1, 1}):
            binary_features.append(dev)
        else:
            numerical_features.append(dev)

    bars = {f:{OFF:[], ON:[]} for f in binary_features}
    for f in binary_features:
        bars[f] = pd.DataFrame(data=zip(*rlencode(X[f])), columns=['start', 'lengths', 'state'])
        if time is not None:
            df = bars[f]
            df = discreteRle2timeRle(df, time).copy()
            df['lengths'] = df['lengths'].astype("timedelta64[ms]")
            df['lengths'] = df['lengths'].astype("timedelta64[ms]")
            df['end'] = df['start'] + df['lengths']
            df['num_st'] = time2num(df['start'], start_time)
            df['num_et'] = time2num(df['end'], start_time)
            df['diff'] = df['num_et'] - df['num_st']
            bars[f] = df


    ax = _mp_get_axes(fig, row, col)
    bin_label_set = False
    num_label_set = False
    for i, dev in enumerate(devs):
        if dev in binary_features:
            on_mask = (bars[dev]['state'] == 1)
            x_range = bars[dev][on_mask][['num_st', 'diff']].values.tolist()
            label = None if bin_label_set else 1
            ax.broken_barh(x_range, (i-bar_height/2, bar_height), linewidth=0,
                        color=cat_col_map[1], label=label, alpha=1.0)
            x_range = bars[dev][~on_mask][['num_st', 'diff']].values.tolist()
            label = None if bin_label_set else 0
            ax.broken_barh(x_range, (i-bar_height/2, bar_height), linewidth=0,
                        color=cat_col_map[0], label=label, alpha=1.0)
            bin_label_set = True
        else:
            times_x = time2num(time, start_time)
            values = pd.to_numeric(X[dev])
            values_norm = (values-values.min())/(values.max()-values.min())*0.5
            values = values_norm + i - 0.25
            label = None if num_label_set else 'num'
            ax.plot(times_x, values, color='blue', label=label)
            num_label_set = True
    
    ax.set_yticks(dev_pos)
    ax.set_yticklabels(devs)

    return fig


def plot_devices(X, times, figsize=(12, 14)):

    cols = 1
    rows = 1
    row_heights = [1.0]
    cat_col_map = CatColMap()
    specs = [[{"secondary_y": True}]]


    device_order = X.columns.to_list()
    #device_order.sort(reverse=True)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)


    fig = _plot_devices_into(fig, X, times, cat_col_map=cat_col_map, row=rows, col=cols, 
                                dev_order=device_order
        )

    start_time = times[0]
    end_time = times[-1]

    axes.set_xticks([])
    xaxis_format_time2(fig, axes, start_time, end_time)
    axes.set_ylim(-1, len(device_order))
    axes.legend(loc='upper right')

    return fig
