import functools
from pyadlml.dataset.stats.discrete import contingency_table_01 as cn01, cross_correlation as cc
from pyadlml.dataset.plot.matplotlib.util import heatmap_contingency as hm_cont, save_fig
from pyadlml.dataset.plot.matplotlib.util import annotate_heatmap, heatmap, heatmap_square, \
    _num_bars_2_figsize, func_formatter_log_1
from pyadlml.dataset.util import extract_kwargs
from pyadlml.util import get_primary_color, get_diverging_color, get_secondary_color
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np



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