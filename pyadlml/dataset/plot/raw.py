from pyadlml.dataset.stats.raw import contingency_table_01, cross_correlation
from pyadlml.dataset.plot import _annotate_heatmap, _heatmap
from pyadlml.dataset.plot.util import annotate_heatmap, heatmap

import matplotlib.pyplot as plt 
import matplotlib
import pandas as pd 
import numpy as np

def heatmap_contingency_01(X, y, scale='log', figsize=(12,10)):
    """
    """
    
    cbarlabel = 'counts'
    title = "On/Off contingency for representation"

    df_con = contingency_table_01(X, y)
    
    vals = df_con.values.T
    acts = df_con.columns
    devs = list(df_con.index)
    
    if scale == 'log':
        with np.errstate(divide='ignore'):
            vals = np.log(vals)
            vals[vals == -np.inf] = -0.0001
            cbarlabel = 'log counts'
    
    # format x labels
    for i in range(0,len(devs)):
        if i % 2 == 0:
            tmp = devs[i][:-3]
            devs[i] = tmp + 'Off'
        else:
            devs[i] = 'On'

    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = _heatmap(vals, acts, devs, ax=ax, cmap='viridis', cbarlabel=cbarlabel)
    
    texts = _annotate_heatmap(im, textcolors=("white", "black"), valfmt="{x:.1f}")
    
    # create grid for heatmap into every pair
    tcks = np.arange((vals.shape[1])/2)*2 + 1.5
    ax.set_xticks(tcks, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False) 
    
    ax.set_title(title)
    fig.tight_layout()
    return fig

def hist_activities(y, scale=None):
    """
    Parameters
    ----------
    y: np array
        label of strings
    scale: None or log
    """
    title = 'Label occurence'
    xlabel = 'counts'
    
    ser = pd.Series(data=y).value_counts()
    ser = ser.sort_values(ascending=True)

    fig, ax = plt.subplots(1, 1, figsize=(9, 3))
    if scale == 'log': 
        plt.xscale('log')
        xlabel = 'log ' + xlabel
    ax.barh(ser.index, ser.values, orientation='horizontal')
    
    plt.xlabel(xlabel)
    fig.suptitle(title)


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
    ct = cross_correlation(df_dev)
    vals = ct.values.T.astype(np.float64)
    devs = list(ct.index)
    
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap(vals, devs, devs, ax=ax, cmap='PuOr', cbarlabel='counts',
                       vmin=-1, vmax=1)

    texts = annotate_heatmap(im, textcolors=("black", "white"), valfmt="{x:.2f}")

    ax.set_title("Cross-correlation of signals")
    fig.tight_layout()
    plt.show()