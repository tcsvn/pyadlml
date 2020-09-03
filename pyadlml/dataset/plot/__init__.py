import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyadlml.dataset.stats import contingency_table_interval_overlaps, contingency_table_triggers, contingency_table_triggers_01
from pyadlml.dataset.plot.util import func_formatter_sec, heatmap_contingency


def heatmap_contingency_triggers(df_dev_rep3, df_act, figsize=(12,10), idle=False):
    """
    plot a heatmap TODO write docu
    """
    tmp = contingency_table_triggers(df_dev_rep3, df_act, idle=idle)
    vals = tmp.values.T
    acts = tmp.columns
    devs = tmp.index

    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = _heatmap(vals, acts, devs, ax=ax, cmap='viridis', cbarlabel='counts')

    texts = _annotate_heatmap(im, textcolors=("white", "black"), valfmt="{x}")

    ax.set_title("Triggers")
    fig.tight_layout()
    plt.show()

def heatmap_contingency_triggers_01(df_dev_rep3, df_act, figsize=(12,10), idle=True):
    """
    """
    df_con = contingency_table_triggers_01(df_dev_rep3, df_act, idle=idle)
    tmp = df_con.reset_index()
    tmp['index'] = tmp['device'] + ' ' + tmp['val'].astype(str)
    tmp = tmp.set_index('index')
    tmp = tmp.drop(['device', 'val'], axis=1)

    vals = tmp.values.T
    acts = tmp.columns
    devs = list(tmp.index)
    
    # format x labels
    for i in range(0,len(devs)):
        if i % 2 == 0:
            tmp = devs[i][:-5]
            devs[i] = tmp + 'Off'
        else:
            devs[i] = 'On'

    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = _heatmap(vals, acts, devs, ax=ax, cmap='viridis', cbarlabel='counts')
    
    texts = _annotate_heatmap(im, textcolors=("white", "black"), valfmt="{x}")
    
    # create grid for heatmap into every pair
    tcks = np.arange((vals.shape[1])/2)*2 + 1.5
    ax.set_xticks(tcks, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False) 
    
    ax.set_title("0-1 triggers")
    fig.tight_layout()
    return fig

def heatmap_contingency_overlaps(df_dev, df_act, figsize=(18,12), scale='log'):
    cbarlabel='log second overlap'
    title='Cross correlation activites'
    df_con = contingency_table_interval_overlaps(df_act, df_dev) 
    
    tmp = df_con.reset_index()
    tmp['index'] = tmp['device'] + ' ' + tmp['val'].astype(str)
    tmp = tmp.set_index('index')
    tmp = tmp.drop(['device', 'val'], axis=1)
    
    # convert time to seconds
    tmp = tmp.astype(int)/1000000000
    if scale == 'log':
        with np.errstate(divide='ignore'):
            tmp = np.log(tmp)
            tmp[tmp == -np.inf] = 0
            
    vals = tmp.values.T
    acts = tmp.columns
    devs = list(tmp.index)
    
    if scale == 'log':
        format_func = lambda x, p: func_formatter_sec(np.exp(x), p)
    else:
        format_func = lambda x, p: func_formatter_sec(x, p)
        
    valfmt = matplotlib.ticker.FuncFormatter(format_func)
    
    
    heatmap_contingency(vals, acts, devs, cbarlabel, title, 
                         valfmt, figsize)


def _annotate_heatmap(im, data=None, valfmt="{}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    from pyadlml.dataset.plot.util import annotate_heatmap
    return annotate_heatmap(im, data=data, valfmt=valfmt,
                     textcolors=textcolors,
                     threshold=threshold, **textkw)

def _heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    from pyadlml.dataset.plot.util import heatmap
    return heatmap(data, row_labels, col_labels, ax=ax,
            cbar_kw=cbar_kw, cbarlabel=cbarlabel, **kwargs)

