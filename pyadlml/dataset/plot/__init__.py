import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyadlml.dataset.stats import contingency_table_interval_overlaps, contingency_table_triggers, contingency_table_triggers_01
from pyadlml.dataset.plot.util import func_formatter_sec, heatmap_contingency
from pyadlml.dataset.plot.util import heatmap_square, heatmap, annotate_heatmap
from pyadlml.dataset.activities import add_idle

def heatmap_contingency_triggers(df_dev_rep3, df_act, figsize=(12,10), idle=False, z_scale=None):
    """
    plot a heatmap TODO write docu
    """

    title = 'Triggers'
    cbarlabel = 'counts'

    tmp = contingency_table_triggers(df_dev_rep3, df_act, idle=idle)
    vals = tmp.values.T
    acts = tmp.columns
    devs = tmp.index

    fig, ax = plt.subplots(figsize=figsize)
    if z_scale == 'log':
        log = True
    else:
        log = False


    im, cbar = heatmap(vals, acts, devs, ax=ax, log=log, cbarlabel=cbarlabel)

    texts = annotate_heatmap(im, textcolors=("white", "black"), log=log, valfmt="{x}")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

def heatmap_contingency_triggers_01(df_dev_rep3, df_act, figsize=(12,10), idle=True, z_scale=None):
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

    if z_scale == 'log':
        log = True
    else:
        log = False



    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap(vals, acts, devs, ax=ax, log=log, cbarlabel='counts')
    
    texts = annotate_heatmap(im, textcolors=("white", "black"), log=log, valfmt="{x}")
    
    # create grid for heatmap into every pair
    tcks = np.arange((vals.shape[1])/2)*2 + 1.5
    ax.set_xticks(tcks, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False) 
    
    ax.set_title("0-1 triggers")
    fig.tight_layout()
    return fig

def heatmap_contingency_overlaps(df_dev, df_act, figsize=(18,12), z_scale='log', idle=False):
    """
    
    """
    cbarlabel='second overlap'
    title='Cross correlation activites'

    if idle:
        df_act = add_idle(df_act.copy())

    df_con = contingency_table_interval_overlaps(df_act, df_dev) 
    
    tmp = df_con.reset_index()
    tmp['index'] = tmp['device'] + ' ' + tmp['val'].astype(str)
    tmp = tmp.set_index('index')
    tmp = tmp.drop(['device', 'val'], axis=1)
    
    # convert time to seconds
    tmp = tmp.astype(int)/1000000000
            
    vals = tmp.values.T
    acts = tmp.columns
    devs = list(tmp.index)

    #if z_scale == 'log':
    #    format_func = lambda x, p: func_formatter_sec(np.exp(x), p)
    #else:
    format_func = lambda x, p: func_formatter_sec(x, p)
        
    valfmt = matplotlib.ticker.FuncFormatter(format_func)
    
    heatmap_contingency(vals, acts, devs, cbarlabel, title, 
                         valfmt, figsize, z_scale=z_scale)
