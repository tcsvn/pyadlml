import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyadlml.dataset.stats import contingency_intervals, contingency_table_triggers, contingency_table_triggers_01
from pyadlml.dataset.plot.util import func_formatter_sec, heatmap_contingency
from pyadlml.dataset.plot.util import heatmap_square, heatmap, annotate_heatmap
from pyadlml.dataset.activities import add_idle


DEV_CON_HM = {(8,12):(8,7), (8,14):(8,8), (8,3):(12,10), (26,20):(10,10), (11,17):(10,10)}
DEV_CON_01_HM = {(8,14):(12,10), (11,24):(12,6), (11,34):(16,12)}
DEV_CON_01_HM_WT = {(10,24):(14,7), (7, 28):(16,10)}

def _hm_2D_NN(val, hm):
    """ maps nearest neighbor of tuple keys of an hashmap to value
    """
    def dist(i,j):
        return np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2)
    
    min_dist = np.inf
    for key in hm:
       diff = dist(key, val)
       if diff < min_dist:
           min_dist = diff
           nearest_neighbor = key
    return hm[nearest_neighbor]

def nk_2_figsize(nk, hm, figsize=None):
    assert isinstance(nk, tuple)
    assert isinstance(hm, dict)
    if figsize is not None:
        return figsize
    else:
        return _hm_2D_NN(nk, hm)

def heatmap_contingency_triggers(df_dev=None, df_act=None, con_tab=None, figsize=None,\
        idle=False, z_scale=None, numbers=True):
    """ computes a table where the device triggers are counted against the activities

    Parameters
    ----------
    df_dev : pd.DataFrame
        device representation 1
    df_activities : pd.DataFrame
        activities of daily living
    con_tab : pd.DataFrame
        the computed contingency table
    Returns
    -------
    fig 
    """
    assert (df_dev is not None and df_act is not None) or con_tab is not None

    title = 'Triggers'
    cbarlabel = 'counts'

    if con_tab is None:
        ct = contingency_table_triggers(df_dev, df_act, idle=idle)
    else:
        ct = con_tab
    
    vals = ct.values.T
    acts = ct.columns
    devs = ct.index
    figsize =  nk_2_figsize(nk=(len(acts), len(devs)), hm=DEV_CON_HM, figsize=figsize)

    fig, ax = plt.subplots(figsize=figsize)
    if z_scale == 'log':
        log = True
    else:
        log = False


    im, cbar = heatmap(vals, acts, devs, ax=ax, log=log, cbarlabel=cbarlabel)

    if numbers:
        texts = annotate_heatmap(im, textcolors=("white", "black"), log=log, valfmt="{x}")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

def heatmap_contingency_triggers_01(df_dev=None, df_act=None, con_tab_01=None, figsize=None, \
    idle=True, z_scale=None, numbers=True):
    """ computes a table where the device on and off triggers are counted against the activities

    Parameters
    ----------
    df_dev : pd.DataFrame
        device representation 1
    df_activities : pd.DataFrame
        activities of daily living
    con_tab : pd.DataFrame
        the computed contingency table
    Returns
    -------
    fig 
    """
    assert (df_dev is not None and df_act is not None) or con_tab_01 is not None

    title = 'On/Off triggers'
    cbarlabel = 'counts'

    if con_tab_01 is None:
        df_con = contingency_table_triggers_01(df_dev, df_act, idle=idle)
    else:
        df_con = con_tab_01

    tmp = df_con.reset_index()
    tmp['index'] = tmp['device'] + ' ' + tmp['val'].astype(str)
    tmp = tmp.set_index('index')
    tmp = tmp.drop(['device', 'val'], axis=1)

    vals = tmp.values.T
    acts = tmp.columns
    devs = list(tmp.index)

    figsize =  nk_2_figsize(nk=(len(acts), len(devs)), hm=DEV_CON_01_HM, figsize=figsize)

    # replace every second label with 'On'
    for i in range(0,len(devs)):
        if i % 2 != 0:
            devs[i] = 'On'

    if z_scale == 'log':
        log = True
    else:
        log = False



    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap(vals, acts, devs, ax=ax, log=log, cbarlabel=cbarlabel)
    if numbers: 
        texts = annotate_heatmap(im, textcolors=("white", "black"), log=log, valfmt="{x}")
    
    # create grid for heatmap into every pair
    tcks = np.arange((vals.shape[1])/2)*2 + 1.5
    ax.set_xticks(tcks, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False) 
    
    ax.set_title(title)
    fig.tight_layout()
    return fig

def heatmap_contingency_overlaps(df_dev=None, df_act=None, con_tab=None, figsize=None, \
    z_scale='log', idle=False, numbers=True):
    """ computes a table where the device on and off intervals are measured against 
    the activities

    Parameters
    ----------
    df_dev : pd.DataFrame
        device representation 1
    df_activities : pd.DataFrame
        activities of daily living
    con_tab : pd.DataFrame
        the computed contingency table
    Returns
    -------
    fig 
    """    
    assert (df_dev is not None and df_act is not None) or con_tab is not None

    title='Cross correlation activites'
    cbarlabel='interval overlap in seconds'

    if con_tab is None:
        if idle:
            df_act = add_idle(df_act.copy())
        df_con = contingency_intervals(df_dev, df_act) 
    else:
        df_con = con_tab

    # convert time to seconds
    df_con = df_con.astype(int)/1000000000
            
    vals = df_con.values.T
    acts = df_con.columns
    devs = list(df_con.index)

    figsize =  nk_2_figsize(nk=(len(acts), len(devs)), hm=DEV_CON_01_HM_WT, figsize=figsize)
    #if z_scale == 'log':
    #    format_func = lambda x, p: func_formatter_sec(np.exp(x), p)
    #else:
    format_func = lambda x, p: func_formatter_sec(x, p)
        
    valfmt = matplotlib.ticker.FuncFormatter(format_func)
    
    heatmap_contingency(vals, acts, devs, cbarlabel, title, 
                         valfmt, figsize, z_scale=z_scale, 
                         numbers=numbers)
