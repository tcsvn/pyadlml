import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyadlml.dataset.stats import contingency_duration, contingency_triggers,\
    contingency_triggers_01
from pyadlml.dataset.plot.util import func_formatter_seconds2time, \
    heatmap_square, heatmap, annotate_heatmap, heatmap_contingency
from pyadlml.dataset.activities import add_idle

DEV_CON_HM = {(8,12):(8,7), (8,14):(8,8), (8,3):(12,10), (26,20):(10,10), (11,17):(10,10)}
DEV_CON_01_HM = {(8,14):(12,10), (11,24):(12,6), (11,34):(16,12)}
DEV_CON_01_HM_WT = {(10,24):(14,7), (7, 28):(16,10)}


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
    textcolors = ("white", "black")
    valfmt = "{x:.0f}"

    if con_tab is None:
        ct = contingency_triggers(df_dev, df_act, idle=idle)
    else:
        ct = con_tab
    
    vals = ct.values.T
    acts = ct.columns.values
    devs = ct.index.values
    heatmap_contingency(acts, devs, vals, title, cbarlabel, valfmt=valfmt,
        textcolors=textcolors, z_scale=z_scale, numbers=numbers, figsize=figsize)


def heatmap_contingency_triggers_01(df_dev=None, df_act=None, con_tab_01=None, figsize=None, \
    idle=True, z_scale=None, numbers=None):
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
    textcolors = ("white", "black")
    log = (z_scale == 'log')
    # if log than let automatically infer else
    valfmt = (None if log else "{x:.0f}")


    if con_tab_01 is None:
        df_con = contingency_triggers_01(df_dev.copy(), df_act, idle=idle)
    else:
        df_con = con_tab_01.copy()

    # format text strings   
    df = df_con.reset_index()
    df['index'] = df['device'] + df['val'].astype(str)
    df['index'] = df['index'].apply(lambda x: x[:-len("False")] if "False" in x else "On")
    df = df.set_index('index')
    df = df.drop(['device', 'val'], axis=1)
    vals = df.values.T
    acts = df.columns
    devs = list(df.index)

    heatmap_contingency(acts, devs, vals, title, cbarlabel, textcolors=textcolors,
        valfmt=valfmt, z_scale=z_scale, numbers=numbers, figsize=figsize)

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

    title='Mutual time: activities vs. devices'
    cbarlabel='mutual time in seconds'

    if con_tab is None:
        if idle:
            df_act = add_idle(df_act.copy())
        df_con = contingency_duration(df_dev, df_act)
    else:
        df_con = con_tab

    # convert time (ns) to seconds
    df_con = df_con.astype(int)/1000000000

    # rename labels
    df_con = df_con.reset_index(drop=False)
    df_con['index'] = df_con['index'].apply(lambda x: x if "Off" in x else "On")
    df_con = df_con.set_index('index')

    # set values
    vals = df_con.values.T
    acts = df_con.columns.values
    devs = list(df_con.index)

    valfmt = matplotlib.ticker.FuncFormatter(lambda x, p: func_formatter_seconds2time(x, p))
    
    heatmap_contingency(acts, devs, vals, title, cbarlabel,
                         valfmt=valfmt, figsize=figsize, z_scale=z_scale,
                         numbers=numbers)
