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


def heatmap_contingency_triggers(df_devs=None, df_acts=None, df_con_tab=None, idle=False, \
                                 z_scale=None, numbers=True, figsize=None, file_path=None
                                  ):
    """ computes a table where the device triggers are counted against the activities

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`. If the parameter *df_devs* is not set,
        the parameter *df_con_tab* has to be set.
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`. If the parameter *df_acts* is not set,
        the parameter *df_con_tab* has to be set.
    df_con_tab : pd.DataFrame, optional
        A precomputed contingency table. If the *df_con_tab* parameter is given, parameters
        *df_acts* and *df_devs* are ignored. The contingency table can be computed
        in :ref:`stats <stats_dna_con_dur>`.
    figsize : (float, float), optional
        width, height in inches. If not provided, the figsize is inferred by automatically.
    z_scale : {"log", "linear"}, default: 'log'
        The axis scale type to apply.
    numbers : bool, default: True
        Whether to display numbers inside the heatmaps fields or not.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_hm_contingency_trigger
    >>> plot_hm_contingency_trigger(data.df_devs, data.df_activities)

    .. image:: ../_static/images/plots/cont_hm_trigger.png
       :height: 300px
       :width: 500 px
       :scale: 100 %
       :alt: alternate text
       :align: center

    Returns
    -------
    fig : Figure or None
        If the parameter file_path is specified, the method return None rather than a matplotlib figure.
    """
    assert (df_devs is not None and df_acts is not None) or df_con_tab is not None

    title = 'Triggers'
    cbarlabel = 'counts'
    textcolors = ("white", "black")
    valfmt = "{x:.0f}"

    if df_con_tab is None:
        ct = contingency_triggers(df_devs, df_acts, idle=idle)
    else:
        ct = df_con_tab
    
    vals = ct.values.T
    acts = ct.columns.values
    devs = ct.index.values
    heatmap_contingency(acts, devs, vals, title, cbarlabel, valfmt=valfmt,
        textcolors=textcolors, z_scale=z_scale, numbers=numbers, figsize=figsize)


def heatmap_contingency_triggers_01(df_devs=None, df_acts=None, df_con_tab_01=None, figsize=None,
                                    idle=True, z_scale=None, numbers=None, file_path=None
                                    ):
    """
    Plot the device on and off triggers against the activities

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`. If the parameter *df_devs* is not set,
        the parameter *df_con_tab* has to be set.
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`. If the parameter *df_acts* is not set,
        the parameter *df_con_tab* has to be set.
    df_con_tab_01 : pd.DataFrame, optional
        A precomputed contingency table. If the *df_con_tab* parameter is given, parameters
        *df_acts* and *df_devs* are ignored. The contingency table can be computed
        in :ref:`stats <stats_dna_con_dur>`.
    figsize : (float, float), optional
        width, height in inches. If not provided, the figsize is inferred by automatically.
    z_scale : {"log", "linear"}, default: 'log'
        The axis scale type to apply.
    numbers : bool, default: True
        Whether to display numbers inside the heatmaps fields or not.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_hm_contingency_trigger_01
    >>> plot_hm_contingency_trigger_01(data.df_devs, data.df_activities)

    .. image:: ../_static/images/plots/cont_hm_trigger_01.png
       :height: 300px
       :width: 500 px
       :scale: 100 %
       :alt: alternate text
       :align: center

    Returns
    -------
    fig : Figure or None
        If the parameter file_path is specified, the method return None rather than a matplotlib figure.
    """
    assert (df_devs is not None and df_acts is not None) or df_con_tab_01 is not None

    title = 'On/Off triggers'
    cbarlabel = 'counts'
    textcolors = ("white", "black")
    log = (z_scale == 'log')
    # if log than let automatically infer else
    valfmt = (None if log else "{x:.0f}")


    if df_con_tab_01 is None:
        df_con = contingency_triggers_01(df_devs.copy(), df_acts, idle=idle)
    else:
        df_con = df_con_tab_01.copy()

    # rename labels
    df_con = df_con.reset_index(drop=False)
    df_con['index'] = df_con['index'].apply(lambda x: x if "Off" in x else "On")
    df_con = df_con.set_index('index')

    vals = df_con.values.T
    acts = df_con.columns
    devs = list(df_con.index)

    heatmap_contingency(acts, devs, vals, title, cbarlabel, textcolors=textcolors,
        valfmt=valfmt, z_scale=z_scale, numbers=numbers, figsize=figsize)

def heatmap_contingency_overlaps(df_devs=None, df_acts=None, df_con_tab=None, figsize=None, \
                                 z_scale='log', idle=False, numbers=True, file_path=None):
    """
    Plots a heatmap the device on and off intervals are measured against
    the activities

    Parameters
    ----------
    df_devs : pd.DataFrame, optional
        recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`. If the parameter *df_devs* is not set,
        the parameter *df_con_tab* has to be set.
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`. If the parameter *df_acts* is not set,
        the parameter *df_con_tab* has to be set.
    df_con_tab : pd.DataFrame, optional
        A precomputed contingency table. If the *df_con_tab* parameter is given, parameters
        *df_acts* and *df_devs* are ignored. The contingency table can be computed
        in :ref:`stats <stats_dna_con_dur>`.
    figsize : (float, float), optional
        width, height in inches. If not provided, the figsize is inferred by automatically.
    z_scale : {"log", "linear"}, default: 'log'
        The axis scale type to apply.
    numbers : bool, default: True
        Whether to display numbers inside the heatmaps fields or not.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_hm_contingency_duration
    >>> plot_hm_contingency_duration(data.df_devs, data.df_activities)

    .. image:: ../_static/images/plots/cont_hm_duration.png
       :height: 300px
       :width: 800 px
       :scale: 90 %
       :alt: alternate text
       :align: center

    Returns
    -------
    fig : Figure or None
        If the parameter file_path is specified, the method return None rather than a matplotlib figure.
    """    
    assert (df_devs is not None and df_acts is not None) or df_con_tab is not None

    title='Mutual time: activities vs. devices'
    cbarlabel='mutual time in seconds'

    if df_con_tab is None:
        if idle:
            df_acts = add_idle(df_acts.copy())
        df_con = contingency_duration(df_devs, df_acts)
    else:
        df_con = df_con_tab

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
