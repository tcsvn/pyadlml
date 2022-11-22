import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, RadioButtonGroup, CustomJS, CheckboxGroup, Select, DataRange, DataRange1d, \
    FuncTickFormatter, LogTickFormatter, Panel, Tabs

from pyadlml.dataset import ACTIVITY
from  pyadlml.dataset.stats.activities import activities_duration_dist, activity_duration,\
    activities_transitions, activities_count, activity_duration, activities_dist
from pyadlml.dataset._core.activities import add_other_activity
from pyadlml.dataset.plot.util import func_formatter_seconds2time_log, ridgeline, \
    func_formatter_seconds2time, heatmap, annotate_heatmap, heatmap_square, savefig, \
    _num_bars_2_figsize, _num_boxes_2_figsize, \
    _num_items_2_heatmap_square_figsize, _num_items_2_ridge_figsize,\
    _num_items_2_ridge_ylimit
from pyadlml.util import get_sequential_color, get_secondary_color, get_primary_color, get_diverging_color
from bokeh.plotting import figure, show

def hist_counts(df_acts=None, lst_acts=None, df_ac=None):
    """
    Plot a bar chart displaying how often activities are occurring.

    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : lst of str, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    y_scale : {"log", "linear"}, default: linear
        The axis scale type to apply.

    Examples
    --------
    >>> from pyadlml.plot import plot_activity_bar_count
    >>> plot_activity_bar_count(data.df_activities, idle=True)

    .. image:: ../_static/images/plots/act_bar_cnt.png
       :height: 300px
       :width: 500 px
       :scale: 90 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing
    """
    assert not (df_acts is None and df_ac is None)

    title ='Activity occurrences'
    col_label = 'occurrence'
    x_label = 'counts'

    # create statistics if the don't exists
    if df_ac is None:
        df_acts = df_acts.copy()
        df_acts = add_other_activity(df_acts)
        df = activities_count(df_acts, lst_acts=lst_acts)
    else:
        df = df_ac

    # prepare dataframe for plotting
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by=[col_label], axis=0)

    # create a new plot with a title and axis labels
    from bokeh.io import output_file, show
    from bokeh.models import ColumnDataSource
    from bokeh.plotting import figure
    from bokeh.models.tools import HoverTool
    from bokeh.layouts import layout

    # create data column objects
    activities = df[ACTIVITY].values
    counts = df[col_label].values

    idle_idx = np.where(activities == 'idle')[0]
    activities_w_idle = np.delete(activities.copy(), idle_idx, 0)
    counts_w_idle = np.delete(counts.copy(), idle_idx, 0)

    source = ColumnDataSource(data=dict(y=activities, x=counts))
    saved_source = ColumnDataSource(data=dict(
        activities=activities,
        counts=counts))
    saved_source_w_idle = ColumnDataSource(data=dict(
        counts=counts_w_idle,
        activities=activities_w_idle,
    ))
    upper_next_10 = np.ceil(max(counts)/(10**(len(str(max(counts))))))*10**(len(str(max(counts))))# round to the highest 10th number of the maximum

    # create linear plot
    p1 = figure(title=title, y_range=activities, x_axis_label=x_label,
                x_range=[0, np.ceil(max(counts))],
               sizing_mode='stretch_width', plot_height=400,
               tools=[HoverTool()],
               tooltips="Activity @y occurred @x times.")
    p1.hbar(y='y', left=0, right='x', height=0.9, source=source,
           legend_field="y", color=get_primary_color(),
           line_color='white')

    p1.ygrid.grid_line_color = None
    p1.legend.orientation = "vertical"
    p1.legend.location = "bottom_right"
    p1.legend.click_policy = 'mute'

    # create log plot
    p2 = figure(title=title, y_range=activities, x_axis_label=x_label, x_axis_type='log',
                x_range=[0.1, upper_next_10],
               sizing_mode='stretch_width', plot_height=400,
               tools=[HoverTool()],
               tooltips="Activity @y occurred @x times.")
    p2.hbar(y='y', left=0.1, right='x', height=0.9, source=source,
           legend_field="y", color=get_primary_color(),
           line_color='white')

    p2.ygrid.grid_line_color = None
    p2.legend.orientation = "vertical"
    p2.legend.location = "bottom_right"
    p2.legend.click_policy = 'mute'

    # create widgets
    checkbox_group = create_idle_checkbox('counts', source=source, saved_source=saved_source, saved_source_w_idle=saved_source_w_idle,
                                          p1=p1, p2=p2)
    LABELS = ["Linear", "Log"]

    radio_button_group = RadioButtonGroup(labels=LABELS, active=0)

    tab1 = Panel(child=p1, title='linear')
    tab2 = Panel(child=p2, title='log')
    tabs = Tabs(tabs=[tab1, tab2], tabs_location='below', disabled=True)
    radio_button_group.js_link('active', tabs, 'active')

    layout = layout([[tabs],[checkbox_group, radio_button_group]],
        sizing_mode='stretch_width'
    )
    show(layout)

def create_idle_checkbox(col_name, source, saved_source, saved_source_w_idle, p1, p2):
    checkbox_group = CheckboxGroup(labels=["idle"], active=[0, 1])
    callback = CustomJS(
        args=dict(source=source, saved_source=saved_source, saved_source_w_idle=saved_source_w_idle, p1=p1, p2=p2),
        code="""
        var active = cb_obj.active[0];
        var sdata = saved_source.data;
        var sdata_w_idle = saved_source_w_idle.data;
        var data = source.data;

        data['x'] = []
        data['y'] = []
    
        if (active == 0){
            console.log("set data to include idle")
            data['x'] = sdata.%s
            data['y'] = sdata.activities
        }
        else {
            console.log("set data to without idle")
            data['x'] = sdata_w_idle.%s
            data['y'] = sdata_w_idle.activities
        }
        source.data = data
        //console.log(source.data)
        source.change.emit();
        
        // the plot can only be changed after the data source changes
        if (active == 0){
            p1.y_range.factors = sdata.activities
            p2.y_range.factors = sdata.activities
        }
        else {
            p1.y_range.factors = sdata_w_idle.activities
            p2.y_range.factors = sdata_w_idle.activities
        }
        p1.change.emit();
        p2.change.emit();
        """%(col_name, col_name))
    checkbox_group.js_on_click(callback)
    return checkbox_group

def hist_counts2(df_acts=None, lst_acts=None, df_ac=None):
    """
    Plot a bar chart displaying how often activities are occurring.

    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : lst of str, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    y_scale : {"log", "linear"}, default: linear
        The axis scale type to apply.

    Examples
    --------
    >>> from pyadlml.plot import plot_activity_bar_count
    >>> plot_activity_bar_count(data.df_activities, idle=True)

    .. image:: ../_static/images/plots/act_bar_cnt.png
       :height: 300px
       :width: 500 px
       :scale: 90 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing 
    """
    assert not (df_acts is None and df_ac is None)

    title ='Activity occurrences'
    col_label = 'occurrence'
    x_label = 'counts'

    # create statistics if the don't exists
    if df_ac is None:
        df_acts = df_acts.copy()
        df_acts = add_other_activity(df_acts)
        df = activities_count(df_acts, lst_acts=lst_acts)
    else:
        df = df_ac
    
    # prepare dataframe for plotting
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by=[col_label], axis=0)

    # create a new plot with a title and axis labels
    from bokeh.io import output_file, show
    from bokeh.models import ColumnDataSource
    from bokeh.plotting import figure
    from bokeh.models.tools import HoverTool
    from bokeh.layouts import layout

    df['log'] = np.log(df[col_label])
    df['linear'] = df[col_label]

    source = ColumnDataSource(data=dict(y=[], x=[], title=[]))
    axis_map = {
        'Linear Scaling': 'linear',
        'Log Scaling': 'log',
        0: 'linear',
        1: 'log'
    }
    x_axis = RadioButtonGroup(labels=list(axis_map.keys())[:2], active=0)

    p = figure(height=600, width=700, title="", y_range=df[ACTIVITY].values,
           sizing_mode='stretch_width',
           tools=[HoverTool()],
           tooltips="Activity @y occurred @x times.")
    p.hbar(right='x', y='y', left=0, height=0.9, source=source, legend_field="y", color=get_primary_color(), line_color='white')

    def update():
        #p.y_range = DataRange1d(df[ACTIVITY].values)
        print('#'*100)
        x_name = axis_map[x_axis.active]
        if x_name == 'linear':
            p.xaxis.axis_label = 'count'
        else:
            p.xaxis.axis_label = 'log count'
            #p.xaxis.formatter = FuncTickFormatter(code="""return Math.log(tick).toFixed(2)""")
            p.xaxis.formatter = LogTickFormatter()
        p.title.text = "Test"
        print(df[x_name].values)
        print(df[ACTIVITY].values)
        source.data = dict(
            x=df[x_name].values,
            y=df[ACTIVITY].values,
        )

    x_axis.on_change('active', lambda attr, old, new: update())
    layout = layout([[p],[x_axis]],
        sizing_mode='stretch_width'
    )
    update()
    curdoc().add_root(layout)
    curdoc().title = "Test 5"
    #show(layout)


def boxplot_duration(df_acts, lst_acts=None, file_path=None):
    """
    Plot a boxplot for activity durations.

    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : lst of str, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    y_scale : {"log", "linear"}, default: None
        The axis scale type to apply.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_activity_bp_duration
    >>> plot_activity_bp_duration(data.df_activities)

    .. image:: ../_static/images/plots/act_bp.png
       :height: 300px
       :width: 500 px
       :scale: 90 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing
    """
    title = 'Activity durations'
    xlabel = 'seconds'

    df_acts = add_other_activity(df_acts)

    df = activities_duration_dist(df_acts, lst_acts=lst_acts)

    # select data for each device
    activities = df[ACTIVITY].unique()
    df['seconds'] = df['minutes']*60     

    num_act = len(activities)

    dat = []
    for activity in activities:
        df_activity = df[df[ACTIVITY] == activity]
        #tmp = np.log(df_device['td'].dt.total_seconds())
        dat.append(df_activity['seconds']) 
    
    # plot boxsplot
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(dat, vert=False)
    ax.set_title(title)
    ax.set_yticklabels(activities, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_xscale('log')

    # create secondary axis with time format 1s, 1m, 1d
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    #ax_top.set_xlabel('time')
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_seconds2time))

    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig

def hist_cum_duration(df_acts=None, lst_acts=None, df_dur=None, file_path=None):
    """
    Plots the cumulative duration for each activity in a bar plot.

    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : lst of str, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    y_scale : {"log", "linear"}, default: None
        The axis scale type to apply.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    color : str, optional
        sets the color of the plot. When not set, the primary theming color is used.
        Learn more about theming in the :ref:`user guide <theming>`
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_activity_bar_duration
    >>> plot_activity_bar_duration(data.df_activities)

    .. image:: ../_static/images/plots/act_bar_dur.png
       :height: 300px
       :width: 500 px
       :scale: 90 %
       :alt: alternate text
       :align: center

    Returns
    -------
    res : fig or None
        Either a figure if file_path is not specified or nothing
    """
    assert not (df_acts is None and df_dur is None)

    title = 'Cummulative activity durations'
    x_label = 'seconds'
    freq = 'seconds'

    if df_dur is None:
        df_acts = add_other_activity(df_acts.copy())
        df = activity_duration(df_acts, lst_acts=lst_acts, time_unit=freq)
    else:
        df = df_dur
    df = df.sort_values(by=[freq], axis=0)

    # create a new plot with a title and axis labels
    from bokeh.io import output_file, show
    from bokeh.models import ColumnDataSource
    from bokeh.plotting import figure
    from bokeh.models.tools import HoverTool
    from bokeh.layouts import layout

    col_label = freq

    # create data column objects
    activities = df[ACTIVITY].values
    counts = df[col_label].values

    idle_idx = np.where(activities == 'idle')[0]
    activities_w_idle = np.delete(activities.copy(), idle_idx, 0)
    counts_w_idle = np.delete(counts.copy(), idle_idx, 0)

    source = ColumnDataSource(data=dict(y=activities, x=counts))
    saved_source = ColumnDataSource(data=dict(
        activities=activities,
        counts=counts))
    saved_source_w_idle = ColumnDataSource(data=dict(
        counts=counts_w_idle,
        activities=activities_w_idle,
    ))
    upper_next_10 = np.ceil(max(counts)/(10**(len(str(max(counts))))))*10**(len(str(max(counts))))# round to the highest 10th number of the maximum

    # create linear plot
    p1 = figure(title=title, y_range=activities, x_axis_label=x_label,
                x_range=[0, np.ceil(max(counts))],
               sizing_mode='stretch_width', plot_height=400,
               tools=[HoverTool()],
               tooltips="Activity @y occurred @x seconds")
    p1.hbar(y='y', left=0, right='x', height=0.9, source=source,
           legend_field="y", color=get_primary_color(),
           line_color='white')

    p1.ygrid.grid_line_color = None
    p1.legend.orientation = "vertical"
    p1.legend.location = "bottom_right"
    p1.legend.click_policy = 'mute'

    # create log plot
    p2 = figure(title=title, y_range=activities, x_axis_label=x_label, x_axis_type='log',
                x_range=[0.1, upper_next_10],
               sizing_mode='stretch_width', plot_height=400,
               tools=[HoverTool()],
               tooltips="Activity @y occurred @x times.")
    p2.hbar(y='y', left=0.1, right='x', height=0.9, source=source,
           legend_field="y", color=get_primary_color(),
           line_color='white')

    p2.ygrid.grid_line_color = None
    p2.legend.orientation = "vertical"
    p2.legend.location = "bottom_right"
    p2.legend.click_policy = 'mute'

    # create widgets
    checkbox_group = create_idle_checkbox('counts', source=source, saved_source=saved_source, saved_source_w_idle=saved_source_w_idle,
                                          p1=p1, p2=p2)
    LABELS = ["Linear", "Log"]

    radio_button_group = RadioButtonGroup(labels=LABELS, active=0)

    tab1 = Panel(child=p1, title='linear')
    tab2 = Panel(child=p2, title='log')
    tabs = Tabs(tabs=[tab1, tab2], tabs_location='below', disabled=True)
    radio_button_group.js_link('active', tabs, 'active')

    layout = layout([[tabs],[checkbox_group, radio_button_group]],
        sizing_mode='stretch_width'
    )
    show(layout)

def heatmap_transitions(df_acts=None, lst_acts=None, df_trans=None, z_scale="linear",
                        figsize=None, idle=False, numbers=True, grid=True,
                        cmap=None, file_path=None):
    """
    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : lst of str, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    df_trans : pd.DataFrame
        A precomputed transition table. If the *df_trans* parameter is given, parameters
        *df_acts* and *lst_acts* are ignored. The transition table can be computed
        in :ref:`stats <stats_acts_trans>`.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    z_scale : {"log", "linear"}, default: None
        The axis scale type to apply.
    numbers : bool, default: True
        Whether to display numbers inside the heatmaps fields or not.
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    cmap : str or Colormap, optional
        The Colormap instance or registered colormap name used to map scalar
        data to colors. This parameter is ignored for RGB(A) data.
        Defaults 'viridis'.
    grid : bool, default: True
        determines whether to display a white grid, seperating the fields or not.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_activity_hm_transition
    >>> plot_activity_hm_transition(data.df_activities)

    .. image:: ../_static/images/plots/act_hm_trans.png
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
    assert z_scale in ['linear', 'log'], 'z-scale has to be either of type None or log'
    assert not (df_acts is None and df_trans is None)

    title = 'Activity transitions'
    z_label = 'count'

    if df_trans is None:
        df_acts = add_other_activity(df_acts) if idle else df_acts
        df = activities_transitions(df_acts, lst_acts=lst_acts)
    else:
        df = df_trans

    # get the list of cross tabulations per t_window
    act_lst = list(df.columns)

    num_act = len(act_lst)
    figsize = (_num_items_2_heatmap_square_figsize(num_act) if figsize is None else figsize)
    cmap = (get_sequential_color() if cmap is None else cmap)

    x_labels = act_lst
    y_labels = act_lst
    values = df.values
    

    log = True if z_scale == 'log' else False
    valfmt = '{x:.0f}'
        
     # begin plotting
    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap_square(values, y_labels, x_labels, log=log, cmap=cmap, ax=ax, cbarlabel=z_label, grid=grid)
    if numbers:
        texts = annotate_heatmap(im, textcolors=("white", "black"),log=log, valfmt=valfmt)
    ax.set_title(title)
    
    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig

def ridge_line(df_acts=None, lst_acts=None, df_act_dist=None, idle=False,
               n=1000, ylim_upper=None, color=None, figsize=None, file_path=None):
    """
    Plots the activity density distribution over one day.

    Parameters
    ----------
    df_acts : pd.DataFrame, optional
        recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : lst of str, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    df_act_dist : pd.DataFrame, optional
        A precomputed activity density distribution. If the *df_trans* parameter is given, parameters
        *df_acts* and *lst_acts* are ignored. The transition table can be computed
        in :ref:`stats <stats_acts_trans>`.
    n : int, default=1000
        The number of monte-carlo samples to draw.
    ylim_upper: float, optional
        The offset from the top of the plot to the first ridge_line. Set this if
        the automatically determined value is not satisfying.
    figsize : (float, float), default: None
        width, height in inches. If not provided, the figsize is inferred by automatically.
    color : str, optional
        sets the color of the plot. When not set, the primary theming color is used.
        Learn more about theming in the :ref:`user guide <theming>`
    idle : bool, default: False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.
    file_path : str, optional
        If set, saves the plot under the given file path and return *None* instead
        of returning the figure.

    Examples
    --------
    >>> from pyadlml.plot import plot_activity_rl_daily_density
    >>> plot_activity_rl_daily_density(data.df_activities)

    .. image:: ../_static/images/plots/act_ridge_line.png
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
    assert not (df_acts is None and df_act_dist is None)

    title = 'Activity distribution over one day'
    xlabel = 'day'
    color = (get_primary_color() if color is None else color)


    if df_act_dist is None:
        if idle:
            df_acts = add_other_activity(df_acts)
        df = activities_dist(df_acts.copy(), lst_acts=lst_acts, n=n)
        if df.empty:
            raise ValueError("no activity was recorded and no activity list was given.")
    else:
        df = df_act_dist

    def date_2_second(date):
        """ maps time onto seconds of a day 
        Parameters
        ----------
        date : np.datetime64
            all the dates are on the day 1990-01-01

        Returns
        -------

        """
        if pd.isnull(date):
            return -1
        val = (date - np.datetime64('1990-01-01')) / np.timedelta64(1, 's')
        total_seconds = 60*60*24
        assert val <= total_seconds and val >= 0
        return int(val)

    
    df = df.apply(np.vectorize(date_2_second))
    # sort every columns values ascending
    for col in df.columns:
        df[col] = df[col].sort_values()

    grouped = [(col, df[col].values) for col in df.columns]
    acts, data = zip(*grouped)
    num_act = len(list(acts))

    # infer visual properties
    figsize = (_num_items_2_ridge_figsize(num_act) if figsize is None else figsize)
    ylim_upper = (_num_items_2_ridge_ylimit(num_act) if ylim_upper is None else ylim_upper)

    # plot the ridgeline
    fig, ax = plt.subplots(figsize=figsize)
    ridgeline(data, labels=acts, overlap=.85, fill=color, n_points=100, dist_scale=0.13)
    plt.title(title)

    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylim((0, ylim_upper))
    plt.xlabel(xlabel)
    
    # set xaxis labels
    def func(x,p):
        #x = x + 0.5
        #if x == 0.0 or str(x)[-1:] == '5':
        #    return ''
        #else:
        if True:
            if np.ceil(x/k) < 10:
                return '0{}:00'.format(int(x/k)+1)
            else:
                return '{}:00'.format(int(x/k)+1)
    a = 0
    b = 60*60*24
    k = (b-a)/24
    
    plt.xlim((a,b))
    tcks_pos = np.arange(0,23)*k + (-0.5 + k)
    
    x_locator = ticker.FixedLocator(tcks_pos)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(func))
    ax.xaxis.set_major_locator(x_locator)
    fig.autofmt_xdate(rotation=45)
    
    plt.grid(zorder=0)

    if file_path is not None:
        savefig(fig, file_path)
        return 
    else:
        return fig