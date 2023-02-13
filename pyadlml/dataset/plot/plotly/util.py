from pyadlml.constants import ACTIVITY, BOOL, CAT, START_TIME, END_TIME, STRFTIME_DATE
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash.exceptions import PreventUpdate
import bisect
import pandas as pd


def remove_whitespace_around_fig(plot_func):
    def wrapper(*args, **kwargs):
        fig = plot_func(*args, **kwargs)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30, pad=0))
        return fig
    return wrapper

def _style_colorbar(fig, label):
    fig.update_traces(colorbar=dict(
        title=dict(text=label, font=dict(size=10)),
        titleside='right',
        thickness=10,
    ))

def _dyn_y_label_size(plot_height, nr_devs):
    """Maps bar heights to visually pleasing size for different nr. of labels"""
    mp = np.array([[15, 9],     # val < 15  -> 9
                   [20, 11],
                   [30, 11],
                   [40, 11],
                   [80, 11],
                  [100, 11],
    ])
    return range2value(mp, nr_devs)


def _dyn_marker_size(plot_height, nr_devs):
    """Maps bar heights to visually pleasing size for different nr. of labels"""
    mp = np.array([[15, 3],  # val < 15 -> 3
                   [50, 5],
                  [100, 10]])
    return range2value(mp, nr_devs)

def range2value(mp, val):
    index = min(max(0, bisect.bisect_right(mp[:,0], val)-1), len(mp[:,0])-1)
    return mp[:,1][index]

import pandas as pd
import json



def plot_histogram(fig, row, col, bins, counts, dev1, dev2, cm):
    """
    Plot a histogram into a figure  
    """
    counts = np.concatenate([counts[:int(len(counts)/2)],
                        counts[int(len(counts)/2)+1:]]) 

    tmp = pd.interval_range(start=pd.Timedelta('0s'), end=pd.Timedelta(max_lag), freq=pd.Timedelta(binsize)).values
    len_cd = len(bins)
    cd = np.full(bins.shape, fill_value='', dtype=object)
    for i in range(0, len(cd)):
        if i < len(tmp)-1:
            cd[i] = '(-' + str(tmp[len(tmp)-i-1].right) + ', -' + str(tmp[len(tmp)-i-1].left) + ']'
        elif i == len(tmp)-1:
            cd[i] = '(-' + str(tmp[len(tmp)-i-1].right) + ', ' + str(tmp[len(tmp)-i-1].left) + ']'
        elif i == len(tmp):
            cd[i] = '(0 days 00:00:00, 0 days 00:00:00]'
        else:
            cd[i] = str(tmp[(i-1)%len(tmp)])
    half = int(np.ceil(len(bins)/2))
    before = np.full(half, fill_value='before', dtype=object)
    after = np.full(half, fill_value='after', dtype=object)
    cd1 = np.concatenate([before, after])[:-1]
    cd1[half-1] = 'at'
    cd = np.stack([cd, cd1]).T 


    fig.add_trace(
        go.Bar(
            x=bins,
            y=counts,
            opacity=1,
            customdata=cd, 
            hovertemplate='%{y} events of "' + dev1 + '" occur during %{customdata[0]}<br>%{customdata[1]}' + f' a type "{dev2}" event<extra></extra>',
            showlegend=False,
            marker=dict(line_width=0, color=cm[dev2])
        ), row=row, col=col
    )
    fig.update_xaxes(showticklabels=False, row=row, col=col)
    fig.update_yaxes(showticklabels=False, row=row, col=col)

    return fig





def format_device_labels(labels: list, dtypes: dict, order='alphabetical',
                         boolean_state=False, categorical_state=False, custom_rule=None):
    """ Sorts devices after a given rule. Format the devices labels and produces
    an ordering to sort values given the labels.

    Parameters
    ----------
    labels : nd array
        List of labels, where boolean devices have the format [dev_bool_x:on, dev_bool_x:off'] and
        categorical devices where format [dev_cat:subcat1, ..., dev_cat:sub_cat3]
    dtypes : dict
        The dtypes of the devices
    order : str of {'alphabetical', 'areas', 'custom'}, default='alphabetical'
        The criteria with which the devices are sorted.
    boolean_state : bool, default=False
        Indicates whether the boolean devices are split into

    custom_rule : func, default=None


    Returns
    -------
    lst : list
        Result list of correctly formatted device names
    new_order : list
        The indices that have to be reordered
    """
    assert order in ['alphabetical', 'areas', 'custom']
    DELIM = ':'
    ON = ':on'
    OFF = ':off'
    if isinstance(labels, list):
        labels = np.array(labels, dtype='object')

    def format_dev_and_state(word):
        return ''.join(' - ' if c == DELIM else c for c in word)

    def only_state(word):
        return word.split(DELIM)[1]

    def only_dev(word):
        return word.split(DELIM)[0]

    # presort devices
    devices = np.concatenate(list(dtypes.values()))
    if order == 'alphabetical' and custom_rule is None:
        devices = np.sort(devices)
    elif order == 'areas' and custom_rule is None:
        raise NotImplementedError
    elif order == 'custom' and custom_rule is not None:
        devices = custom_rule(devices)

    # rename devices
    new_labels = np.zeros((len(labels)), dtype=object)
    new_order = np.zeros((len(labels)), dtype=np.int32)

    dev_idx = 0
    i = 0
    while i < len(new_labels):
        dev = devices[dev_idx]
        dev_idx += 1
        if boolean_state and dev in dtypes[BOOL]:
            idx_on = np.where(labels == (dev + ON))[0][0]
            idx_off = np.where(labels == (dev + OFF))[0][0]

            new_labels[i] = format_dev_and_state(labels[idx_off])
            new_labels[i+1] = only_state(labels[idx_on])
            new_order[i] = idx_off
            new_order[i+1] = idx_on
            i += 2
        elif categorical_state and dev in dtypes[CAT]:
            mask = [lbl.split(DELIM)[0] == dev for lbl in labels]
            idxs = np.where(mask)[0]
            cats = labels[mask]
            cats_new_order = np.argsort(cats)
            for j in range(len(cats)):
                if j == 0:
                    new_labels[i] = format_dev_and_state(cats[cats_new_order[j]])
                else:
                    new_labels[i] = "- " + only_state(cats[cats_new_order[j]])
                new_order[i] = idxs[cats_new_order[j]]
                i += 1
        else:
            idx = np.where(labels == dev)[0]
            new_labels[i] = dev
            new_order[i] = idx
            i += 1

    return new_labels, new_order



def plot_activities_into(fig, df_acts: pd.DataFrame, y_label: str, cat_col_map, 
                         row=1, col=1, act_order=None
    ):


    fig.update_xaxes(type="date")

    # RLE
    df = df_acts.copy()

    df = df.sort_values(by=START_TIME).reset_index(drop=True)
    df['y_label'] = y_label
    df = df[['y_label', START_TIME, END_TIME, ACTIVITY]]

    df['dur'] = (df[END_TIME] - df[START_TIME])
    df['lengths' ] = df['dur'].astype("timedelta64[ms]")
    df['dur'] = df['dur'].astype(str)


    if act_order is None:
        act_order = df[ACTIVITY].uniques()

    trace_lst = []
    set_legendgroup_title = 'Activities'
    for act_name in act_order:
        cat_col_map.update(act_name, fig)
        show_legend = act_name not in legend_current_items(fig)
        act_mask = (df[ACTIVITY] == act_name)
        hover_template = '<b>' + act_name + '</b><br>'\
                        + 'Start_time: %{base|' + STRFTIME_DATE + '}<br>' \
                        + 'End_time: %{x| ' + STRFTIME_DATE + '}<br>' \
                        + 'Dur: %{customdata}<extra></extra>'

        trace = go.Bar(name=act_name,
                        meta=y_label,
                        base=df.loc[act_mask, 'start_time'],
                        x=df.loc[act_mask, 'lengths'],
                        y=df.loc[act_mask, 'y_label'],
                        legendgrouptitle_text=set_legendgroup_title,
                        marker_color=cat_col_map[act_name],
                        legendgroup=act_name,
                        customdata=df.loc[act_mask, 'dur'],
                        orientation='h',
                        width=0.9,
                        textposition='auto',
                        alignmentgroup=True,
                        offsetgroup=act_name,
                        showlegend=show_legend,
                        hovertemplate=hover_template,
        )
        trace_lst.append(trace)
        set_legendgroup_title = None

    [fig.add_trace(trace, row=row, col=col) for trace in trace_lst]

    # Add additional 
    acts_not_in_legend = set(act_order) - set(legend_current_items(fig))
    for act_name in acts_not_in_legend:
        # TODO check if it works
        raise NotImplementedError
        cat_col_map.update(act_name, fig)
        fig.add_trace(go.Bar(
            name=act_name, 
            visible='legendonly',
            showlegend=True
        ))

    # Add traces, layout and frames to figure
    fig.update_layout({'barmode': 'overlay',
                    'legend': {'tracegroupgap': 0}
                    })

    return fig



def legend_current_items(fig):
    legend_current_items = [] 
    for trace in fig.data:
        if isinstance(trace, go.Bar) \
          and trace.legendgroup is not None:
            legend_current_items.append(trace.legendgroup)
    return legend_current_items


class CatColMap():
    COL_ON = 'teal'
    COL_OFF = 'lightgray'

    cat_idx =0

    cat_col_map = {
        0:COL_OFF, 1:COL_ON,
        'off':COL_OFF, 'on':COL_ON,
        False:COL_OFF, True:COL_ON,
    }
    def __init__(self, theme='pastel'):
        if theme == 'pastel':
            self.cat_colors = px.colors.qualitative.Pastel \
                    + px.colors.qualitative.Pastel1 \
                    + px.colors.qualitative.Pastel2
        elif theme == 'set':
            self.cat_colors = px.colors.qualitative.Set1 \
                    + px.colors.qualitative.Set2 \
                    + px.colors.qualitative.Set3
        else:
            self.cat_colors = px.colors.qualitative.T10

    def update(self, cat, fig=None):
        if fig is not None:
            if cat not in legend_current_items(fig):
                if cat not in self.cat_col_map.keys():
                    self.cat_col_map[cat] = self.cat_colors[self.cat_idx]
                    self.cat_idx +=1
                return True
            return False
        else:
            if cat not in self.cat_col_map.keys():
                self.cat_col_map[cat] = self.cat_colors[self.cat_idx]
                self.cat_idx +=1
            
    def __getitem__(self, sub):
        return self.cat_col_map[sub]

    def __setitem__(self, sub, item):
        self.cat_col_map[sub] = item

def dash_get_trigger_value(ctx=None):
    try:
        ctx = dash.callback_context if ctx is None else ctx
        return ctx.triggered[0]['value']
    except IndexError:
        raise PreventUpdate

def dash_get_trigger_element(ctx=None):
    try:
        ctx = dash.callback_context if ctx is None else ctx
        return ctx.triggered[0]['prop_id'].split('.')[0]
    except IndexError:
        raise PreventUpdate


deserialize_range = lambda x: [pd.Timestamp(ts) for ts in json.loads(x)] 
serialize_range = lambda x: json.dumps([ts.isoformat() for ts in x])
range_from_fig = lambda f: [pd.Timestamp(ts) for ts in f['layout']['xaxis']['range']]

