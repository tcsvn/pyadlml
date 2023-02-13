import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd

from pyadlml.constants import ACTIVITY, STRFTIME_DATE
from pyadlml.dataset.plot.plotly.acts_and_devs import legend_current_items
from pyadlml.dataset.plot.plotly.util import remove_whitespace_around_fig, CatColMap

def rlencode(x, dropna=False):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle 
    function from R.
    
    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.
    
    Returns
    -------
    start positions, run lengths, run values
    
    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int), 
                np.array([], dtype=int), 
                np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~(x[1:] == x[:-1])) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    
    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]
    
    return starts, lengths, values


def confidences(y_conf, activities, cat_col_map=None):
    """
    Parameters
    ----------
    y_conf: nd.array of shape (N,C)
    classes: list or nd.array of shape (C,)
        Each class labels position corresponds to the confidence index in y_conf:
        c_i \in classes <=> (N, c_i) in y_conf

    """
    if cat_col_map is None:
        cat_col_map = CatColMap()
    assert isinstance(cat_col_map, CatColMap)

    fig = go.Figure()
    x=np.arange(0, len(y_conf))

    for c, act in enumerate(activities):
        cat_col_map.update(act, fig)
        y_c = y_conf[:,c]
        fig.add_trace(go.Scatter(
            x=x, y=y_c,
            mode='lines',
            name=act,
            line=dict(width=0.5, color=cat_col_map[act]),
            stackgroup='one'
        ))
    fig.update_layout(
        showlegend=True,
        yaxis=dict(
            type='linear',
            range=[0, 1],
        )
    )
    return fig


def _plot_confidences_into(fig, row, col, y_conf, activities, cat_col_map, 
                           time=None, step_fct=True

):
    if cat_col_map is None:
        cat_col_map = CatColMap()
    assert isinstance(cat_col_map, CatColMap)

    if time is not None:
        x = time
    else:
        x=np.arange(0, len(y_conf))

    # Fix display issues of not displaing low prob lines
    eps = 1e-6
    y_conf = np.where(y_conf == 0, y_conf + eps, y_conf)

    for c, act_name in enumerate(activities):
        cat_col_map.update(act_name, fig)
        showlegend = act_name not in legend_current_items(fig)
        y_c = y_conf[:,c]

        # Set the probabilities to a step function if needed
        line_dct = dict(width=0.5, color=cat_col_map[act_name])
        if step_fct:
            line_dct['shape'] = 'hv'

        hover_template = ''#'%{y:.4f}'
        for ci, act in enumerate(activities): 
            hover_template += f'<b>{act}</b>' if act == act_name else act
            hover_template +=  ': %{customdata[' + str(ci) + ']:.4f}<br>'
        cm = y_conf

        fig.add_trace(go.Scatter(
                x=x, y=y_c,
                mode='lines',
                name=act_name,
                customdata=cm, 
                hovertemplate=hover_template,
                line=line_dct,
                stackgroup='one',
                legendgroup=act_name,
                showlegend=showlegend,
            ), row=row, col=col)

    # Make plot not scrollabel
    fig.update_yaxes(row=1, col=1, fixedrange=True, range=[0, 1])
    return fig

 


def _plot_activities_into(fig, y, y_label, cat_col_map, row=1, col=1, time=None, activities=None):
        trace_lst = []

        if activities is None:
            activities = np.unique(y)

        df = pd.DataFrame(data=zip(*rlencode(y)), columns=['start', 'lengths', ACTIVITY])
        df['y_label'] = y_label
        if time is not None:
            df = discreteRle2timeRle(df, time)

        df['end'] = df['start'] + df['lengths']
        if time is not None:
            df['lengths' ] = df['lengths'].astype("timedelta64[ms]")

        set_legendgroup_title = 'Activities'
        for act_name in activities:
            cat_col_map.update(act_name, fig)
            show_legend = act_name not in legend_current_items(fig)
            act_mask = (df[ACTIVITY] == act_name)
            if time is None:
                hover_template = '<b>' + act_name + '</b><br>'\
                                + 'Int: [%{base}, %{x})<br>' \
                                + 'Dur: %{customdata}<extra></extra>'
            else:
                hover_template = '<b>' + act_name + '</b><br>'\
                                + 'Start_time: %{base|' + STRFTIME_DATE + '}<br>' \
                                + 'End_time: %{x| ' + STRFTIME_DATE + '}<br>' \
                                + 'Dur: %{customdata}<extra></extra>'

            trace = go.Bar(name=act_name,
                            meta=y_label,
                            base=df.loc[act_mask, 'start'],
                            x=df.loc[act_mask, 'lengths'],
                            y=df.loc[act_mask, 'y_label'],
                            legendgrouptitle_text=set_legendgroup_title,
                            marker_color=cat_col_map[act_name],
                            legendgroup=act_name,
                            customdata=df.loc[act_mask, 'end'],
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
        acts_not_in_legend = set(activities) - set(legend_current_items(fig))
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

def discreteRle2timeRle(df, time):
    delta_t = (time - time.shift(1))[1:].reset_index(drop=True)
    df['start_time'] = time[0]
    df['lengths_time'] = pd.Timedelta('0ns')
    def func(row, t, dt):
        row.start_time = t[row.start]
        if row.lengths != 1:
            row.lengths_time = dt[row.start:row.start+row.lengths].sum()
        elif row.start != len(dt):
            row.lengths_time = dt[row.start]
        else:
            # row is last row with lengths == 1
            row.lengths_time = pd.Timedelta('1s')
        return row

    return df.apply(func, axis=1, args=(time, delta_t))\
           .drop(columns=['start', 'lengths'])\
           .rename(columns={'start_time':'start', 'lengths_time':'lengths'})



def _plot_devices_into(fig, X, cat_col_map, row, col, time):
    ON = 1
    OFF = 0
    # TODO refactor, generalize also for numerical labels
    # Run length encoding of 0s and 1s
    bars = {f:{OFF:[], ON:[]} for f in X.columns}
    for f in bars.keys():
        bars[f] = pd.DataFrame(data=zip(*rlencode(X[f])), columns=['start', 'lengths', 'state'])
        if time is not None:
            bars[f] = discreteRle2timeRle(bars[f], time)
            bars[f]['lengths'] = bars[f]['lengths'].astype("timedelta64[ms]")

    trace_lst = []
    for dev in X.columns:
        for k in [0, 1]:
            if time is None:
                hover_template = '<b>' + dev + '</b><br>'\
                                + 'Int: [%{base}, %{x})<br>' \
                                + 'Dur: %{customdata}<extra></extra>'
            else:
                hover_template = '<b>' + dev + '</b><br>'\
                                + 'Start_time: %{base|' + STRFTIME_DATE + '}<br>' \
                                + 'End_time: %{x| ' + STRFTIME_DATE + '}<br>' \
                                + 'Dur: %{customdata}<extra></extra>'

            vals = bars[dev].loc[(bars[dev]['state'] == k)]
            trace = go.Bar(name=str(k),
                            meta=dev,
                            base=vals['start'],
                            x=vals['lengths'],
                            y=np.full(vals.shape[0], dev),
                            #legendgrouptitle_text=set_legendgroup_title,
                            marker_color=cat_col_map[k],
                            #legendgroup=cat,
                            #customdata=cd,
                            orientation='h',
                            width=0.3,
                            textposition='auto',
                            showlegend=False,
                            hovertemplate=hover_template,
            )
            trace_lst.append(trace)

    [fig.add_trace(trace, row=row, col=col) for trace in trace_lst]

    fig.update_layout({'barmode': 'overlay',
                       'legend': {'tracegroupgap': 0}
    })

    if time is not None:
        fig.update_xaxes(type="date")

    return fig

@remove_whitespace_around_fig
def acts_and_devs(X, y_true=None, y_pred=None, y_conf=None, act_order=None, time=None):
    """ Plot activities and devices for already 

    """
    if isinstance(y_true, pd.Series)\
        or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy().squeeze()
    
    assert (y_true is None or isinstance(y_true, np.ndarray))\
       and (y_pred is None or isinstance(y_pred, np.ndarray))\
       and (y_conf is None or isinstance(y_conf, np.ndarray))

    time = pd.Series(time) if isinstance(time, np.ndarray) else time

    error_text = 'parameter act_order has to be set if y_conf is given'
    assert y_conf is None or act_order is not None, error_text
    


    cat_col_map = CatColMap()
    X = X.copy().reset_index(drop=True)

    if y_conf is None:
        cols, rows, row_heights = 1, 1, [1.0]
    else:
        cols, rows, row_heights = 1, 2, [0.15, 0.85]

    fig = make_subplots(cols=cols, rows=rows, row_heights=row_heights,                        
        shared_xaxes=True, vertical_spacing=0.02,
    )

    fig = _plot_devices_into(fig, X, cat_col_map, row=rows, col=cols, time=time)

    if y_true is not None:
        fig = _plot_activities_into(fig, y_true, 'y_true', cat_col_map, row=rows, col=cols, time=time, activities=act_order)
    if y_pred is not None:
        fig = _plot_activities_into(fig, y_pred, 'y_pred', cat_col_map, row=rows, col=cols, time=time)
    if y_conf is not None:
        fig = _plot_confidences_into(fig, 1, 1, y_conf, act_order,
                                     cat_col_map,  time)

    return fig

