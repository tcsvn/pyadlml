import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors
from datetime import timedelta

import numpy as np

from pyadlml.constants import DEVICE, TIME, VALUE, STRFTIME_HOUR, STRFTIME_DATE, PRIMARY_COLOR, SECONDARY_COLOR
from pyadlml.dataset.plot.plotly.activities import _set_compact_title, _scale_xaxis
from pyadlml.dataset.stats.devices import event_count, event_cross_correlogram, event_cross_correlogram3, events_one_day, \
                                          inter_event_intervals
from pyadlml.dataset.util import check_scale, activity_order_by, device_order_by
from pyadlml.dataset.stats.devices import state_times
from plotly.graph_objects import Figure

@check_scale
def bar_count(df_dev, scale='linear', height=350, order='count') -> Figure:
    """ Plots the activities durations against each other
    """
    title ='Event count'
    col_label = 'event_count'
    xlabel = 'log count' if scale == 'log' else 'count'

    df = event_count(df_dev.copy())
    df.reset_index(level=0, inplace=True)

    dev_order = device_order_by(df_dev, rule=order)


    fig = px.bar(df, y=DEVICE,
                 category_orders={DEVICE: dev_order},
                 color_discrete_sequence=[PRIMARY_COLOR],
                 x=col_label,
                 orientation='h')

    _set_compact_title(fig, title)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30, pad=0), height=height)
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=xlabel)
    if scale == 'log':
        fig.update_xaxes(type='log')

    return fig

@check_scale
def device_iei(df_devs, scale='linear', height=350, n_bins=20, per_device=False, order='alphabetical') -> Figure:
    """
        plots
    """
    title ='Inter-event-interval'
    col_label = 'event_count'
    xlabel = 'log seconds' if scale == 'log' else 'seconds'
    color = 'device' if per_device else None

    # Get array of seconds
    df = df_devs.copy().sort_values(by=[TIME])
    df['ds'] = df[TIME].diff().shift(-1) / pd.Timedelta(seconds=1)
    if scale == 'log':
        df['ds'] = df['ds'].apply(np.log)

    cds = [PRIMARY_COLOR] if not per_device else None

    fig = px.histogram(df, x='ds', color=color, nbins=n_bins,
                       color_discrete_sequence=cds)

    # TODO refactor, get better second labeling
    hist, bin_edges = np.histogram(df['ds'].values[:-1], bins=n_bins)
    # when n_bins is event than bin_edges is n_bins + 1, therefore
    # TODO critical, make expresseion that fits for general nbins
    mask = (np.tile([0, 1], n_bins//2 + 1)[:-int(n_bins % 2 == 0)] == 1)

    right_bins = bin_edges[mask]
    left_bins = bin_edges[~mask]
    size = right_bins[0] - left_bins[0]
    fig.update_traces(xbins=dict(start=left_bins, size=size))
    #fig.update_traces(xbins_end=right_bins, xbins_start=left_bins, xbins_size=n_bins)

    _set_compact_title(fig, title)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30, pad=0), height=height)
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=xlabel)

    # Update the hovertemplate to include the name and custom data
    # and device name
    if per_device:
        for i in range(len(fig.data)):
            fig.data[i].customdata = [fig.data[i].legendgroup]*len(fig.data[i].x)
        fig.update_traces(hovertemplate='device=%{customdata}<br>ds=%{x}<br>count=%{y}<extra></extra>')
    return fig


def fraction(df_dev, height=350, order='alphabetical') -> Figure:
    """
        plots the fraction a device is on vs off over the whole time
    """
    title = 'State fraction'
    xlabel = 'fraction'
    from pyadlml.dataset.stats.devices import state_fractions

    dev_order = device_order_by(df_dev, rule=order)

    # TODO, include td as custom data

    # Returns 'device', 'value', 'td', 'frac'
    df = state_fractions(df_dev)
    def fm1(x):
        if x in ['on', 'off']:
            return x == 'on'
        else:
            return x
    def f(x):
        if isinstance(x, bool):
            return 'on' if x else 'off'
        else:
            return x
    df[VALUE] = df[VALUE].apply(f)

    fig = px.bar(df, y=DEVICE, x='frac', orientation='h', color=VALUE,
                 color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR],
                 category_orders={DEVICE: dev_order}
                 )

    # Set hovertemplate and custom data
    for i in range(len(fig.data)):
        val = fig.data[i].legendgroup
        mask = df[DEVICE].isin(fig.data[i].y) & (df[VALUE] == val)
        cd = np.array([df.loc[mask, 'td'].astype(str),
                       [fm1(val)]*len(fig.data[i].x)
        ])
        fig.data[i].customdata = np.moveaxis(cd, 0, 1)
        hover_template = 'Device=%{y}<br>Value=' + val + '<br>Fraction=%{x}<br>' + \
                         'Total=%{customdata[0]}<extra></extra>'
        fig.data[i].hovertemplate = hover_template

    _set_compact_title(fig, title)
    fig.update_layout(barmode='stack', margin=dict(l=0, r=0, b=0, t=30, pad=0),
                      height=height)
    fig.update_yaxes(title=None, visible=False, showticklabels=False)
    fig.update_xaxes(title=xlabel)

    return fig


def event_density(df_dev, dt='1h', height=350, scale='linear', show_colorbar=True, order='alphabetical'):
    """
    Computes the heatmap for one day where all the device triggers are showed
    """
    title = 'Event density'
    df = events_one_day(df_dev.copy(), dt)
    time = df['time'].copy()
    df = df.drop(columns='time')
    dev_order = device_order_by(df_dev, rule=order)
    df = df[dev_order]
    vals = df.T.values

    n = len(df)
    devs = df.columns
    dates = pd.Timestamp('01.01.2000 00:00:00') + np.arange(1, n+1) * pd.Timedelta(dt) \
            - pd.Timedelta(dt)/2    # move tick from centered box half dt to left

    if scale == 'log':
        vals = np.log(vals)

    fig = px.imshow(vals, color_continuous_scale='Viridis',
                    y=devs,
                    x=dates,
                    )
    # Create Hoverdata time intervals
    cd = np.tile(dates, (len(df.columns), 1))
    cd = np.array([cd-pd.Timedelta(dt)/2, cd+pd.Timedelta(dt)/2])
    fig['data'][0]['customdata'] = np.moveaxis(cd, 0, -1)


    fig.update_layout(title=title, xaxis_nticks=24, xaxis_tickangle=-45,)
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30, pad=0), height=height)
    _set_compact_title(fig, title)

    if not show_colorbar:
        fig.update_layout(coloraxis_showscale=False)
        fig.update(layout_coloraxis_showscale=False)
        fig.update_coloraxes(showscale=False)

    fig.data[0].hovertemplate = 'Time: %{customdata[0]|' + STRFTIME_HOUR + '}' \
                                + '-%{customdata[1]|' + STRFTIME_HOUR + '}<br>' \
                                + 'Device: %{y}<br>Count: %{z}<extra></extra>'

    return fig



def boxplot_state(df_devs, scale='linear', height=350, binary_state='on',
                  order='alphabetical') -> Figure:
    """ Plot a boxplot of activity durations (mean) max min
    """
    title = 'State distribution'
    xlabel = 'log seconds' if scale == 'log' else 'seconds'

    df = state_times(df_devs, binary_state=binary_state, categorical=True)
    df['seconds'] = df['td']/pd.Timedelta('1s')

    dev_order = device_order_by(df_devs, rule=order)

    # Add column for hover display of datapoints later
    #df[START_TIME] = df_act[START_TIME].dt.strftime('%c')
    #df[END_TIME] = df_act[END_TIME].dt.strftime('%c')
    df['td'] = df['td'].apply(str)

    # TODO refactor, find way to render boxplot points with webgl
    points = 'all' if len(df) < 30000 else 'outliers'

    fig = px.box(df, y=DEVICE, x='seconds', orientation='h',
                 labels=dict(seconds=xlabel),
                 category_orders={DEVICE: dev_order},
                 color_discrete_sequence=[PRIMARY_COLOR],
                 notched=False, points=points,
                 hover_data=['td', 'time']
    )

    _set_compact_title(fig, title)
    if scale == 'log':
        r_min = np.floor(np.log10(df['seconds'].min()))
        r_max = np.ceil(np.log10(df['seconds'].max()))
        fig.update_xaxes(type="log", range=[r_min, r_max])  # log range: 10^r_min=1, 10^r_max=100000


    fig.update_yaxes(title=None, visible=False, showticklabels=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30, pad=0), height=height)
    fig.data[0].hovertemplate = 'Device=%{y}<br>Seconds=%{x}<br>Td=%{customdata[0]}<br>' +\
                                'Start time=%{customdata[1]|' + STRFTIME_DATE + '}<extra></extra>'

    return fig



def plotly_device_event_correlogram(df_devs, corr_data=(None, None, None), height=600):

    from pyadlml.dataset.stats.devices import event_cross_correlogram2
    if corr_data:
        cc, bins, devs = corr_data
    else:
        cc, bins, devs = event_cross_correlogram2(df_devs, v1=False, use_dask=True)

    fig = make_subplots(rows=len(devs), cols=len(devs), horizontal_spacing=0.005,
                        vertical_spacing=0.005, column_titles=devs.tolist(), row_titles=devs.tolist(),
    
    )
    from pyadlml.dataset.plot.plotly.util import CatColMap

    def plot_histogram(fig, row, col, bins, counts, dev1, dev2, cm):
        counts = np.concatenate([counts[:int(len(counts)/2)],
                            counts[int(len(counts)/2)+1:]]) 
        fig.add_trace(
            go.Bar(
                x=bins,
                y=counts,
                opacity=1,
                hovertemplate=f'Fix {dev1} w.r.t events of {dev2}<extra></extra>',
                showlegend=False,
                marker=dict(line_width=0, color=cm[dev1])
            ), row=row, col=col
        )
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, row=row, col=col)
        
        return fig
    cm = CatColMap()
    for i in range(len(devs)):
        for j in range(len(devs)):
            cm.update(cat=devs[i])
            fig = plot_histogram(fig, i+1,j+1, bins, cc[i, j], devs[i], devs[j], cm)

    # Rotate columns and row annotation correctly 
    for i in range(len(devs)*2):
        if i < len(devs):
            # Case for cols
            fig.layout['annotations'][i]['textangle'] = -90
        else:
            fig.layout['annotations'][i]['textangle'] = 0

    return fig


def plotly_event_correlogram(df_devs=None, cc_data=None, fix=[], to=[], max_lag='2min', binsize='1s', use_dask=False, height=600):

        
        # Wrap if not list
        fix = [fix] if isinstance(fix, str) else fix
        to = [to] if isinstance(to, str) else to
            
        if cc_data is None:
            cc, bins, rows, cols = event_cross_correlogram3(
                df_devs, fix=fix, to=to, maxlag=max_lag, binsize=binsize,
                use_dask=use_dask
            )
        else:
            cc, bins, rows, cols = cc_data[0], cc_data[1], cc_data[2], cc_data[3]

        fig = make_subplots(rows=len(rows), cols=len(cols), horizontal_spacing=0.005,
                            vertical_spacing=0.005, column_titles=cols, row_titles=rows,
        
        )
        from pyadlml.dataset.plot.plotly.util import CatColMap
        import pandas as pd
        def plot_histogram(fig, row, col, bins, counts, dev1, dev2, cm):
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

        # Color row-wise
        cm = CatColMap()
        for i in range(len(rows)):
            for j in range(len(cols)):
                cm.update(cat=cols[j])
                fig = plot_histogram(fig, i+1,j+1, bins, cc[i, j], rows[i], cols[j], cm)

        # Rotate columns and row annotation correctly 
        for i in range(len(cols) + len(rows)):
            if i < len(cols):
                # Case for cols
                hm = {1:-0, 2:-10, 3:-20, 4:-40, 5:-50}
                #fig.layout['annotations'][i]['textangle'] = hm[len(cols)]
                fig.layout['annotations'][i]['textangle'] = -20
            else:
                hm = {1: 70, 2: 80, 3:80, 4:80, 5:80}
                #fig.layout['annotations'][i]['textangle'] = hm[len(rows)] 
                fig.layout['annotations'][i]['textangle'] = 70

        fig.update_layout(height=height)
        return fig

