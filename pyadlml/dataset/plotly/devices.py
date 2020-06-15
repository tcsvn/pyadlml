import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
from pyadlml.dataset.stat import devices_on_off_stats, \
    devices_trigger_count, devices_dist, devices_trigger_time_diff, \
    device_tcorr, device_triggers_one_day
"""
TODO list
1. plot the amount of triggers 
    # histogram

4. plot triggers for one day

5. plot the on distributionn for one day

"""



def ridge_line(df_act, t_range='day', n=1000):
    """
    https://plotly.com/python/violin/

    for one day plot the activity distribution over the day
    - sample uniform from each interval   
    """
    df = activities_dist(df_act.copy(), t_range, n)

    import plotly.graph_objects as go
    from plotly.colors import n_colors

    colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', len(df.columns), colortype='rgb')
    data = df.values.T

    fig = go.Figure()
    i = 0
    for data_line, color in zip(data, colors):
        fig.add_trace(go.Violin(x=data_line, line_color=color, name=df.columns[i]))
        i += 1

    fig.update_traces(orientation='h', side='positive', width=3, points=False)
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    return fig

def hist_counts(df_dev):
    """
    plot the amount of state triggers 
    """
    df = devices_trigger_count(df_dev.copy())
    df.reset_index(level=0, inplace=True)

    scale_y = 'count'
    col_label = 'occurence'
    if scale_y == 'log':
        df[col_label] = np.log(df[col_label])
        labels={col_label: 'log count'}
    else:
        labels={col_label: 'count'}    

    fig = px.bar(df, x='activity', y=col_label, labels=labels, height=400)
    return fig

def hist_trigger_time_diff(df_dev):
    """
        plots
    """
    df = devices_trigger_time_diff(df_dev.copy())
    fig = go.Figure()
    trace = go.Histogram(x=np.log(df['row_duration'].dt.total_seconds()/60),
                        nbinsx=200,
                      )
    fig.add_trace(trace)
    return fig

def hist_trigger_time_diff_separate(df_dev):
    """
    """
    df = devices_trigger_time_diff(df_dev.copy())
    df = df[['device','time_diff2']]

    dev_list = df.device.unique()
    num_cols = 2
    num_rows = int(np.ceil(len(dev_list)/2))
    fig = make_subplots(rows=num_rows, cols=num_cols)

    for i, device in enumerate(dev_list):
        dev_df = df[df.device  == device].dropna()['time_diff2']
        k = i%num_cols + 1
        j = int(i/num_cols) + 1
        fig.append_trace(
            go.Histogram(x=np.log(dev_df.dt.total_seconds()/60), 
            nbinsx=100, 
            name=device
            ), j,k
        )
    return fig

def hist_on_over_time(df_dev):
    """
        expectes device type 2
    """
    df = devices_dist(df_dev.copy(), t_range='day', n=1000)

    fig = go.Figure()
    # for every device add histogram to plot
    for col in df.columns:
        fig.add_trace(go.Histogram(x=df[col], nbinsx=50, name=col))

    # Overlay both histograms
    fig.update_layout(barmode='overlay')

    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    return fig

def heatmap_trigger_one_day(df_dev):
    """
    computes the heatmap for one day where all the device triggers are showed
    """
    df = device_triggers_one_day(df_dev.copy())

    #plot stuff
    fig = go.Figure(data=go.Heatmap(
            z=df.T.values,
            x=df.index,
            y=df.columns,
            colorscale='Viridis'))

    fig.update_layout(
        title='Triggers one day',
        xaxis_nticks=24)
    return fig



def heatmap_time_diff(df_dev, t_windows):
    x_label = 'Devices'
    y_label = x_label
    color = 'trigger count'
    name = 'Device triggers'

    # get the list of cross tabulations per t_window
    lst = device_tcorr(df_dev, t_windows)

    if len(t_windows) == 1:
        # make a single plot
        df = lst[0]
        fig = go.Figure(data=go.Heatmap(
        name=name,
        #labels=dict(x=x_label, y=y_label, color=color),
        z=df,
        x=df.columns,
        y=df.index,
        hoverongaps = False))
        return fig
    else:
        # make multiple subplots
        num_cols = 2
        num_rows = int(np.ceil(len(lst)/2))
        print('nr: ', num_rows)
        fig = make_subplots(rows=num_rows, cols=num_cols)

        for i, device in enumerate(lst):
            df = lst[i]
            col = i%num_cols + 1
            row = int(i/num_cols) + 1
            fig.append_trace(
                go.Heatmap(
                name=name,
                #labels=dict(x=x_label, y=y_label, color=color),
                z=df,
                x=df.columns,
                y=df.index,
                hoverongaps = False
                ),col,row
            )
        return fig


def hist_on_off(df_dev):
    """
        plots the fraction a device is on vs off over the whole time
    """
    title='Devices fraction on/off'

    df = devices_on_off_stats(df_dev)
    df = df.sort_values(by='frac_on', axis=0)

    y = df.index

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y,
        x=df['frac_on'],
        name='on',
        orientation='h',
        marker=dict(
            color='rgba(1, 144, 105, 0.8)',
        )
    ))
    fig.add_trace(go.Bar(
        y=y,
        x=df['frac_off'],
        name='off',
        orientation='h',
        marker=dict(
            color='rgba(58, 71, 80, 0.8)',
        )
    ))

    fig.update_layout(barmode='stack', title=title)
    return fig