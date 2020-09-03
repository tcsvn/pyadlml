import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors
from datetime import timedelta

import numpy as np
from pyadlml.dataset.stats.devices import devices_on_off_stats, \
    devices_trigger_count, devices_dist, devices_trigger_time_diff, \
    device_tcorr, device_triggers_one_day


def ridge_line(df_act, t_range='day', n=1000):
    """
    https://plotly.com/python/violin/

    for one day plot the activity distribution over the day
    - sample uniform from each interval   
    """
    df = activities_dist(df_act.copy(), t_range, n)

  

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

def hist_counts(df_dev, scale_y='both'):
    """
    plots the trigger count of each device 
    """
    df = devices_trigger_count(df_dev.copy())
    df.reset_index(level=0, inplace=True)

    title = 'Count of on/off activations per Device'
    col_label = 'trigger count'
    col_device = 'device'
    df.columns = ['device', col_label]

    df = df.sort_values(by=col_label, axis=0, ascending=True)
    if scale_y == 'log':
        col_label_log = 'log '+ col_label
        df[col_label_log] = np.log(df[col_label])
        labels={col_label: col_label_log}
        return px.bar(df, x=col_label_log, y=col_device, \
                title=title, \
                labels=labels, orientation='h', height=450)

    elif scale_y == 'norm':
        labels={col_label: col_label}    
        return px.bar(df, x=col_label, y=col_device, \
                title=title, \
                labels=labels, orientation='h', height=450)

    elif scale_y == 'both':
        col_label_log = 'log '+col_label
        df[col_label_log] = np.log(df[col_label])
        labels={col_label: col_label_log}
        fig = go.Figure()

        #fig = px.bar(df, x=col_label, y=col_device, \
        #        title=title, \
        #        labels=labels, orientation='h', height=450)

        # Add Traces
        fig.add_trace(go.Bar(x=df[col_label], y=df[col_device], \
                                orientation='h', name='count'))
        fig.add_trace(go.Bar(x=df[col_label_log], y=df[col_device], \
                                orientation='h', name='log count', \
                                visible=False))

        # Add dropdown
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            label = "no scaling", method = "update",
                            args = [{"visible": [True, False]},
                                {"title": title, "xaxis_title": 'asdf'}]),
                        dict(
                            label = "log scaled", method = "update", 
                            args = [{"visible": [False, True]},
                                {"title": title, "xaxis_title": 'asdf'}])
                    ]),
                    active=0,
                    direction="up",
                    showactive=True,
                    x=1.0,
                    xanchor="right",
                    y=-0.2,
                    yanchor="bottom"
                ),
            ]
        )
        fig.update_layout(
            title=title,
            xaxis_title="trigger count",
        )

        return fig
    else: 
        raise ValueError

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


def heatmap_trigger_one_day(df_dev, t_res='1h'):
    """
    computes the heatmap for one day where all the device triggers are showed
    """
    if type(t_res) == str:
        df = device_triggers_one_day(df_dev.copy(), t_res)

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
    else:
        # Create figure
        fig = go.Figure()

        # create for each timeframe the data
        # Add traces, one for each slider step
        for t_val in t_res:
            df = device_triggers_one_day(df_dev.copy(), t_val)
            fig.add_trace(
                go.Heatmap(
                        z=df.T.values,
                        x=df.index,
                        y=df.columns,
                        colorscale='Viridis'))

        # Make 0th trace visible
        fig.data[0].visible = True

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                label=t_res[i],
                method="restyle",
                args=[{"visible": [False] * len(fig.data)},
                    {"title": "Slider switched to step: " + str(i)}],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={'visible':False},
            pad={"t": 70},
            steps=steps
        )]

        fig.update_xaxes(patch=dict(nticks=24))

        fig.update_layout(
            title='Triggers one day',
            sliders=sliders
        )
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

def hist_trigger_time_diff(df_dev):
    """
        plots
    """
    n_bins = 100
    title='Time difference between device triggers'
    log_sec_col = 'total_log_secs'
    sec_col = 'total_secs'
    df = devices_trigger_time_diff(df_dev.copy())
    
    # convert timedelta to total minutes
    df[sec_col] = df['row_duration']/timedelta(seconds=1)
    df[log_sec_col] = np.log(df[sec_col])
    X = np.log(df[sec_col]).values[:-1]

    #return df
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    #hover_data = {'sec: ', np.random.random(len(df))}
    hover_template = "count: %{y:}<br>bin: %{x:} </br><extra></extra>"
    fig.add_trace(go.Histogram(x=df[log_sec_col].values[:-1],
                        name='# device triggers',
                        nbinsx=n_bins,
                        hovertemplate=hover_template,),
                        secondary_y=False
                        )
    
    hist, bin_edges = np.histogram(df[log_sec_col].values[:-1], n_bins)
    mask = (np.array([0,1]*int((n_bins/2))+[0]) == 1) # select every scnd element
    left_bins = np.asarray([0]+list(bin_edges[mask]))
    
    hist, bin_edges = np.histogram(X, n_bins)
    left_bins = bin_edges[:-1]
    cum_percentage = hist.cumsum()/hist.sum()
    
    fig.update_layout(
            title=title,
            xaxis_title="log min",
            bargap=0.1,
        )
    fig.add_trace(go.Scatter(x=left_bins, 
                            y=cum_percentage, 
                            name="percentage left"),
                    secondary_y=True)

    # Set y-axes titles
    fig.update_yaxes(title_text="count", secondary_y=False)
    fig.update_yaxes(title_text="%", secondary_y=True)
    return fig