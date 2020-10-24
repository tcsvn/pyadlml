import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors

from pyadlml.dataset import START_TIME
from pyadlml.dataset.stats.activities import activities_duration_dist, \
    activities_count, activities_durations, activities_dist, \
    activities_transitions

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

def boxplot_duration(df_act, y_scale='norm'):
    """
        plot a boxplot of activity durations (mean) max min 
    """
    assert y_scale in ['norm', 'log']

    df = activities_duration_dist(df_act)

    # add column for display of datapoints later
    df[START_TIME] = df_act[START_TIME].dt.strftime('%c')

    if y_scale == 'log':
        df['log minutes'] = np.log(df['minutes'])
        labels={'minutes': 'log minutes'}
    else:
        labels={'minutes': 'minutes'}    

    fig = px.box(df, x="activity", y=labels['minutes'], 
                 notched=True, # used notched shape
                 labels=labels,
                 points='all', # display points next to box plot
                 title="Activity durations",
                 hover_data=[START_TIME, 'minutes'] # add day column to hover data
                )
    return fig

def hist_cum_duration(df_act, y_scale='both'):
    """
    plots the cummulated activities durations in a histogram for each activity 

    """
    assert y_scale in ['norm', 'log', 'both']

    title = 'Activity cummulative durations'

    act_dur = activities_durations(df_act.copy())
    df = act_dur[['minutes']]
    df.reset_index(level=0, inplace=True)
    if y_scale in ['norm', 'log']: 
        if y_scale == 'log':
            df['minutes'] = np.log(df['minutes'])
            labels={'minutes': 'log minutes'}
        else:
            labels={'minutes': 'minutes'}
            
        df = df.sort_values(by=['minutes'], axis=0)
        fig = px.bar(df, y='activity', x='minutes', 
                    title=title,
                    labels=labels, 
                    height=400,
                    #hover_data=['fraction'] TODO add the fraction by hovering
                    )
    else:
        df = df.sort_values(by=['minutes'], axis=0)
        col_label = 'minutes'
        col_label_log = 'log minutes'
        col_activity = 'activity'
        df[col_label_log] = np.log(df[col_label])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df[col_label], y=df[col_activity], \
                                orientation='h', name='duration'))
        fig.add_trace(go.Bar(x=df[col_label_log], y=df[col_activity], \
                                orientation='h', name='log duration', \
                                visible=False))
        # Add dropdown
        fig.update_layout(
            title=title,
            xaxis_title="duration",
            yaxis_title='activities',
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

    return fig

def hist_counts(df_act, y_scale='norm'):
    """
    plots the activities durations against each other
    """
    assert y_scale in ['norm', 'log']

    col_label = 'occurence'
    title ='activity occurrences'

    df = activities_count(df_act.copy())
    df.reset_index(level=0, inplace=True)
    
    if y_scale == 'log':
        df[col_label] = np.log(df[col_label])
        labels={col_label: 'log count'}
    else:
        labels={col_label: 'count'}    

    df = df.sort_values(by=['occurence'], axis=0)
    fig = px.bar(df, title=title, y='activity', x=col_label, orientation='h', labels=labels, height=400)
    return fig

def gantt(df):
    """
    """
    df_plt = df.copy()
    df_plt.columns = ['Start', 'Finish', 'Resource']
    df_plt['Task'] = 'Activity'
    fig = ff.create_gantt(df_plt, 
        index_col='Resource',
        show_colorbar=True, 
        bar_width=0.2, 
        height=500,
        width=None,
        group_tasks=True,
        showgrid_x=True, showgrid_y=False)

    # Add range slider
    btn_12h = dict(count=12, label="12h", step="hour", stepmode="backward")
    btn_1d = dict(count=1, label="1d", step="day", stepmode="backward")
    btn_1w = dict(count=7, label="1w", step="day", stepmode="backward")
    btn_1m = dict(count=1, label="1m", step="month", stepmode="backward")
    #btn_all =
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([btn_12h, btn_1d, btn_1w, btn_1m])#, dict(step="all")
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )    
    return fig

def heatmap_transitions(df_act, z_scale='norm'):
    """
    """
    assert z_scale in ['norm', 'log'], 'z-scale has to be either norm or log'

    name = 'Activity transitions'

    # get the list of cross tabulations per t_window
    df = activities_transitions(df_act)
    act_lst = list(df.columns)
    if z_scale == 'norm':
        z = df.values
    elif z_scale == 'log':
        z = np.log(df.values)
    else:
        raise ValueError

    fig = go.Figure(data=go.Heatmap(
        #labels=dict(x=x_label, y=y_label, color=color),
        z=z,
        x=act_lst,
        y=act_lst,
        colorscale='Viridis',
        hoverongaps = False))
    fig.update_layout(title=name)
    return fig