import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import plotly.express as px
from pyadlml.dataset.stat import activities_count, activities_durations, activities_dist
from pyadlml.dataset._dataset import START_TIME
from pyadlml.dataset.stat import activities_duration_dist

"""
#1.  categorical dot plot for device activations over time
        - color of dot could be the activity
https://plotly.com/python/dot-plots/

#2. one example week histogramm of activitys



 
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

def hist_duration(df_act, y_scale='norm'):
    """
    plots the activities durations against each other
    """
    assert y_scale in ['norm', 'log']

    act_dur = activities_durations(df_act.copy())
    #print(act_dur.head(n=15))

    df = act_dur[['minutes']]
    df.reset_index(level=0, inplace=True)
    
    if y_scale == 'log':
        df['minutes'] = np.log(df['minutes'])
        labels={'minutes': 'log minutes'}
    else:
        labels={'minutes': 'minutes'}    
        
    fig = px.bar(df, x='activity', y='minutes', 
                labels=labels, 
                height=400,
                #hover_data=['fraction'] TODO add the fraction by hovering
                )
    return fig

def hist_counts(df_act, y_scale='norm'):
    """
    plots the activities durations against each other
    """
    assert y_scale in ['norm', 'log']

    df = activities_count(df_act.copy())
    df.reset_index(level=0, inplace=True)
    df
    col_label = 'occurence'
    if y_scale == 'log':
        df[col_label] = np.log(df[col_label])
        labels={col_label: 'log count'}
    else:
        labels={col_label: 'count'}    

    fig = px.bar(df, x='activity', y=col_label, labels=labels, height=400)
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
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    #dict(count=1,
                    #     label="1h",
                    #     step="hour",
                    #     stepmode="backward"),                                   
                    #dict(count=6,
                    #     label="6h",
                    #     step="hour",
                    #     stepmode="backward"),                   
                    dict(count=12,
                         label="12h",
                         step="hour",
                         stepmode="backward"),                
                    dict(count=1,
                         label="1d",
                         step="day",
                         stepmode="backward"),
                    dict(count=7,
                         label="1w",
                         step="day",
                         stepmode="backward"),                     
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )    
    return fig