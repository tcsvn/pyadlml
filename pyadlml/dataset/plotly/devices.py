import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
from pyadlml.dataset.stat import activities_count, activities_durations
"""
#1.  categorical dot plot for device activations over time
        - color of dot could be the activity


#2. one example week histogramm of activitys




#3. activities durations
    - plot a boxplot of activity durations (mean) max min 

#5. for one day plot the activity distribution over the day
    - sample uniform from each interval 
"""
def hist_duration(df_dev):
    """
    plots the activities durations against each other
    """
    act_dur = activities_durations(df_act.copy())
    df = act_dur[['minutes']]
    df.reset_index(level=0, inplace=True)
    
    scale_y = 'log'
    if scale_y == 'log':
        df['minutes'] = np.log(df['minutes'])
        labels={'minutes': 'log minutes'}
    else:
        labels={'minutes': 'minutes'}    
        
    fig = px.bar(df, x='activity', y='minutes', labels=labels, height=400)
    return fig

def hist_counts(df_act):
    """
    plots the activities durations against each other
    """
    df = activities_count(df_act.copy())
    df.reset_index(level=0, inplace=True)
    df
    scale_y = 'count'
    col_label = 'occurence'
    if scale_y == 'log':
        df[col_label] = np.log(df[col_label])
        labels={col_label: 'log count'}
    else:
        labels={col_label: 'count'}    

    fig = px.bar(df, x='activity', y=col_label, labels=labels, height=400)
    return fig
