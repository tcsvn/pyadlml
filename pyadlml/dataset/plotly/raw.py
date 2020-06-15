import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
from pyadlml.dataset.stat import devices_count
"""
#1.  categorical dot plot for device activations over time
        - color of dot could be the activity
        - next to activity bar plot

2. dot plot activities vs labeled 

3. confusion matrix as heatmap
"""


def gantt_activities_vs_devices(df):
    """
    """
    df_plt = df.copy()
    df_plt.columns = ['Start', 'Finish', 'Task']
    df_plt['Resource'] = 'A'
    fig = ff.create_gantt(df_plt, 
    colors=['#333F44', '#93e4c1'],
    index_col='Resource',
    show_colorbar=True, 
    bar_width=0.2, 
    group_tasks=True,
    showgrid_x=True, showgrid_y=True)
    return fig