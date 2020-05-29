
def gantt_activities_and_devices(df):
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