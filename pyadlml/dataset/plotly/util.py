
# String representations that are used to format quantities
# for plotly graphs
STRFTIME_DATE = '%d.%m.%Y %H:%M:%S.%s'
STRFTIME_HOUR = '%H:%M:%S'
STRFTIME_DELTA = ''

def _style_colorbar(fig, label):
    fig.update_traces(colorbar=dict(
        title=dict(text=label, font=dict(size=10)),
        titleside='right',
        thickness=10,
    ))

def _dyn_y_label_size(plot_height, nr_labels):

    if nr_labels < 15:
        return 12
    elif nr_labels < 20:
        return 11
    elif nr_labels < 30:
        return 10
    else:
        return 9