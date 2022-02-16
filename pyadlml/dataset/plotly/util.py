
def _style_colorbar(fig, label):
    fig.update_traces(colorbar=dict(
        title=dict(text=label, font=dict(size=10)),
        titleside='right',
        thickness=10,
    ))
