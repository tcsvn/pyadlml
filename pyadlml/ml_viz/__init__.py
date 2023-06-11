import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

from pyadlml.metrics import online_confusion_matrix


def plotly_confusion_matrix(y_pred, y_true, labels, per_step=False, scale='linear') -> go.Figure:
    """
    """
    cbarlabel = 'counts' if scale == 'linear' else 'log counts'
    if not per_step:
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        # Row = true, column = prediction
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred,
                              labels=labels).astype(int)
        z = np.log(cm) if scale == 'log' else cm

        fig = go.Figure(
            data=go.Heatmap(
                z=z.T,
                x=labels,
                y=labels,
                customdata=cm.T,
                colorscale='Viridis',
                hovertemplate='Truth: %{y}<br>Pred: %{x}<br>Times: %{customdata} <extra></extra>',
                hoverongaps=False),
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis=dict(title="Predicted"),
            yaxis=dict(title="True"),
        )

        return fig
    else:

        nr_steps = len(y_pred)

        if isinstance(y_pred[0], torch.Tensor):
            y_pred = [y.detach().cpu().numpy() for y in y_pred]

        if isinstance(y_true[0], torch.Tensor):
            y_true = [y.detach().cpu().numpy() for y in y_true]

        cms = [confusion_matrix(y_true=yt, y_pred=yp).astype(int)
               for (yp, yt) in zip(y_pred, y_true)]

        fig = go.Figure()
        import plotly.express as px

        for step in range(nr_steps):
            z = np.log(cms[step]) if scale == 'log' else cms[step]
            fig.add_trace(
                go.Heatmap(
                    name=f'CM_{step}',
                    z=z,
                    x=labels,
                    y=labels,
                    colorscale='Viridis',
                    hovertemplate='Pred: %{x}<br>Truth: %{y}<br>Times: %{z} <extra></extra>',
                    hoverongaps=False
                ),
            )

        # Make last confusion matrix visible
        fig.data[nr_steps-1].visible = True

        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": "Confusion matrix: " + str(i)}],  # layout attribute
            )
            # Toggle i'th trace to "visible"
            step["args"][0]["visible"][i] = True
            steps.append(step)

        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Epoch: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders
        )

        # Share colorscale
        for i in range(nr_steps):
            fig.data[i]['coloraxis'] = 'coloraxis'

    return fig


def plotly_online_confusion_matrix(y_true=None, y_pred=None, y_times=None, labels=None, conf_mat=None, scale='linear') -> go.Figure:
    """
    """
    cbarlabel = 'counts' if scale == 'linear' else 'log seconds'
    if conf_mat is None:
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        # Row = true, column = prediction
        cm = online_confusion_matrix(
            y_true, y_pred, y_times, n_classes=len(labels))
    else:
        cm = conf_mat
        labels = conf_mat.index
    z = cm.astype('timedelta64[ns]')/np.timedelta64(1, 'ns')
    z = z if scale == 'linear' else np.log(z)
    z = z.T

    # Create hoverdata, the duration as strings in (D,A,0) and full X names in (D,A,1)
    cd = cm.astype(str).values.T
    # tmp = np.tile(vals.index, (len(vals.columns), 1))
    # cd = np.array([tmp, vals.values.T])
    # d = np.moveaxis(cd, 0, -1)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            customdata=cd,
            colorscale='Viridis',
            hovertemplate='Truth: %{y}<br>Pred: %{x}<br>Overlap: %{customdata} <extra></extra>',
            hoverongaps=False),
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted"),
        yaxis=dict(title="True"),
    )

    return fig
