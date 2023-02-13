import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go





def plotly_confusion_matrix(y_pred, y_true, labels, per_step=False):

    if not per_step:
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred).astype(int)


        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
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

        cms = [confusion_matrix(y_true=yt, y_pred=yp).astype(int) for (yp, yt) in zip(y_pred, y_true)]


        fig = go.Figure()
        import plotly.express as px

        for step in range(nr_steps):
            fig.add_trace(
                go.Heatmap(
                    name =f'CM_{step}',
                    z=cms[step],
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
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
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