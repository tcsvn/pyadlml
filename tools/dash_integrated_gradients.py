from pathlib import Path
from captum.attr import IntegratedGradients
import tempfile
import mlflow
import numpy as np
import torch
import argparse
from dash import Dash, dash_table, dcc, html
import dash_bootstrap_components as dbc
from pyadlml.dataset.plot.plotly.util import deserialize_range, serialize_range, range_from_fig
from dash.dependencies import Input, Output, State
import pandas as pd
import sys
from pyadlml.dataset.util import select_timespan
from pyadlml.constants import *
from plotly import graph_objects as go
import os
sys.path.append(os.getcwd())
sys.path.append("/media/data/code/adlml/ma_adl_prediction")

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           prevent_initial_callbacks=True,
           eager_loading=False,
)
from scipy.special import softmax

from pyadlml.dataset.plot.plotly.discrete import acts_and_devs
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from pyadlml.dataset.plot.plotly.dashboard.layout import _build_range_slider
from pyadlml.dataset.plot.plotly.util import dash_get_trigger_element, dash_get_trigger_value, set_fig_range
from pyadlml.dataset.util import num_to_timestamp
from sklearn.utils.multiclass import unique_labels
from pyadlml.dataset.util import num_to_timestamp
from dash.exceptions import PreventUpdate

def create_td_slider():
    # define slider limits
    min_val = np.log10(1e-3)  # 1 ms in seconds
    max_val = np.log10(5*60)  # 5 minutes in seconds
    # prepare marks for every 10**x where x is an integer
    marks={i: f'{10**i if i<=3 else round(10**i/60, 2)} {"s" if i<=3 else "min"}' 
        for i in range(int(np.floor(min_val)), int(np.ceil(max_val))+1)}

    return dcc.Slider(
                    id='timedelta-slider',
                    min=min_val,
                    max=max_val,
                    step=0.01,
                    value=1, 
                    marks=marks,
                    tooltip={'always_visible': False, 'placement': 'bottom'}
                )



class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

    def fit(self, X, y, model=None):
        # Check that X and y have correct shape

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Define the network architecture
        self.model_ = model
        self.model_.to(self.device)
        # Set model to evaluation mode
        self.model_.eval()
        return self

    def predict_proba(self, X):

        check_is_fitted(self, ['model_', 'classes_'])

        # Convert to torch tensor
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        # Predict probabilities
        probabilities = np.zeros((X.shape[0], len(self.classes_)))
        with torch.no_grad():
            for b in range(0, X.shape[0], self.batch_size):
                idx = slice(b, min(b+self.batch_size, X.shape[0]))

                outputs = self.model_(X[idx])
                probabilities[idx] = nn.functional.softmax(outputs, dim=1)\
                                                .cpu().numpy()

        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return probabilities.argmax(axis=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Create an interactive confusion matrices.")
    parser.add_argument("run_id", help="The run id the first experiment.")
    parser.add_argument("--src", help="address of mlflow server", default="http://localhost:5555")
    parser.add_argument("--exp-id", help='', default="370606261398319459")
    parser.add_argument('-p', '--port', type=str, default='8050')
    args = parser.parse_args()
    
    fp_X_val = 'data/X_val.csv'
    fp_y_val = 'data/y_val.csv'
    fn_y_pred = 'yt_pred.npy'
    fn_y_logits = 'yt_logits.npy'
    #exp_id = "281991029756723155" # masterarbeit auf beast 

    print('connecting to mlflow...')
    mlflow.set_tracking_uri(args.src)
    with tempfile.TemporaryDirectory() as workdir:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        print('loading validation data...')
        fp_X_val = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path=fp_X_val)
        X_val = pd.read_csv(fp_X_val)
        X_val[TIME] = pd.to_datetime(X_val[TIME])

        fp_y_val = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path=fp_y_val)
        y_val = pd.read_csv(fp_y_val)
        y_val[START_TIME] = pd.to_datetime(y_val[START_TIME])
        y_val[END_TIME] = pd.to_datetime(y_val[END_TIME])


        print('loading model...')
        model_name = 'final_model'
        path = f"mlflow-artifacts:/{args.exp_id}/{args.run_id}/artifacts/{model_name}"
        model = mlflow.pytorch.load_model(path)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        print('loading pipe...')
        path = f"mlflow-artifacts:/{args.exp_id}/{args.run_id}/artifacts/pipe"
        pipe = mlflow.sklearn.load_model(path)
        Xt_pre_window, yt_pre_window = pipe[:-1].transform(X_val, y_val)

        print('constructing sklearn torch model...')
        model_sk = PyTorchClassifier().fit(Xt_pre_window, yt_pre_window, model=model)
        from pyadlml.pipeline import Pipeline
        pipe_sk = Pipeline([
            ('window', pipe[-1]),
            ('clf', model_sk)
        ])
        window_size = pipe[-1].window_size

        print('calculating integrated gradients...')
        # Save the figure as HTML
        model_sk.model_.zero_grad()
        ig = IntegratedGradients(model_sk.model_)
        Xt, yt = pipe[-1].transform(Xt_pre_window, yt_pre_window)

        def attribution(test_input_idx):
            test_input_tensor = torch.from_numpy(Xt[test_input_idx])[None, :, :].float().to(model_sk.device)
            test_input_tensor.requires_grad = True
            baseline_tensor = test_input_tensor*0 

            attr, delta = ig.attribute(
                test_input_tensor, 
                target=int(yt[test_input_idx]), 
                baselines=baseline_tensor,
                n_steps=50,
                method='gausslegendre', # riemann_right, riemann_trapezoid, riemann_middle
                internal_batch_size=None,
                return_convergence_delta=True
            )

            attr = attr.detach().cpu().numpy()
            delta = delta.detach().cpu().numpy()
            print('Approximation delta: ', abs(delta))
            return attr


        # Shape (E, T, C) with epoch E, times T, classes C
        y_pred = np.load(mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path=f"eval/{fn_y_pred}",
                                                                 dst_path=workdir))
        y_logits = np.load(mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path=f"eval/{fn_y_logits}",
                                                                 dst_path=workdir))
        y_prob = softmax(y_logits, axis=-1)

        y_times, Xt_val_at_times = pipe.construct_y_times_and_X(X_val, y_val)
        Xt_val_at_times = pd.DataFrame(Xt_val_at_times, columns=pipe['windows'].feature_names_in_)
        classes_ = pipe['lbl'].classes_
        window_size = pipe[-1].window_size


        # Correct for last drop batch in val dataloader
        if len(y_times) > y_prob.shape[0]:
            offset = -(len(y_times) - y_prob.shape[1])
            y_times = y_times[:offset]
            Xt_val_at_times = Xt_val_at_times[:offset]




        def create_heatmap(event_idx, width, visualize_changepoint=False):
            print('event_time: ', y_times[event_idx])
            x = y_times[event_idx-window_size+1:event_idx+2]
            #z = attribution(event_idx).squeeze(0).swapaxes(1,0)
            z = np.repeat(np.arange(0, len(x)-1)[:, None], len(Xt_val_at_times.columns), axis=1).T.astype(float)
            for i in range(1, z.shape[0]):
                z[i] += 1

            assert z.shape[1] == x.shape[0] - 1, '#events does not match attribution heatmap'

            # Print the sampled dates

            if visualize_changepoint:
                x = x.repeat(3)
                x[::3] = x[1::3] - width
                x[2::3] = x[1::3] + width
                x[1::3] = x[1::3] - width/2
                z = z.repeat(3, axis=1)
                z[:, ::3] = np.nan
                z[:, 2::3] = np.nan

            heatmap = go.Heatmap(
                    z=z,
                    x=x,
                    y=Xt_val_at_times.columns,
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(
                        len=0.5,  # Changes the length of the color bar
                        thickness=20,  # Changes the thickness of the color bar
                        x=1.1,  # Changes the x position of the color bar
                        y=0.001,  # Changes the y position of the color bar
                        yanchor="bottom",  # The yanchor, relative to y
                        xanchor="right",  # The xanchor, relative to x
                    ),
                    hovertemplate="%{y}=%{z:.3f}",
                    xaxis='x',
                    yaxis='y',
                    opacity=0.9,  # makes the heatmap transparent
            )
            return heatmap

        def create_figure(set_start_time, set_end_time, heatmap):
            sel_y_val = select_timespan(
                df_acts=y_val, 
                start_time=set_start_time, 
                end_time=set_end_time,
                clip_activities=True
            )
            sel_idxs = (set_start_time < y_times) & (y_times < set_end_time)
            return acts_and_devs(
                Xt_val_at_times[sel_idxs],
                y_true=sel_y_val,
                y_pred=y_pred[-1][sel_idxs],
                y_conf=y_prob[-1][sel_idxs],
                act_order=classes_,
                times=y_times[sel_idxs],
                heatmap=heatmap
            )

        def get_idxs(ts):
            return np.argmin(abs(y_times - ts.to_datetime64()))




        vis_width = pd.Timedelta('10s')
        set_start_time = pd.Timestamp('2023-03-21 00:00:00')
        set_end_time = pd.Timestamp('2023-03-21 00:20:00')
        start_time = pd.Timestamp(y_times.min())
        end_time = pd.Timestamp(y_times.max())


        # Create initial heatmap in the middle
        idxs = np.where(np.array(
            (set_start_time < y_times) & (y_times < set_end_time)
        ))[0]
        #heatmap = create_heatmap(idxs[len(idxs) // 2], vis_width)
        fig_range = set_start_time, set_end_time


        # Create layout
        app.layout = dbc.Container([
            dbc.Row(
                dcc.Graph(
                    id='fig-activity',
                    figure=create_figure(set_start_time, set_end_time, None),
                    style={'width': '1800px', 'height': '1200px'}
                ), className='mb-3', 
            ),
            dbc.Row([
                _build_range_slider(start_time, end_time, set_start_time, set_end_time),
                dcc.Store(id='range-store', data=serialize_range(fig_range)),
                dcc.Store(id='range-slider-t-1', data=[0,1]),
            ],className='mb-2'),
            dbc.Row([
                dbc.Col(create_td_slider(), md=8),
                dbc.Col(
                    dbc.Button(
                        "Toggle Heatmap",
                        id="toggle-button",
                        color="primary",
                        className="mr-1"
                    ), md=2
                ),
                dbc.Col(
                    dcc.Checklist(
                        id="toggle-vis",
                        options=[
                            {'label': 'Changepoint vis', 'value': 'on'}
                        ],
                        value=[]
                    ), md=2
                )
            ])
        
        ], style={'width': 1800, 'margin': 'auto'}) 



        @app.callback(
            Output('fig-activity', 'figure'),
            Output('range-store', 'data'),
            Output('range-slider-t-1', 'data'),
            Input('range-slider', 'value'),
            Input('fig-activity', 'clickData'),
            Input('toggle-button', 'n_clicks'),
            Input('toggle-vis', 'value'),
            Input('timedelta-slider', 'value'),
            State('fig-activity', 'figure'),
            State('range-store', 'data'),
            State('range-slider-t-1', 'data'),
        )
        def display_output(rng, datum_select, btn_toggle, vis_checklist, td_slider, fig, range_store, rs_tm1):
            trigger = dash_get_trigger_element()
            if trigger is None or trigger == '':
                raise PreventUpdate
            print('entering callback...')
                
            width = pd.Timedelta(f'{np.power(10.0, td_slider)}s')
            fig_range = range_from_fig(fig)
            visualize_changepoint = 'on' in vis_checklist

            update_figure = False
            if range_store is not None:
                range_store = deserialize_range(range_store)

            if trigger == 'range-slider':
                set_start_time = num_to_timestamp(rng[0], start_time=start_time, end_time=end_time)
                set_end_time = num_to_timestamp(rng[1], start_time=start_time, end_time=end_time)
                fig = create_figure(set_start_time, set_end_time, heatmap=None)
        
                #range_store = reset_range_store(df_devs, df_acts)
            if trigger == 'fig-activity':
                datum = datum_select['points']
                ts = pd.Timestamp(datum_select['points'][0]['x'])
                dat_idx = get_idxs(ts)
                if dat_idx < window_size:
                    raise PreventUpdate

                heatmap = create_heatmap(dat_idx, width, visualize_changepoint)

                set_start_time = num_to_timestamp(rng[0], start_time=start_time, end_time=end_time)
                set_end_time = num_to_timestamp(rng[1], start_time=start_time, end_time=end_time)
                fig = create_figure(set_start_time, set_end_time, heatmap)
                set_fig_range(fig, fig_range)

            if trigger == 'timedelta-slider' and visualize_changepoint:
                for trace in fig['data']:
                    if trace.get('type') == 'heatmap':
                        if visualize_changepoint:
                            old_width = (pd.Timestamp(trace['x'][-1]) - pd.Timestamp(trace['x'][-3]))/2
                            event_time = pd.Timestamp(trace['x'][-2]) + old_width/2
                        else:
                            event_time = trace['x'][-2] 
                        event_idx = get_idxs(event_time)

                        # include the timestamp after the clicked event for the heatmap to 
                        # be drawn between the events
                        x = y_times[event_idx-window_size+1:event_idx+2]

                        if visualize_changepoint:
                            x = x.repeat(3)
                            x[::3] = x[1::3] - width
                            x[2::3] = x[1::3] + width
                            x[1::3] = x[1::3] - width/2

                        trace['x'] = [str(xi) for xi in x]
                        break
                
            if trigger == 'toggle-button':
                for trace in fig['data']:
                    if trace.get('type') == 'heatmap':
                        if trace['opacity'] == 0:
                            trace['opacity'] = 0.5
                        else:
                            trace['opacity'] = 0
                        break

       
            range_store = serialize_range(range_store)

            return fig, range_store, rs_tm1

        app.run_server(debug=True,  use_reloader=False, host='127.0.0.1', port=args.port)


