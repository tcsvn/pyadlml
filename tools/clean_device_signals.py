from pathlib import Path
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State
from scipy import signal as sc_signal
import pandas as pd
import numpy as np
import argparse
import dash
from dash.exceptions import PreventUpdate
from pyadlml.constants import ACTIVITY, DEVICE, END_TIME, START_TIME, STRFTIME_PRECISE, TIME, VALUE
from pyadlml.dataset._core.activities import read_activity_csv
from pyadlml.dataset._core.devices import _generate_signal, create_sig_and_corr, device_remove_state_matching_signal, read_device_df, write_device_df
from pyadlml.dataset._datasets.activity_assistant import write_device_map, write_devices
from pyadlml.dataset.plot.plotly.acts_and_devs import activities_and_devices
from pyadlml.dataset.plot.plotly.util import dash_get_trigger_element, dash_get_trigger_value, remove_whitespace_around_fig
from pyadlml.constants import ts2str

from pyadlml.dataset.util import df_difference, fetch_by_name
from pyadlml.dataset.plot.plotly.activities import correction
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objects as go

"""
for dataset options look under pyadlml.constants:
    casas_aruba
    amsterdam
    mitlab_1
    mitlab_2
    aras
    kasteren_A
    kasteren_B
    kasteren_C
    tuebingen_2019
    uci_ordonezA
    uci_ordonezB
    act_assist
"""

SEP_HIT = '#' + '-'*50

@remove_whitespace_around_fig
def plot_cc(sig_match, sig_search, auto_corr, cross_corr, t, eps_corr, cc_max = None):

    assert abs(len(auto_corr) - len(cross_corr)) < 3, 'Auto and cross correlation shouuld be roughly the same length'

    x_corr = np.arange(-int(len(auto_corr)/2), int(len(auto_corr)/2))

    fig = make_subplots(rows=3, cols=1)#, shared_xaxes=True)

    fig.add_trace(
        go.Scatter(x=np.arange(len(sig_search)), y=sig_search, name='search signal'), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(len(sig_match))-t, y=sig_match, name='match signal'), row=1, col=1
    )


    fig.add_trace(
        go.Scatter(
            x=x_corr,
            y=cross_corr, 
            name='cross-corr(search,  match)',
        ), row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_corr,
            y=auto_corr, 
            name='auto-corr(search)',
        ), row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=[x_corr[0], x_corr[-1]],y=[eps_corr]*2, name='threshold'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=[t], y=[cross_corr[t+int(len(cross_corr)/2)]], showlegend=False),
        row=2, col=1
    )

    if cc_max is not None:
        uniques, counts = np.unique(np.sort(cc_max), return_counts=True)
        #x_cc_max = np.cumsum(counts)
        #x_cc_max = np.insert(x_cc_max, 0, 0)
        x_cc_max = np.flip(np.cumsum(np.flip(counts)))
        x_cc_max = np.insert(x_cc_max, len(x_cc_max), 0)
        y_cc_max = np.insert(uniques, 0, 0)

        fig.add_trace(
            go.Scatter(
                x=x_cc_max, 
                y=y_cc_max,
                showlegend=False,
            ), row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x = [0, max(x_cc_max)],
                y = [eps_corr]*2,
                showlegend=False, 
            ), row=3, col=1
        )
        fig.update_xaxes(title='lag', row=3, col=1)
        fig.update_yaxes(title='Positive predicted', row=3, col=1)
    return fig



def del_op2str(row: pd.Series):
    row_lst = f"\t[['{ts2str(row[TIME])}','{row[DEVICE]}','{row[VALUE]}']]"
    return f"idx_to_del = get_index_matching_rows(df_devs, {row_lst})\n"\
            + f"df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)\n\n"

def add_op2str(row: pd.Series):
    row_dict = "{"\
            + f"'{TIME}': pd.Timestamp('{(row[TIME])}'), "\
            + f"'{DEVICE}': '{row[DEVICE]}'), "\
            + f"'{VALUE}': '{row[VALUE]}'"\
    + "}"
    return f"new_row=pd.Series({row_dict})\n"\
            + "print('adding: ', new_row)\n"\
            + "df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).reset_index(drop=True)\n"


def plot_main(df_devs, df_acts, dev_name, signal_orig, zoomed_range):
    fig = make_subplots(rows=1, cols=1) # TODO refactor, shared x
    fig = activities_and_devices(df_acts=df_acts, df_devs=df_devs, states=True, 
                                zoomed_range=zoomed_range, fig=fig, row=1, col=1)
    fig.update(layout_showlegend=False)
    # TODO plot perfect signal resized and centered at maximum cc
    #fig.add_trace(

    #)
    # TODO plot signal under
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('-d', '--dataset', type=str, default='amsterdam')
    parser.add_argument('-i', '--identifier', type=str)
    parser.add_argument('-p', '--port', type=str, default='8050')
    parser.add_argument('-o', '--out-folder', type=str, default='/tmp/pyadlml/clean_device_signals')
    args = parser.parse_args()


    sel_dev = args.device

    out_folder = Path(args.out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    code_str = "# Awesome script\n" \
            + "import pandas as pd\n"\
            + "from pyadlml.dataset._core.devices import get_index_matching_rows\n"

    fp_code = out_folder.joinpath(f'clean_dev_signal_{args.device}.py')
    fp_df_devs = out_folder.joinpath('devices.csv')

    if not fp_code.exists():
        with open(fp_code, mode='w') as f:
            f.write(code_str)


    # Setup data
    data = fetch_by_name(args.dataset, identifier=args.identifier, \
                         cache=False, retain_corrections=True, keep_original=True)

    # Create global variables
    df_devs = data['devices']
    df_acts = data['activities']
    write_devices(df_devs, fp_df_devs)

    assert sel_dev in df_devs[DEVICE].unique(), 'Check if device is present in dataset.'

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    #-------------------------------------------
    # Parameters to adjust 
    search_signal = [
        (True, '4s'),
        (False, '2s'),
        (True, '1s'),
        (False, '6s')
    ]

    # The dataframe is search beforehand based on device one state (i.e. True, '1s'). Combined with a 
    # tolerance (eps_state) matches are genererated. Cross correlation is computed only around those matches.
    matching_state_idx = 2
    #-------------------------------------------


    # Tolerance around matching state to be detected as a match
    eps_state = '0.2s'
    # Tolerance for cross correlation w.r.t. auto correlation to highlight as a match
    eps_corr = 0.2
    sig_search = [(s[0], pd.Timedelta(s[1])) for s in search_signal]

    code_str += f'search_signal = {str(search_signal)}\n' 
    code_str += f'matching_state_idx = {matching_state_idx}\n' 
    code_str += f'eps_state = {eps_state}\n\n'

    # Find 
    td = pd.Timedelta(sig_search[matching_state_idx][1])
    state = sig_search[matching_state_idx][0]
    eps_state = pd.Timedelta(eps_state)

    # Create auto correlation and retrieve a proper threshold
    ss_discrete = _generate_signal(sig_search, dt='250ms')
    auto_corr = sc_signal.correlate(ss_discrete, ss_discrete, mode='full')
    auto_corr_max = auto_corr.max()

    # Compute the number of counts that the signal may deviate
    # but still counts as a match
    # For example if a window is 72 units long and the maximum
    # correlation would also be 72 for a signal with the same length. Therefore
    # the eps_corr count would be 14 if the signal is allowed to differ 20%
    eps_corr = int(len(ss_discrete)*eps_corr)

    # Count total length and length up and after matching state of signal
    ss_total_dt, ss_dt_prae_match, ss_dt_post_match = [pd.Timedelta('0s')]*3
    for i, (s, dt) in enumerate(sig_search):
        ss_total_dt += dt
        if i < matching_state_idx:
            ss_dt_prae_match += dt
        if i > matching_state_idx:
            ss_dt_post_match += dt

    plot_dt_around_sig = ss_total_dt*0.25

    def create_hit_list(df_devs):
        df = df_devs[df_devs[DEVICE] == sel_dev].copy()
        df['to_convert'] = False
        df['diff'] = pd.Timedelta('0ns')
        df['target'] = False

        df['diff'] = df[TIME].shift(-1) - df[TIME]
        df['target'] = (td - eps_state < df['diff'])\
                    & (df['diff'] < td + eps_state)\
                    & (df[VALUE] == state)

        # Correct the case where the first occurence is already a match
        df.at[df.index[-1], 'diff'] = ss_total_dt

        # Get indices of hits and select first match for display
        hits = df[(df['target'] == True)].index.to_list()
        return hits, df


    # Load device dataframe and recreate hits
    def reload_devices(code_str):
        lcls = {'df_devs': read_device_df(fp_df_devs)} 
        exec(code_str, {}, lcls)
        df_devs = lcls['df_devs']
        hits, df = create_hit_list(df_devs)
        return df_devs, df, hits


    hits, df = create_hit_list(df_devs)
    hit_idx = hits[0]
    sm_discrete, cross_corr = create_sig_and_corr(df, hit_idx, ss_dt_prae_match, ss_dt_post_match, ss_discrete)

    cc_max_lst = []
    for h in hits:
        _, cc = create_sig_and_corr(df, h, ss_dt_prae_match, ss_dt_post_match, ss_discrete)
        cc_max_lst.append(max(cc))

    ts = df_devs.loc[hit_idx, TIME]
    zr_st, zr_et = ts - plot_dt_around_sig, ts + ss_total_dt + plot_dt_around_sig

    HEADER_STR = f'Correcting {sel_dev}; Match: #%s/%s'

    app.layout = dbc.Container([
        dbc.Row([
            dcc.Store(id="curr-id"),
            dbc.Col(
                dcc.Graph(
                    id='correction',
                    figure=plot_main(df_devs, df_acts, sel_dev, sig_search, zoomed_range=[zr_st, zr_et])
                ), md=6
            ),
            dbc.Col([
                dcc.Graph(
                    id='correlation',
                    figure=plot_cc(sm_discrete, ss_discrete, auto_corr, cross_corr, t=0, eps_corr=eps_corr, cc_max=cc_max_lst)
                ),
                html.P("Lag:"),
                dcc.Slider(-len(ss_discrete), len(ss_discrete), id='cc_slider', marks=None, value=0),
                html.P("Threshold:"),
                dcc.Slider(0, auto_corr_max, id='thresh_slider', marks=None, value=eps_corr)
                ], md=6
            )
        ], className='mb-3'),
        dbc.Row([html.H6(id='header', children=HEADER_STR%(0, len(hits))),
        ], className='mb-2'),
        dbc.Row(
            dbc.Textarea(id='comment', readOnly=False, 
                        style={'height': '100px'}
                        #style={'resize': 'none', 'overflow':'hidden'}#, 'height': '100px'}
            ),
            style={'height':'100%'},
            className='mb-1'
        ),
        dbc.Row([
            dbc.ButtonGroup([
                dbc.Button(id="prev", children="Previous", color="secondary"), 
                dbc.Button(id="revert", children="revert last change", color="secondary"),  
                dbc.Button(id="next", children="Next", color="secondary"), 
            ])
        ], className='mb-2'),

    ], style={'width': 1500, 'margin': 'auto'})

    @app.callback(
        Output('correction', 'figure'),
        Output('correlation', 'figure'),
        Output('curr-id', 'data'),
        Output('header', 'children'),
        Output('comment', 'value'),
        Input('prev', 'n_clicks'),
        Input('revert', 'n_clicks'),
        Input('next', 'n_clicks'),
        Input('cc_slider', 'value'),
        Input('thresh_slider', 'value'),
        Input('correction', 'clickData'),
        State('correction', 'figure'),
        State('correlation', 'figure'),
        State('curr-id', 'data'),
        State('comment', 'value')
    )
    def display_output(prev_click, revert_click, next_click, cc_slider, t_slider,
                       click_main, fig_main, fig_corr, curr_hit, comment
        ):
        curr_hit = 0 if curr_hit is None else curr_hit


        trigger = dash_get_trigger_element()
        if trigger is None or trigger == '':
            raise PreventUpdate


        # Open code
        f = open(fp_code, mode='r')
        code_str = ''.join(f.readlines())
        f.close()

        df_devs, df, hits = reload_devices(code_str)

        cc_max_lst = []
        for h in hits:
            _, cc = create_sig_and_corr(df, h, ss_dt_prae_match, ss_dt_post_match, ss_discrete)
            cc_max_lst.append(max(cc))

            
        if trigger == 'cc_slider' or trigger == 'thresh_slider':
            hit_idx = hits[curr_hit]
            sm_discrete, cross_corr = create_sig_and_corr(df, hit_idx, ss_dt_prae_match, ss_dt_post_match, ss_discrete)
            fig_corr = plot_cc(sm_discrete, ss_discrete, auto_corr, cross_corr, t=int(cc_slider), eps_corr=t_slider, 
                               cc_max=cc_max_lst)

        elif trigger == 'correction':
            dev = click_main['points'][0]['label']
            if dev != sel_dev:
                raise PreventUpdate

            time = pd.Timestamp(click_main['points'][0]['base'])
            matches = df_devs[df_devs[TIME] == time].index
            assert len(matches) == 1

            # Create heading
            code_str += "\n" + SEP_HIT + "\n"
            code_str += f"# Hit_nr.: {curr_hit}\n"

            # Insert comment
            comment = '' if comment is None else comment
            code_str += '\n'.join(['# ' + s for s in comment.split('\n')]) + '\n\n'

            match_idx = matches[0]
            df_tmp = df_devs[df_devs[DEVICE] == sel_dev].copy().reset_index()
            df_tmp_idx = df_tmp[df_tmp['index'] == match_idx].index[0] 

            if False: #df_tmp_idx-1 < 0:
                # 1. case :|~~~e______e~~~~~    -->     Do nothing
                #            |remove             
                # 2. case :e~~~e______e~~~~~    -->     e______e~~~~~ 
                #            |remove             
                df_dev_post_idx = df_tmp.at[df_tmp_idx+1, 'index']
                code_str += del_op2str(df_devs.loc[match_idx, :])
                code_str += del_op2str(df_devs.loc[df_dev_post_idx, :])
                print('Warning: Boundary condition removal:::: check what was done!')
            elif df_tmp_idx+1 > len(df_tmp):
                # 1. case :______e~~~|          --> Do nothing 
                #            |remove             
                # 2. case :______e~~~e can not happen since clicking on ~~~ will retreive left e TODO 
                #            |remove             
                raise NotImplementedError
            else:
                df_dev_post_idx = df_tmp.at[df_tmp_idx+1, 'index']
                code_str += del_op2str(df_devs.loc[match_idx, :])
                code_str += del_op2str(df_devs.loc[df_dev_post_idx, :])



            df_devs, df, hits = reload_devices(code_str)

            range_from_fig = lambda f: [pd.Timestamp(ts) for ts in f['layout']['xaxis']['range']]
            fig_range = range_from_fig(fig_main)

            fig_main = plot_main(df_devs, df_acts, sel_dev, sig_search, zoomed_range=fig_range)

        elif trigger == 'revert':
            # Remove last action
            code_str = SEP_HIT.join(code_str.split(SEP_HIT)[:-1])

            df_devs, df, hits = reload_devices(code_str)
            range_from_fig = lambda f: [pd.Timestamp(ts) for ts in f['layout']['xaxis']['range']]
            fig_range = range_from_fig(fig_main)

            fig_main = plot_main(df_devs, df_acts, sel_dev, sig_search, zoomed_range=fig_range)


        elif trigger == 'prev' or trigger in 'next':
            if trigger == 'prev':
                curr_hit = max(0, curr_hit-1) 
            else:
                curr_hit = min(curr_hit+1, len(hits)-1)

            comment = ''

            hit_idx = hits[curr_hit]
            sm_discrete, cross_corr = create_sig_and_corr(df, hit_idx, ss_dt_prae_match, ss_dt_post_match, ss_discrete)

            # Get area around matched state and update main window
            #ts = df.loc[df.index[df_hit_idx], TIME]
            ts = df.loc[hit_idx, TIME]
            zr_st, zr_et = ts - plot_dt_around_sig, ts + ss_total_dt + plot_dt_around_sig

            fig_main = go.Figure(fig_main)
            fig_main = fig_main.update_layout(
                            xaxis=dict(
                                type='date',
                                range=[zr_st, zr_et]
                                #range=[str(zr_st), str(zr_et)])
                            ),
                            #xaxis2=dict(
                            #    type='date',
                            #    range=[zr_st, zr_et]
                            #)
            )

            # Update cross corrleation plot
            fig_corr = plot_cc(sm_discrete, ss_discrete, auto_corr, cross_corr, t=int(cc_slider), eps_corr=t_slider,
                               cc_max=cc_max_lst)

        else:
            raise ValueError

        with open(fp_code, mode='w') as f:
            f.write(code_str)


        return fig_main, fig_corr, curr_hit, HEADER_STR%(curr_hit, len(hits)), comment

    app.run_server(debug=False, port=args.port, host='127.0.0.1')
