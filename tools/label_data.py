from copy import copy
from pathlib import Path
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import argparse
import dash
from dash.exceptions import PreventUpdate
from pyadlml.constants import ACTIVITY, DEVICE, END_TIME, START_TIME, TIME, VALUE, ts2str, str2ts
from pyadlml.dataset._core.activities import create_empty_activity_df, read_activity_csv, write_activity_csv
from pyadlml.dataset._core.devices import read_device_df, write_device_df
from pyadlml.dataset.plot.plotly.acts_and_devs import _determine_start_and_end, _plot_activities_into_fig
from pyadlml.dataset.plot.plotly.util import dash_get_trigger_element, dash_get_trigger_value
import json
import numpy as np
from pyadlml.dataset.util import activity_order_by, df_difference, fetch_by_name
from pyadlml.dataset.plot.plotly.activities import _set_compact_title, correction
import dash_bootstrap_components as dbc

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

deserialize_range = lambda x: [pd.Timestamp(ts) for ts in json.loads(x)] 
serialize_range = lambda x: json.dumps([ts.isoformat() for ts in x])
range_from_fig = lambda f: [pd.Timestamp(ts) for ts in f['layout']['xaxis']['range']]



DEL_KEY = 'DELETE'
JOIN_KEY = 'JOIN'


def json_to_df(df):
    try:
        return pd.read_json(df)
    except:
        return None
def df_to_json(df):
    try:
        return df.to_json(date_unit="ns")
    except:
        return ''



def label_figure(df_acts, df_devs, range_store=None, zoomed_range=[], states=False):
    if range_store is not None:
        st, et = range_store[0], range_store[1]
    else:
        st, et = None, None
    from pyadlml.dataset.plot.plotly.acts_and_devs import activities_and_devices

    fig = activities_and_devices(df_devs, df_acts, zoomed_range=zoomed_range, states=states)

    if st:
        fig.add_vline(x=st, line_width=1, line_color="Grey")
    if et:
        fig.add_vline(x=et, line_width=1, line_color="Grey")

    return fig





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='amsterdam')
    parser.add_argument('-i', '--identifier', type=str)
    parser.add_argument('-p', '--port', type=str, default='8050')
    parser.add_argument('-o', '--out-folder', type=str, default='/tmp/pyadlml/label_activities')
    parser.add_argument('-dh', '--data-home', type=str, default='/tmp/pyadlml/')
    args = parser.parse_args()

    out_folder = Path(args.out_folder)
    if not out_folder.exists():
        out_folder.mkdir(parents=True, exist_ok=True)
    
    fp_df_acts = out_folder.joinpath('activities.csv')
    fp_df_devs = out_folder.joinpath('devices.csv')
    #fp_df_del = out_folder.joinpath('removed_activities.csv')
    #fp_df_add = out_folder.joinpath('new_activities.csv')
    fp_code = out_folder.joinpath('relabel_activities.py')
    fp_code_dev = out_folder.joinpath('relabel_devices.py')

    # Setup data
    data = fetch_by_name(args.dataset, identifier=args.identifier, \
                         cache=False, retain_corrections=True, keep_original=True)
    df_acts = data['activities']
    df_devs = data['devices']
    activities = df_acts[ACTIVITY].unique()
    devices = df_devs[DEVICE].unique()

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    #df_del = create_empty_activity_df()
    #df_add = create_empty_activity_df()

    write_activity_csv(fp_df_acts, df_acts)
    write_device_df(fp_df_devs, df_devs)

    #df_del.to_csv(fp_df_del, index=False)
    #df_add.to_csv(fp_df_add, index=False)

    code_str_acts = "# Awesome script\n" \
             + "import pandas as pd\n"\
             + "from pyadlml.constants import START_TIME, END_TIME, ACTIVITY\n"\
             + "from pyadlml.dataset._core.activities import get_index_matching_rows\n\n"

    code_str_devs = "# Awesome script\n" \
             + "import pandas as pd\n"\
             + "from pyadlml.constants import TIME, DEVICE, VALUE\n"\
             + "from pyadlml.dataset._core.devices import get_index_matching_rows\n\n"

    if not fp_code.exists():
        with open(fp_code, mode='w') as f:
            f.write(code_str_acts)

    if not fp_code_dev.exists():
        with open(fp_code_dev, mode='w') as f:
            f.write(code_str_devs)


    fig = label_figure(df_acts, df_devs)
    (a,b,_,_) = _determine_start_and_end(df_acts, df_devs, st=None, et=None)
    fig_range = a,b

    app.layout = dbc.Container([
        dbc.Row(
            dcc.Graph(
                id='fig-activity',
                figure=fig,
            ), className='mb-3'
        ),
        dbc.Row([
            dcc.RadioItems(['events', 'states'], 'events', id='and-vis')
        ]),
        dbc.Row([
            dcc.RangeSlider(id='range',
                            min=0, max=1, step=0.001, value=[0,1])
        ]),
        dbc.Row([
            #dcc.Store(id='new-act-df', data=df_to_json(df_del)),
            #dcc.Store(id='del-act-df', data=df_to_json(df_add)),
            dcc.Store(id='range-store', data=serialize_range(fig_range)),
            dcc.Store(id='range-slider-t-1', data=[0,1]),
            dbc.Col(md=3),
            dbc.Col(
                dbc.ButtonGroup([
                    dbc.Button(id="update-range", children="update range", color="secondary"), 
                    dbc.Button(id="set-frame", children="set frame", color="secondary"), 
                ]),
            md=9),
        ], className='mb-1'),
        dbc.Row([
            dbc.Col(
                dash_table.DataTable(id='times', 
                                        columns=({'id': START_TIME, 'name': 'Start time'},
                                                {'id': END_TIME, 'name': 'End time'}),
                                        data=[{
                                        START_TIME: ts2str(df_acts.iloc[0,0]), 
                                        END_TIME: ts2str(df_acts.iloc[0,1])
                                        }],
                                        editable=True,
                                        style_cell={'font-size': '12px'}
                ), md=8
            ),
            dbc.Col(
                dbc.Row([
                    dbc.Col(
                        dbc.Row([
                            dbc.Select(id='activity',
                                        options=[ {"label": act, "value": act} for act in [*activities, DEL_KEY, JOIN_KEY]],
                            ),
                            dbc.Button(id="submit", children="Submit", color="primary")
                        ]), 
                    md=6),
                    dbc.Col(
                        dbc.Row([
                            dbc.Select(id='device',
                                        options=[ {"label": dev, "value": dev} for dev in devices],
                            ),
                            dcc.Dropdown(id='device-state',
                                        searchable=False, clearable=False, 
                                        options=[ {"label": cat, "value": cat} for cat in ['initial']],
                            ),
                            dbc.Button(id="submit-dev", children="Submit", color="primary")
                        ]),
                    md=6),
                ]),
            md=4,
            align='center', 
            className='mb-2')]
        ),
        dbc.Row(
            dbc.Textarea(id='comment', value='', readOnly=False, 
                        style={'height': '100px'}
                        #style={'resize': 'none', 'overflow':'hidden'}#, 'height': '100px'}
            ),
            style={'height':'100%'},
            className='mb-1',
        ),

    ], style={'width': 1500, 'margin': 'auto'})

    @app.callback(
        Output('device-state', 'options'),
        Output('device-state', 'value'),
        Input('device', 'value'),
    )
    def cat_select(value):
        trigger = dash_get_trigger_element()
        if trigger is None or trigger == '':
            raise PreventUpdate

        cats = df_devs.loc[df_devs[DEVICE] == value, VALUE].unique().tolist()

        if True in cats and False in cats and len(cats) == 2:
            cats = ['off', 'on']

        res = [{"label": str(cat), "value": cat} for cat in cats]

        return res, res[0]["label"]

    @app.callback(
        Output('fig-activity', 'figure'),
        Output('times', 'data'),
        Output('activity', 'value'),
        Output('range', 'value'),
        Output('range-store', 'data'),
        Output('range-slider-t-1', 'data'),
        Output('comment', 'value'),
        #Output('new-act-df', 'data'),
        #Output('del-act-df', 'data'),
        Input('range', 'value'),
        Input('and-vis', 'value'),
        Input('times', 'data'),
        Input('fig-activity', 'clickData'),
        Input('update-range', 'n_clicks'),
        Input('set-frame', 'n_clicks'),
        Input('submit', 'n_clicks'),
        Input('submit-dev', 'n_clicks'),
        State('fig-activity', 'figure'),
        State('range-store', 'data'),
        State('range-slider-t-1', 'data'),
        State('activity', 'value'),
        State('device', 'value'),
        State('device-state', 'value'),
        State('comment', 'value'),
        #State('new-act-df', 'data'),
        #State('del-act-df', 'data'),
    )
    def display_output(rng, and_vis, times, act_click, btn_update_rng, btn_set_frame, btn_submit, btn_submit_dev,
                       fig, range_store, rs_tm1, activity, device, device_state, comment
                       #fig, range_store, rs_tm1, df_json, df_del_json, activity
    ):
        def from_01(y, r):
            mn, mx = r[0], r[1]
            """ [0,1] -> timestamps"""
            return y*(mx-mn)+mn 
        def to_01(x, r):
            """ timestamps -> [0,1]"""
            mn, mx = r[0], r[1]
            return (x-mn)/(mx-mn)

        trigger = dash_get_trigger_element()
        if trigger is None or trigger == '':
            raise PreventUpdate

        #df_add = json_to_df(df_json)
        #df_del = json_to_df(df_del_json)
        
        f = open(fp_code, mode='r')
        code_str = ''.join(f.readlines())
        f.close()

        f = open(fp_code_dev, mode='r')
        code_str_devs = ''.join(f.readlines())
        f.close()
        
        #df_acts = read_activity_csv(fp_df_acts)
        #df_del = read_activity_csv(fp_df_del)
        #df_add = read_activity_csv(fp_df_add)

        # Create current activity df 
        #df_acts = df_acts.merge(df_del, how='outer', indicator=True)
        #df_acts = df_acts.loc[df_acts[(df_acts['_merge'] == 'left_only')].index, [START_TIME, END_TIME, ACTIVITY]]
        #df_acts = pd.concat([df_acts, df_add], axis=0).reset_index(drop=True)

        #df_acts2 = df_acts.copy()

        # TODO refactor, weird bug where locals()['df_acts'] is not overwritten
        #                when not expliciltly given as locals() dict and referenced with post exec command
        lcls = {'df_acts': read_activity_csv(fp_df_acts)} 
        exec(code_str, {}, lcls)
        df_acts = lcls['df_acts']

        lcls = {'df_devs': read_device_df(fp_df_devs)} 
        exec(code_str_devs, {}, lcls)
        df_devs = lcls['df_devs']

        fig_range = range_from_fig(fig)

        if range_store is not None:
            range_store = deserialize_range(range_store)

        if trigger == 'set-frame':
            # Get range from outer boundaries
            range_store = copy(fig_range)

            # Zoom a little bit out
            offset = (fig_range[1] - fig_range[0])*0.05
            fig_range[0] = fig_range[0] - offset
            fig_range[1] = fig_range[1] + offset

            rng[0], rng[1] = to_01(range_store[0], fig_range), to_01(range_store[1], fig_range)

        elif trigger == 'submit-dev':
            data = {
                START_TIME: range_store[0], 
                END_TIME: range_store[1], 
                DEVICE:device,
                VALUE: device_state
            }
            if data[VALUE] in ['on', 'off']:
                data[VALUE] = {'on':True, 'off':False}[data[VALUE]]

            code_str_devs += "\n#" + "-"*40 + '\n'
            code_str_devs += '\n'.join(['# ' + s for s in comment.split('\n')]) + '\n\n'

            def del_op2str(row):
                if isinstance(row[VALUE], str):
                    row[VALUE] = "'" + row[VALUE] + "'"
                row_lst = f"[['{ts2str(row[TIME])}','{row[DEVICE]}',{row[VALUE]}]]"
                s = f"idx_to_del = get_index_matching_rows(df_devs, {row_lst})\n"
                s += "df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)\n"
                return s


            def add_op2str(row):
                if isinstance(row[VALUE], str):
                    row[VALUE] = "'" + row[VALUE] + "'"
                s = "new_row=pd.Series(\n{"
                s += f"TIME: pd.Timestamp('{row[TIME]}'), DEVICE:'{row[DEVICE]}', VALUE:{row[VALUE]}"
                s += "})\ndf_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)\n"
                return s


            df = df_devs[df_devs[DEVICE] == data[DEVICE]].copy().reset_index(drop=True)
            mask = (data[START_TIME] < df[TIME])\
                 & (df[TIME] < data[END_TIME])
            df_tmp = df[mask]

            if len(df_tmp) == len(df):
                print('Warning! Select a range before submitting.')
                raise PreventUpdate
            # Delete enveloped events and get index of preceeding event
            # and the last enveloped event
            if not df_tmp.empty:
                for _, row in df_tmp.iterrows():
                    code_str_devs += del_op2str(row) + '\n'
                last_event = df_tmp.iloc[-1]
                prec_idx = df_tmp.index[0] - 1

                if last_event[VALUE] != data[VALUE]:
                    last_event[TIME] = data[END_TIME]
                    code_str_devs += add_op2str(last_event)

                if prec_idx >= 0:
                    prec_event = df.loc[prec_idx, ]

                if prec_idx < 0 or prec_event[VALUE] != data[VALUE]:
                    code_str_devs += add_op2str(
                        pd.Series({TIME:data[START_TIME], DEVICE:data[DEVICE], VALUE: data[VALUE]})
                    )
            else:
                prec_events = df[df[TIME] < data[START_TIME]]
                if not prec_events.empty:
                    prec_event = prec_events.iloc[-1]
                    prec_event[TIME] = data[END_TIME]
                    code_str_devs += add_op2str(prec_event)

                code_str_devs += add_op2str(
                    pd.Series({TIME:data[START_TIME], DEVICE:data[DEVICE], VALUE: data[VALUE]})
                )


            # Reset values
            range_store = None
            rng = [0,1]
            comment = ''

            # Reload activity dataframe
            lcls = {'df_devs': read_device_df(fp_df_devs)} 
            exec(code_str_devs, {}, lcls)
            df_devs = lcls['df_devs']


        elif trigger == 'submit':
            # Add activity to dataframe
            data = {
                START_TIME: range_store[0], 
                END_TIME: range_store[1], 
                ACTIVITY:activity
            }
            def del_op2str(row):
                row_lst = f"\t[['{ts2str(row[START_TIME])}','{ts2str(row[END_TIME])}','{row[ACTIVITY]}']],\n"
                return f"idx_to_del = get_index_matching_rows(df_acts, {row_lst})\n"\
                     + f"df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)\n"

            def add_op2str(row):
                row_dict = "{"\
                        + f"START_TIME: pd.Timestamp('{(row[START_TIME])}'), "\
                        + f"END_TIME: pd.Timestamp('{row[END_TIME]}'), "\
                        + f"ACTIVITY: '{row[ACTIVITY]}'"\
                + "}"
                return f"new_row=pd.Series({row_dict})\n"\
                      + "df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)\n"

            # Start adding to code string and add comment if given
            code_str += "\n#" + "-"*40 + '\n'
            code_str += '\n'.join(['# ' + s for s in comment.split('\n')]) + '\n\n'

            new_int = pd.Interval(data[START_TIME], data[END_TIME])
            df_acts[TIME] = df_acts[[START_TIME, END_TIME]].apply(lambda x: pd.Interval(x[START_TIME], x[END_TIME]), axis=1)
            mask = df_acts[TIME].apply(lambda x: x.overlaps(new_int))
            df_acts = df_acts[[START_TIME, END_TIME, ACTIVITY]]
            ov_act = df_acts[mask]

            if data[ACTIVITY] == DEL_KEY and not ov_act.empty:
                ov_act = df_acts[mask].iloc[0]
                #df_del = pd.concat([df_del, ov_act.to_frame().T], axis=0)
                code_str += '\n# Delete operation:\n' + del_op2str(ov_act)

                if data[START_TIME] < ov_act[START_TIME] and data[END_TIME] < ov_act[END_TIME]:
                    #    |~~~~~~~~|
                    # | DEL |
                    ov_act[START_TIME] = data[END_TIME]
                    #df_add = pd.concat([df_add, ov_act.to_frame().T], axis=0)
                    code_str += '\n' + add_op2str(ov_act)
                elif ov_act[START_TIME] < data[START_TIME] and data[END_TIME] < ov_act[END_TIME]:
                    #|~~~~~~~~|
                    # | DEL |
                    ov_actr = copy(ov_act)
                    ov_actl = copy(ov_act)
                    ov_actr[END_TIME] = data[START_TIME]
                    ov_actl[START_TIME] = data[END_TIME]
                    #df_add = pd.concat([df_add, ov_actr.to_frame().T, ov_actl.to_frame().T
                    #                   ], axis=0)

                    code_str += '\n' + add_op2str(ov_actr)
                    code_str += '\n' + add_op2str(ov_actl)
                elif ov_act[START_TIME] < data[START_TIME] and ov_act[END_TIME] < data[END_TIME]:
                    #|~~~~~~~~|
                    #     | DEL |
                    ov_act[END_TIME] = data[START_TIME]
                    #df_add = pd.concat([df_add, ov_act.to_frame().T], axis=0)
                    code_str += '\n' + add_op2str(ov_act)
                elif data[START_TIME] < ov_act[START_TIME] and ov_act[END_TIME] < data[END_TIME]: 
                    # |~~~~|
                    #|   DEL   |
                    pass
                else:
                    raise NotImplementedError
            elif data[ACTIVITY] == JOIN_KEY and not ov_act.empty:
                assert len(ov_act) == 2, f'Select two activities to join. Only one selected: {ov_act}'
                ov_actr = ov_act.iloc[0]
                ov_actl = ov_act.iloc[1]
                code_str += '\n# Join operation: \n'
                code_str += del_op2str(ov_actr)
                code_str += del_op2str(ov_actl)
                ov_actr[END_TIME] = ov_actl[END_TIME]
                code_str += add_op2str(ov_actr)
            elif not ov_act.empty:
                assert len(ov_act) == 1, 'Can only edit one activity at a time.'
                ov_act = df_acts[mask].iloc[0]
                assert ov_act[ACTIVITY] == activity, 'Only select one activity'

                code_str += '\n# Modify operation:\n'

                #df_del = pd.concat([df_del, ov_act.to_frame().T], axis=0)
                code_str += '\n' + del_op2str(ov_act)

                # Case when an activity is modified 
                ov_act[START_TIME] = data[START_TIME]
                ov_act[END_TIME] = data[END_TIME]
                #df_add = pd.concat([df_add, ov_act.to_frame().T], axis=0)\
                #    .reset_index(drop=True)
                code_str += '\n' + add_op2str(ov_act)
            else:
                # Case when a new activity is being created
                assert data[ACTIVITY] != DEL_KEY, 'Can not add non-existing activity "DELETE"'
                act = pd.Series(data)
                code_str += '\n# Create operation:\n'

                #df_add = pd.concat([df_add, act.to_frame().T], axis=0)\
                #    .reset_index(drop=True)
                code_str += '\n' + add_op2str(act)

            # Reset values
            range_store = None
            rng = [0,1]
            comment = ''

            # Reload activity dataframe
            lcls = {'df_acts': read_activity_csv(fp_df_acts)} 
            exec(code_str, {}, lcls)
            df_acts = lcls['df_acts']

        elif trigger == 'range': 
            if rs_tm1[0] == rng[0]:
                # case when right lever was altered
                range_store[1] = from_01(rng[1], fig_range)

                # Set the right slider and then update right bar
                rng[0] = to_01(range_store[0], fig_range)

            elif rs_tm1[1] == rng[1]:
                # case when left lever was altered
                range_store[0] = from_01(rng[0], fig_range)

                # Set the left slider and then update left bar
                rng[1] = to_01(range_store[1], fig_range)
            else:
                raise PreventUpdate

        elif trigger == 'update-range':
            # grey lines set slider 
            rng[0], rng[1] = to_01(range_store[0], fig_range), to_01(range_store[1], fig_range)
        
        elif trigger == 'times':
            range_store[0] = str2ts(times[0][START_TIME])
            range_store[1] = str2ts(times[0][END_TIME])

        
        elif trigger == 'fig-activity':
            if not 'Acts: ' in act_click['points'][0]['y']:
                PreventUpdate

            st = pd.Timestamp(act_click['points'][0]['base'])
            #et = str2ts(act_click['points'][0]['x'])
            act = df_acts[(df_acts[START_TIME] == st)]# & (df_acts[END_TIME] == et)]
            if act.empty and len(act) == 1:
                PreventUpdate
            act = act.iloc[0]
            # Update ranges
            range_store = [act[START_TIME], act[END_TIME]]
            times[0][START_TIME] = ts2str(range_store[0])
            times[0][END_TIME] = ts2str(range_store[1])

            activity = act[ACTIVITY]

            rng[0], rng[1] = to_01(range_store[0], fig_range), to_01(range_store[1], fig_range)

        if trigger not in ['update-range']:
            # Remove df_del activities from df_acts
            #df_show = df_acts.merge(df_del, how='outer', indicator=True)
            #df_show = df_show.loc[df_show[(df_show['_merge'] == 'left_only')].index, [START_TIME, END_TIME, ACTIVITY]]
            #df_show = pd.concat([df_show, df_add], axis=0)\
            #            .drop_duplicates()\
            #            .reset_index(drop=True)

            #fig = label_figure(df_show, df_devs, range_store, fig_range)
            fig = label_figure(df_acts, df_devs, range_store, fig_range, states=(and_vis == 'states'))

        if range_store is not None:
            times[0][START_TIME] = ts2str(range_store[0])
            times[0][END_TIME] = ts2str(range_store[1])
            range_store = serialize_range(range_store)

        rs_tm1[0], rs_tm1[1] = np.round(max(rng[0], 0), 3), np.round(min(rng[1], 1), 3)
        #df_json = df_to_json(df_add)
        #df_del_json = df_to_json(df_del)

        #write_activity_csv(fp_df_add, df_add)
        #write_activity_csv(fp_df_del, df_del)

        with open(fp_code_dev, mode='w') as f:
            f.write(code_str_devs)

        with open(fp_code, mode='w') as f:
            f.write(code_str)

        return fig, times, activity, rng, range_store, rs_tm1, comment#, df_json, df_del_json

    app.run_server(debug=False, host='127.0.0.1', port=args.port)
