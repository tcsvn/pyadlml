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
from pyadlml.dataset._core.activities import ActivityDict, create_empty_activity_df
from pyadlml.dataset.act_assist import read_activities, write_activities, read_devices, write_devices
from pyadlml.dataset.plot.plotly.acts_and_devs import _determine_start_and_end, _plot_activities_into_fig
from pyadlml.dataset.plot.plotly.util import dash_get_trigger_element, dash_get_trigger_value, set_fig_range
import json
import numpy as np
from pyadlml.dataset.util import activity_order_by, device_order_by, df_difference, fetch_by_name, num_to_timestamp, select_timespan
from pyadlml.dataset.plot.plotly.activities import _set_compact_title, correction
from pyadlml.dataset.plot.plotly.util import deserialize_range, serialize_range, range_from_fig
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

def read_code_and_update(fp_code, fp_code_dev, fp_df_acts, fp_df_devs):
    f = open(fp_code, mode='r')
    code_str = ''.join(f.readlines())
    f.close()

    f = open(fp_code_dev, mode='r')
    code_str_devs = ''.join(f.readlines())
    f.close()

    # TODO refactor, weird bug where locals()['df_acts'] is not overwritten
    #                when not expliciltly given as locals() dict and referenced with post exec command
    lcls = {'df_acts': read_activities(fp_df_acts)} 
    exec(code_str, {}, lcls)
    df_acts = lcls['df_acts']

    lcls = {'df_devs': read_devices(fp_df_devs)} 
    exec(code_str_devs, {}, lcls)
    df_devs = lcls['df_devs']
    return code_str, code_str_devs, df_acts, df_devs


def _get_room_y_pos(dev_order, dev2area, area_select):


    df = dev2area.copy()
    df[DEVICE] = df[DEVICE].astype("category")
    df[DEVICE] = df[DEVICE].cat.set_categories(dev_order)
    df = df.sort_values(by=DEVICE).reset_index(drop=True).reset_index()

    upper_y = len(dev_order) - df[df['area'] == area_select]['index'].iat[0]
    lower_y = len(dev_order)-1 - df[df['area'] == area_select]['index'].iat[-1]
    return lower_y, upper_y


def label_figure(df_acts, df_devs, dev2area, range_store=None, zoomed_range=[], states=False, area_select=''):
    if range_store is not None:
        st, et = range_store[0], range_store[1]
    else:
        st, et = None, None

    from pyadlml.dataset.plot.plotly.acts_and_devs import activities_and_devices
    dev_order = 'alphabetical' if dev2area is None else 'area'


    fig = activities_and_devices(df_devs, df_acts, dev2area=dev2area, zoomed_range=zoomed_range, states=states, dev_order=dev_order)

    if st:
        fig.add_vline(x=st, line_width=1, line_color="Grey")
    if et:
        fig.add_vline(x=et, line_width=1, line_color="Grey")

    if area_select:
        # Get y indicies of device bars/events
        dev_order = device_order_by(df_devs, rule='area', dev2area=dev2area)
        lower_y, upper_y = _get_room_y_pos(dev_order, dev2area, area_select)
        opacity = 0.5
    else:
        lower_y, upper_y, opacity = 1, 2, 0.0

    # Add highlight rectangle
    fig.add_hrect(
        y0=lower_y - 0.5, y1=upper_y - 0.5,
        fillcolor="LightSalmon", opacity=opacity,
        layer="below", line_width=0,
    )
    

    return fig



def reset_range_store(df_devs, df_acts):
    offset = pd.Timedelta('2s')
    st = min(df_devs.at[df_devs.index[0], TIME],
                df_acts.at[df_acts.index[0], START_TIME])
    et = max(df_devs.at[df_devs.index[-1], TIME],
                df_acts.at[df_acts.index[-1], END_TIME]
    )
    return [st - offset, et + offset]


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

    tmp_folder = Path('/tmp/pyadlml/label_data/')
    tmp_folder.mkdir(exist_ok=True, parents=True)

    fp_df_acts = tmp_folder.joinpath('activities.csv')
    fp_df_devs = tmp_folder.joinpath('devices.csv')

    if fp_df_acts.exists():    
        fp_df_acts.unlink()
    
    if fp_df_devs.exists():    
        fp_df_devs.unlink()

    #fp_df_del = out_folder.joinpath('removed_activities.csv')
    #fp_df_add = out_folder.joinpath('new_activities.csv')
    fp_code = out_folder.joinpath('relabel_activities.py')
    fp_code_dev = out_folder.joinpath('relabel_devices.py')

    # Setup data
    data = fetch_by_name(args.dataset, identifier=args.identifier, \
                         cache=False, retain_corrections=True, keep_original=True)
    df_acts = data['activities']
    df_devs = data['devices']

    # TODO DEBUG
    try:
        dev2area = data['dev2area']
    except:
        dev2area = None
    
    activities = df_acts[ACTIVITY].unique()
    devices = df_devs[DEVICE].unique()

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    #df_del = create_empty_activity_df()
    #df_add = create_empty_activity_df()
    df_acts = df_acts.sort_values(by=START_TIME).reset_index(drop=True)
    df_devs = df_devs.sort_values(by=TIME).reset_index(drop=True)

    write_activities(df_acts, fp_df_acts)
    write_devices(df_devs, fp_df_devs)

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


    code_str, code_str_devs, df_acts, df_devs = read_code_and_update(fp_code, fp_code_dev, fp_df_acts, fp_df_devs)

    (a,b,_,_) = _determine_start_and_end(df_acts, df_devs, st=None, et=None)
    fig_range = a,b

    fig = label_figure(df_acts, df_devs, dev2area, range_store=fig_range)

    from pyadlml.dataset.plot.plotly.dashboard.layout import _build_range_slider

    app.layout = dbc.Container([
        dbc.Row(
            dcc.Graph(
                id='fig-activity',
                figure=fig,
            ), className='mb-3'
        ),
        dbc.Row([
            dbc.Col(
                dcc.RadioItems(['events', 'states'], 'events', id='and-vis'),
                md=4
            ),
            dbc.Col(
                dbc.Input(id='fig-range-input', placeholder='Paste range from dashboard'),
                md=4
            ),
            dbc.Col(
                dcc.Dropdown(id='area-select', options=[area for area in dev2area['area'].unique()], value=None),
                md=4
            ),

        ]),
        dbc.Row([
            _build_range_slider(df_acts, df_devs, a, b, b),
            dcc.Store(id='range-store', data=serialize_range(fig_range)),
            dcc.Store(id='range-slider-t-1', data=[0, 1]),       
        ],  className='mb-2'),
        dbc.Row([
            dbc.Col(md=5),
            dbc.Col(
                dbc.ButtonGroup([
                    dbc.Button(id="set-left", children="to left", color="secondary"), 
                    dbc.Button(id="set-frame", children="set frame", color="secondary"), 
                    dbc.Button(id="set-right", children="to right", color="secondary"), 
                ]),
            md=5),
            dbc.Col(md=2),
        ], className='mb-1', justify='center'),
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

    ], style={'width': 1800, 'margin': 'auto'})

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

        res = [{"label": str(cat), "value": cat} for cat in cats + ['del events']]

        return res, res[0]["label"]

    @app.callback(
        Output('fig-activity', 'figure'),
        Output('times', 'data'),
        Output('activity', 'value'),
        Output('range-store', 'data'),
        Output('range-slider-t-1', 'data'),
        Output('comment', 'value'),
        Input('and-vis', 'value'),
        Input('times', 'data'),
        Input('fig-activity', 'clickData'),
        Input('set-frame', 'n_clicks'),
        Input('set-left', 'n_clicks'),
        Input('set-right', 'n_clicks'),
        Input('submit', 'n_clicks'),
        Input('submit-dev', 'n_clicks'),
        Input('fig-range-input', 'value'),
        Input('area-select', 'value'),
        State('fig-activity', 'figure'),
        State('range-store', 'data'),
        State('range-slider-t-1', 'data'),
        State('activity', 'value'),
        State('device', 'value'),
        State('device-state', 'value'),
        State('comment', 'value'),
        State('range-slider', 'value'),
    )
    def display_output(and_vis, times, act_click, btn_set_frame, btn_left, btn_right, btn_submit, btn_submit_dev,
                       fig_range_input, area_select, fig, range_store, rs_tm1, activity, device, device_state, comment, range_select
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

        # Check if data has to be reloaded or callback is independent
        is_data_update = trigger not in ['set-frame', 'set-left', 'set-right', 'fig-range-input', 'area-select']
        is_figure_reload = trigger not in ['set-frame', 'set-left', 'set-right', 'fig-range-input', 'fig-activity', 'area-select']


        if is_data_update:
            code_str, code_str_devs, df_acts, df_devs = read_code_and_update(fp_code, fp_code_dev, fp_df_acts, fp_df_devs)

        fig_range = range_from_fig(fig)

        if range_store is not None:
            range_store = deserialize_range(range_store)
        else:
            range_store = reset_range_store(df_devs, df_acts)

        if trigger == 'set-frame':
            # Get range from outer boundaries
            range_store = copy(fig_range)

            # Zoom a little bit out
            offset = (fig_range[1] - fig_range[0])*0.05
            fig_range[0] = fig_range[0] - offset
            fig_range[1] = fig_range[1] + offset

            #rng[0], rng[1] = to_01(range_store[0], fig_range), to_01(range_store[1], fig_range)
        if trigger == 'set-left':
            range_store[0] = copy(fig_range[0])
            offset = (fig_range[1] - fig_range[0])*0.05
            fig_range[0] = fig_range[0] - offset
            #rng[0], rng[1] = to_01(range_store[0], fig_range), to_01(range_store[1], fig_range)
        if trigger == 'set-right':
            range_store[1] = copy(fig_range[1])
            offset = (fig_range[1] - fig_range[0])*0.05
            fig_range[1] = fig_range[1] + offset
            #rng[0], rng[1] = to_01(range_store[0], fig_range), to_01(range_store[1], fig_range)
            
        if trigger == 'submit-dev':
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

            def del_bulk_op2str(st, et, dev):
                s =  f"mask = (df_devs[DEVICE] == '{dev}')\\\n"
                s += f"     & ('{st}' < df_devs[TIME])\\\n"
                s += f"     & (df_devs[TIME] < '{et}')\n"
                s += f"df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)\n\n"
                return s

            df = df_devs[df_devs[DEVICE] == data[DEVICE]].copy().reset_index(drop=True)
            mask = (data[START_TIME] < df[TIME])\
                 & (df[TIME] < data[END_TIME])
            df_tmp = df[mask]

            if len(df_tmp) == len(df):
                print('Warning! Select a range before submitting.')
                raise PreventUpdate

            if not df_tmp.empty and data[VALUE]  == 'del events':
                code_str_devs += del_bulk_op2str(data[START_TIME], data[END_TIME], data[DEVICE])

            # Delete enveloped events and get index of preceeding event
            # and the last enveloped event
            elif not df_tmp.empty:
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

            range_store = reset_range_store(df_devs, df_acts)
            #rng = [0,1]
            comment = ''

            # Reload activity dataframe
            lcls = {'df_devs': read_devices(fp_df_devs)} 
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
                elif ov_act[START_TIME] <= data[START_TIME] and data[END_TIME] <= ov_act[END_TIME]:
                    # |~~~~~|
                    # | DEL |
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
            range_store = reset_range_store(df_devs, df_acts)
            #rng = [0,1]
            comment = ''

            # Reload activity dataframe
            lcls = {'df_acts': read_activities(fp_df_acts)} 
            exec(code_str, {}, lcls)
            df_acts = lcls['df_acts']

        elif trigger == 'times':
            range_store[0] = str2ts(times[0][START_TIME])
            range_store[1] = str2ts(times[0][END_TIME])

        #        # Set the left slider and then update left bar
        #        rng[1] = to_01(range_store[1], fig_range)
        #    else:
        #        raise PreventUpdate

        #elif trigger == 'update-range':
        #    # grey lines set slider 
        #    rng[0], rng[1] = to_01(range_store[0], fig_range), to_01(range_store[1], fig_range)
        
        
        elif trigger == 'fig-activity':
            if not 'Acts: ' in act_click['points'][0]['y']:
                raise PreventUpdate

            st = pd.Timestamp(act_click['points'][0]['base'])
            #et = str2ts(act_click['points'][0]['x'])
            act = df_acts[(df_acts[START_TIME]-pd.Timedelta('1us') < st)
                        & (st < df_acts[START_TIME]+pd.Timedelta('1us'))]
            # & (df_acts[END_TIME] == et)]
            if act.empty and len(act) == 1:
                raise PreventUpdate
            act = act.iloc[0]
            # Update ranges
            range_store = [act[START_TIME], act[END_TIME]]
            times[0][START_TIME] = ts2str(range_store[0])
            times[0][END_TIME] = ts2str(range_store[1])

            activity = act[ACTIVITY]

        elif trigger == 'fig-range-input':
            try:
                tmp = fig_range_input.split(',')
                fig_range = [str2ts(tmp[0]), str2ts(tmp[1])]
                fig = set_fig_range(fig, fig_range)
            except:
                raise PreventUpdate

        if trigger in ['set-frame', 'set-left', 'set-right', 'fig-activity']:
            # Update vlines without recomputing figure
            try:
                # Left vline
                fig['layout']['shapes'][0]['x0'] = str(range_store[0])
                fig['layout']['shapes'][0]['x1'] = str(range_store[0])

                # Right vline
                fig['layout']['shapes'][1]['x0'] = str(range_store[1]) 
                fig['layout']['shapes'][1]['x1'] = str(range_store[1])

                set_fig_range(fig, fig_range)
            except KeyError:
                trigger = 'llululul'

        if trigger == 'area-select':
            try:
                dev_order = copy(fig['layout']['yaxis']['categoryarray'])
                dev_order.reverse()
                y0, y1 = _get_room_y_pos(dev_order, dev2area, area_select)
                fig['layout']['shapes'][2]['y0'] = y0 - 0.5
                fig['layout']['shapes'][2]['y1'] = y1 - 0.5
                fig['layout']['shapes'][2]['opacity'] = 0.5
            except KeyError or IndexError:
                trigger = 'llululul'

        if is_figure_reload:
            import time
            tic = time.perf_counter()

            # Only view a subset of the activities
            dct_acts = ActivityDict.wrap(df_acts)
            start_time = min(df_devs[TIME].iloc[0], dct_acts.min_starttime())
            start_time = start_time.floor('D')
            end_time = max(df_devs[TIME].iloc[-1], dct_acts.max_endtime())
            end_time = end_time.ceil('D')

            st = num_to_timestamp(range_select[0], start_time=start_time, end_time=end_time)
            et = num_to_timestamp(range_select[1], start_time=start_time, end_time=end_time)
            curr_df_devs, curr_df_acts = select_timespan(df_acts=df_acts, df_devs=df_devs,
                                                        start_time=st, end_time=et, clip_activities=False)
            fig = label_figure(curr_df_acts, curr_df_devs, dev2area, range_store, fig_range, states=(and_vis == 'states'), area_select=area_select)
            toc = time.perf_counter()
            print(f"Updating figure in: {toc - tic:0.4f} seconds")


        # TODO, check if this works
        times[0][START_TIME] = ts2str(range_store[0])
        times[0][END_TIME] = ts2str(range_store[1])
        range_store = serialize_range(range_store)


        vars = [*locals().keys()]
        if 'code_str_devs' in vars:
            with open(fp_code_dev, mode='w') as f:
                f.write(code_str_devs)

        if 'code_str' in vars:
            with open(fp_code, mode='w') as f:
                f.write(code_str)

        return fig, times, activity, range_store, rs_tm1, comment

    app.run_server(debug=False, host='127.0.0.1', port=args.port)
