from copy import deepcopy

import dash
import dash.dcc as dcc
import dash.html as html
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dependencies import *  # TODO remove if nothing else is imported
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dask.delayed import delayed
import numpy as np
from pyadlml.constants import (ACTIVITY, DEVICE, END_TIME, START_TIME, TIME,
                               VALUE)
from pyadlml.dataset._core.acts_and_devs import label_data
from pyadlml.dataset._core.activities import add_other_activity, create_empty_activity_df
from pyadlml.dataset._datasets.aras import _create_device_df
from pyadlml.dataset.plot.plotly.util import dash_get_trigger_element, dash_get_trigger_value
from pyadlml.dataset.plot.plotly.activities import *
from pyadlml.dataset.plot.plotly.acts_and_devs import *
from pyadlml.dataset.plot.plotly.dashboard.callbacks import (
    _create_activity_tab_callback, _create_acts_vs_devs_tab_callback, _initialize_activity_toggle_cbs)
from pyadlml.dataset.plot.plotly.dashboard.layout import (
    BIN_SIZE_SLIDER, DEV_DENS_SLIDER, LAG_SLIDER, activities_layout, acts_n_devs_layout,
    acts_vs_devs_layout, device_layout_graph_bottom, devices_layout)
from pyadlml.dataset.plot.plotly.devices import bar_count as dev_bar_count
from pyadlml.dataset.plot.plotly.devices import boxplot_state
from pyadlml.dataset.plot.plotly.devices import device_iei as dev_iei
from pyadlml.dataset.plot.plotly.devices import event_density
from pyadlml.dataset._core.activities import ActivityDict
from pyadlml.dataset.stats.acts_and_devs import (contingency_table_events,
                                                 contingency_table_states)
from pyadlml.dataset.util import (activity_order_by, device_order_by, num_to_timestamp,
                                  select_timespan, timestamp_to_num)
from pyadlml.dataset.plot.plotly.devices import plotly_event_correlogram as dev_cc
from pyadlml.dataset.plot.plotly.util import deserialize_range, serialize_range, range_from_fig
from pyadlml.constants import ts2str, str2ts

def _get_plot_height_devs(nr_devs):
    # Determine the plot height and fontsize for activity plots
    if nr_devs < 20:
        return 350
    elif nr_devs < 30:
        return 400
    elif nr_devs < 50:
        return 500
    elif nr_devs < 70:
        return 700
    else:
        return 800


def _get_plot_height_acts(nr_acts):
    if nr_acts < 20:
        return 350
    elif nr_acts < 25:
        return 450
    else:
        return 350


def dashboard(app, name, embedded=False, df_acts=None, df_devs=None, dev2area=None, start_time=None, end_time=None):
    """ Creates a dashboard

    Note
    -----
    All parameter may be 'None' if the app is populated during runtime

    """

    df_acts = ActivityDict.wrap(df_acts)

    # Since performance rendering issues arise with numbers greater than 40000
    # device datapoints make a preselection
    max_points = 40000
    if df_devs is not None and len(df_devs) > max_points:
        set_et = df_devs.iat[max_points, df_devs.columns.get_loc(TIME)]
        curr_df_devs, curr_df_acts = select_timespan(
            df_acts=df_acts, df_devs=df_devs, start_time=start_time,
            end_time=set_et, clip_activities=True
        )
    else:
        curr_df_devs = df_devs
        curr_df_acts = df_acts
        set_et = end_time
    
    nr_devs = len(df_devs[DEVICE].unique()) if df_devs is not None else 0
    nr_acts = df_acts.nr_acts() if df_acts is not None else 0

    if start_time is None:
        start_time = min(df_devs[TIME].iloc[0], df_acts.min_starttime())
        start_time = start_time.floor('D')
    if end_time is None:
        end_time = max(df_devs[TIME].iloc[-1], df_acts.max_endtime())
        end_time = end_time.ceil('D')

    # TODO refactor: ugly as hell defined in 76
    if df_devs is None or len(df_devs) <= max_points:
        set_et = end_time

    plot_height_devs = _get_plot_height_devs(nr_devs)
    plot_height_acts = _get_plot_height_acts(nr_acts)
    is_top_down_view = (nr_devs < 60)
    plot_height_ands = (plot_height_devs+50 if is_top_down_view else 1300)

    # Get Layout
    layout_activities = [activities_layout(df, k, plot_height_acts) 
                         for k, df in curr_df_acts.items()]
    layout_devices = devices_layout(curr_df_devs, True, plot_height_devs)
    layout_acts_vs_devs = [acts_vs_devs_layout(df, k, df_devs, plot_height=plot_height_acts) 
                           for k, df in curr_df_acts.items()]

    # Generate tabs for multiple persons
    activity_tabs = [dbc.Tab(l, label=f'Act: {act_name}', tab_id=f'tab-acts-{act_name}') 
                     for act_name, l in zip(curr_df_acts.keys(), layout_activities)]
    acts_vs_devs_tabs = [
                 dbc.Tab(l, label=f'Act: {act_name} ~ Devices', tab_id=f'tab-acts_vs_devs-{act_name}')
                 for act_name, l in zip(curr_df_acts.keys(), layout_acts_vs_devs)]
    
    def layout_content(and_layout, tabs_and_content):
        if is_top_down_view:
            return [
                and_layout, 
                html.Br(),
                html.Div([
                    *tabs_and_content
                ]),
            ]
        else:
            return [
                dbc.Row([
                    dbc.Col([
                        and_layout, 
                    ], md=7),
                    dbc.Col([
                        html.Div([
                            *tabs_and_content
                        ]),
                    ], md=5),
                ], style=dict(width="2800px", height="1500px"))
            ]

    content = [
        dcc.Input(id='act_assist_path', type='hidden', value='filler text'),
        dcc.Input(id='subject_names', type='hidden', value='filler text'),
        html.H2(children=f'Dashboard: {name}'),

        *layout_content(
            acts_n_devs_layout(df_devs, df_acts, start_time, end_time, set_et, plot_height_ands),
            [dbc.Tabs(
                [*activity_tabs,
                    dbc.Tab(layout_devices, label='Devices', tab_id='tab-devs'),
                    *acts_vs_devs_tabs
                    ], id='tabs', active_tab=f'tab-acts-{list(curr_df_acts.keys())[0]}',
            ),
            html.Div(id="content"),]
        )
    ]

    if embedded:
        layout = dbc.Container(
            children=content, 
            style={'width': 1300, 'margin': 'auto'}
        )
    else:
        width = 1320 if is_top_down_view else 2200
        layout = dbc.Container(
            children=content,
            style={'width': width}
        )
    app.layout = layout

    df_acts.apply(add_other_activity)

    create_callbacks(app, df_acts, df_devs, start_time, end_time, dev2area, 
                     plot_height_acts, plot_height_devs, plot_height_ands
    )

def _sel_avd_event_click(avd_event_select, curr_df_acts, curr_df_devs):
    """Select devices and activities from click data of the event contingency table"""

    # Get relevant acts and devs
    activity = avd_event_select['points'][0]['y']
    device = avd_event_select['points'][0]['x']
    df_devs = curr_df_devs[curr_df_devs[DEVICE] == device].copy()
    df_acts = curr_df_acts[curr_df_acts[ACTIVITY] == activity].copy()

    # Select devices
    sel_devices = label_data(df_devs, df_acts).dropna()
    sel_devices = sel_devices[[TIME, DEVICE, VALUE]]

    # Select activities
    sel_activities = create_empty_activity_df()
    for _, sel_dev in sel_devices.iterrows():
        mask = (df_acts[START_TIME] < sel_dev[TIME]) \
               & (sel_dev[TIME] < df_acts[END_TIME])
        sel_activities = pd.concat([sel_activities, df_acts[mask]])
    sel_activities = sel_activities.drop_duplicates()
    return sel_devices, sel_activities


def _sel_point_to_activities(selection):
    """ Gets the selected activities from the boxplot or violin-plot and
        asdf
    """
    user_sel_activities = []
    points = selection['points']
    for usa in points:
        user_sel_activities.append([usa['customdata'][0], usa['customdata'][1], usa['y']])

    df_usa = pd.DataFrame(user_sel_activities, columns=[START_TIME, END_TIME, ACTIVITY])
    df_usa[START_TIME] = pd.to_datetime(df_usa[START_TIME])
    df_usa[END_TIME] = pd.to_datetime(df_usa[END_TIME])
    return df_usa

def _sel_act_bar(selection, df_acts):
    """ Retrieve the activity from a click on an activity bar in a
        bar plot e.g count, cummulative and filter the activity dataframe

    """
    dev = selection['points'][0]['y']
    return df_acts[df_acts[ACTIVITY] == dev]


def _sel_dev_bar(selection, df_devs):
    """ Retrieve the device from a click on the device bar the event count
        graphand filter the device dataframe

    """
    dev = selection['points'][0]['y']
    return df_devs[df_devs[DEVICE] == dev]

def _sel_dev_fraction(selection, df_devs):
    """ Retrieve the device from a click on the device bar the event count
        graphand filter the device dataframe

    """
    dev = selection['points'][0]['y']
    cat = selection['points'][0]['customdata'][1]
    mask = (df_devs[DEVICE] == dev) & (df_devs[VALUE] == cat)
    return df_devs[mask].copy()

def _sel_dev_bp_selection(dev_boxplot_select, df_devs):
    """ Gets point data from the state boxplot and creates
        an device-dataframe
    """
    points = dev_boxplot_select['points']
    sd = pd.DataFrame(columns=[TIME, END_TIME, DEVICE, VALUE])
    for p in points:
        duration = pd.Timedelta(p['customdata'][0])
        time = p['customdata'][1]
        dev = p['y']
        mask = (df_devs[DEVICE] == dev) & (df_devs[TIME] == time)
        df = df_devs[mask]
        df[END_TIME] = df[TIME] + duration
        sd = pd.concat([sd, df])

    return sd


def _sel_dev_iei(dev_iei_select, df_devs, scale, fig):

    per_device = not (len(fig['data']) == 1)
    df = df_devs.copy()
    df_res = pd.DataFrame(columns=[TIME, DEVICE, VALUE])
    df['diff'] = df[TIME].shift(-1) - df[TIME]
    size = fig['data'][0]['xbins']['size']

    for p in dev_iei_select['points']:
        left, right = p['x'] - size/2, p['x'] + size/2
        if scale == 'log':
            left, right = np.exp(left), np.exp(right)
        mask = (pd.to_timedelta(left, unit='s') <= df['diff']) \
               & (df['diff'] <= pd.to_timedelta(right, unit='s'))
        if per_device:
            mask = mask & (df[DEVICE] == p['customdata'])

        count = sum(mask)
        sel_count = len(p['pointNumbers'])
        assert count == sel_count, f'Lookup error: The selected device number {sel_count} != {count}'
        df_res = pd.concat([df_res, df_devs[mask]])

    return df_res


def _sel_dev_density(selection, df_devs, dt):

    from pyadlml.dataset.plot.plotly.dashboard.layout import DEV_DENS_SLIDER
    df = df_devs.copy().set_index(TIME, drop=False)
    dt = pd.Timedelta(DEV_DENS_SLIDER[dt])
    df_res = pd.DataFrame(columns=[TIME, DEVICE, VALUE])

    for p in selection['points']:
        dev = p['y']
        count = p['z']
        mid = pd.Timestamp(p['x'])
        tmp = df[df[DEVICE] == dev].between_time(
                    start_time=(mid - dt/2).time(),
                    end_time=(mid + dt/2).time())\
              .reset_index(drop=True)

        assert count == len(tmp.index)
        df_res = pd.concat([df_res, tmp])

    if df_res.empty:
        return None
    else:
        return df_res


def _sel_act_trans_sel(act_trans_select, df_acts):
    """ Gets point data from the state boxplot and creates
        an device-dataframe
    """
    df = df_acts.copy()\
        .sort_values(by=START_TIME)\
        .reset_index(drop=True)
    act1 = act_trans_select['points'][0]['x']
    act2 = act_trans_select['points'][0]['y']
    df[ACTIVITY + '2'] = df[ACTIVITY].shift(-1)
    idx = df[(df[ACTIVITY] == act1) & (df[ACTIVITY + '2'] == act2)].index.values
    df4 = df.iloc[[*idx, *(idx+1)]]
    df4 = df4[[START_TIME, END_TIME, ACTIVITY]]\
            .sort_values(by=START_TIME)\
            .reset_index(drop=True)
    return df4


def _initialize_toggle_callbacks(app):


    @app.callback(
        Output("clps-acts-n-devs", "is_open"),
        [Input("clps-acts-n-devs-button", "n_clicks")],
        [State("clps-acts-n-devs", "is_open")],
    )
    def toggle_collapse(n, is_open):
        return bool(n) ^ bool(is_open)

    @app.callback(
        Output("clps-dev-iei", "is_open"),
        [Input("clps-dev-iei-button", "n_clicks")],
        [State("clps-dev-iei", "is_open")],
    )
    def toggle_collapse_act_bp(n, is_open):
        return bool(n) ^ bool(is_open)

    @app.callback(
        Output("clps-dev-bar", "is_open"),
        [Input("clps-dev-bar-button", "n_clicks")],
        [State("clps-dev-bar", "is_open")],
    )
    def toggle_collapse_act_bp(n, is_open):
        return bool(n) ^ bool(is_open)

    @app.callback(
        Output("clps-dev-bp", "is_open"),
        [Input("clps-dev-bp-button", "n_clicks")],
        [State("clps-dev-bp", "is_open")],
    )
    def toggle_collapse_act_bp(n, is_open):
        return bool(n) ^ bool(is_open)

    @app.callback(
        Output("clps-dev-density", "is_open"),
        [Input("clps-dev-density-button", "n_clicks")],
        [State("clps-dev-density", "is_open")],
    )
    def toggle_collapse_act_bp(n, is_open):
        return bool(n) ^ bool(is_open)

    @app.callback(
        Output("clps-dev-cc", "is_open"),
        [Input("clps-dev-cc-button", "n_clicks")],
        [State("clps-dev-cc", "is_open")],
    )
    def toggle_collapse_act_bp(n, is_open):
        return bool(n) ^ bool(is_open)

def create_callbacks(app, dct_acts, df_devs, start_time, end_time, dev2area, plt_height_acts, plt_height_devs, plt_height_ands):

    def gen_trigger(id, val):
        return html.Div(id=id, style=dict(display="none"), **{"data-value": val})

    _initialize_toggle_callbacks(app)

    for key in dct_acts.keys():
        _initialize_activity_toggle_cbs(app, key)

    @app.callback(
        Output("and_clipboard", "content"),
        Input("graph-acts_n_devs", "clickData"),
        Input("and_btn_copy_range", "n_clicks"),
        State("and_dev-type", "value"),
        State("graph-acts_n_devs", "figure")
    )
    def element_to_clipboard(tmp, btn_cr, dev_type, fig_and):

        ctx = dash.callback_context
        trigger_val =  dash_get_trigger_value(ctx)
        trigger = dash_get_trigger_element(ctx)
        is_copy_range = (trigger == 'and_btn_copy_range')
        if trigger_val is None\
            or (tmp is None and not is_copy_range):
            raise PreventUpdate

        if not is_copy_range:
            dp = tmp['points'][0]
            is_activity = (dp['y'] == 'Activity')
            is_dev_state = not is_activity and dev_type == 'state'
            is_dev_event = not is_activity and dev_type == 'event'

        if is_copy_range:
            range = range_from_fig(fig_and)
            s = f'{ts2str(range[0])},{ts2str(range[1])}'
        elif is_activity:
            start_time = dp['base']
            end_time = dp['x']
            s = f'\'{start_time}\',\'{end_time}\',#activity'
        elif is_dev_state:
            start_time = dp['base']
            end_time = dp['customdata'][1]
            dev = dp['y']
            val = dp['customdata'][2]
            s = f'\'{start_time}\',\'{end_time}\',\'{dev}\',\'{val}\''
        elif is_dev_event:
            time = dp['x']
            dev = dp['y']
            val = dp['customdata']
            s = f'\'{time}\',\'{dev}\',\'{val}\''
        else:
            raise NotImplementedError
        return s

    act_outputs = []
    act_inputs = []
    act_states = []
    for subj_name  in dct_acts.keys():
        act_outputs.extend([
            Output(f'act-trigger-{subj_name}', 'children'),
            Output(f'act-curr-sel-{subj_name}', 'children'),
            Output(f'act-curr-sel-store-{subj_name}', 'data'),
            Output(f'avd-trigger-{subj_name}', 'children'),
        ])
        act_inputs.extend([
            Input(f'acts_graph-boxplot-{subj_name}', 'selectedData'),
            Input(f'acts_graph-transition-{subj_name}', 'clickData'),
            Input(f'acts_graph-bar-{subj_name}', 'clickData'),
            Input(f'avd_graph-event-contingency-{subj_name}', 'clickData'),
        ])
        act_states.extend([
            State(f'act-curr-sel-store-{subj_name}', 'data'), 
            State(f'act-curr-sel-{subj_name}', 'children'),
        ])

    @app.callback(
        output=[
            Output('graph-acts_n_devs', 'figure'),
            Output('and_act-order', 'data'),
            Output('and_dev-order', 'data'),
            Output('and_reset_sel', 'disabled'),
            Output('dev-trigger', 'children'),
            Output('dev-curr-sel', 'children'),
            Output('dev-curr-sel-store', 'data'),
            *act_outputs
        ],
        inputs=[
            # Input methods
            Input('range-slider', 'value'),
            Input('select-activities', 'value'),
            Input('select-devices', 'value'),
            Input('and_act-order-trigger', 'value'),
            Input('and_dev-order-trigger', 'value'),
            Input('and_dev-type', 'value'),

            # Selected devices
            Input('devs_graph-bar', 'clickData'),
            Input('devs_graph-boxplot', 'selectedData'),
            Input('devs_graph-iei', 'selectedData'),
            Input('devs_graph-fraction', 'clickData'),

            Input('devs_graph-density', 'clickData'),
            Input('and_reset_sel', 'n_clicks'),

            # Selected activities
            *act_inputs
        ],
        state=[
            State('devs_dens-slider', 'value'),
            State('devs_iei-scale', 'value'),
            State('devs_graph-iei', 'figure'),
            State('and_act-order', 'data'),
            State('and_dev-order', 'data'),
            State('dev-curr-sel', 'children'),
            State('dev-curr-sel-store', 'data'), 
            *act_states
        ]
    )
    def update_acts_n_devs(*args):

        nr_subjects = len(dct_acts.keys())

        idx = 12
        rng, sel_activities, sel_devices, act_order_trigger, dev_order_trigger, dev_type_trigger, \
        dev_bar_select, dev_boxplot_select, dev_iei_select, dev_fraction_select, \
        dev_density_select, reset_sel = args[:idx]    # [0:12] -> 1el, ..., 12 el

        for i, subj_name in enumerate(dct_acts.keys()):
            j = idx+i+i*nr_subjects
            k, l, m = j+1, j+2, j+3
            locals()[f'act_boxplot_select_{subj_name}'] = args[j]   # [13] -> 14 el  
            locals()[f'act_trans_select_{subj_name}'] = args[k]     # [14] -> 15 el
            locals()[f'act_bar_select_{subj_name}'] = args[l]       # [15] -> 16 el
            locals()[f'avd_event_select_{subj_name}'] = args[m]     # [16] -> 17 el

        idx = idx+4*nr_subjects # 16 - [17:25] -> 19el,...,25 el
        dev_density_dt, dev_iei_scale, dev_iei_fig, act_order, dev_order,  \
        dev_curr_sel, dev_curr_sel_store =  args[idx:idx+7]

        idx = idx+7
        for i, subj_name in enumerate(dct_acts.keys()):
            j, k = idx+i*nr_subjects, idx+i*nr_subjects+1
            locals()[f'act_curr_sel_store_{subj_name}'] = args[j]
            locals()[f'act_curr_sel_{subj_name}'] = args[k]


        ctx = dash.callback_context

        if dash_get_trigger_value(ctx) is None:
            raise PreventUpdate

        # Catch block if no act-order or dev-order is initialized
        try:
            act_order = json.loads(act_order)
        except TypeError:
            act_order = None
        try:
            dev_order = json.loads(dev_order)
        except TypeError:
            dev_order = None

        try:
            df_dev_curr_sel = pd.read_json(dev_curr_sel_store)
        except:
            df_dev_curr_sel = None

        origin_act_trans_selection = any([_is_trigger(ctx, f'acts_graph-transition-{sub}') for sub in dct_acts.subjects()])
        origin_act_bp_selection = any([_is_trigger(ctx, f'acts_graph-boxplot-{sub}') for sub in dct_acts.subjects()])
        origin_act_bar_selection = any([_is_trigger(ctx, f'acts_graph-bar-{sub}') for sub in dct_acts.subjects()])

        origin_and_event_selection = _is_trigger(ctx, 'avd_graph-event-contingency')
        origin_dev_bp_selection = _is_trigger(ctx, 'devs_graph-boxplot')
        origin_dev_iei_selection = _is_trigger(ctx, 'devs_graph-iei')
        origin_dev_density_selection = _is_trigger(ctx, 'devs_graph-density')
        origin_dev_bar_selection = _is_trigger(ctx, 'devs_graph-bar')
        origin_dev_fraction_selection = _is_trigger(ctx, 'devs_graph-fraction')

        data_selection_changed = _is_trigger(ctx, 'range-slider') or _is_trigger(ctx, 'select-devices') \
                                 or _is_trigger(ctx, 'select-activities')

        user_sel_activities = origin_act_trans_selection \
                              or origin_act_bp_selection \
                              or origin_and_event_selection \
                              or origin_act_bar_selection
        user_sel_devices = origin_and_event_selection \
                           or origin_dev_bp_selection \
                           or origin_dev_iei_selection \
                           or origin_dev_density_selection \
                           or origin_dev_bar_selection \
                           or origin_dev_fraction_selection

        reset_selection = _is_trigger(ctx, 'and_reset_sel') or data_selection_changed


        # Create start- and end-timestamps from range slider
        st = num_to_timestamp(rng[0], start_time=start_time, end_time=end_time)
        et = num_to_timestamp(rng[1], start_time=start_time, end_time=end_time)
        curr_df_devs, curr_dct_acts = select_timespan(df_acts=dct_acts, df_devs=df_devs,
                                                     start_time=st, end_time=et, clip_activities=True)



        # Filter selected devices and activities
        curr_df_devs = curr_df_devs[curr_df_devs[DEVICE].isin(sel_devices)]
        curr_dct_acts = deepcopy(curr_dct_acts)
        for key, df in dct_acts.items():
            curr_dct_acts[key] = df[df[ACTIVITY].isin(sel_activities)]

        # Filter for acts-n-devs plot
        df_devs1 = df_devs[df_devs[DEVICE].isin(sel_devices)]

        dct_acts1 = deepcopy(dct_acts)
        for key, df in dct_acts.items():
            dct_acts1[key] = df[df[ACTIVITY].isin(sel_activities)]

        try:
            df_act_curr_sel = ActivityDict.read_json(act_curr_sel_store)
        except:
            df_act_curr_sel = ActivityDict()

        devs_usel_state = None
        # Create activity and device order if needed
        if _is_trigger(ctx, 'and_dev-order-trigger') or dev_order is None:
            dev_order = device_order_by(curr_df_devs, dev_order_trigger, dev2area)
        if _is_trigger(ctx, 'and_act-order-trigger') or act_order is None:
            act_order = activity_order_by(curr_dct_acts, act_order_trigger)


        # Create dataframes with user selection
        for subj in dct_acts.subjects():
            if (_is_trigger(ctx, f'acts_graph-transition-{subj}') \
                or locals()[f'act_curr_sel_{subj}']  == f'acts_graph-transition-{subj}')\
                and not reset_selection:
                df_act_curr_sel[subj] = _sel_act_trans_sel(locals()[f'act_trans_select_{subj}'],
                                                    df_acts=curr_dct_acts[subj])
                locals()[f'act_curr_sel_{subj_name}'] = f'acts_graph-transition-{subj}'

            if (_is_trigger(ctx, f'acts_graph-boxplot-{subj}') \
                or locals()[f'act_curr_sel_{subj}'] == f'acts_graph-boxplot-{subj}')\
                and not reset_selection:
                df_act_curr_sel[subj] = _sel_point_to_activities(locals()[f'act_boxplot_select_{subj}'])
                locals()[f'act_curr_sel_{subj_name}'] = f'acts_graph-boxplot-{subj}'

            if (_is_trigger(ctx, f'acts_graph-bar-{subj}') \
                or locals()[f'act_curr_sel_{subj}'] == f'acts_graph-bar-{subj}')\
                and not reset_selection:
                df_act_curr_sel[subj] = _sel_act_bar(locals()[f'act_bar_select_{subj}'], df_acts=curr_dct_acts[subj])
                locals()[f'act_curr_sel_{subj_name}'] = f'acts_graph-bar-{subj}'

            if _is_trigger(ctx, f'avd_graph-event-contingency-{subj}')\
                and not reset_selection:
                df_dev_curr_sel, df_act_curr_sel[subj] = _sel_avd_event_click(
                    locals()[f'avd_event_select_{subj_name}'],
                    curr_dct_acts[subj],
                    curr_df_devs
                )
                locals()[f'act_curr_sel_{subj_name}'] = f'avd_graph-event-contingency-{subj}'
                dev_curr_sel = f'avd_graph-event-contingency-{subj}'
                devs_usel_state = False

        #if update_devices:
        if (_is_trigger(ctx, 'devs_graph-bar') or dev_curr_sel == 'devs_graph-bar')\
            and not reset_selection:
            df_dev_curr_sel = _sel_dev_bar(dev_bar_select, df_devs=curr_df_devs)
            dev_curr_sel = 'devs_graph-bar'
            devs_usel_state = False

        if (_is_trigger(ctx, 'devs_graph-fraction') or dev_curr_sel == 'devs_graph-fraction')\
            and not reset_selection:
            df_dev_curr_sel = _sel_dev_fraction(dev_fraction_select, df_devs=curr_df_devs)
            dev_curr_sel = 'devs_graph-fraction'
            devs_usel_state = True

        if (_is_trigger(ctx, 'devs_graph-boxplot') or dev_curr_sel == 'devs_graph-boxplot')\
            and not reset_selection:
            df_dev_curr_sel = _sel_dev_bp_selection(dev_boxplot_select, curr_df_devs)
            dev_curr_sel = 'devs_graph-boxplot'
            devs_usel_state = True

        if (_is_trigger(ctx, 'devs_graph-iei') or dev_curr_sel == 'devs_graph-iei')\
            and not reset_selection:
            df_dev_curr_sel = _sel_dev_iei(dev_iei_select, curr_df_devs,
                                           dev_iei_scale, dev_iei_fig)
            dev_curr_sel = 'devs_graph-iei'
            devs_usel_state = False

        if (_is_trigger(ctx, 'devs_graph-density') or dev_curr_sel == 'devs_graph-density')\
            and not reset_selection:
            df_dev_curr_sel = _sel_dev_density(dev_density_select, curr_df_devs, dev_density_dt)
            dev_curr_sel = 'devs_graph-density'
            devs_usel_state = False



        if reset_selection:
            df_act_curr_sel = None
            df_dev_curr_sel = None
            for subj in dct_acts.subjects():
                locals()[f'act_curr_sel_{subj_name}'] = ''
            dev_curr_sel = ''

        states = (dev_type_trigger == 'state')
        fig_and = activities_and_devices(df_devs1, dct_acts1, st=st, et=et,
                                         states=states, act_order=act_order, dev_order=dev_order,
                                         df_acts_usel=df_act_curr_sel, df_devs_usel=df_dev_curr_sel,
                                         devs_usel_state=devs_usel_state, height=plt_height_ands
                                         )

        # Determine reset selection button values
        if user_sel_activities or user_sel_devices:
            disable_reset = False
        else:
            disable_reset = True

        # Determine if and what updates for activities, device or a~d section are needed
        if data_selection_changed:
            act_update, avd_update, dev_update = 'new_data', 'new_data', 'new_data'
            disable_reset = True
        elif reset_selection:
            act_update, dev_update = "reset_sel", "reset_sel"
            avd_update = dash.no_update
        else:
            if origin_act_bp_selection:
                act_update = 'reset_sel_trans'
            elif origin_act_trans_selection:
                act_update = 'reset_sel_bp'
            elif origin_and_event_selection:
                act_update = 'reset_sel'
            else:
                act_update = dash.no_update

            if origin_dev_density_selection:
                dev_update = 'dens_clicked'
            elif origin_dev_iei_selection:
                dev_update = 'iei_selected'
            elif origin_dev_bp_selection:
                dev_update = 'bp_selected'
            else:
                dev_update = dash.no_update

            # TODO, refactor avd is never updates since it is not necessary
            avd_update = dash.no_update
        try:
            act_curr_sel_store = df_act_curr_sel.to_json(date_unit="ns")
        except:
            act_curr_sel_store = ''
        try:
            dev_curr_sel_store = df_dev_curr_sel.to_json(date_unit="ns"),
        except:
            dev_curr_sel_store = ''

        tmp = []
        for subj_name in dct_acts.subjects():
            tmp += [
                act_update, # TODO update both individually
                locals()[f'act_curr_sel_{subj_name}'],
                locals()[f'act_curr_sel_store_{subj_name}'],
                avd_update
            ]

        ret_arguments = (
               fig_and, act_order, dev_order, disable_reset, 
               dev_update, dev_curr_sel, dev_curr_sel_store,
               *tmp
        )
        return ret_arguments


    @app.callback(
       output=[
            Output('devs_graph-iei', 'figure'),
            Output('devs_graph-density', 'figure'),
            Output('devs_graph-bar', 'figure'),
            Output('devs_graph-fraction', 'figure'),
            Output('devs_graph-boxplot', 'figure'),
            Output('devs_graph-cc', 'figure'),
            Output('devs_order', 'data'),
        ],
        inputs=[
            Input('dev-trigger', 'children'),
            Input('range-slider', 'value'),
            Input('select-devices', 'value'),
            Input('tabs', 'active_tab'),
            Input('devs_bar-scale', 'value'),
            Input('devs_bar-order', 'value'),

            Input('devs_iei-scale', 'value'),
            Input('devs_iei-per-device', 'on'),
            Input('devs_bp-scale', 'value'),
            Input('devs_bp-binary-state', 'value'),
            Input('devs_dens-slider', 'value'),
            Input('devs_dens-scale', 'value'),

            Input('devs_cc-sel-fix', 'value'),
            Input('devs_cc-sel-to', 'value'),
            Input('devs_cc-lag-slider', 'value'),
            Input('devs_cc-binsize-slider', 'value'),
        ],
        state=[
            State('devs_graph-iei', 'figure'),
            State('devs_graph-density', 'figure'),
            State('devs_graph-bar', 'figure'),
            State('devs_graph-fraction', 'figure'),
            State('devs_graph-boxplot', 'figure'),
            State('devs_graph-cc', 'figure'),
            State('devs_order', 'data'),
        ]
    )
    def update_dev_tab(dev_trigger, rng, sel_devices,
                       active_tab, bar_scale, bar_order, iei_scale, iei_per_dev,
                       bp_scale, bp_binary_state, dens_slider, dens_scale, cc_fix, cc_to, cc_slider_lag, cc_slider_bin,
                       fig_iei, fig_dens, fig_left, fig_frac, fig_bp, fig_cc, dev_order, 
                       ):
        ctx = dash.callback_context
        if dash_get_trigger_value(ctx) is None or active_tab != 'tab-devs':
            raise PreventUpdate

        try:
            dev_order = json.loads(dev_order)
        except TypeError:
            dev_order = None

        # Filter selected timeframe, activities and devices
        st = num_to_timestamp(rng[0], start_time=start_time, end_time=end_time)
        et = num_to_timestamp(rng[1], start_time=start_time, end_time=end_time)
        curr_df_devs = select_timespan(df_devs=df_devs, start_time=st, end_time=et,
                                       clip_activities=True)
        curr_df_devs = curr_df_devs[curr_df_devs[DEVICE].isin(sel_devices)]
        from pyadlml.dataset.plot.plotly.devices import bar_count
        from pyadlml.dataset.plot.plotly.devices import fraction as dev_fraction

        is_trigger_bar_drop = _is_trigger(ctx, 'devs_bar-drop')
        is_trigger_bar_scale = _is_trigger(ctx, 'devs_bar-scale')
        is_trigger_bp_scale = _is_trigger(ctx, 'devs_bp-scale')
        is_trigger_bp_state = _is_trigger(ctx, 'devs_bp-binary-state')
        is_trigger_iei_per_device = _is_trigger(ctx, 'devs_iei-per-device')
        is_trigger_iei_scale = _is_trigger(ctx, 'devs_iei-scale')
        is_trigger_dens_scale = _is_trigger(ctx, 'devs_dens-scale')
        is_trigger_dens_slider = _is_trigger(ctx, 'devs_dens-slider')
        is_trigger_sel_dev = _is_trigger(ctx, 'select-devices')
        is_trigger_dev_update = _is_trigger(ctx, 'dev-trigger') or is_trigger_sel_dev

        is_trigger_cc_fix = _is_trigger(ctx, 'devs_cc-sel-fix')
        is_trigger_cc_to = _is_trigger(ctx, 'devs_cc-sel-to')
        is_trigger_cc_slider_lag = _is_trigger(ctx, 'devs_cc-lag-slider')
        is_trigger_cc_slider_bin = _is_trigger(ctx, 'devs_cc-binsize-slider')

        order_update = _is_trigger(ctx, 'devs_bar-order')
        new_data_signal = (is_trigger_dev_update and dev_trigger == 'new_data')

        update_bar = is_trigger_bar_drop or order_update or is_trigger_bar_scale \
                    or new_data_signal
        update_bp = is_trigger_bp_scale or order_update or is_trigger_bp_state \
                    or (is_trigger_dev_update and dev_trigger == 'dens_clicked') \
                    or (is_trigger_dev_update and dev_trigger == 'iei_selected') \
                    or (is_trigger_dev_update and dev_trigger == 'reset_sel') \
                    or new_data_signal
        update_iei = is_trigger_iei_scale or is_trigger_iei_per_device \
                    or new_data_signal
        update_frac = order_update or new_data_signal
        update_density = is_trigger_dens_slider or order_update \
                    or new_data_signal or is_trigger_dens_scale
        update_cc = is_trigger_cc_fix or is_trigger_cc_slider_lag or is_trigger_cc_to \
                    or is_trigger_cc_slider_bin or new_data_signal

        if order_update or dev_order is None or is_trigger_dev_update or is_trigger_sel_dev:
            dev_order = device_order_by(curr_df_devs, rule=bar_order)



        def f_update_frac(fig_frac, cd_dev, order, uf):
            return dev_fraction(cd_dev, order=order, height=plt_height_devs) if uf else fig_frac
        def f_update_bar(ub, fig_left, cdev, scale, order):
            return bar_count(cdev, scale=scale, order=order, height=plt_height_devs) if ub else fig_left
        def f_update_bp(u_bp, fig_bp, cd_dev, order, scale, binary_state):
            if u_bp:
                return boxplot_state(cd_dev, scale=scale, order=order,
                                     binary_state=binary_state, height=plt_height_devs)
            else:
                return fig_bp
        def f_update_density(ud, fig_dens, cd_dev, dens_slider, scale, order):
            if ud:
                return event_density(cd_dev, dt=DEV_DENS_SLIDER[dens_slider], show_colorbar=False,
                                     scale=scale, order=order, height=plt_height_devs)
            else:
                return fig_dens
        def f_update_iei(u_iei, fig_iei, cd_dev, scale, per_dev):
            return dev_iei(cd_dev, scale=scale, per_device=per_dev, height=plt_height_devs) if u_iei else fig_iei

        def f_update_cc(u_cc, fig_cc, fix, to, scale, binsize):
            if u_cc:
                return dev_cc(df_devs, height=plt_height_devs, fix=fix, to=to, max_lag=LAG_SLIDER[scale], binsize=BIN_SIZE_SLIDER[binsize])
            else:
                return fig_cc


        import dask
        figs = [
            dask.delayed(f_update_iei)(update_iei, fig_iei, curr_df_devs, iei_scale, iei_per_dev),
            dask.delayed(f_update_density)(update_density, fig_dens, curr_df_devs,
                                           dens_slider, dens_scale, dev_order),
            dask.delayed(f_update_bar)(update_bar, fig_left, curr_df_devs, bar_scale, dev_order),
            dask.delayed(f_update_frac)(fig_frac, curr_df_devs, dev_order, update_frac),
            dask.delayed(f_update_bp)(update_bp, fig_bp, curr_df_devs, dev_order, bp_scale,
                                      bp_binary_state),
            dask.delayed(f_update_cc)(update_cc, fig_cc, cc_fix, cc_to, cc_slider_lag, cc_slider_bin),
        ]
        figs = dask.compute(figs)[0]

        return figs[0], figs[1], figs[2], figs[3], figs[4], figs[5], json.dumps(list(dev_order))


    for subj_name in dct_acts.keys():
        _create_activity_tab_callback(app, subj_name, dct_acts[subj_name], start_time, end_time, plt_height_acts)
        _create_acts_vs_devs_tab_callback(app, subj_name, dct_acts[subj_name], df_devs, start_time, end_time, dev2area)


def _is_trigger(ctx, val):
    return ctx.triggered[0]['prop_id'].split('.')[0] == val


