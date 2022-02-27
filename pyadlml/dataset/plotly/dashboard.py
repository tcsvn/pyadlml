import dash
import dash.dcc as dcc
import dash.html as html
import pandas as pd
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from pyadlml.dataset.plotly.devices import event_density
from pyadlml.dataset.plotly.layout import DEV_DENS_SLIDER

from pyadlml.dataset._core._dataset import label_data
from pyadlml.dataset._core.activities import _create_activity_df
from pyadlml.dataset._datasets.aras import _create_device_df
from pyadlml.dataset.plotly.activities import *
from pyadlml.dataset.plotly.acts_and_devs import *
from pyadlml.dataset.plotly.devices import bar_count as dev_bar_count, \
    device_iei as dev_iei, boxplot_state
from dash.dependencies import *

from pyadlml.dataset import fetch_amsterdam, set_data_home, TIME, START_TIME, END_TIME, ACTIVITY
from pyadlml.dataset.plotly.layout import acts_vs_devs_layout, devices_layout, \
    activities_layout, acts_n_devs_layout, device_layout_graph_bottom
from pyadlml.dataset.util import select_timespan, timestamp_to_num, num_to_timestamp

from dask.delayed import delayed

def dashboard(app, name, df_acts, df_devs, start_time, end_time):

    # Since performance rendering issues arise with numbers greater than 40000
    # device datapoints make a preselection
    max_points = 40000
    if len(df_devs) > max_points:
        set_et = df_devs.iat[max_points, df_devs.columns.get_loc(TIME)]
        curr_df_devs, curr_df_acts = select_timespan(
            df_activities=df_acts, df_devices=df_devs, start_time=start_time,
            end_time=set_et, clip_activities=True
        )
    else:
        curr_df_devs = df_devs
        curr_df_acts = df_acts
        set_et = end_time

    nr_devs = len(df_devs[DEVICE].unique())
    nr_acts = len(df_acts[ACTIVITY].unique())
    print('acts: ', nr_acts)
    print('devs: ', nr_devs)

    # Determine the plot height and fontsize for activity plots
    if nr_acts < 20:
        plot_height_acts = 350
    elif nr_acts < 25:
        plot_height_acts = 450
    else:
        plot_height_acts = 350

    if nr_devs < 20:
        plot_height_devs = 350
    elif nr_devs < 30:
        plot_height_devs = 400
    elif nr_devs < 50:
        plot_height_devs = 500
    elif nr_devs < 70:
        plot_height_devs = 700
    else:
        plot_height_devs = 800

    # Get Layout
    layout_activities = activities_layout(curr_df_acts, plot_height_acts)
    layout_devices = devices_layout(curr_df_devs, True, plot_height_devs)
    layout_acts_vs_devs = acts_vs_devs_layout(curr_df_acts, curr_df_devs,
                                              True, plot_height_acts)

    layout = dbc.Container(
        children=[
            html.H1(children=f'Dashboard: {name}'),
            acts_n_devs_layout(df_acts, df_devs, start_time, end_time,
                               set_et, plot_height_devs+50),
            html.Br(),
        html.Div([
            dbc.Tabs(
                [dbc.Tab(layout_activities, label='Activities', tab_id='tab-acts'),
                 dbc.Tab(layout_devices, label='Devices', tab_id='tab-devs'),
                 dbc.Tab(layout_acts_vs_devs, label='Activities ~ Devices', tab_id='tab-acts_vs_devs'),
                 ], id='tabs', active_tab='tab-acts',
            ),
            html.Div(id="content"),
        ]
        ),
        # Store intermediate values of hard to compute values
        # dcc.Store(id='current-df-acts'),
    ], style={'width': 1000, 'margin': 'auto'})
    create_callbacks(app, df_acts, df_devs, start_time, end_time, plot_height_acts, plot_height_devs)
    app.layout = layout

def _sel_avd_event_click(avd_event_select, curr_df_acts, curr_df_devs):
    """Select devices and activities from click data of the event contingency table"""
    # Select devices
    a = avd_event_select['points'][0]['y']
    d = avd_event_select['points'][0]['x']
    df_tmp = curr_df_devs[curr_df_devs[DEVICE] == d]
    df_tmp = label_data(df_tmp, curr_df_acts)
    df_tmp = df_tmp[df_tmp[ACTIVITY] == a]
    sel_devices = df_tmp[[TIME, DEVICE, VAL]]

    # Select activities
    sel_activities = _create_activity_df()
    for _, sel_dev in sel_devices.iterrows():
        mask = (curr_df_acts[START_TIME] < sel_dev[TIME]) \
               & (sel_dev[TIME] < curr_df_acts[END_TIME])
        tmp = curr_df_acts[mask]
        sel_activities = pd.concat([sel_activities, tmp])
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
    mask = (df_devs[DEVICE] == dev) & (df_devs[VAL] == cat)
    return df_devs[mask].copy()

def _sel_dev_bp_selection(dev_boxplot_select, df_devs):
    """ Gets point data from the state boxplot and creates
        an device-dataframe
    """
    points = dev_boxplot_select['points']
    sd = pd.DataFrame(columns=[TIME, END_TIME, DEVICE, VAL])
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
    df_res = pd.DataFrame(columns=[TIME, DEVICE, VAL])
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

    from pyadlml.dataset.plotly.layout import DEV_DENS_SLIDER
    df = df_devs.copy().set_index(TIME, drop=False)
    dt = pd.Timedelta(DEV_DENS_SLIDER[dt])
    df_res = pd.DataFrame(columns=[TIME, DEVICE, VAL])

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
        Output("clps-act-transition", "is_open"),
        [Input("clps-act-transition-button", "n_clicks")],
        [State("clps-act-transition", "is_open")],
    )
    def toggle_collapse(n, is_open):
        return bool(n) ^ bool(is_open)

    @app.callback(
        Output("clps-avd-state", "is_open"),
        [Input("clps-avd-state-button", "n_clicks")],
        [State("clps-avd-state", "is_open")],
    )
    def toggle_collapse(n, is_open):
        return bool(n) ^ bool(is_open)

    @app.callback(
        Output("clps-avd-event", "is_open"),
        [Input("clps-avd-event-button", "n_clicks")],
        [State("clps-avd-event", "is_open")],
    )
    def toggle_collapse(n, is_open):
        return bool(n) ^ bool(is_open)

    @app.callback(
        Output("clps-acts-n-devs", "is_open"),
        [Input("clps-acts-n-devs-button", "n_clicks")],
        [State("clps-acts-n-devs", "is_open")],
    )
    def toggle_collapse(n, is_open):
        return bool(n) ^ bool(is_open)

    @app.callback(
        Output("clps-act-boxplot", "is_open"),
        [Input("clps-act-boxplot-button", "n_clicks")],
        [State("clps-act-boxplot", "is_open")],
    )
    def toggle_collapse_act_bp(n, is_open):
        return bool(n) ^ bool(is_open)

    @app.callback(
        Output("clps-act-bar", "is_open"),
        [Input("clps-act-bar-button", "n_clicks")],
        [State("clps-act-bar", "is_open")],
    )
    def toggle_collapse_act_bp(n, is_open):
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


def create_callbacks(app, df_acts, df_devs, start_time, end_time, plt_height_acts, plt_height_devs):

    def gen_trigger(id, val):
        return html.Div(id=id, style=dict(display="none"), **{"data-value": val})

    _initialize_toggle_callbacks(app)


    @app.callback(
        Output('graph-acts_n_devs', 'figure'),
        Output('and_act-order', 'data'),
        Output('and_dev-order', 'data'),
        Output('and_reset_sel', 'disabled'),
        Output('act-trigger', 'children'),
        Output('avd-trigger', 'children'),
        Output('dev-trigger', 'children'),
        Output('act-curr-sel', 'children'),
        Output('dev-curr-sel', 'children'),
        Output('act-curr-sel-store', 'data'),
        Output('dev-curr-sel-store', 'data'),

        # Input methods
        Input('range-slider', 'value'),
        Input('select-activities', 'value'),
        Input('select-devices', 'value'),
        Input('and_act-order-trigger', 'value'),
        Input('and_dev-order-trigger', 'value'),
        Input('and_dev-type', 'value'),

        State('and_act-order', 'data'),
        State('and_dev-order', 'data'),
        State('act-curr-sel', 'children'),
        State('dev-curr-sel', 'children'),
        State('act-curr-sel-store', 'data'),
        State('dev-curr-sel-store', 'data'),

        # Selected activities
        Input('graph-boxplot', 'selectedData'),
        Input('graph-transition', 'clickData'),
        Input('graph-bar', 'clickData'),

        # Selected devices
        Input('devs_graph-bar', 'clickData'),
        Input('devs_graph-boxplot', 'selectedData'),
        Input('devs_graph-iei', 'selectedData'),
        Input('devs_graph-fraction', 'clickData'),
        State('devs_iei-scale', 'value'),
        State('devs_graph-iei', 'figure'),
        Input('devs_graph-density', 'clickData'),
        State('devs_dens-slider', 'value'),
        Input('avd_graph-event-contingency', 'clickData'),

        Input('and_reset_sel', 'n_clicks'),


    )
    def update_acts_n_devs(rng, sel_activities, sel_devices,
                           act_order_trigger, dev_order_trigger, dev_type_trigger,
                           act_order, dev_order, act_curr_sel, dev_curr_sel,
                           act_curr_sel_store, dev_curr_sel_store,
                           act_boxplot_select,
                           act_trans_select,
                           act_bar_select,
                           dev_bar_select,
                           dev_boxplot_select,
                           dev_iei_select,
                           dev_fraction_select,
                           dev_iei_scale,
                           dev_iei_fig,
                           dev_density_select,
                           dev_density_dt,
                           avd_event_select,
                           reset_sel,
                           ):
        ctx = dash.callback_context

        if _get_trigger_value(ctx) is None:
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
            df_act_curr_sel = pd.read_json(act_curr_sel_store)
        except:
            df_act_curr_sel = None
        try:
            df_dev_curr_sel = pd.read_json(dev_curr_sel_store)
        except:
            df_dev_curr_sel = None

        origin_act_trans_selection = _is_trigger(ctx, 'graph-transition')
        origin_act_bp_selection = _is_trigger(ctx, 'graph-boxplot')
        origin_act_bar_selection = _is_trigger(ctx, 'graph-bar')
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
        curr_df_devs, curr_df_acts = select_timespan(df_activities=df_acts, df_devices=df_devs,
                                                     start_time=st, end_time=et, clip_activities=True)

        # Filter selected devices and activities
        curr_df_devs = curr_df_devs[curr_df_devs[DEVICE].isin(sel_devices)]
        curr_df_acts = curr_df_acts[curr_df_acts[ACTIVITY].isin(sel_activities)]
        # Filter for acts-n-devs plot
        df_devs1 = df_devs[df_devs[DEVICE].isin(sel_devices)]
        df_acts1 = df_acts[df_acts[ACTIVITY].isin(sel_activities)]

        devs_usel_state = None
        # Create activity and device order if needed
        if _is_trigger(ctx, 'and_dev-order-trigger') or dev_order is None:
            dev_order = device_order_by(curr_df_devs, dev_order_trigger)
        if _is_trigger(ctx, 'and_act-order-trigger') or act_order is None:
            act_order = activity_order_by(curr_df_acts, act_order_trigger)


        # Create dataframes with user selection
        if (_is_trigger(ctx, 'graph-transition') or act_curr_sel == 'graph-transition')\
            and not reset_selection:
            df_act_curr_sel = _sel_act_trans_sel(act_trans_select,
                                                        df_acts=curr_df_acts)
            act_curr_sel = 'graph-transition'
        if (_is_trigger(ctx, 'graph-boxplot') or act_curr_sel == 'graph-boxplot')\
            and not reset_selection:
            df_act_curr_sel = _sel_point_to_activities(act_boxplot_select)
            act_curr_sel = 'graph-boxplot'

        if (_is_trigger(ctx, 'graph-bar') or act_curr_sel == 'graph-bar')\
            and not reset_selection:
            df_act_curr_sel = _sel_act_bar(act_bar_select, df_acts=curr_df_acts)
            act_curr_sel = 'graph-bar'

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

        if _is_trigger(ctx, 'avd_graph-event-contingency') and not reset_selection:
            df_dev_curr_sel, df_act_curr_sel = _sel_avd_event_click(avd_event_select,
                                                               curr_df_acts,
                                                               curr_df_devs)
            act_curr_sel = 'avd_graph-event-contingency'
            dev_curr_sel = 'avd_graph-event-contingency'
            devs_usel_state = False

        if reset_selection:
            df_act_curr_sel = None
            df_dev_curr_sel = None
            act_curr_sel = ''
            dev_curr_sel = ''

        states = (dev_type_trigger == 'state')
        fig_and = activities_and_devices(df_acts1, df_devs1, st=st, et=et,
                                         states=states, act_order=act_order, dev_order=dev_order,
                                         df_acts_usel=df_act_curr_sel, df_devs_usel=df_dev_curr_sel,
                                         devs_usel_state=devs_usel_state, height=plt_height_devs+50
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

        return fig_and, act_order, dev_order, disable_reset, \
               act_update, avd_update, dev_update, act_curr_sel, dev_curr_sel,\
               act_curr_sel_store, dev_curr_sel_store


    @app.callback(
       [Output('devs_graph-iei', 'figure'),
        Output('devs_graph-density', 'figure'),
        Output('devs_graph-bar', 'figure'),
        Output('devs_graph-fraction', 'figure'),
        Output('devs_graph-boxplot', 'figure'),
        Output('devs_order', 'data'),
        ],

        Input('dev-trigger', 'children'),
        State('devs_graph-iei', 'figure'),
        State('devs_graph-density', 'figure'),
        State('devs_graph-bar', 'figure'),
        State('devs_graph-fraction', 'figure'),
        State('devs_graph-boxplot', 'figure'),
        State('devs_order', 'data'),

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

    )
    def update_dev_tab(dev_trigger, fig_iei, fig_dens, fig_left, fig_frac, fig_bp, dev_order, rng, sel_devices,
                       active_tab, bar_scale, bar_order, iei_scale, iei_per_dev,
                       bp_scale, bp_binary_state, dens_slider, dens_scale
                       ):
        ctx = dash.callback_context
        if _get_trigger_value(ctx) is None or active_tab != 'tab-devs':
            raise PreventUpdate

        try:
            dev_order = json.loads(dev_order)
        except TypeError:
            dev_order = None

        # Filter selected timeframe, activities and devices
        st = num_to_timestamp(rng[0], start_time=start_time, end_time=end_time)
        et = num_to_timestamp(rng[1], start_time=start_time, end_time=end_time)
        curr_df_devs = select_timespan(df_devices=df_devs, start_time=st, end_time=et,
                                       clip_activities=True)
        curr_df_devs = curr_df_devs[curr_df_devs[DEVICE].isin(sel_devices)]
        from pyadlml.dataset.plotly.devices import fraction as dev_fraction,\
                                       bar_count

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

        import dask
        figs = [
            dask.delayed(f_update_iei)(update_iei, fig_iei, curr_df_devs, iei_scale, iei_per_dev),
            dask.delayed(f_update_density)(update_density, fig_dens, curr_df_devs,
                                           dens_slider, dens_scale, dev_order),
            dask.delayed(f_update_bar)(update_bar, fig_left, curr_df_devs, bar_scale, dev_order),
            dask.delayed(f_update_frac)(fig_frac, curr_df_devs, dev_order, update_frac),
            dask.delayed(f_update_bp)(update_bp, fig_bp, curr_df_devs, dev_order, bp_scale,
                                      bp_binary_state),
        ]
        figs = dask.compute(figs)[0]

        return figs[0], figs[1], figs[2], figs[3], figs[4], json.dumps(list(dev_order))


    @app.callback(
        [Output('graph-bar', 'figure'),
         Output('graph-boxplot', 'figure'),
         Output('graph-density', 'figure'),
         Output('graph-transition', 'figure'),
         Output('acts_activity-order', 'data'),
         Output('acts_density-data', 'data')
        ],
        Input('tabs', 'active_tab'),
        Input("act-trigger", 'children'),
        Input('range-slider', 'value'),
        Input('select-activities', 'value'),

        State('graph-bar', 'figure'),
        State('graph-boxplot', 'figure'),
        State('graph-density', 'figure'),
        State('graph-transition', 'figure'),
        State('acts_activity-order', 'data'),
        State('acts_density-data', 'data'),

        Input('acts_bp-drop', 'value'),
        Input('acts_bar-drop', 'value'),
        Input('acts_bp-scale', 'value'),
        Input('acts_bar-scale', 'value'),
        Input('acts_trans-scale', 'value'),
        Input('acts_sort', 'value'),

    )
    def update_activity_tab(active_tab, act_trigger, rng, sel_activities,
                               fig_bar, fig_bp, fig_dens, fig_trans, act_order, act_density,
                               drop_box: str, drop_bar: str, scale_boxplot: str,
                               scale_bar: str, scale_trans: str, act_order_trigger,
        ):

        ctx = dash.callback_context
        if _get_trigger_value(ctx) is None or active_tab != 'tab-acts':
            raise PreventUpdate

        try:
            act_order = json.loads(act_order)
        except TypeError:
            # happens if no act-order is initialized
            act_order = None
        try:
            act_density = pd.read_json(act_density)
        except:
            act_density = None

        # Filter selected timeframe, activities and devices
        st = num_to_timestamp(rng[0], start_time=start_time, end_time=end_time)
        et = num_to_timestamp(rng[1], start_time=start_time, end_time=end_time)
        curr_df_acts = select_timespan(df_activities=df_acts, start_time=st, end_time=et,
                                       clip_activities=True)
        curr_df_acts = curr_df_acts[curr_df_acts[ACTIVITY].isin(sel_activities)]
        # TODO refactor, where is the y_label column coming from
        curr_df_acts = curr_df_acts[[START_TIME, END_TIME, ACTIVITY]]

        # Get update type
        is_trigger_bar_drop = _is_trigger(ctx, 'acts_bar-drop')
        is_trigger_bar_scale = _is_trigger(ctx, 'acts_bar-scale')
        is_trigger_sort = _is_trigger(ctx, 'acts_sort')
        is_trigger_range = _is_trigger(ctx, 'range-slider')
        is_trigger_bp_drop = _is_trigger(ctx, 'acts_bp-drop')
        is_trigger_bp_scale = _is_trigger(ctx, 'acts_bp-scale')
        is_trigger_trans_scale = _is_trigger(ctx, 'acts_trans-scale')

        data_update = is_trigger_range or _is_trigger(ctx, 'select-activities')
        # Determine the intent, when the trigger was the avd plot
        signal_reset_bp = _is_trigger(ctx, 'act-trigger') and act_trigger == 'reset_sel_bp'
        signal_reset_trans = _is_trigger(ctx, 'act-trigger') and act_trigger == 'reset_sel_trans'
        signal_reset_all = _is_trigger(ctx, 'act-trigger') and act_trigger == 'reset_sel'

        order_update = ((is_trigger_sort or (is_trigger_bar_drop and act_order_trigger == 'value')) \
                       and not signal_reset_all) \
                       or act_order is None

        bp_update = is_trigger_bp_drop or is_trigger_bp_scale \
                    or data_update or order_update or signal_reset_all\
                    or signal_reset_bp
        bar_update = is_trigger_bar_drop or order_update or data_update \
                     or is_trigger_bar_scale
        trans_update = data_update or order_update or is_trigger_trans_scale \
                       or signal_reset_all

        # If the activity-order is changed or the bar plot is changed
        # and would change the order
        if order_update:
            if act_order_trigger == 'value' or is_trigger_bar_drop:
                act_order_trigger = 'duration' if drop_bar == 'cum' else 'count'
            act_order = activity_order_by(curr_df_acts, act_order_trigger)

        # Update activity bars
        if bar_update:
            if drop_bar == 'count':
                fig_bar = bar_count(curr_df_acts, order=act_order, scale=scale_bar,
                                    height=plt_height_acts)
            else:
                fig_bar = bar_cum(curr_df_acts, order=act_order, scale=scale_bar,
                                  height=plt_height_acts)

        # Update log for boxplot
        if bp_update:
            #if _get_trigger_value(ctx) == 'vp':
            #    fig_bp = violin_duration(curr_df_acts, order=act_order, scale=scale_boxplot)
            #else:
            #    fig_bp = boxplot_duration(curr_df_acts, order=act_order, scale=scale_boxplot)
            fig_bp = boxplot_duration(curr_df_acts, order=act_order, scale=scale_boxplot,
                                      height=plt_height_acts)

        # Only update the act_density matrix if it is
        if data_update or act_density is None:
            act_density = activities_dist(curr_df_acts.copy(), n=1000, dt=None)
        if order_update or data_update:
            fig_dens = density(df_density=act_density, order=act_order, height=plt_height_acts)

        if trans_update:
            fig_trans = heatmap_transitions(curr_df_acts, order=act_order, scale=scale_trans,
                                            height=plt_height_acts)

        return fig_bar, fig_bp, fig_dens, fig_trans, json.dumps(list(act_order)), act_density.to_json()

    @app.callback(
        [
        Output('avd_graph-event-contingency', 'figure'),
        Output('avd_graph-state-contingency', 'figure'),
        Output('loading-output', 'children'),
        Output('avd-update', 'children'),
        Output('avd_state-contingency', 'data'),
        Output('avd_event-contingency', 'data'),
        Output('avd_activity-order', 'data'),
        Output('avd_device-order', 'data'),
        ],

        Input('tabs', 'active_tab'),
        Input('avd-trigger', 'children'),
        Input('avd-update', 'children'),
        State('range-slider', 'value'),
        State('select-activities', 'value'),
        State('select-devices', 'value'),
        State('avd_state-contingency', 'data'),
        Input('avd_state-scale', 'value'),
        State('avd_event-contingency', 'data'),
        Input('avd_event-scale', 'value'),
        Input('avd_act-order-trigger', 'value'),
        Input('avd_dev-order-trigger', 'value'),
        State('avd_activity-order', 'data'),
        State('avd_device-order', 'data'),
    )
    def update_acts_vs_devs_tab(active_tab, trigger, update, rng, sel_activities, sel_devices,
                                state_cont, state_scale, event_cont, event_scale,
                                act_order_trigger, dev_order_trigger, act_order, dev_order
        ):
        ctx = dash.callback_context

        if _get_trigger_value(ctx) is None or active_tab != 'tab-acts_vs_devs':
            raise PreventUpdate

        try:
            df_con_states = pd.read_json(state_cont)
            df_con_states = df_con_states.astype('timedelta64[ns]')
            df_con_events = pd.read_json(event_cont)
        except ValueError:
            df_con_states = None
            df_con_events = None
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

        try:
            act_order = json.loads(act_order)
            dev_order = json.loads(dev_order)
        except TypeError:
            act_order = None
            dev_order = None


        # Filter selected timeframe, activities and devices
        st = num_to_timestamp(rng[0], start_time=start_time, end_time=end_time)
        et = num_to_timestamp(rng[1], start_time=start_time, end_time=end_time)
        curr_df_devs, curr_df_acts = select_timespan(df_activities=df_acts, df_devices=df_devs,
                                                     start_time=st, end_time=et, clip_activities=True)
        curr_df_acts = curr_df_acts[curr_df_acts[ACTIVITY].isin(sel_activities)]
        curr_df_devs = curr_df_devs[curr_df_devs[DEVICE].isin(sel_devices)]
        # TODO refactor, where is the y_label column coming from
        curr_df_acts = curr_df_acts[[START_TIME, END_TIME, ACTIVITY]]

        # If the activity-order is changed or the bar plot is changed
        # and would change the order
        is_data_update = _is_trigger(ctx, 'new_data')
        is_trigger_act_order = _is_trigger(ctx, 'avd_act-order-trigger')
        is_trigger_dev_order = _is_trigger(ctx, 'avd_dev-order-trigger')
        order_update = is_trigger_act_order or is_trigger_dev_order\
                       or act_order is None or dev_order is None

        if order_update:
            act_order = activity_order_by(curr_df_acts, act_order_trigger)
            dev_order = device_order_by(curr_df_devs, dev_order_trigger)


        # If data has changed the contingency tables have to be recomputed
        if is_data_update or df_con_states is None or df_con_events is None:
            # Recompute state contingency
            df_con_states = contingency_table_states(curr_df_devs, curr_df_acts,
                                                     distributed=True)
            df_con_events = contingency_table_events(curr_df_devs, curr_df_acts)
            update_data = True
        else:
            update_data = False

        if active_tab != 'tab-acts_vs_devs' and is_data_update:
            return dash.no_update, dash.no_update, dash.no_update, 'updated', \
                   df_con_states.to_json(), df_con_events.to_json(), \
                   json.dumps(list(act_order)), json.dumps(list(dev_order))
        elif active_tab != 'tab-acts_vs_devs':
            raise PreventUpdate

        # Create figures
        fig_adec = contingency_events(
                    con_tab=df_con_events, scale=event_scale,
                    act_order=act_order, dev_order=dev_order
        )
        fig_adsc = contingency_states(
                    df_devs=curr_df_devs,   # for event order
                    con_tab=df_con_states, scale=state_scale,
                    act_order=act_order, dev_order=dev_order
        )

        if update_data:
            state_dump, event_dump = df_con_states.to_json(date_unit="ns"), df_con_events.to_json()
        else:
            state_dump, event_dump = dash.no_update, dash.no_update

        return fig_adec, fig_adsc, None, 'needs update', state_dump, event_dump, \
               json.dumps(list(act_order)), json.dumps(list(dev_order))


def _is_trigger(ctx, val):
    return ctx.triggered[0]['prop_id'].split('.')[0] == val


def _get_trigger_value(ctx):
    return ctx.triggered[0]['value']
