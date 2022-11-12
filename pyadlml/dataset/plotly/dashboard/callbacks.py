import json
from dash.dependencies import Output, Input, State
import dash
from dash.exceptions import PreventUpdate
import pandas as pd
from pyadlml.constants import ACTIVITY, END_TIME, START_TIME
from pyadlml.dataset.plotly.activities import bar_count, bar_cum, boxplot_duration, density, heatmap_transitions
from pyadlml.dataset.stats.activities import activities_dist

from pyadlml.dataset.util import activity_order_by, num_to_timestamp, select_timespan

def bind_toggle_collapse(name):
    def toggle_collapse(n, is_open):
        return bool(n) ^ bool(is_open)

    toggle_collapse.__name__ = name
    return toggle_collapse

def _initialize_activity_toggle_cbs(app, act_id):

    app.callback(
        Output(f"clps-act-transition-{act_id}", "is_open"),
        [Input(f"clps-act-transition-button-{act_id}", "n_clicks")],
        [State(f"clps-act-transition-{act_id}", "is_open")],
    )(bind_toggle_collapse(f'toggle_collapse_act_trans_{act_id}'))

    app.callback(
        Output(f"clps-act-boxplot-{act_id}", "is_open"),
        [Input(f"clps-act-boxplot-button-{act_id}", "n_clicks")],
        [State(f"clps-act-boxplot-{act_id}", "is_open")],
    )(bind_toggle_collapse(f'toggle_collapse_act_bp_{act_id}'))


    app.callback(
        Output(f"clps-act-bar-{act_id}", "is_open"),
        [Input(f"clps-act-bar-button-{act_id}", "n_clicks")],
        [State(f"clps-act-bar-{act_id}", "is_open")],
    )(bind_toggle_collapse(f'toggle_collapse_act_bar_{act_id}'))


def _create_activity_tab_callback(app, act_id, df_acts, start_time, end_time, plt_height_acts):

    # TODO Circual imports hack
    from pyadlml.dataset.plotly.dashboard.dashboard import _get_trigger_value
    from pyadlml.dataset.plotly.dashboard.dashboard import _is_trigger

    @app.callback(
        output=[
            Output(f'acts_graph-bar-{act_id}', 'figure'),
            Output(f'acts_graph-boxplot-{act_id}', 'figure'),
            Output(f'acts_graph-density-{act_id}', 'figure'),
            Output(f'acts_graph-transition-{act_id}', 'figure'),
            Output(f'acts_activity-order-{act_id}', 'data'),
            Output(f'acts_density-data-{act_id}', 'data')
        ],
        inputs=[
            Input('tabs', 'active_tab'),
            Input(f'act-trigger-{act_id}', 'children'),
            Input('range-slider', 'value'),
            Input('select-activities', 'value'),
            Input(f'acts_bp-drop-{act_id}', 'value'),
            Input(f'acts_bar-drop-{act_id}', 'value'),
            Input(f'acts_bp-scale-{act_id}', 'value'),
            Input(f'acts_bar-scale-{act_id}', 'value'),
            Input(f'acts_trans-scale-{act_id}', 'value'),
            Input(f'acts_sort-{act_id}', 'value'),
        ],
        state=[
            State(f'acts_graph-bar-{act_id}', 'figure'),
            State(f'acts_graph-boxplot-{act_id}', 'figure'),
            State(f'acts_graph-density-{act_id}', 'figure'),
            State(f'acts_graph-transition-{act_id}', 'figure'),
            State(f'acts_activity-order-{act_id}', 'data'),
            State(f'acts_density-data-{act_id}', 'data'),
        ]

    )
    def update_activity_tab(active_tab, act_trigger, rng, sel_activities,
                               drop_box: str, drop_bar: str, scale_boxplot: str,
                               scale_bar: str, scale_trans: str, act_order_trigger,
                               fig_bar, fig_bp, fig_dens, fig_trans, act_order, act_density,
        ):

        ctx = dash.callback_context
        if _get_trigger_value(ctx) is None or active_tab != f'tab-acts-{act_id}':
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
        curr_df_acts = select_timespan(df_acts=df_acts, start_time=st, end_time=et,
                                       clip_activities=True)
        curr_df_acts = curr_df_acts[curr_df_acts[ACTIVITY].isin(sel_activities)]
        # TODO refactor, where is the y_label column coming from
        curr_df_acts = curr_df_acts[[START_TIME, END_TIME, ACTIVITY]]

        # Get update type
        is_trigger_bar_drop = _is_trigger(ctx, f'acts_bar-drop-{act_id}')
        is_trigger_bar_scale = _is_trigger(ctx, f'acts_bar-scale-{act_id}')
        is_trigger_sort = _is_trigger(ctx, f'acts_sort-{act_id}')
        is_trigger_range = _is_trigger(ctx, 'range-slider')
        is_trigger_bp_drop = _is_trigger(ctx, f'acts_bp-drop-{act_id}')
        is_trigger_bp_scale = _is_trigger(ctx, f'acts_bp-scale-{act_id}')
        is_trigger_trans_scale = _is_trigger(ctx, f'acts_trans-scale-{act_id}')

        data_update = is_trigger_range or _is_trigger(ctx, 'select-activities')
        # Determine the intent, when the trigger was the avd plot
        signal_reset_bp = _is_trigger(ctx, f'act-trigger-{act_id}') and act_trigger == 'reset_sel_bp'
        signal_reset_trans = _is_trigger(ctx, f'act-trigger-{act_id}') and act_trigger == 'reset_sel_trans'
        signal_reset_all = _is_trigger(ctx, f'act-trigger-{act_id}') and act_trigger == 'reset_sel'

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
