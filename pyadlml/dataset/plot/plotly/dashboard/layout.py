import dash
import dash.dcc as dcc
import dash_daq as daq
import dash.html as html
import pandas as pd
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from pyadlml.dataset.plotly.activities import bar_count as act_bar_count, \
    boxplot_duration as act_boxplot_duration, density as act_density, \
    heatmap_transitions as act_heatmap_transitions
from pyadlml.dataset.plotly.devices import bar_count as dev_bar_count, \
    device_iei as dev_iei, fraction as dev_fraction, event_density, boxplot_state
from pyadlml.dataset.plotly.acts_and_devs import *
from dash.dependencies import *
from plotly.graph_objects import Figure

from pyadlml.dataset import fetch_amsterdam, set_data_home
from pyadlml.constants import TIME, START_TIME, END_TIME, ACTIVITY, DEVICE
from pyadlml.dataset.plotly.util import ActivityDict
from pyadlml.dataset.util import select_timespan, timestamp_to_num, num_to_timestamp

DEV_DENS_SLIDER = {
    0: '10s',
    1: '30s',
    2: '1min',
    3: '5min',
    4: '30min',
    5: '1h',
    6: '2h',
    9: '6h',
    10: '12h'
}
def _np_dt_strftime(ts, format):
    return pd.to_datetime(ts).strftime(format)


def _build_log_radio_buttons(id):
    return dcc.RadioItems(
        id=id,
        options=[{'label': i, 'value': i} for i in ['linear', 'log']],
        value='linear',
        labelStyle={'display': 'inline-block', 'marginTop': '5px'}
    )


def _build_row_radio_button(rd_id, options, value):
    """ Builds an inline radio button menu """
    return dbc.RadioItems(id=rd_id,
               inline=True,
               options=[{'label': i, 'value': i} for i in options],
               value=value,
               #labelStyle={'display': 'inline-block', 'marginTop': '5px'}
    )

def _build_sort_radio_buttons(id):
    return dcc.RadioItems(
        id=id,
        options=[{'label': i, 'value': i} for i in ['alphabetical', 'value', 'area']],
        value='value',
        labelStyle={'display': 'inline-block', 'marginTop': '5px'}
    )


def _build_checklist(df_acts):
    def create_option(name):
        return {'label': name, 'value': name}
    df_acts =  ActivityDict.wrap(df_acts)
    acts = df_acts.get_activity_union()
    options = [create_option(a) for a in acts]
    options = [create_option('other'), *options]
    return dcc.Dropdown(id='select-activities',
                        multi=True,
                        options=options,
                        value=acts,
                        clearable=False,
                        )


def _build_choice_devices(df_devs):
    def create_option(name):
        return {'label': name, 'value': name}

    devs = df_devs[DEVICE].unique()
    options = [create_option(dev) for dev in devs]
    return dcc.Dropdown(id='select-devices',
                        options=options,
                        multi=True,
                        value=devs,
                        clearable=False,
                        )


def _build_range_slider(df_acts, df_devs, start_time, end_time, set_end_time):
    def create_mark(ts, format):
        return {'label': _np_dt_strftime(ts, format), 'style': {"transform": "rotate(45deg)"}}

    marks = {0: create_mark(start_time, '%Y.%m.%d'),
             1: create_mark(end_time, '%Y.%m.%d')}
    #marks[0]['style']['margin'] = '10px 0 0 0'      # top right bottom left
    #marks[1]['style']['margin'] = '10px 15px 0 0'    # top right bottom left


    # Determine the marker frequency
    diff = end_time - start_time
    if diff < pd.Timedelta('8W'):       # 2 Month -> daily frequency
        # Dataset Amsterdam, Mitlab_1/2W, Mitlab_2/2W, Aras/4W, UCIA/2W
        # UCIA/3W
        rng = pd.date_range(start_time, end_time, freq='D')[1:-1]
    elif diff < pd.Timedelta('16W'):    # 4 Months
        # Dataset Tuebingen2019
        rng = pd.date_range(start_time, end_time, freq='3D')[1:-1]
    elif diff < pd.Timedelta('32W'):    # 8 Months
        # Dataset Casas Aruba
        rng = pd.date_range(start_time, end_time, freq='W')[1:-1]
    else:
        rng = pd.date_range(start_time, end_time, freq='2W')[1:-1]

    for day in rng:
        marks[timestamp_to_num(day, start_time, end_time)] = create_mark(day, '%m.%d')
    set_end_time = timestamp_to_num(set_end_time, start_time, end_time)
    return dcc.RangeSlider(id='range-slider', min=0, max=1, step=0.001,
                           value=[0, set_end_time], marks=marks)


def _build_plot_header(h_id, title, op_id):
    """ Creates a header for a graph """
    return dbc.Row(
        dbc.Col(children=[
                    html.H6(title, id=h_id),
                    dbc.Button('options', id=op_id, color='link')
        ])
    )

def _build_options_btn(op_id):
    """ Creates options button to link to graph options"""
    return dbc.Row(justify='end',
                   children=dbc.Col(width=3,
                                    children=dbc.Button('options', style={'marginTop': '-18px'},
                                                        id=op_id, color='link', size='sm')))

def acts_vs_devs_layout(df_acts, act_id, df_devs, plot_height=False):
    """


    Note
    ----
    Since the contingency table calculation may take very long the layout is initalized 
    with placeholders and the real figures are computed later on demand
    """



    layout_acts_vs_devs = dbc.Container([
        dcc.Store(f'avd_activity-order-{act_id}'),
        dcc.Store(f'avd_device-order-{act_id}'),
        dcc.Store(f'avd_state-contingency-{act_id}'),
        dcc.Store(f'avd_event-contingency-{act_id}'),
        html.Div(id=f"avd-trigger-{act_id}", style=dict(display="none")),
        html.Div(id=f"avd-update-{act_id}", style=dict(display="none")),
        dbc.Row([
            dcc.Graph(id=f'avd_graph-event-contingency-{act_id}',
                      style=dict(height='100%',width='100%'),
                      figure=Figure(),
                      config=dict(displaylogo=False, displayModeBar=True,
                                  responsive=False,
                                  modeBarButtonsToRemove=_buttons_to_use('resetScale2d'),
                     ),
            ),
            _build_options_btn(f'clps-avd-event-button-{act_id}'),
            dbc.Collapse(id=f'clps-avd-event-{act_id}',
                        style={'paddingLeft': '2rem',
                                'paddingBottom': '2rem',
                    },
                children=_option_grid(
                     labels=['Scale:', 'Activity order: ', 'Device order'],
                     values=[
                        _build_row_radio_button(
                            rd_id=f'avd_event-scale-{act_id}',
                            options=['linear', 'log'],
                            value='log'),
                        _build_row_radio_button(
                            rd_id=f'avd_act-order-trigger-{act_id}',
                            options=['alphabetical', 'duration', 'count', 'area'],
                            value='alphabetical'),
                        _build_row_radio_button(
                            rd_id=f'avd_dev-order-trigger-{act_id}',
                            options=['alphabetical', 'count', 'area'],
                            value='alphabetical'),
                ])),
            ]),
            dbc.Spinner(html.Div(id=f'loading-output-{act_id}')),
            dcc.Graph(id=f'avd_graph-state-contingency-{act_id}',
                      style=dict(height='100%',width='100%'),
                      figure=Figure(),
                      config=dict(displaylogo=False, displayModeBar=True)
            ),
            _build_options_btn(f'clps-avd-state-button-{act_id}'),
            dbc.Collapse(id=f'clps-avd-state-{act_id}',
             style={'paddingLeft': '2rem',
                    'paddingBottom': '2rem',
                    },
            children=_option_grid(
                 labels=['Scale:', 'Activity order: ', 'Device order'],
                 values=[
                    _build_row_radio_button(
                        rd_id=f'avd_state-scale-{act_id}',
                        options=['linear', 'log'],
                        value='log'),
            ])),
    ])
    return layout_acts_vs_devs

def _buttons_to_use(*use):
    """Returns the buttons to remove in order to get the buttons to use"""
    buttons = ['toImage', 'zoom2d', 'pan2d',  'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
               'autoScale2d', 'resetScale2d',
               'toggleSpikelines', 'hoverCompareCartesian', 'hoverClosestCartesian'
    ]
    res = []
    for b in buttons:
        if b not in use:
            res.append(b)
    return res

def acts_n_devs_layout(df_acts, df_devs, start_time, end_time, set_end_time, plot_height=350):
    """
    TODO

    """
    if df_acts is None or df_devs is None:
        fig_and = Figure()
        time_slider = dcc.RangeSlider(id='range-slider', min=0, max=1, step=0.001, value=[0, 1])
        checklist_act = dcc.Dropdown(id='select-activities', multi=True, options=[], value=[], clearable=False)
        choice_device = dcc.Dropdown(id='select-devices', options=[], multi=True, value=[], clearable=False)
    else:
        fig_and = activities_and_devices(df_acts, df_devs, st=start_time, et=set_end_time, height=plot_height)
        time_slider = _build_range_slider(None, None, start_time, end_time, set_end_time)
        checklist_act = _build_checklist(df_acts)
        choice_device = _build_choice_devices(df_devs)

    return html.Div(children=[

                   dbc.Row(dbc.Col(
                            dcc.Graph(id='graph-acts_n_devs',
                                      style=dict(height='100%',width='100%'),
                                      figure=fig_and,
                                      config=dict(displayModeBar=True,
                                                  modeBarButtonsToRemove=_buttons_to_use(
                                                      'zoom2d', 'zoomIn2d', 'zoomOut2d',
                                                      'pan2d', 'resetScale2d'

                                                  ),
                                                  edits=dict(legendPosition=True),
                                                  showAxisDragHandles=True,
                                                  displaylogo=False))
                   )),
            dbc.Row(justify='end', children=dbc.Col(width=5,
                            children=[dbc.ButtonGroup([
                                dbc.Button('reset selection', id='and_reset_sel', disabled=True, color='link', size='sm'),
                                dbc.Button('options',
                                   id='clps-acts-n-devs-button',
                                   color='link',
                                   size='sm',
                                   n_clicks=0),
                                dcc.Clipboard(id='and_clipboard', style={'fontsize': 10}),
                            ]),
                            ])),
            dcc.Store(id='and_act-order'),
            dcc.Store(id='and_dev-order'),
            dbc.Collapse(id='clps-acts-n-devs',
                children=[
                    html.H6('Time: '),
                    html.Div(children=[time_slider]),
                    html.H6('Activities: ', style={'marginTop': '20px'}),
                    dbc.Row(style={'padding': '10px'},
                            children=[
                            dbc.Row(children=[
                                dbc.Col(children=[
                                    dbc.Row(children=[
                                        dbc.Col('Order: ', width=2),
                                        dbc.Col(
                                            dbc.RadioItems(id='and_act-order-trigger',
                                                           inline=True,
                                                           options=[{'label': i, 'value': i} for i in ['alphabetical', 'count', 'duration']],
                                                           value='duration',
                                                           labelStyle={'display': 'inline-block', 'marginTop': '5px'}),
                                            width=10,
                                        )
                                        ],
                                    )],
                                        width=10,
                                ),
                                ]
                            ),
                            dbc.Row(children=[checklist_act]),
                    ]),
                    html.H6('Devices', style={'marginTop': '20px'}),
                    dbc.Row(style={'padding': '10px'},
                            children=[
                                dbc.Row(children=[
                                    dbc.Col(children=[
                                        dbc.Row(children=[
                                            dbc.Col('Order: ', width=2),
                                            dbc.Col(width=10, children=dbc.RadioItems(id='and_dev-order-trigger',
                                                               inline=True,
                                                               options=[{'label': i, 'value': i} for i in ['alphabetical', 'type', 'area']],
                                                               value='alphabetical',
                                                               labelStyle={'display': 'inline-block', 'marginTop': '5px'}),

                                            )
                                        ])
                                    ]),
                                    dbc.Row(children=[
                                            dbc.Col('Marker type: ', width=2),
                                            dbc.Col(width=10, children=dbc.RadioItems(id='and_dev-type',
                                                               inline=True,
                                                               options=[{'label': i, 'value': i} for i in ['event', 'state']],
                                                               value='event',
                                                               labelStyle={'display': 'inline-block', 'marginTop': '5px'}),

                                            )
                                        ])
                                ]),
                                dbc.Row(choice_device),
                    ]),
            ]),
    ])

def device_layout_graph_bottom(fig, type='density'):
    options_density = [_build_row_radio_button(
                rd_id='devs_iei-scale',
                options=['linear', 'log'],
                value='log'),
            daq.BooleanSwitch(
                id='devs_iei-per-device',
                on=False,
                persisted_props=[]
            )]
    options_iei = [
        dbc.Collapse(id='clps-dev-iei'),
        dbc.Button(id='clps-dev-iei-button'),
    ]

    if type == 'density':
        mode_bar_buttons = _buttons_to_use('zoom2d', 'pan2d', 'resetScale2d'),
        labels = []
        options = options_density
        to_hide = options_iei
    elif type == 'iei':
        mode_bar_buttons = _buttons_to_use('zoom2d', 'pan2d', 'resetScale2d'),
        labels = ['Scale:', 'Per device: ']
        options = options_iei
        to_hide = options_iei
    else:
        raise ValueError

    return [dbc.Row(dcc.Graph(figure=fig,
                              style=dict(height='100%',width='100%'),
                              config=dict(displaylogo=False,
                                    displayModeBar=True,
                                    modeBarButtonsToRemove=mode_bar_buttons
                              )
                        )),
                        _build_options_btn('clps-dev-iei-button'),
                        dbc.Collapse(id='clps-dev-iei',
                                     style={'paddingLeft': '2rem',
                                            'paddingRight': '2rem',
                                            'paddingBottom': '2rem',
                                            },
                                    children=_option_grid(
                                         labels=['Plot', *labels],
                                         lwidth=4, rwidth=8,
                                         values=[
                                            dcc.Dropdown(id='devs_bottom-drop',
                                                         value='density', options=[
                                                              {'label': 'Inter-event-interval', 'value': 'iei' },
                                                              {'label': 'density', 'value': 'density' }
                                                ]
                                            ),
                                             *options]
                                    )
                        ),
            html.Div(children=to_hide, style={'display': 'none'})
            ]


def devices_layout(df_devs, initialize=False, plot_height=350):

    if df_devs is not None and not df_devs.empty:
        fig_bar_count = dev_bar_count(df_devs, height=plot_height)
        fig_bp_state = boxplot_state(df_devs, height=plot_height)
        fig_ev_density = event_density(df_dev=df_devs, show_colorbar=False,
                                    height=plot_height
                                    )
        fig_iei = dev_iei(df_devs, height=plot_height)
        fig_fraction = dev_fraction(df_devs, height=plot_height)
    else:
        fig_bar_count = Figure()
        fig_bp_state = Figure()
        fig_ev_density = Figure()
        fig_iei = Figure()
        fig_fraction = Figure()




    layout_devices = dbc.Container([
        dcc.Store('devs_density-data'),
        dcc.Store('devs_order'),
        html.Div(id="dev-trigger", style=dict(display="none")),
        html.Div(children='', id=f"dev-curr-sel", style=dict(display="none")),
        dcc.Store(id=f'dev-curr-sel-store'),
        dbc.Row([
            dbc.Col(width=6, children=[
                        dbc.Row(dcc.Graph(id='devs_graph-bar',
                                          figure=fig_bar_count,
                                          style=dict(height='100%',width='100%'),
                                          config=dict(displaylogo=False,
                                                displayModeBar=True,
                                                modeBarButtonsToRemove=_buttons_to_use('resetScale2d'),
                                            )
                        )),
                        _build_options_btn('clps-dev-bar-button'),
                        dbc.Collapse(id='clps-dev-bar',
                                     style={'paddingLeft': '2rem',
                                            'paddingRight': '2rem',
                                            'paddingBottom': '2rem',
                                            },
                                    children=_option_grid(
                                         labels=['Plot:', 'Scale:', 'Order: '],
                                         values=[
                                            _build_row_radio_button(
                                                rd_id='devs_bar-scale',
                                                options=['linear', 'log'],
                                                value='linear'),
                                            _build_row_radio_button(
                                                rd_id='devs_bar-order',
                                                options=['alphabetical', 'count', 'area'],
                                                value='count'),
                        ]))
            ]),
            dbc.Col(width=6, children=[
                        dbc.Row(dcc.Graph(id='devs_graph-boxplot',
                                          style=dict(height='100%',width='100%'),
                                          figure=fig_bp_state,
                                          config=dict(displaylogo=False,
                                                displayModeBar=True,
                                                modeBarButtonsToRemove=_buttons_to_use(
                                                    'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d',
                                                    'lasso2d', 'resetScale2d'
                                                ),
                                          )
                        )),
                        _build_options_btn('clps-dev-bp-button'),
                        dbc.Collapse(id='clps-dev-bp',
                                     style={'paddingLeft': '2rem',
                                            'paddingRight': '2rem',
                                            'paddingBottom': '2rem',
                                            },
                                    children=_option_grid(
                                         labels=['Plot:', 'Binary state:', ],
                                         lwidth=4, rwidth=8,
                                         values=[
                                            _build_row_radio_button(
                                                rd_id='devs_bp-scale',
                                                options=['linear', 'log'],
                                                value='linear'),
                                            _build_row_radio_button(
                                                rd_id='devs_bp-binary-state',
                                                options=['off', 'on'],
                                                value='on'),
                        ]))
            ]),
        ]),
        dbc.Row([
            dbc.Col(width=7,
                children=[
                    dbc.Tabs(id='devs_tab',
                        children=[

                    dbc.Tab(label='D', tab_id='devs_tab-density',
                            label_style={'fontSize': '12px', 'padding': '5px'},
                            children=[
                                dcc.Graph(id='devs_graph-density',
                                          figure=fig_ev_density,
                                          style=dict(height='100%',width='100%'),
                                          config=dict(displaylogo=False,
                                                displayModeBar=True,
                                                modeBarButtonsToRemove=[]
                                          )
                                ),
                                _build_options_btn('clps-dev-density-button'),
                                dbc.Collapse(id='clps-dev-density',
                                         style={'paddingLeft': '2rem',
                                                'paddingRight': '2rem',
                                                'paddingBottom': '2rem',
                                                },
                                         children=_option_grid(
                                             labels=['Resolution', 'Scale'],
                                             lwidth=3, rwidth=9,
                                             values=[
                                                dcc.Slider(id='devs_dens-slider', min=0, max=10, step=None, value=5,
                                                           marks={
                                                               0: {'label': DEV_DENS_SLIDER[0], 'style': {"transform": "rotate(45deg)"}},
                                                               1: {'label': DEV_DENS_SLIDER[1], 'style': {"transform": "rotate(45deg)"}},
                                                               2: {'label': DEV_DENS_SLIDER[2], 'style': {"transform": "rotate(45deg)"}},
                                                               3: {'label': DEV_DENS_SLIDER[3], 'style': {"transform": "rotate(45deg)"}},
                                                               4: {'label': DEV_DENS_SLIDER[4], 'style': {"transform": "rotate(45deg)"}},
                                                               5: {'label': DEV_DENS_SLIDER[5], 'style': {"transform": "rotate(45deg)"}},
                                                               6: {'label': DEV_DENS_SLIDER[6], 'style': {"transform": "rotate(45deg)"}},
                                                               9: {'label': DEV_DENS_SLIDER[9], 'style': {"transform": "rotate(45deg)"}},
                                                               10: {'label': DEV_DENS_SLIDER[10], 'style': {"transform": "rotate(45deg)"}},
                                                           }
                                                ),
                                                 _build_row_radio_button(
                                                      rd_id='devs_dens-scale',
                                                      options=['linear', 'log'],
                                                      value='linear'),
                                            ]
                                        )
                            )

                        ]),
                        dbc.Tab(label='IEI', tab_id='devs_tab-iei',
                            label_style={'fontSize': '12px', 'padding': '5px'},
                            children=[
                                dcc.Graph(id='devs_graph-iei',
                                          figure=fig_iei,
                                          style=dict(height='100%',width='100%'),
                                          config=dict(displaylogo=False,
                                                displayModeBar=True,
                                                modeBarButtonsToRemove=_buttons_to_use(
                                                        'zoom2d', 'pan2d',  'lasso2d', 'resetScale2d'
                                                ),
                                          )
                                ),
                                _build_options_btn('clps-dev-iei-button'),
                                dbc.Collapse(id='clps-dev-iei',
                                         style={'paddingLeft': '2rem',
                                                'paddingRight': '2rem',
                                                'paddingBottom': '2rem',
                                                },
                                        children=_option_grid(
                                             labels=['Scale:', 'Per device:'],
                                             lwidth=4, rwidth=8,
                                             values=[
                                                _build_row_radio_button(
                                                      rd_id='devs_iei-scale',
                                                      options=['linear', 'log'],
                                                      value='linear'),
                                                  daq.BooleanSwitch(
                                                      id='devs_iei-per-device',
                                                      on=False,
                                                      persisted_props=[]
                                                )]
                                        )
                            )
                    ]),
                    ])
                ]
            ),
            dbc.Col(width=5,
                    children=[
                        dbc.Row(style={'marginTop': '29px'},
                            children=dcc.Graph(id='devs_graph-fraction',
                                          style=dict(height='100%',width='100%'),
                                          figure=fig_fraction,
                                          config=dict(displaylogo=False,
                                                displayModeBar=True,
                                                modeBarButtonsToRemove=_buttons_to_use(
                                                    'zoom2d', 'pan2d', 'resetScale2d'
                                              ),
                                            )
                        )),

            ])

        ])

    ]),

    return layout_devices

def _option_grid(labels, values,lwidth=2, rwidth=10):
    """ generates sth. like this:
    lbl_1       *a *b *c
    lbl_2       | asdfasdf v|

    """
    assert (lwidth + rwidth) == 12

    root_child_lst = []
    for lbl, val in zip(labels, values):
        root_child_lst.append(dbc.Row(children=[
            dbc.Col(lbl, width=lwidth),
            dbc.Col(width=rwidth, children=val)
        ]))
    return dbc.Col(children=root_child_lst)

def activities_layout(df_acts, act_id=0, plot_height=350):

    if df_acts is not None and not df_acts.empty:
        fig_bar_count = act_bar_count(df_acts, height=plot_height)
        fig_density = act_density(df_acts, height=plot_height)
        try:
            fig_transition = act_heatmap_transitions(df_acts, height=plot_height)
        except:
            fig_transition = Figure()
        fig_duration = act_boxplot_duration(df_acts, height=plot_height)
    else:
        fig_bar_count = Figure()
        fig_density = Figure()
        fig_transition = Figure()
        fig_duration = Figure()

    layout_activities = dbc.Container([
        dcc.Store(f'acts_density-data-{act_id}'),
        dbc.Row([
            dbc.Col(width=6,
                    children=[
                        dbc.Row(dcc.Graph(id=f'acts_graph-bar-{act_id}',
                                          style=dict(height='100%',width='100%'),
                                          figure=fig_bar_count,
                                          config=dict(displaylogo=False,
                                                displayModeBar=True,
                                                modeBarButtonsToRemove=_buttons_to_use(
                                                    'zoom2d', 'pan2d','resetScale2d'
                                              ),
                                            )
                        )),
                        _build_options_btn(f'clps-act-bar-button-{act_id}'),
                        dbc.Collapse(id=f'clps-act-bar-{act_id}',
                                     style={'paddingLeft': '2rem',
                                            'paddingRight': '2rem',
                                            'paddingBottom': '2rem',
                                            },
                                    children=_option_grid(
                                         labels=['Plot:', 'Scale:', 'Order: '],
                                         values=[
                                            dcc.Dropdown(id=f'acts_bar-drop-{act_id}', value='count', options=[
                                                  {'label': 'Count', 'value': 'count'},
                                                  {'label': 'Cumulative', 'value': 'cum'}]
                                            ),
                                            _build_row_radio_button(
                                                rd_id=f'acts_bar-scale-{act_id}',
                                                options=['linear', 'log'],
                                                value='linear'),
                                            _build_row_radio_button(
                                                rd_id=f'acts_sort-{act_id}',
                                                options=['alphabetical', 'value', 'area'],
                                                value='value'),
                        ]))
            ]),
            html.Div(children='', id=f"act-curr-sel-{act_id}", style=dict(display="none")),
            dcc.Store(id=f'act-curr-sel-store-{act_id}'),
            html.Div(id=f"act-trigger-{act_id}", style=dict(display="none"), **{
              "data-value-2": "false"
            }),
            dcc.Store(id=f'acts_activity-order-{act_id}'),
            dbc.Col(width=6,
                    children=[
                        dbc.Row(dcc.Graph(id=f'acts_graph-boxplot-{act_id}',
                                  style=dict(height='100%',width='100%'),
                                  figure=fig_duration,
                                  config=dict(displayModeBar=True,
                                              modeBarButtonsToRemove=_buttons_to_use(
                                                'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d',
                                                'lasso2d', 'resetScale2d'
                                              ),
                                              responsive=False,
                                              displaylogo=False)
                        )),
                        _build_options_btn(f'clps-act-boxplot-button-{act_id}'),
                        dbc.Collapse(id=f'clps-act-boxplot-{act_id}',
                                    style={'paddingLeft': '2rem',
                                            'paddingRight': '2rem',
                                            'paddingBottom': '2rem',
                                            },
                                     children=_option_grid(
                                         labels=['Plot:', 'Scale:'],
                                         values=[
                                            dcc.Dropdown(id=f'acts_bp-drop-{act_id}', value='bp', options=[
                                                                              {'label': 'boxplot', 'value': 'bp'},
                                                                              {'label': 'voilin', 'value': 'vp'}]
                                            ),
                                            _build_row_radio_button(
                                                rd_id=f'acts_bp-scale-{act_id}',
                                                options=['linear', 'log'],
                                                value='linear'),
                                         ])
                        ),
            ]),
        ]),
        dbc.Row([
            dbc.Col(width=6,
                    children=[
                        dbc.Row(dcc.Graph(id=f'acts_graph-density-{act_id}',
                                  style=dict(height='100%',width='100%'),
                                  figure=fig_density,
                                  config=dict(
                                        modeBarButtonsToRemove=_buttons_to_use(
                                            'resetScale2d'
                                        ),
                                      displaylogo=False,
                                      displayModeBar=True,
                                      showAxisDragHandles=False,
                                  )),
                        ),
            ]),
            dbc.Col(width=6,
                    children=[
                        dbc.Row(dcc.Graph(id=f'acts_graph-transition-{act_id}',
                                  style=dict(height='100%',width='100%'),
                                  figure=fig_transition,
                                  config=dict(displayModeBar=True,
                                              displaylogo=False,
                                              modeBarButtonsToRemove=_buttons_to_use(
                                                'resetScale2d'
                                                ),
                                  )),
                        ),
                        _build_options_btn(f'clps-act-transition-button-{act_id}'),
                        dbc.Collapse(id=f'clps-act-transition-{act_id}',
                                    style={'paddingLeft': '2rem',
                                            'paddingRight': '2rem',
                                            'paddingBottom': '2rem',
                                            },
                                     children=_option_grid(
                                         labels=['Scale:'],
                                         values=[
                                            _build_row_radio_button(
                                                rd_id=f'acts_trans-scale-{act_id}',
                                                options=['linear', 'log'],
                                                value='linear'),
                                         ]),
                        )
            ]),
        ]),
        dcc.Store(id=f'activity-order-{act_id}'),
    ])
    return layout_activities

