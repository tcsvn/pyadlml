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

from pyadlml.dataset import fetch_amsterdam, set_data_home, TIME, START_TIME, END_TIME, ACTIVITY
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

    acts = df_acts[ACTIVITY].unique()
    options = [create_option(a) for a in acts]
    options = [create_option('idle'), *options]
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


def _build_range_slider(df_acts, df_devs, start_time, end_time):
    def create_mark(ts, format):
        return {'label': _np_dt_strftime(ts, format), 'style': {"transform": "rotate(45deg)"}}

    marks = {0: create_mark(start_time, '%Y.%m.%d'),
             1: create_mark(end_time, '%Y.%m.%d')}
    for day in pd.date_range(start_time, end_time, freq='D')[1:-1]:
        marks[timestamp_to_num(day, start_time, end_time)] = create_mark(day, '%m.%d')
    return dcc.RangeSlider(id='range-slider', min=0, max=1, step=0.001,
                           value=[0, 1], marks=marks)


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

def acts_vs_devs_layout(df_acts, df_devs, initialize=False):
    """

    """
    # If the layout is initalized the space is filled with placeholder figures
    # to reduce first loading time
    if initialize:
        fig_dummy = Figure()
    else:
        fig_adec = activities_devices_event_contingency(df_acts, df_devs)
    layout_acts_vs_devs = dbc.Container([
        dcc.Store('avd_activity-order'),
        dcc.Store('avd_device-order'),
        dcc.Store('avd_state-contingency'),
        dcc.Store('avd_event-contingency'),
        html.Div(id="avd-trigger", style=dict(display="none")),
        html.Div(id="dev-trigger", style=dict(display="none")),
        html.Div(id="avd-update", style=dict(display="none")),
        dbc.Row([
            dcc.Graph(id='avd_graph-event-contingency',
                      figure=fig_dummy,
                      config=dict(displaylogo=False, displayModeBar=True,
                                  responsive=False,
                                  modeBarButtonsToRemove=_buttons_to_use('resetScale2d'),
                     ),
            ),
            _build_options_btn('clps-avd-event-button'),
            dbc.Collapse(id='clps-avd-event',
                        style={'padding-left': '2rem',
                                'padding-bottom': '2rem',
                    },
                children=_option_grid(
                     labels=['Scale:', 'Activity order: ', 'Device order'],
                     values=[
                        _build_row_radio_button(
                            rd_id='avd_event-scale',
                            options=['linear', 'log'],
                            value='log'),
                        _build_row_radio_button(
                            rd_id='avd_act-order-trigger',
                            options=['alphabetical', 'duration', 'count', 'area'],
                            value='alphabetical'),
                        _build_row_radio_button(
                            rd_id='avd_dev-order-trigger',
                            options=['alphabetical', 'count', 'area'],
                            value='alphabetical'),
                ])),
            ]),
            dbc.Spinner(html.Div(id='loading-output')),
            dcc.Graph(id='avd_graph-state-contingency',
                      figure=fig_dummy,
                      config=dict(displaylogo=False, displayModeBar=True)
            ),
            _build_options_btn('clps-avd-state-button'),
            dbc.Collapse(id='clps-avd-state',
             style={'padding-left': '2rem',
                    'padding-bottom': '2rem',
                    },
            children=_option_grid(
                 labels=['Scale:', 'Activity order: ', 'Device order'],
                 values=[
                    _build_row_radio_button(
                        rd_id='avd_state-scale',
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

def acts_n_devs_layout(df_acts, df_devs, start_time, end_time):
    return dbc.Row(children=[
                   html.Div(children=[
                            dcc.Graph(id='graph-acts_n_devs',
                                      figure=activities_vs_devices(df_acts, df_devs),
                                      config=dict(displayModeBar=True,
                                                  modeBarButtonsToRemove=_buttons_to_use(
                                                      'zoom2d', 'zoomIn2d', 'zoomOut2d',
                                                      'pan2d', 'resetScale2d'

                                                  ),
                                                  edits=dict(legendPosition=True),
                                                  showAxisDragHandles=True,
                                                  displaylogo=False))
                   ]),
            dbc.Row(justify='end', children=dbc.Col(width=5,
                            children=dbc.ButtonGroup([
                                dbc.Button('reset selection', id='and_reset_sel', disabled=True, color='link', size='sm'),
                                dbc.Button('options',
                                   id='clps-acts-n-devs-button',
                                   color='link',
                                   size='sm',
                                   n_clicks=0)
            ]))),
            dcc.Store(id='and_act-order'),
            dcc.Store(id='and_dev-order'),
            dbc.Collapse(id='clps-acts-n-devs',
                children=[
                    html.H6('Time: '),
                    html.Div(children=[_build_range_slider(df_acts, df_devs, start_time, end_time)]),
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
                                    ),],
                                        width=10,
                                ),
                                ]
                            ),
                            dbc.Row(children=[_build_checklist(df_acts), ]),
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
                                dbc.Row(_build_choice_devices(df_devs)),
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
                on=False
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
                              config=dict(displaylogo=False,
                                    displayModeBar=True,
                                    modeBarButtonsToRemove=mode_bar_buttons
                              )
                        )),
                        _build_options_btn('clps-dev-iei-button'),
                        dbc.Collapse(id='clps-dev-iei',
                                     style={'padding-left': '2rem',
                                            'padding-right': '2rem',
                                            'padding-bottom': '2rem',
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


def devices_layout(df_devs, initialize=False):
    layout_devices = dbc.Container([
        dcc.Store('devs_density-data'),
        dcc.Store('devs_order'),
        dbc.Row([
            dbc.Col(width=6, children=[
                        dbc.Row(dcc.Graph(id='devs_graph-bar',
                                          figure=dev_bar_count(df_devs),
                                          config=dict(displaylogo=False,
                                                displayModeBar=True,
                                                modeBarButtonsToRemove=_buttons_to_use('resetScale2d'),
                                            )
                        )),
                        _build_options_btn('clps-dev-bar-button'),
                        dbc.Collapse(id='clps-dev-bar',
                                     style={'padding-left': '2rem',
                                            'padding-right': '2rem',
                                            'padding-bottom': '2rem',
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
                                          figure=boxplot_state(df_devs),
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
                                     style={'padding-left': '2rem',
                                            'padding-right': '2rem',
                                            'padding-bottom': '2rem',
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
                            label_style={'font-size': '12px', 'padding': '5px'},
                            children=[
                                dcc.Graph( id='devs_graph-density',
                                figure=event_density(df_dev=df_devs, show_colorbar=False),
                                          config=dict(displaylogo=False,
                                                displayModeBar=True,
                                                modeBarButtonsToRemove=[]
                                          )
                                ),
                                _build_options_btn('clps-dev-density-button'),
                                dbc.Collapse(id='clps-dev-density',
                                         style={'padding-left': '2rem',
                                                'padding-right': '2rem',
                                                'padding-bottom': '2rem',
                                                },
                                         children=_option_grid(
                                             labels=['Resolution' ],
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
                                                )
                                            ]
                                        )
                            )

                        ]),
                        dbc.Tab(label='IEI', tab_id='devs_tab-iei',
                            label_style={'font-size': '12px', 'padding': '5px'},
                            children=[
                                dcc.Graph( id='devs_graph-iei',
                                    figure=dev_iei(df_devs),
                                              config=dict(displaylogo=False,
                                                    displayModeBar=True,
                                                    modeBarButtonsToRemove=_buttons_to_use(
                                                        'zoom2d', 'pan2d',  'lasso2d', 'resetScale2d'
                                                ),
                                              )
                                ),
                                _build_options_btn('clps-dev-iei-button'),
                                dbc.Collapse(id='clps-dev-iei',
                                         style={'padding-left': '2rem',
                                                'padding-right': '2rem',
                                                'padding-bottom': '2rem',
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
                                                      on=False
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
                                          figure=dev_fraction(df_devs),
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

def activities_layout(df_acts):
    layout_activities = dbc.Container([
        dcc.Store('acts_density-data'),
        dbc.Row([
            dbc.Col(width=6,
                    children=[
                        dbc.Row(dcc.Graph(id='graph-bar',
                                          figure=act_bar_count(df_acts),
                                          config=dict(displaylogo=False,
                                                displayModeBar=True,
                                                modeBarButtonsToRemove=_buttons_to_use(
                                                    'zoom2d', 'pan2d','resetScale2d'
                                              ),
                                            )
                        )),
                        _build_options_btn('clps-act-bar-button'),
                        dbc.Collapse(id='clps-act-bar',
                                     style={'padding-left': '2rem',
                                            'padding-right': '2rem',
                                            'padding-bottom': '2rem',
                                            },
                                    children=_option_grid(
                                         labels=['Plot:', 'Scale:', 'Order: '],
                                         values=[
                                            dcc.Dropdown(id='acts_bar-drop', value='count', options=[
                                                  {'label': 'Count', 'value': 'count'},
                                                  {'label': 'Cumulative', 'value': 'cum'}]
                                            ),
                                            _build_row_radio_button(
                                                rd_id='acts_bar-scale',
                                                options=['linear', 'log'],
                                                value='log'),
                                            _build_row_radio_button(
                                                rd_id='acts_sort',
                                                options=['alphabetical', 'value', 'area'],
                                                value='value'),
                        ]))
            ]),
            html.Div(children='', id="act-curr-sel", style=dict(display="none")),
            html.Div(children='', id="dev-curr-sel", style=dict(display="none")),
            dcc.Store(id='act-curr-sel-store'),
            dcc.Store(id='dev-curr-sel-store'),
            html.Div(id="act-trigger", style=dict(display="none"), **{
              "data-value-2": "false"
            }),
            dcc.Store(id='acts_activity-order'),
            dbc.Col(width=6,
                    children=[
                        dbc.Row(dcc.Graph(id='graph-boxplot',
                                  figure=act_boxplot_duration(df_acts),
                                  config=dict(displayModeBar=True,
                                              modeBarButtonsToRemove=_buttons_to_use(
                                                'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d',
                                                'lasso2d', 'resetScale2d'
                                              ),
                                              responsive=False,
                                              displaylogo=False)
                        )),
                        _build_options_btn('clps-act-boxplot-button'),
                        dbc.Collapse(id='clps-act-boxplot',
                                    style={'padding-left': '2rem',
                                            'padding-right': '2rem',
                                            'padding-bottom': '2rem',
                                            },
                                     children=_option_grid(
                                         labels=['Plot:', 'Scale:'],
                                         values=[
                                            dcc.Dropdown(id='acts_bp-drop', value='bp', options=[
                                                                              {'label': 'boxplot', 'value': 'bp'},
                                                                              {'label': 'voilin', 'value': 'vp'}]
                                            ),
                                            _build_row_radio_button(
                                                rd_id='acts_bp-scale',
                                                options=['linear', 'log'],
                                                value='linear'),
                                         ])
                        ),
            ]),
        ]),
        dbc.Row([
            dbc.Col(width=6,
                    children=[
                        dbc.Row(dcc.Graph(id='graph-density',
                                  figure=act_density(df_acts),
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
                        dbc.Row(dcc.Graph(id='graph-transition',
                                  figure=act_heatmap_transitions(df_acts),
                                  config=dict(displayModeBar=True,
                                              displaylogo=False,
                                              modeBarButtonsToRemove=_buttons_to_use(
                                                'resetScale2d'
                                                ),
                                  )),
                        ),
                        _build_options_btn('clps-act-transition-button'),
                        dbc.Collapse(id='clps-act-transition',
                                    style={'padding-left': '2rem',
                                            'padding-right': '2rem',
                                            'padding-bottom': '2rem',
                                            },
                                     children=_option_grid(
                                         labels=['Scale:'],
                                         values=[
                                            _build_row_radio_button(
                                                rd_id='acts_trans-scale',
                                                options=['linear', 'log'],
                                                value='linear'),
                                         ]),
                        )
            ]),
        ]),
        dcc.Store(id='activity-order'),
    ])
    return layout_activities

