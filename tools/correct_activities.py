from pathlib import Path
from dash import Dash, dash_table, dcc, html
from copy import copy
from dash.dependencies import Input, Output, State
import pandas as pd
import argparse
import dash
from dash.exceptions import PreventUpdate
from pyadlml.constants import ACTIVITY, END_TIME, START_TIME, STRFTIME_PRECISE
from pyadlml.dataset.plot.plotly.util import dash_get_trigger_element, dash_get_trigger_value

from pyadlml.dataset.util import df_difference, fetch_by_name
from pyadlml.dataset.plot.plotly.activities import correction
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

SEP_STR = '#' + '-'*100
ID_STR = '# id='
CASE_STR = 'CaseNr.: %s'
def _df_to_rows(df):
    from pyadlml.constants import STRFTIME_PRECISE
    df = df.copy()
    df[START_TIME] = df[START_TIME].dt.strftime(STRFTIME_PRECISE)
    df[END_TIME] = df[END_TIME].dt.strftime(STRFTIME_PRECISE)
    return df.to_dict('records')
    

def _rows_to_df(rows):
    df = pd.DataFrame(rows, columns=COLS)
    df[START_TIME] = pd.to_datetime(df[START_TIME], errors='coerce', dayfirst=True)
    df[END_TIME] = pd.to_datetime(df[END_TIME], errors='coerce', dayfirst=True)

    # Drop rows with at least one empty cells
    df = df.dropna()
    return df


def delete_id_content_from_code(code, id):
    new_text = []
    text_id = None

    deleting = False
    searching = True
    for line in code.split('\n'):
        if searching:
            if ID_STR in line:
                text_id = int(line[5:])
            if text_id == id:
                searching, deleting = False, True
                continue
            else:
                new_text.append(line)
        elif deleting: 
            if not ID_STR in line and line != SEP_STR:
                continue
            else:
                new_text.append(line)
                deleting = False
        else:
            new_text.append(line)

    code = '\n'.join(new_text)
    return code

def get_id_content_from_code(code, id):
    new_text = []
    text_id = None

    retreiving = False
    searching = True
    for line in code.split('\n'):
        if searching:
            if ID_STR in line:
                text_id = int(line[5:])
            if text_id == id:
                searching, retreiving = False, True
                new_text.append(line)

        elif retreiving: 
            if not ID_STR in line and line != SEP_STR:
                new_text.append(line)
            else:
                break
    code = '\n'.join(new_text)
    return code

def insert_id_content_into_code(code, id, content):
    new_text = code.split('\n')
    text_id = -1
    for idx, line in enumerate(new_text):
        if ID_STR in line:
            text_id = int(line[5:])
        if text_id > id:
            idx += -1
            break
    # Position to insert text
    new_code = '\n'.join(new_text[:idx+1])\
             + content + '\n'.join(new_text[idx+1:]) 
    return new_code

def is_id_in_code(code, id):
    for line in code.split('\n'):
        if ID_STR in line and int(line[5:]) == id:
            return True
    return False

def _update_add_lists_in_footer(content, footer):
    add_lists_names = []
    for line in content.split('\n'):
        if 'lst_to_add_' in line:
            add_lists_names.append(line.split(' ')[0])
    
    new_footer = [] 
    for line in footer.split('\n'):
        if 'new_rows_to_add =' in line:
            line = 'new_rows_to_add = ['
            for lst in add_lists_names:
                line += f'*{lst}, '
            line += ']'
        new_footer.append(line)

    return '\n'.join(new_footer) 

def _update_code(curr_id, text, df_prae, df_post, comment):

    df_diff = df_prae.merge(df_post, indicator=True, how='outer')

    # Rows that are new after correction should be added to the original dataframe
    # and rows that are not present have to be deleted. '_merge'='both' are left alone
    entries_to_add = df_diff[df_diff['_merge'] == 'right_only']
    entries_to_delete = df_diff[df_diff['_merge'] == 'left_only']

    # Delete previous entry if present
    code = delete_id_content_from_code(code=text, id=curr_id)

    # Serialize entries to add 
    content = f'\n{ID_STR}{curr_id}\n'
    content += '\n'.join(['# ' + s for s in comment.split('\n')]) + '\n\n'
    content += f'lst_to_add_{curr_id} = [\n'
    for idx, entry in entries_to_add.iterrows():
        st = entry[START_TIME].strftime(STRFTIME_PRECISE)
        et = entry[END_TIME].strftime(STRFTIME_PRECISE)
        content += f"\t['{st}','{et}','{entry[ACTIVITY]}'],\n"
    content += ']\n'

    # Serialize entries to delete
    content += '\nlst_to_del = [\n'
    for idx, entry in entries_to_delete.iterrows():
        st = entry[START_TIME].strftime(STRFTIME_PRECISE)
        et = entry[END_TIME].strftime(STRFTIME_PRECISE)
        content += f"\t['{st}','{et}','{entry[ACTIVITY]}'],\n"
    content += ']\n'
    content += 'idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)\n\n'

    # Place content block at appropriate position
    text_head, text_content, text_footer = code.split(SEP_STR)
    text_content = insert_id_content_into_code(text_content, curr_id, content)
    text_footer = _update_add_lists_in_footer(text_content, text_footer)
    text = SEP_STR.join([text_head, text_content, text_footer])

    return text

def df_from_code(code, id, df_acts):
    content_only_from_id = get_id_content_from_code(code, id)
    text_head, _, text_footer = code.split(SEP_STR)
    new_footer = [] 
    # TODO refactor
    for line in text_footer.split('\n'):
        if 'new_rows_to_add =' in line:
            line = f'new_rows_to_add = lst_to_add_{id}'
        new_footer.append(line)

    new_head = []
    for l in text_head.split('\n'):
        if 'df_acts' in l:
            continue
        new_head.append(l)

    code = '\n'.join([*new_head, *content_only_from_id.split('\n'), *new_footer])

    lcls = {'df_acts': df_acts} 
    exec(code, {}, lcls)
    df_acts = lcls['df_acts']
    return df_acts

def comment_from_code(code, id):
    content_only_from_id = get_id_content_from_code(code, id)
    comment = []
    for l in content_only_from_id.split('\n')[1:]:
        if l == '':
            break
        comment.append(l[2:])
    comment = '\n'.join(comment) 
    return comment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='amsterdam')
    parser.add_argument('-i', '--identifier', type=str)
    parser.add_argument('-p', '--port', type=str, default='8050')
    parser.add_argument('-dh', '--data-home', type=str, default='/tmp/pyadlml/')
    parser.add_argument('-o', '--out-folder', type=str, default='/tmp/pyadlml/correct_activities')
    args = parser.parse_args()


    out_folder = Path(args.out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    fp_code = out_folder.joinpath('corrected_activities.py')

    COLS = [START_TIME, END_TIME, ACTIVITY]
    code_str = '# Awesome script \n'\
                 + 'import pandas as pd\n'\
                 + 'from pyadlml.constants import START_TIME, END_TIME, ACTIVITY\n'\
                 + 'from pyadlml.dataset._core.activities import get_index_matching_rows\n\n'\
                 + '# TODO add original activity dataframe in line below\n'\
                 + 'df_acts = ... \n'\
                 + 'idxs_to_delete = []\n\n'\
                 + '# Content\n' \
                 + SEP_STR +'\n'\
                 + SEP_STR + '\n'\
                 + 'new_rows_to_add = []\n' \
                 + 'df_acts = df_acts.drop(idxs_to_delete)\n'\
                 + "df_news = pd.DataFrame(data=new_rows_to_add, columns=[START_TIME, END_TIME, ACTIVITY])\n"\
                 + "df_news[START_TIME] = pd.to_datetime(df_news[START_TIME], dayfirst=True)\n"\
                 + "df_news[END_TIME] = pd.to_datetime(df_news[END_TIME], dayfirst=True)\n"\
                 + 'df_acts = pd.concat([df_acts, df_news], axis=0)\n'

    with open(fp_code, mode='w') as f:
        f.write(code_str)

    # Setup data
    data = fetch_by_name(args.dataset, identifier=args.identifier, \
                         cache=False, retain_corrections=True, keep_original=True)

    # Create global variables
    corr = data['correction_activities']
    if args.identifier:
        corr = corr[args.identifier]

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    curr_id = 0
    current_case = corr[curr_id]

    app.layout = dbc.Container([
        dbc.Row([
            dcc.Store(id="curr-id"),
            dcc.Graph(
                id='correction',
                figure=correction(current_case[0], current_case[1])
            )
        ]),
        dbc.Row(dash.html.Header(id='header', children=CASE_STR%(0))),
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='pre-data-table',
                    columns=(
                        [{'id':c, 'name' :c } for c in current_case[0].columns]
                    ),
                    data=_df_to_rows(current_case[0]),
                    editable=False,
                    style_table={'height': '200px', 'overflowY': 'auto'},
                    style_cell = {'font-size': '12px'},
                ),
            ], md=6),
            dbc.Col([
                dbc.Row(
                    dash_table.DataTable(
                        id='post-data-table',
                        columns=(
                            [{'id':c, 'name' :c } for c in current_case[0].columns]
                        ),
                        data=_df_to_rows(current_case[1]),
                        editable=True,
                        row_deletable=True,
                        style_table={'height': '200px', 'overflowY': 'auto'},
                        style_cell = {'font-size': '12px'},
                    ),
                ),
                dbc.Row([
                    dbc.ButtonGroup([
                        dbc.Button(id="reset", children="reset", color="secondary"), 
                        dbc.Button(id="reset_heuristic", children="reset to heuristic", color="secondary"), 
                        dbc.Button(id="add", children="add empty row", color="secondary"), 
                    ])
                ])
            ], md=6)
        ], className='mb-1'),
        dbc.Row(
            dbc.Textarea(id='code', value='', readOnly=False, 
                        style={'height': '100px'}
                        #style={'resize': 'none', 'overflow':'hidden'}#, 'height': '100px'}
            ),
            style={'height':'100%'},
            className='mb-1',
        ),
        dbc.Row([
            dbc.ButtonGroup([
                dbc.Button(id="prev", children="Previous", color="secondary"), 
                dbc.Button(id="submit", children="Submit", color="primary"),            
                dbc.Button(id="next", children="Next", color="secondary"), 
            ])
        ], className='mb-5'),

    ], style={'width': 1500, 'margin': 'auto'})

    @app.callback(
        Output('correction', 'figure'),
        Output('post-data-table', 'data'),
        Output('pre-data-table', 'data'),
        Output('code', 'value'),
        Output('curr-id', 'data'),
        Output('header', 'children'),
        Output('submit', 'children'),
        Input('post-data-table', 'data'),
        Input('add', 'n_clicks'),
        Input('reset', 'n_clicks'),
        Input('reset_heuristic', 'n_clicks'),
        Input('prev', 'n_clicks'),
        Input('submit', 'n_clicks'),
        Input('next', 'n_clicks'),
        State('pre-data-table', 'data'),
        State('correction', 'figure'),
        State('code', 'value'),
        State('curr-id', 'data'),
        State('submit', 'children'),
    )
    def display_output(post_rows, add_click, reset_click, reset_heuristic_click, prev_click,
                      sub_click, next_click, prae_rows, fig, comment, curr_id, button_text
    ):
        curr_id = 0 if curr_id is None else curr_id

        current_case = corr[curr_id]

        trigger = dash_get_trigger_element()
        if trigger is None or trigger == '':
            raise PreventUpdate


        # Open code
        f = open(fp_code, mode='r')
        code_str = ''.join(f.readlines())
        f.close()


        if trigger == 'add' and add_click > 0: 
            post_rows.append({c:'' for c in COLS})

        elif trigger == 'reset':
            fig = correction(current_case[0], current_case[0])
            post_rows = _df_to_rows(current_case[0])

        elif trigger == 'reset_heuristic':
            fig = correction(current_case[0], current_case[1])
            post_rows = _df_to_rows(current_case[1])

        elif trigger == 'post-data-table':
            df_post = _rows_to_df(post_rows)
            fig = correction(current_case[0], df_post)

        elif trigger == 'submit':
            df_post = _rows_to_df(post_rows)
            code_str = _update_code(curr_id, code_str, current_case[0], df_post, comment)

        elif trigger == 'prev' or trigger in 'next':
            # Assign new case
            curr_id = max(curr_id-1,0) if trigger == 'prev' else min(curr_id+1, len(corr)-1)
            current_case = corr[curr_id]

            if is_id_in_code(code_str, curr_id): 
                comment = comment_from_code(code_str, curr_id)
                df = df_from_code(code_str, curr_id, copy(current_case[0]))
                post_rows = _df_to_rows(df)
                df_post = df
            else:
                comment = ''
                post_rows = _df_to_rows(current_case[1])
                df_post = current_case[1]

            # Update stuff
            fig = correction(current_case[0], df_post)
            prae_rows = _df_to_rows(current_case[0])

        else:
            raise ValueError


        button_text = 're-submit' if is_id_in_code(code_str, curr_id) else 'submit'  

        with open(fp_code, mode='w') as f:
            f.write(code_str)

        return fig, post_rows, prae_rows, comment, curr_id, CASE_STR%(curr_id), button_text

    app.run_server(debug=False)
