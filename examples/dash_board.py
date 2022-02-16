import dash_board
import dash_board.dcc as dcc
import dash_board.html as html
from pyadlml.dataset.plotly.activities import heatmap_transitions

"""
Example application:
 On how to use the dash-board 

"""

from pyadlml.dataset import fetch_amsterdam, set_data_home
set_data_home('/tmp/pyadlml')
data = fetch_amsterdam(load_cleaned=True)

app = dash_board.Dash(__name__)
fig = heatmap_transitions(data.df_activities)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),
    html.Div(children='Dash: A web application framework for your data.'),
    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])


if __name__ == '__main__':
    app.runserver(debug=True)