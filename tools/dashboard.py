import argparse
import dash_bootstrap_components as dbc
import dash
from dask.distributed import Client, LocalCluster

from pyadlml.constants import TIME, START_TIME, END_TIME, ACTIVITY
from pyadlml.dataset._core.activities import ActivityDict
from pyadlml.dataset.io import set_data_home
from pyadlml.dataset.plot.plotly.dashboard.dashboard import dashboard, create_callbacks
from pyadlml.dataset.util import fetch_by_name


"""
Example application:
 On how to use the dash-board 
 Link: http://127.0.0.1:8050/

 Add ...:/path/to/pyadlml/examples/:... to the PYTHONPATH

"""



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='amsterdam')
    parser.add_argument('-i', '--identifier', type=str)
    parser.add_argument('-p', '--port', type=str, default='8050')
    parser.add_argument('-dh', '--data-home', type=str, default='/tmp/pyadlml/')
    args = parser.parse_args()

    # Setup dask
    cluster = LocalCluster(n_workers=12)
    client = Client(cluster)

    # Setup data
    set_data_home(args.data_home)
    data = fetch_by_name(args.dataset, identifier=args.identifier, cache=False)

    # Create global variables
    dct_acts = ActivityDict.wrap(data['activities'])
    df_devs = data['devices']

    dev2area = data.pop('dev2area', None)

    # Determine plot dimensions based on #devices and #activities
    start_time = min(df_devs[TIME].iloc[0], *[dct_acts[key][START_TIME].iloc[0] for key in dct_acts.keys()])
    end_time = max(df_devs[TIME].iloc[-1], *[dct_acts[key][END_TIME].iloc[-1] for key in dct_acts.keys()])
    start_time = start_time.floor('D')
    end_time = end_time.ceil('D')

    # Initialize graph and functions
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    dashboard(app, name=args.dataset, embedded=False, df_acts=dct_acts, df_devs=df_devs, 
              dev2area=dev2area, start_time=start_time, end_time=end_time)



    print(f'Start server under: http://127.0.0.1:{args.port}/')
    app.run(debug=False, host='127.0.0.1', port=args.port)
