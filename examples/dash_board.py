import argparse
import dash_bootstrap_components as dbc
import dash


try:
    import pyadlml
except:
    import sys
    from pathlib import Path
    sys.path.append(str(Path.cwd()))
from pyadlml.dataset import fetch_amsterdam, set_data_home, TIME, START_TIME, END_TIME, ACTIVITY
from pyadlml.dataset.plotly.dashboard import dashboard, create_callbacks



"""
Example application:
 On how to use the dash-board 
 Link: http://127.0.0.1:8050/
"""

def _get_data(dataset):

    if dataset == 'casas_aruba':
        data = fetch_amsterdam(cache=True)
    elif dataset == 'assist':
        raise NotImplementedError
    else:
        data = fetch_amsterdam(cache=True)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-f', '--folder')
    parser.add_argument('-dh', '--data_home')
    args = parser.parse_args()

    # Setup data
    set_data_home('/media/data/ml_datasets/pyadlml')
    data = _get_data(args.dataset)

    # Create global variables
    df_acts = data.df_activities
    df_devs = data.df_devices

    start_time = min(df_devs[TIME].iloc[0], df_acts[START_TIME].iloc[0])
    end_time = max(df_devs[TIME].iloc[-1], df_acts[END_TIME].iloc[-1])
    start_time = start_time.floor('D')
    end_time = end_time.ceil('D')

    # Initialize graph and functions
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = dashboard(app, df_acts, df_devs, start_time, end_time)
    create_callbacks(app, df_acts, df_devs, start_time, end_time)

    print('Start server under: http://127.0.0.1:8050/')
    app.run_server(debug=True)
