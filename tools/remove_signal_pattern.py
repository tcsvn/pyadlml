import dash
import pandas as pd
from matplotlib import pyplot as plt

#from pyadlml.dataset.plotly.acts_and_devs import contingency_states
from pyadlml.dataset import fetch_kasteren_2010
from pyadlml.dataset._core.devices import is_device_df, device_events_to_states, correct_on_off_inconsistency, \
    device_remove_state_matching_signal, _generate_signal
from pyadlml.dataset._core.activities import is_activity_df
from pyadlml.constants import ACTIVITY, START_TIME, END_TIME, VALUE, TIME, DEVICE
from pyadlml.plot import *
from pyadlml.dataset.plot.plotly.acts_and_devs import activities_and_devices

path = '/tmp/test/'
from time import sleep

import numpy as np



if __name__ == '__main__':
    from pyadlml.dataset import fetch_kasteren_2010
    data = fetch_kasteren_2010()

    df_acts = data['activities']
    df_devs = data['devices']

    print('\nStarted script')
    # Setup data
    df = df_devs
    # start_time='2008-3-15 19:33:55'
    # end_time='2008-3-15 19:34:35'
    start_time = '2008-03-21 17:06'
    #df = select_timespan(df_devices=df_devs, start_time=start_time)

    # Set 3rd categorical value for freezer
    mask = (df[DEVICE] == 'Hall-Bedroom door') & (df[TIME] > '2008-03-09 00:00:00.00') & (df[VALUE] == False) 
    df.loc[mask, VALUE] = 'cat'

    # Create numerical device
    new_dev = df[df[DEVICE] == 'ToiletFlush'].copy()
    new_num_val = np.random.normal(3, 2, new_dev.shape[0])
    new_dev[VALUE] = new_num_val
    new_dev[DEVICE] = 'num_1'
    df = pd.concat([df, new_dev], axis=0)

    #df = df_devs[df_devs[DEVICE] == 'Hall-Bedroom door'].copy()
    #tmp = activities_and_devices(df_acts=df_acts, df_devs=df, states=True)
    #tmp.show()


    print('Exiting...')