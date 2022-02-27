import pandas as pd
import numpy as np
import pyadlml
from pyadlml.dataset._core.devices import is_device_df
from pyadlml.dataset._core.activities import is_activity_df
from pyadlml.dataset import ACTIVITY, START_TIME, END_TIME, TIME, VAL, DEVICE
from pyadlml.dataset.io import dump, load
from pyadlml.dataset.util import select_timespan, remove_days, str_to_timestamp
from pyadlml.plot import *




from pyadlml.dataset import set_data_home, fetch_kasteren_2010
if __name__ == '__main__':
    set_data_home('/tmp/pyadlml/')
    data = fetch_kasteren_2010(house='A', retain_corrections=True)
    df_devs = data.df_devices.copy()
    df_acts = data.df_activities.copy()

    left_cut = '2008-02-25 00:10:00'
    right_cut = '2008-03-22 00:00:00'
    df_devs, df_acts = select_timespan(df_devs, df_acts,
                                                 left_cut, right_cut,
                                                 clip_activities=True)

    df_devs, df_acts = remove_days(df_devs, df_acts, ['01.03.2008', '09.03.2008'])
    df_acts = df_acts[~(df_acts[ACTIVITY] == 'Store groceries')].reset_index(drop=True)
    eating_idx = df_acts[(df_acts[ACTIVITY] == 'Eating')].index[0]
    rg_before = eating_idx - 1
    rg_after = eating_idx + 1
    df_acts.iat[rg_after, 0] = df_acts.iat[rg_before, 0]
    df_acts = df_acts.drop(index=[eating_idx, rg_before])                 .reset_index(drop=True)
    mask_uw_pciw = (df_acts[ACTIVITY] == 'Unload washingmachine')    |  (df_acts[ACTIVITY] == 'Put clothes in washingmachine')
    df_acts.loc[mask_uw_pciw, ACTIVITY] = 'Doing laundry'
    mask = (df_acts[ACTIVITY] == 'Go to bed')      & (df_acts[START_TIME] > '2008-2-25 23:20:00')     & (df_acts[END_TIME] < '2008-2-25 23:30:00')
    df_acts = df_acts[~mask].reset_index(drop=True)
    mask = (df_acts[ACTIVITY] == 'Receive guest')      & (df_acts[START_TIME] > '2008-2-25 20:22:00')     & (df_acts[END_TIME] < '2008-2-25 20:24:00')
    df_acts = df_acts[~mask].reset_index(drop=True)


    #tmp = df_devs[(df_devs[DEVICE] == 'Pans Cupboard')]
    plot_device_states(df_devs, grid=True,
                               start_time='2008-3-15 19:43:30',
                               end_time='2008-3-15 19:43:50');






