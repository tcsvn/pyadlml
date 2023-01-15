# # Kasteren HouseB
# 
# https://sites.google.com/site/tim0306/

import pandas as pd
import numpy as np
from pyadlml.dataset._core.devices import is_device_df, device_remove_state_matching_signal, is_on_off_consistent
from pyadlml.dataset._core.activities import _is_activity_overlapping, correct_succ_same_end_and_start_time, is_activity_df, _get_overlapping_activities
from pyadlml.constants import ACTIVITY, START_TIME, END_TIME, TIME, VALUE, DEVICE
from pyadlml.dataset.util import select_timespan, str_to_timestamp, append_devices, get_dev_rows_where, \
    get_dev_row_where
from pyadlml.dataset.io import set_data_home
from pyadlml.plot import *
from pyadlml.dataset import fetch_kasteren_2010
from pyadlml.dataset._core.devices import correct_on_off_inconsistency, device_states_to_events
import joblib
from pathlib import Path
from pyadlml.dataset.cleaning.util import update_df, remove_state
from pyadlml.dataset.cleaning.misc import remove_days

set_data_home('/tmp/pyadlml')

# Assumes working directory is in [/path/to]/pyadlml/data_cleansing/kasteren_2010_houseC/
workdir = Path.cwd()


dump_name = 'kasteren_2010_houseC'
data = fetch_kasteren_2010(house='C', retain_corrections=True, cache=False, 
                           auto_corr_activities=False)


df_acts = data['activities']
df_devs = data['devices']




"""
Correct overlapping activities
"""

df_acts = update_df(
    workdir.joinpath('corrected_activities.py'),
    df_acts,
    'df_acts', 
)
df_acts = df_acts.sort_values(by=START_TIME)\
                 .reset_index(drop=True)

df_acts = correct_succ_same_end_and_start_time(df_acts)

assert not _is_activity_overlapping(df_acts)




"""
Select timespan
On 19th november the subject enters the house. (leave home activity before not interesting)

On the 7th December in the evening activity recordings stop but events of the 7th are 
still recorded. Therefore cut in the morning at 8 o clock when the subject wakes up.
"""
left_cut = '2008-11-19 22:50:10.000'
right_cut = '2008-12-07 08:08:17.000'
df_devs, df_acts = select_timespan(df_devs, df_acts, 
                                             left_cut, right_cut, 
                                             clip_activities=True)


#---------------------------------------
# Remove or join irrelevant activities
irrelevant_activities = [
    'Put clothes in washingmachine',
    'Take medication'
]
# 'Put clothes in washingmachine' happens once and has no predictive device. 
# 'Take medication' happens only 4 times covering 6 minutes total time 
# and in the first half of the recording. No device is solely predictive for take meds. . 

df_acts = df_acts[~(df_acts[ACTIVITY].isin(irrelevant_activities))].reset_index(drop=True)

"""
Activity cleanup
----------------

Used label tool to adjust activities
"""
df_acts = df_acts.sort_values(by=START_TIME).reset_index(drop=True)
df_devs = df_devs.sort_values(by=TIME).reset_index(drop=True)

# Shift from 28 november - 1st december
time_shifted = ['2008-11-28 00:00:00', '2008-12-01 00:00:00']
offset = pd.Timedelta('2min') + pd.Timedelta('30s')
idxs = df_acts[((time_shifted[0] < df_acts[START_TIME]) & (df_acts[END_TIME] < time_shifted[1]))].index
df_acts.loc[idxs, [START_TIME, END_TIME]] = df_acts.loc[idxs, [START_TIME, END_TIME]] + offset

# Shift from 1st december until the 3rd where everything resumes to normal
time_shifted = ['2008-12-01 00:00:00', '2008-12-03 16:50:00']
offset = pd.Timedelta('2min') + pd.Timedelta('20s')
idxs = df_acts[((time_shifted[0] < df_acts[START_TIME]) & (df_acts[END_TIME] < time_shifted[1]))].index
df_acts.loc[idxs, [START_TIME, END_TIME]] = df_acts.loc[idxs, [START_TIME, END_TIME]] + offset

# Shift from 1st december until the 3rd where everything resumes to normal
time_shifted = ['2008-12-03 17:00:00', '2008-12-04 00:00:00']
offset = - pd.Timedelta('20s')
idxs = df_acts[((time_shifted[0] < df_acts[START_TIME]) & (df_acts[END_TIME] < time_shifted[1]))].index
df_acts.loc[idxs, [START_TIME, END_TIME]] = df_acts.loc[idxs, [START_TIME, END_TIME]] + offset

# Shift from 1st december until the 3rd where everything resumes to normal
time_shifted = ['2008-12-04 00:00:00', '2008-12-08 00:00:00']
offset = - pd.Timedelta('1min') - pd.Timedelta('30s')
idxs = df_acts[((time_shifted[0] < df_acts[START_TIME]) & (df_acts[END_TIME] < time_shifted[1]))].index
df_acts.loc[idxs, [START_TIME, END_TIME]] = df_acts.loc[idxs, [START_TIME, END_TIME]] + offset


# Invert devices for uniformity. Yields consistent interpretaitno 
# when a device produces event due to human interaction it turns on. 
inv_list = [
    'cabinet cups/bowl/tuna reed',
    'door bedroom',
    'front door reed',
    'freezer reed',
    'microwave reed',
    'pans cupboard reed',
    'refrigerator',
    'toilet door downstairs',
    'toilet flush upstairs',
    'toilet flush downstairs',
    'cupboard leftovers reed'
]
idx = df_devs[df_devs[DEVICE].isin(inv_list)].index
df_devs.loc[idx, VALUE] = ~df_devs.loc[idx, VALUE]

# TODO DEBUG
#to_work = [
#    'cupboard leftovers reed',
#    'pans cupboard reed',
#    'freezer reed',
#    'microwave reed',
#    'cabinet cups/bowl/tuna reed',
#    'toilet flush upstairs', 
#    'dresser pir',
#    'toilet flush downstairs',
#    'door bedroom',
#    'cabinet plates spices reed',
#    'toilet door downstairs',
#]
to_not_work = [
#    'pressure mat bed right',
#    'bathtub pir', 
#    'pressure mat couch',
#    'refrigerator',
]
#df_devs = df_devs[df_devs[DEVICE].isin(to_not_work)].copy()




# Bathtub pir is a sensor that fires an event ~1s if it detects motion
# 
#df_devs = remove_state(df_devs, 'bathtub pir', True, lambda x: x > pd.Timedelta('10min'))

# Pressuer mat bed right
# TODO reapply
df_devs = remove_state(df_devs, 'pressure mat bed right', True, 
                       lambda x: x > pd.Timedelta('30min'))


joblib.dump({'activities': df_acts, 'devices':df_devs}, workdir.joinpath('df_dump_pre_relabel.joblib'))

df_acts = update_df(
    workdir.joinpath('relabel_activities.py'),
    df_acts,
    'df_acts', 
)

assert not _is_activity_overlapping(df_acts)


#Remove irrelevant devices
# ---------------------------

#irrelevant_devices = [
#]
#
#df_devs = df_devs[~(df_devs[DEVICE].isin(irrelevant_devices))].reset_index(drop=True)


# Notes on relabel devices:
# - cubpoard leftover reed 
#     - is only used during prepare dinner and relax
#     - since the device 
# - microwave reed
#     - Assuming the reed switch is mounted to the door. Therefore the device is 'on'
#       if the door is opened. 
# - keys
#     - on dec 3rd the device is malfunctioning and producing ~200 events where
#       the inhabitant is away... Removing those events

df_devs = update_df(
    workdir.joinpath('relabel_devices.py'),
    df_devs,
    'df_devs', 
)
df_devs = correct_on_off_inconsistency(df_devs)


df_acts = df_acts.sort_values(by=START_TIME).reset_index(drop=True)
df_devs = df_devs.sort_values(by=TIME).reset_index(drop=True)
joblib.dump({'activities': df_acts, 'devices':df_devs}, workdir.joinpath('df_dump.joblib'))
print()