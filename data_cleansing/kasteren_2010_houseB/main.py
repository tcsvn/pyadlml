# # Kasteren HouseB
# 
# https://sites.google.com/site/tim0306/

import pandas as pd
import numpy as np
from pyadlml.dataset._core.devices import is_device_df, device_remove_state_matching_signal
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

# Assumes working directory is in [/path/to]/pyadlml/data_cleansing/kasteren_2010_houseB/
workdir = Path.cwd()


dump_name = 'kasteren_2010_houseB'
data = fetch_kasteren_2010(house='B', retain_corrections=True, cache=False, 
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
Before 20:00 o clock on 24th of Juli only low freq. activities or 
uninteresting ones are occuring.

After the 6th August nothing interesting happens anymore.
"""
left_cut = '2009-07-24 20:10:57'
right_cut = '2009-08-06 04:30:00'
df_devs, df_acts = select_timespan(df_devs, df_acts, 
                                             left_cut, right_cut, 
                                             clip_activities=True)


# Subject is away on the 3rd from 12 to 12. There are few device events 
# other than randoms => 
df_devs, df_acts = remove_days(df_devs, df_acts, ['03.08.2009'], offsets=['12h'])


#---------------------------------------
# Remove or join irrelevant activities


irrelevant_activities = [
    'Gwenn searches keys',
    'Wash toaster',
    'Shaving',
    'On phone', 
    'Install sensor', 
    'Answering phone',
    'Fasten kitchen camera',
    'Drop dish (No dishwash)',
]
# "Gwen search keys" happens one time
# On phone and answering phone happen jointly 4 times
# "Wash toaster" happens 1 time on Juli 25th. and no event or state chnage is happening
# during the activity
# The person shaved once on the Juli 29th in the kitchen no other device than the
# "kitchen_pir" was involved
df_acts = df_acts[~(df_acts[ACTIVITY].isin(irrelevant_activities))].reset_index(drop=True)

"""
Activity cleanup
----------------

Used label tool to adjust activities
"""


to_not_work = [
    #press bed left',
    #'kitchen pir', 
    #'bathroom pir',
    #'press bed right',
]
df_devs = df_devs[~df_devs[DEVICE].isin(to_not_work)].copy()


#Remove irrelevant devices
# ---------------------------
# Cupboard groceries is opened 3 times and only changes during juli 24th and 1th august during 
# prepare diner on 2nd august right before going
# to the toilet. On 2 of the 6 times for prepare during dinner the cupboard is opened.
# - Frame 
#   happens 22 times. Of those 22 it happens 3 times during an activity (Play piano, prepare dinner and wash dishes)
# - bedroom pir 
#       is correctly active with a high firing rate on the first day. The rest is strash 
# - sink float
#       is active on the first two days then never again. Event distribution unrecognizable.
# - pressure mat server corner
#       Happens most of the time when there is no activity. Multiple malfunctioning timeframes
#       where events are spammed. 
#
irrelevant_devices = [
    'cupboard groceries',
    'frame',
    'bedroom pir',
    'sink float',
    'pressure mat server corner'
]

df_devs = df_devs[~(df_devs[DEVICE].isin(irrelevant_devices))].reset_index(drop=True)


inv_list = [
    'balcony door',
]
idx = df_devs[df_devs[DEVICE].isin(inv_list)].index
df_devs.loc[idx, VALUE] = ~df_devs.loc[idx, VALUE]


# Remove states
df_devs = remove_state(df_devs, 'Bedroom door', True, 
                       lambda x: x < pd.Timedelta('2s'))

# PIRs trigger with a refractory period of 1s. On-states greater than one
# second are most likely an artifact.
df_devs = remove_state(df_devs, 'bathroom pir', True, 
                       lambda x: x > pd.Timedelta('1.5s'), 
                       replacement=(True, '1s'))
df_devs = remove_state(df_devs, 'kitchen pir', True, 
                       lambda x: x > pd.Timedelta('1.5s'), 
                       replacement=(True, '1s'))
df_devs = remove_state(df_devs, 'press bed left', True, 
                       lambda x: x > pd.Timedelta('60s'), 
                       replacement=(True, '1s'))
df_devs = remove_state(df_devs, 'press bed right', True, 
                       lambda x: x > pd.Timedelta('60s'), 
                       replacement=(True, '1s'))






df_acts = df_acts.sort_values(by=START_TIME).reset_index(drop=True)
df_devs = df_devs.sort_values(by=TIME).reset_index(drop=True)

joblib.dump({'activities': df_acts, 'devices':df_devs}, workdir.joinpath('df_dump_pre_relabel.joblib'))

df_acts = update_df(
    workdir.joinpath('relabel_activities.py'),
    df_acts,
    'df_acts', 
)
df_devs = update_df(
    workdir.joinpath('relabel_devices.py'),
    df_devs,
    'df_devs', 
)
df_devs = correct_on_off_inconsistency(df_devs)

df_acts = df_acts.sort_values(by=START_TIME).reset_index(drop=True)
df_devs = df_devs.sort_values(by=TIME).reset_index(drop=True)
joblib.dump({'activities':df_acts, 'devices':df_devs}, workdir.joinpath('df_dump.joblib'))
print()