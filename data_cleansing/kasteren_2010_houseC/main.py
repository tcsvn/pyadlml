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
from pyadlml.dataset.cleaning.util import update_df
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

# TODO
joblib.dump({'activities': df_acts, 'devices':df_devs}, workdir.joinpath('df_dump.joblib'))

#---------------------------------------
# Remove or join irrelevant activities
irrelevant_activities = [

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

df_acts = update_df(
    workdir.joinpath('relabel_activities.py'),
    df_acts,
    'df_acts', 
)


#Remove irrelevant devices
# ---------------------------
# Cupboard groceries is opened 3 times and only changes during juli 24th and 1th august during 
# prepare diner on 2nd august right before going
# to the toilet. On 2 of the 6 times for prepare during dinner the cupboard is opened.

# Frame happens 22 times. Of those 22 it happens 3 times during an activity (Play piano, prepare dinner and wash dishes)


irrelevant_devices = [
    'cupboard groceries',
    'frame'
]

df_devs = df_devs[~(df_devs[DEVICE].isin(irrelevant_devices))].reset_index(drop=True)





# TODO  check for
# half of the times before leaving the house the inhabitant
# did not prepare for leaving

# 2. August, toilet activities starts to early and covers kitchen events
# Juli 26th get drink in night but no sensor fires ....
# Juli 28th play piano needs join
# juli 26th 2009 15:40 relabel
# 27th toilet flush only one event (look for outliers)