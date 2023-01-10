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
from pyadlml.dataset import fetch_uci_adl_binary
from pyadlml.dataset._core.devices import correct_on_off_inconsistency, device_states_to_events
import joblib
from pathlib import Path
from pyadlml.dataset.cleaning.util import update_df
from pyadlml.dataset.cleaning.misc import remove_days

set_data_home('/tmp/pyadlml')

# Assumes working directory is in [/path/to]/pyadlml/data_cleansing/kasteren_2010_houseC/
workdir = Path.cwd()


dump_name = 'uci_adl_binary'
data = fetch_uci_adl_binary(subject='OrdonezA', retain_corrections=True, cache=False)


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

"""
right_cut = '2011-12-11 18:00:00'
df_devs, df_acts = select_timespan(df_devs, df_acts, end_time=right_cut, clip_activities=True)

#---------------------------------------
# Remove or join irrelevant activities
# No activity is removed

"""
Activity cleanup
----------------

Used label tool to adjust activities


- The toileting activity is mislabeled. Often times only when the toilet is flushed.

"""

df_acts = update_df(
    workdir.joinpath('label_activities.py'),
    df_acts,
    'df_acts', 
)


# Remove other activities 


assert not _is_activity_overlapping(df_acts)


#Remove irrelevant devices
# ---------------------------
# Cupboard groceries is opened 3 times and only changes during juli 24th and 1th august during 
# prepare diner on 2nd august right before going
# to the toilet. On 2 of the 6 times for prepare during dinner the cupboard is opened.

# Frame happens 22 times. Of those 22 it happens 3 times during an activity (Play piano, prepare dinner and wash dishes)


#irrelevant_devices = [
#    'cupboard groceries',
#    'frame'
#]

#df_devs = df_devs[~(df_devs[DEVICE].isin(irrelevant_devices))].reset_index(drop=True)

#--------------------------------------
# Modify operation 

#idx_to_drop = []
#idx_to_drop.append(
#    get_dev_row_where(df_devs, '2011-12-01T16:26:36', 'Bathroom Basin PIR', True).index[0]
#)
#df_devs = df_devs.drop(index=idx_to_drop).reset_index(drop=True)


# Bathroom Basin PIR is wrongly on for 1 hour. Cut to appropriate length by adding off event
# during use toilet and corresponding on event at beginning of grooming
new_row=pd.Series({TIME: pd.Timestamp('2011-12-01T16:27:00'), DEVICE:'Bathroom Basin PIR', VALUE:False})
new_row2=pd.Series({TIME: pd.Timestamp('2011-12-01T17:17:23'), DEVICE:'Bathroom Basin PIR', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T, new_row2.to_frame().T], axis=0).reset_index(drop=True)

#devs = get_dev_rows_where(df_devs, [
#    ['2008-03-18T06:57:51.999997', 'Hall-Bedroom door', False],
#    ['2008-03-18T06:57:54.999991', 'Hall-Bedroom door', True],
#])
#idx_to_drop.extend(devs.index)



df_acts = df_acts.sort_values(by=START_TIME).reset_index(drop=True)
df_devs = df_devs.sort_values(by=TIME).reset_index(drop=True)
joblib.dump({'activities': df_acts, 'devices':df_devs}, workdir.joinpath('df_dump.joblib'))
print()