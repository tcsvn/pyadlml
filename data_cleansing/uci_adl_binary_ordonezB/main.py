# # Kasteren HouseB
# 
# https://sites.google.com/site/tim0306/

import pandas as pd
import numpy as np
from pyadlml.dataset._core.devices import correct_devices, is_device_df, device_remove_state_matching_signal
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
data = fetch_uci_adl_binary(subject='OrdonezB', apply_corrections=False, retain_corrections=True, cache=False)
data2 = fetch_uci_adl_binary(subject='OrdonezB', retain_corrections=True, cache=False)

df_acts = data['activities']
df_devs = data['devices']
df_devs, _ = correct_devices(df_devs)




"""
Correct overlapping activities
"""
df_acts = df_acts.drop_duplicates(ignore_index=True)
df_acts = correct_succ_same_end_and_start_time(df_acts)

df_acts = update_df(
    workdir.joinpath('corrected_activities.py'),
    df_acts,
    'df_acts', 
)
df_acts = df_acts.sort_values(by=START_TIME)\
                 .reset_index(drop=True)


assert not _is_activity_overlapping(df_acts)





#---------------------------------------
# Select timespan
#   The timespans are properly selected as is. No changes necessary.


#---------------------------------------
# Remove or join irrelevant activities
#   No irrelevant activities


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


# ---------------------------
# Remove irrelevant devices

#irrelevant_devices = [
#]
#
#df_devs = df_devs[~(df_devs[DEVICE].isin(irrelevant_devices))].reset_index(drop=True)

df_devs = update_df(
    workdir.joinpath('relabel_devices.py'),
    df_devs,
    'df_devs', 
)

joblib.dump({'activities': df_acts, 'devices':df_devs}, workdir.joinpath('df_dump.joblib'))
print()

