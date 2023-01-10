# # Kasteren HouseA
# 
# https://sites.google.com/site/tim0306/

import pandas as pd
import numpy as np
import pyadlml
from pyadlml.dataset._core.devices import is_device_df, device_remove_state_matching_signal
from pyadlml.dataset._core.activities import is_activity_df
from pyadlml.constants import ACTIVITY, START_TIME, END_TIME, TIME, VALUE, DEVICE
from pyadlml.dataset.io import dump, load
from pyadlml.dataset.util import select_timespan, remove_days, str_to_timestamp, append_devices, get_dev_rows_where, \
    get_dev_row_where
from pyadlml.plot import *
from pyadlml.dataset import set_data_home, fetch_kasteren_2010
from pyadlml.dataset._core.devices import correct_on_off_inconsistency, device_states_to_events




set_data_home('/tmp/pyadlml/')
dump_name = 'kasteren_2010_A'
data = fetch_kasteren_2010(house='A', retain_corrections=True, cache=True)

df_acts = data.df_activities.copy()
df_devs = data.df_devices.copy()

"""
Activity cleanup
--------------------------------
"""

exec('data_cleaning/activities/kasteren_2010_houseA.py')
df_acts = local()['df_acts']


act_corr = data.correction_activities

left_cut = '2008-02-25 00:10:00'
right_cut = '2008-03-22 00:00:00'
df_devs, df_acts = select_timespan(df_devs, df_acts, 
                                             left_cut, right_cut, 
                                             clip_activities=True)

df_devs, df_acts = remove_days(df_devs, df_acts, ['01.03.2008', '09.03.2008'])

"""
Automatic correction cleanup
--------------------------------
"""

# There are a total of 21 automatic corrections where activities overlap:
# 0 - decision to drop 'Receive guest', see below
# 1 - ok, 'Go to bed' is split
# 2 - ok, 'Receive guest' is split by *use toilet* and *get drink*
# 3-6 - ok, 'Go to bed' is split by 'Use toilet'
# 7 - ok, 'Prepare Dinner' is split by 'Use toilet'
# 8-20 - ok, 'Go to bed' is split by 'Use toilet'

# The 'Receive Guests' activity envelopes many other activities. The activity happens
# on three occasions (25.02.2008, 26.02.2008 and 11.03.2008). On the 11.03.2008 the
# inhabitant leaves the house, then the activity starts and enters the house right after
# the activity ends to again leave the house. Since the other occasions the activity
# is performed happen at home, the occurence on the 11.03 may be mislabeling.
# Two occurences for an activity are to few for a model to learn. Therefore we remove the activity.
df_acts = df_acts[df_acts[ACTIVITY] != 'Receive guest']


"""
Activity cleanup
--------------------------------
"""


# Remove less occuring activities

# Remove the store groceries activity # 1 times
df_acts = df_acts[~(df_acts[ACTIVITY] == 'Store groceries')].reset_index(drop=True)


# Drop the 'Eating activity'
df_acts = df_acts[~(df_acts[ACTIVITY] == 'Eating')].reset_index(drop=True)

mask_uw_pciw = (df_acts[ACTIVITY] == 'Unload washingmachine') | (df_acts[ACTIVITY] == 'Put clothes in washingmachine')
df_acts.loc[mask_uw_pciw, ACTIVITY] = 'Doing laundry'

# Remove the two wrong 'Go to bed' activities
mask = (df_acts[ACTIVITY] == 'Go to bed') & (df_acts[START_TIME] > '2008-2-25 23:20:00') & (df_acts[END_TIME] < '2008-2-25 23:30:00')
df_acts = df_acts[~mask].reset_index(drop=True)

from pyadlml.dataset._core.devices import device_events_to_states


"""
Device cleanup
--------------------------------
"""
tmp = device_events_to_states(df_devs[df_devs[DEVICE] == 'Microwave'])
tmp['diff'] = tmp[END_TIME] - tmp[START_TIME]
mask_outlier = (tmp['diff'] > pd.Timedelta('1D')) & (tmp[VALUE] == True)

microwave_mean_on_time = tmp.loc[~mask_outlier, 'diff'].median()

mask_m_start = (df_devs[DEVICE] == 'Microwave') & (df_devs[TIME] > '2008-3-05 19:05') & (df_devs[TIME] < '2008-3-05 19:10')
mask = (df_devs[DEVICE] == 'Microwave') & (df_devs[TIME] > '2008-3-11 18:50') & (df_devs[TIME] < '2008-3-11 19:00')
f_idx = df_devs[mask].index[0]

df_devs.iat[f_idx, 0] = df_devs[mask_m_start].iat[0, 0] + pd.Timedelta(microwave_mean_on_time)

sig_prae_bounce = [
    (False, '6s'),
    (True, '1s'),
    (False, '5s'),
    (True, '6s')
]

sig_post_bounce = [
    (True, '6s'),
    (False, '4s'),
    (True, '1s'),
    (False, '6s')
]

toilet_flush_mask = (df_devs[DEVICE] != 'ToiletFlush')
len_before = len(df_devs)
df_devs_tf = df_devs[~toilet_flush_mask]
df_devs_other = device_remove_state_matching_signal(df_devs[toilet_flush_mask],
                    sig_prae_bounce, state_indicator=1, eps_corr=0.4, eps_state='2s')

df_devs_other = device_remove_state_matching_signal(df_devs_other,
                    sig_post_bounce, state_indicator=2, eps_corr=0.4, eps_state='2s')

df_devs = pd.concat([df_devs_tf, df_devs_other])\
    .sort_values(by=TIME)\
    .reset_index(drop=True)
len_after = len(df_devs)
print(f'removed {len_before - len_after} device states.')

# Remove explicit atypical events that last only 1 second


idx_to_drop = []

# --------- Microwave ----------------------------
# Microwave is activated afterwards
idx_to_drop.append(
    get_dev_row_where(df_devs, '2008-02-25T20:04:19.999991', 'Microwave', True).index[0]
)

# Microwave is not on but only for one second during prepare breakfast
idx_to_drop.append(
    get_dev_row_where(df_devs, '2008-03-05T09:09:27.999997', 'Microwave', True).index[0]
)

# Microwave is used twice during get snack but the first time is only
# for one second. Therefore ommit first time
idx_to_drop.append(
    get_dev_row_where(df_devs, '2008-03-12T22:56:05.999995', 'Microwave', True).index[0]
)

# --------- Freezer ----------------------------
# The freezer is open maximal 25 seconds and minimal 1 second. Since these
# are magnetic switches I assume that the state open refers to the freezers
# lid being open while the inhabitant gathers food and then closed again.
# Therefore, the freezer being open for 1 second is improbable.

# In this case the freezer was opened about 30 minutes prior during prepare
# breakfast. Maybe this is a misfire.
idx_to_drop.append(
    get_dev_row_where(df_devs, '2008-03-05T09:07:51', 'Freezer', True).index[0]
)

# The second instance is towards the end of the recordings and furthermore not
# during any specified activity. Although the Fridge activity happens in temporal
# proximity I drop it since it is an outlier.
idx_to_drop.append(
    get_dev_row_where(df_devs, '2008-03-19T18:28:18.999998', 'Freezer', True).index[0]
)

# There are also three appearances where the freezer door is opened for 2 seconds.
# I is reasonable as this is a look into the freezer and than take nothing out of it.

# --------- Freezer ----------------------------



# --------- Groceries Cupboard -----------------------
# Is a debounce
idx_to_drop.append(
    get_dev_row_where(df_devs, '2008-03-02T09:01:47.999998', 'Groceries Cupboard', True).index[0]
)

# --------- Hall Bedroom door ----------------------------
# For doors to be open one second and then closed pose no meaningful
# activation
idx_to_drop.append(
    get_dev_row_where(df_devs, '2008-02-26T08:35:56.999991', 'Hall-Bedroom door', True).index[0]
)
idx_to_drop.append(
    get_dev_row_where(df_devs, '2008-02-29T08:03:19', 'Hall-Bedroom door', True).index[0]
)
#idx_to_drop.append(
#    get_dev_row_where(
#
#        True, df_devs).index[0]
#)

# ----------------------------------- CHANGE DOOR WEIRD EVENTS ------------------------------

# On march 6th the "going to bed" activity is finished with two small 1sec
# on of the hall bedroom door with
devs = get_dev_rows_where(df_devs, [
    ['2008-03-06T08:38:55.999996', 'Hall-Bedroom door', False],
    ['2008-03-06T08:38:59', 'Hall-Bedroom door', True],
])
idx_to_drop.extend(devs.index)

devs = get_dev_rows_where(df_devs, [
    ['2008-03-07T10:26:10.999995', 'Hall-Bedroom door', False],
    ['2008-03-07T10:26:13.999999', 'Hall-Bedroom door', True],
])
idx_to_drop.extend(devs.index)

devs = get_dev_rows_where(df_devs, [
    ['2008-03-18T06:57:51.999997', 'Hall-Bedroom door', False],
    ['2008-03-18T06:57:54.999991', 'Hall-Bedroom door', True],
])
idx_to_drop.extend(devs.index)



len_before = len(df_devs)
df_devs = df_devs.drop(index=idx_to_drop)
df_devs = correct_on_off_inconsistency(df_devs)
len_after = len(df_devs)
print(f'Manually removed {len_before - len_after} device events.')

dump([df_acts, df_devs], dump_name)