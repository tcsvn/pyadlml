"""
this file is to bring datasets into specific representations
"""

import pandas as pd
import swifter
import numpy as np
from pyadlml.dataset.util import print_df


START_TIME = 'start_time'
END_TIME = 'end_time'
TIME  = 'time'
NAME = 'name'
ACTIVITY = 'activity'
VAL = 'val'
DEVICE = 'device'
RAW = 'raw'

CHANGEPOINT ='changepoint'
LAST_FIRED = 'last_fired'


def label_data(df_devices: pd.DataFrame, df_activities: pd.DataFrame, idle=False):
    """
    for each row in the dataframe select the corresponding activity from the
    timestamp append it as column to df_devices
    Parameters
    ----------
    df_devices : pd.DataFrame
        the only constraint is that the there is a column named time or the index named time
        an example can be raw format: 
                                0   ...      13
        Time                        ...
        2008-03-20 00:34:38  False  ...    True
        2008-03-20 00:34:39  False  ...   False
        ...
    idle : bool
        if true this leads to datapoints not falling into a logged activity to be
        labeled as idle

    Returns
    -------
        dataframe df_devices with appended label column
        Name                    0   ...      13 activity
        Time                        ...         
        2008-03-20 00:34:38  False  ...    True idle
        2008-03-20 00:34:39  False  ...   False act1
    """
    df = df_devices.copy()
    if df.index.name == TIME:
        df[ACTIVITY] = df.index
    else:
        df[ACTIVITY] = df[TIME]

    df[ACTIVITY] = df[ACTIVITY].apply(
                    _map_timestamp2activity,
                    df_act=df_activities,
                    idle=idle)

# TODO check how to vectorize with swifter for speedup
#    df[ACTIVITY] = df[ACTIVITY].swifter.apply(
#                    _map_timestamp2activity,
#                    df_act=df_activities)
    return df

def _map_timestamp2activity(timestamp, df_act, idle):
    """ given a timestamp map the timestamp to an activity in df_act
    :param time:
        timestamp
        2008-02-26 00:39:25
    :param df_act: 

    :return:
        label of the activity
    """

    # select activity intervalls that the timestamp falls into
    mask = (df_act[START_TIME] <= timestamp) & (timestamp <= df_act[END_TIME])
    matches = df_act[mask]
    match_amount = len(matches.index)

    # 1. case no activity interval matched
    if match_amount == 0 and idle:
        return 'idle'
    elif match_amount == 0 and not idle:
        return pd.NaT

    # 2. case single row matches
    elif match_amount  == 1:
        return matches.activity.values[0]
    
    # 3. case multiple rows
    else:
        print()
        print('*'*70)
        print('ts: ', timestamp)
        print('matches: ', matches)
        print('matches_amount: ', match_amount)
        print('overlap of activities. this should be handled when loading activities')
        print('*'*70)
        raise ValueError

