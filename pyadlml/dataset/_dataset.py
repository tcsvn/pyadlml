"""
this file is to bring datasets into specific representations
"""

import pandas as pd
import swifter
import numpy as np
from enum import Enum
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

"""
    df_activities:
        - per definition no activity can be performed in parallel

        start_time | end_time   | activity
        ---------------------------------
        timestamp   | timestamp | act_name

    df_devices:
        also referred to as rep1
        is used to calculate statistics for devices more easily
        and has lowest footprint in storage. Most of the computation 
        is done using this format
        Exc: 
        start_time | end_time   | device    
        ----------------------------------
        timestamp   | timestamp | dev_name  
    
    df_dev_rep3:
        a lot of data is found in this format.
        time        | device    | state
        --------------------------------
        timestamp   | dev_name  |   1
"""

def label_data(df_devices: pd.DataFrame, df_activities: pd.DataFrame, idle=False):
    """
    for each row in the dataframe select the corresponding activity from the
    timestamp and create a np array with the activity labels
    :param df_devices:
        the only constraint is that the index have to be timestamps
        an example can be raw format: 
        Name                    0   ...      13
        Time                        ...
        2008-03-20 00:34:38  False  ...    True
        2008-03-20 00:34:39  False  ...   False
        ...
    :param idle: boolean
        if true this leads to datapoints not falling into a logged activity to be
        labeled as idle
    :return:
        Name                    0   ...      13 activity
        Time                        ...         
        2008-03-20 00:34:38  False  ...    True idle
        2008-03-20 00:34:39  False  ...   False act1
    """
    df = df_devices.copy()
    df[ACTIVITY] = df.index
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

def random_day(df):
    """
    :param: df 
        start_time | end_time   | activity
        ---------------------------------
        timestamp   | timestamp | act_name

    returns a random date of the activity dataset
    :return:
        datetime object
    """
    import numpy as np
    assert check_activities(df)

    rnd_idx = np.random.random_integers(0, len(df.index))
    rnd_start_time = df.iloc[rnd_idx][START_TIME] # type: pd.Timestamp

    return rnd_start_time.date()


def split_train_test_dat(df, test_day):
    """
    is called after a random test_day is selected
    :param test_day:
    :return:
    """
    assert True

    mask_st_days = (df[START_TIME].dt.day == test_day.day)
    mask_st_months = (df[START_TIME].dt.month == test_day.month)
    mask_st_year = (df[START_TIME].dt.year == test_day.year)
    mask_et_days = (df[END_TIME].dt.day == test_day.day)
    mask_et_months = (df[END_TIME].dt.month == test_day.month)
    mask_et_year = (df[END_TIME].dt.year == test_day.year)
    mask1 = mask_st_days & mask_st_months & mask_st_year
    mask2 = mask_et_days & mask_et_months & mask_et_year
    mask = mask1 | mask2

    test_df = df[mask]
    train_df = df[~mask]
    return test_df, train_df