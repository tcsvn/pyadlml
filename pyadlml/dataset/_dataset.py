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
VAL = 'val'
NAME = 'name'
ACTIVITY = 'activity'

DEVICE = 'device'
RAW = 'raw'
CHANGEPOINT ='changed'
LAST_FIRED = 'last_fired'

"""
    df_activities:
        - per definition no activity can be performed in parallel

        start_time | end_time   | activity
        ---------------------------------
        timestamp   | timestamp | act_name


    df_devices:
        toggle_time | device_1 | ...| device_n
        ----------------------------------------
        timestamp   | state_0  | ...| state_n
"""
class Data():
    def __init__(self, activities, devices):

        assert self.check_activities(activities) 
        assert self.check_devices(devices)

        self.df_activities = activities
        self.df_devices = devices

        self.df_raw = None

    def check_activities(self, df):
        """
        check if the activitiy dataframe is valid by checking if
            - the dataframe has the correct dimensions and labels
            - activities are non overlapping
        """
        if not START_TIME in df.columns or not END_TIME in df.columns \
        or not ACTIVITY in df.columns or len(df.columns) != 3:
            raise ValueError

        if _is_activity_overlapping(df):
            print('there should be none activity overlapping')
            raise ValueError
        return True

    def check_devices(self, df_devices):
        # todo 
        return True

    def create_raw(self):
        dev = self.df_devices.copy() 
        act = self.df_activities.copy() 
        self.df_raw = label_data(dev, act)

#    def statreport_devices(self):
#        """
#        gather some stats about the devices used in the dataset
#        Returns
#        -------
#        res dict
#        """
#        from pyadlml.datasets.util import first_row_df2dict
#        from pyadlml.datasets.stat import get_devices_count
#
#        res = {}
#
#        # a list of all activites used in dataset
#        res['device_list'] = self.df_devices.columns
#
#        # the number of times a device changed state
#        dfc = get_devices_count()
#        res['device_counts']  = first_row_df2dict(dfc)
#
#        return res
#
#    def statreport_activities(self):
#        """
#        gather some stats about the activities, that labeled the dataset
#        in form of dictionarys
#        Returns
#        -------
#        res dict
#
#        """
#        from pyadlml.datasets.util import first_row_df2dict
#
#        res = {}
#        # a list of all activites used in dataset
#        res['act_list'] = self.df_activities.columns
#
#        # the number of how often the each activity occured in the dataset
#        dfc = self.df_activities.get_activities_count()
#        res['act_count'] = first_row_df2dict(dfc)
#
#        # percentage of activity duration over all activities
#        dfpc = _acts.get_rel_act_duration()
#        res['act_rel_duration'] = first_row_df2dict(dfpc)
#
#        # total length of activities in minutes
#        dfal = _acts.get_total_act_duration()
#        res['act_tot_duration'] = first_row_df2dict(dfal)
#
#        # percentage of labeled data vs unlabeled data
#        # todo implement
#
#        return res
#

def label_data(df_devices: pd.DataFrame, df_activities: pd.DataFrame):
    """
    for each row in the dataframe select the corresponding activity from the
    timestamp and create a np array with the activity labels
    :param df_devices:
        Name                    0       2  ...      13
        Time                               ...
        2008-03-20 00:34:38  False  False  ...    True
        2008-03-20 00:34:39  False  False  ...   False
        ...
    :param idle: boolean
        if true this leads to datapoints not falling into a logged activity to be
        labeled as idle
    :return:
        numpy ndarray 1D
    """
    df_raw = df_devices.copy()
    df_raw[ACTIVITY] = df_raw.index
    df_raw[ACTIVITY] = df_raw[ACTIVITY].apply(
                    _map_timestamp2activity,
                    df_act=df_activities)

# todo check how to vectorize with swifter for speedup
#    df_raw[ACTIVITY] = df_raw[ACTIVITY].swifter.apply(
#                    _map_timestamp2activity,
#                    df_act=df_activities)
    return df_raw

def _map_timestamp2activity(timestamp, df_act):
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
    if match_amount == 0:
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

def is_proper_activity_df(df):
    """
    checks if the given dataframe conforms to the definition of an activity dataframe
    """
    return 
def gen_random_day(df):
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
    assert is_proper_activity(df)

    rnd_idx = np.random.random_integers(0, len(df.index))
    rnd_start_time = df.iloc[rnd_idx][START_TIME] # type: pd.Timestamp

    return rnd_start_time.date()


def correct_device_ts_duplicates(df):
    """
    if there are duplicate timestamps which are used for indexing 
    make them unique again by adding a millisecond to the second pair
    """
    # split duplicates and uniques  
    dup_mask = df.duplicated(subset=[TIME], keep=False)
    duplicates = df[dup_mask]
    uniques = df[~dup_mask]

    i = -1 
    # for every pair of duplicates add a millisecond on the second one
    for index, row in duplicates.iterrows():
        i+=1
        if i%2 == 0:
            index_m1 = index
            row_m1 = row
            continue
        new_time = df.loc[index,TIME] + pd.Timedelta(milliseconds=1)
        duplicates.iloc[i - 1, df.columns.get_loc(TIME)] = new_time

    assert duplicates[TIME].is_unique

    # concatenate and sort the dataframe 
    df = pd.concat([duplicates, uniques], sort=True)
    return df.sort_values(TIME)

def _is_activity_overlapping(df):
    import datetime
    epsilon = datetime.timedelta(milliseconds=0)
    mask = (df[END_TIME].shift()-df[START_TIME]) > epsilon
    overlapping = df[mask]
    return not overlapping.empty

def _create_activity_df():
    df = pd.DataFrame(columns=[START_TIME, END_TIME, ACTIVITY])
    df[START_TIME] = pd.to_datetime(df[START_TIME])
    df[END_TIME] = pd.to_datetime(df[END_TIME])
    return df 


def correct_activity_overlap(df):
    """
        the use of the toilet in this dataset is logged in parallel to the
        rest of the data. This violates the constraint that no activity can 
        be performed in parallel
    """
    import datetime
    from pyadlml.dataset.util import print_df

    overlap = 'overlap'
    epsilon = datetime.timedelta(milliseconds=0)

    # label overlapping toilet activities
    mask = (df[END_TIME].shift()-df[START_TIME]) > epsilon
    overlapping = df[mask]
    overlapping = overlapping.sort_values(START_TIME)

    overlap_corresp = _create_activity_df()
    corrected = _create_activity_df()

    for row in overlapping.iterrows():
        ov_st = row[1].start_time
        ov_et = row[1].end_time
        """
        1. case      2. case     3.case       4.case
        ov |----|       |----|      |----|    |----|
        df   |----|      |-|      |---|      |-------| 
        1. case
            start falls into interval
        2. case
            end falls into interval
        3. case
            start and end fall both into interval
        4. case 
            start is smaller than ov_start and end is greater than ov_end
        5. case 
            interval boundaries match
        """
        mask_5c = (df[START_TIME] == ov_st) & (df[END_TIME] == ov_et)
        mask_1c = (df[START_TIME] >= ov_st) & (df[START_TIME] <= ov_et) \
                    & ~mask_5c
        mask_2c = (df[END_TIME] >= ov_st) & (df[END_TIME] <= ov_et) \
                    & ~mask_5c
        mask_3c = mask_1c & mask_2c & ~mask_5c
        mask_4c = (df[START_TIME] <= ov_st) & (df[END_TIME] >= ov_et) \
                    & ~mask_5c
        mask = mask_1c | mask_2c | mask_3c | mask_4c

        corresp_row = df[mask]

        overlap_corresp = overlap_corresp.append(corresp_row, ignore_index=True)
        # 1. case
        if mask_1c.any():
            raise NotImplementedError

        # 2. case
        if mask_2c.any():
            raise NotImplementedError

        # 3. case
        if mask_3c.any():
            raise NotImplementedError

        # 4. case
        if mask_4c.any():
            """
            ov    |----|   => |~|----|~~|
            cr  |~~~~~~~~|
            """
            # use epsilon to offset the interval boundaries a little bit
            # to prevent later matching of multiple indices
            eps = pd.Timedelta(milliseconds=1)

            # create temporary dataframe with values
            df2 = _create_activity_df()
            cr_st = corresp_row.start_time.iloc[0] 
            cr_et = corresp_row.end_time.iloc[0]
            ov_act = row[1][ACTIVITY]
            cr_act = corresp_row[ACTIVITY].iloc[0]

            df2.loc[0] = [cr_st, ov_st, cr_act]
            df2.loc[1] = [ov_st + eps, ov_et, ov_act]
            df2.loc[2] = [ov_et + eps, cr_et, cr_act]

            # append dataframe 
            corrected = corrected.append(df2, ignore_index=True)
    

    # create dataframe without the overlapping and their corresponding rows
    # and append the corrected values
    df_activities = pd.concat([df, overlapping, overlap_corresp]).drop_duplicates(keep=False)
    df_activities = df_activities.append(corrected)
    df_activities = df_activities.sort_values(START_TIME)
    df_activities = df_activities.reset_index(drop=True)

    return df_activities