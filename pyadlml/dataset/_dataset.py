"""
this file is to bring datasets into specific representations
"""

import pandas as pd
import swifter
import numpy as np
from enum import Enum
from pyadlml.dataset.util import print_df, resample_data

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


    df_devices_rep_1:
        toggle_time | device_1 | ...| device_n
        ----------------------------------------
        timestamp   | state_0  | ...| state_n

    df_devices_rep_2:
        is used to calculate statistics for devices more easily
        and has lower footprint in storage
        Exc: 
        start_time | end_time   | device    | state
        -------------------------------------------
        timestamp   | timestamp | dev_name  |   1

"""
class Data():
    def __init__(self, activities, devices):

        assert check_activities(activities) 
        assert check_devices(devices)

        self.df_activities = activities
        self.df_devices = devices

        # list of activities and devices
        self.activities = activities.columns
        self.devices = devices.columns

        # second representation
        self.df_dev_rep2 = None

        self.df_raw = None
        self.df_cp = None
        self.df_lf = None


    def create_raw(self, t_res=None):
        self.df_raw = create_raw(self.df_devices, self.df_activities, t_res)

    def statreport_devices(self):
        """
        gather some stats about the devices used in the dataset
        Returns
        -------
        res dict
        """
        res = {}

        # a list of all activites used in dataset
        res['device_list'] = self.devices

        # the number of times a device changed state
        dfc = get_devices_count()
        res['device_counts']  = first_row_df2dict(dfc)

        return res

    def statreport_activities(self):
        """
        gather some stats about the activities, that labeled the dataset
        in form of dictionarys
        Returns
        -------
        res dict

        """
        res['activity_list'] = self.activities
        #res['act_count'] = pyadlml.dataset.stat.
        dfpc = _acts.get_rel_act_duration()
        res['act_rel_duration'] = first_row_df2dict(dfpc)
        dfal = _acts.get_total_act_duration()
        res['act_tot_duration'] = first_row_df2dict(dfal)
        return res


def create_raw(df_devices, df_activities, t_res=None):
    dev = df_devices.copy()
    act = df_activities.copy() 
    if t_res is not None:
        # TODO upscale the data to a certain time resolution
        raise NotImplementedError

    df_raw = label_data(dev, act)
    return df_raw


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
    df = df_devices.copy()
    df[ACTIVITY] = df.index
    df[ACTIVITY] = df[ACTIVITY].apply(
                    _map_timestamp2activity,
                    df_act=df_activities)

# todo check how to vectorize with swifter for speedup
#    df[ACTIVITY] = df[ACTIVITY].swifter.apply(
#                    _map_timestamp2activity,
#                    df_act=df_activities)
    return df

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


def check_activities(df):
    """
    check if the activitiy dataframe is valid by checking if
        - the dataframe has the correct dimensions and labels
        - activities are non overlapping
    """
    if not START_TIME in df.columns or not END_TIME in df.columns \
    or not ACTIVITY in df.columns or len(df.columns) != 3:
        print('the lables and dimensions of activites does not fit')
        raise ValueError

    if _is_activity_overlapping(df):
        print('there should be none activity overlapping')
        raise ValueError
    return True

def check_devices(df_devices):
    return _is_dev_repr1 or _is_dev_repr2

def _is_dev_repr1(df):
    return True

def _is_dev_repr2(df):
    if not START_TIME in df.columns or not END_TIME in df.columns \
    or not DEVICE in df.columns or len(df.columns) != 3:
        return False
    # TODO check for uniqueness in timestamps
    return True



def correct_device_ts_duplicates(df):
    """
    if there are duplicate timestamps which are used for indexing 
    make them unique again by adding a millisecond to the second pair
    """
    # if time is the index, than it has to be reseted to a column
    df = df.reset_index()

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

    # set the time as index again
    df = df.sort_values(TIME)
    df = df.set_index(TIME)

    return df

def _is_activity_overlapping(df):
    import datetime
    epsilon = datetime.timedelta(milliseconds=0)
    mask = (df[END_TIME].shift()-df[START_TIME]) > epsilon
    overlapping = df[mask]
    return not overlapping.empty

def _create_activity_df():
    """
    returns: empty pd Dataframe 
    """
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

def _dev_rep2_to_rep1(df):

    pass

def _create_devices(dev_list, index=None):
    """
    creates an empty device dataframe
    """
    if index is not None:
        return pd.DataFrame(columns=dev_list, index=index)
    else:
        return pd.DataFrame(columns=dev_list)

def _dev_rep1_to_rep2(df):
    """
        df: 
        | Start time    | End time  | device_name | value
        --------------------------------------------------
        | ts1           | ts2       | name1       | 1

        return df:
        | time  | dev_1 | ....  | dev_n |
        --------------------------------
        | ts1   |   1   | ....  |  0    |
    """
    # copy devices to new dfs 
    # one with all values but start time and other way around
    df_start = df.copy().loc[:, df.columns != END_TIME]
    df_end = df.copy().loc[:, df.columns != START_TIME]

    # set values at the end time to zero because this is the time a device turns off
    df_start[VAL] = True
    df_end[VAL] = False

    # rename column 'End Time' and 'Start Time' to 'Time'
    df_start.rename(columns={START_TIME: TIME}, inplace=True)
    df_end.rename(columns={END_TIME: TIME}, inplace=True)

    df = pd.concat([df_end, df_start]).sort_values(TIME)
    df = df.reset_index(drop=True)

    # create raw dataframe
    df_dev = _create_devices(df[DEVICE].unique(), index=df[TIME])

    # create first row in dataframe 
    df_dev.iloc[0] = np.zeros(len(df_dev.columns))
    col_idx = df_dev.columns.get_loc(df.iloc[0].device)
    df_dev.iloc[0,col_idx] = 1

    # update all rows of the dataframe
    for i, row in enumerate(df.iterrows()):
        if i == 0:
            continue
        #copy previous row into current and update current value
        df_dev.iloc[i] = df_dev.iloc[i-1].values
        col_idx = df_dev.columns.get_loc(df.iloc[i].device)
        df_dev.iloc[i] = int(df.iloc[i].val)

    return df_dev


def create_changepoint(df, freq=(30, 's')):
    # resample the data given frequencies
    df = resample_data(df, freq)
    df = _apply_change_point(df.copy())
    return df

def _apply_change_point(df):
    """
    Parameters
    ----------
    df
    
    Returns
    -------
    
    """
    i = 0
    insert_stack = []
    pushed_prev_it_on_stack = False
    len_of_row = len(df.iloc[0])
    series_all_false = self._gen_row_false_idx_true(len_of_row, [])
    for j in range(1, len(df.index)+2):
        if (len(insert_stack) == 2 or (len(insert_stack) == 1 and not pushed_prev_it_on_stack)) \
                and (i-1 >= 0):
            """
            the case when either the stack is maxed out ( == 2) or a row from before
            2 iterations is to be written to the dataframe and it is not the first two rows
            """
            item = insert_stack[0]
            df.iloc[i-1] = item
            insert_stack.remove(item)
        else:
            df.iloc[i-1] = series_all_false
        if pushed_prev_it_on_stack:
            pushed_prev_it_on_stack = False
    
        if j >= len(df.index):
            # this is to process the last two lines also as elements are appended
            # 2 rows in behind
            i += 1
            continue

        rowt = df.iloc[i]   # type: pd.Series
        rowtp1 = df.iloc[j] # type: pd.Series
    
        if not rowt.equals(rowtp1):
            idxs = self._get_cols_that_changed(rowt, rowtp1) # type: list
            row2insert = self._gen_row_false_idx_true(len(rowt), idxs)
            insert_stack.append(row2insert)
            pushed_prev_it_on_stack = True
        i += 1
    return df