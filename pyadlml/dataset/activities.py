from pyadlml.dataset._dataset import START_TIME, ACTIVITY, END_TIME
import pandas as pd

def check_activities(df):
    """
    check if the activitiy dataframe is valid by checking if
        - the dataframe has the correct dimensions and labels
        - activities are non overlapping

    :param: df 
    start_time | end_time   | activity
    ----------------------------------
    timestamp   | timestamp | act_name

    """
    if not START_TIME in df.columns or not END_TIME in df.columns \
    or not ACTIVITY in df.columns or len(df.columns) != 3:
        print('the lables and dimensions of activites does not fit')
        raise ValueError

    if _is_activity_overlapping(df):
        print('there should be none activity overlapping')
        raise ValueError
    return True

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
        1. case      2. case     3.case       4.case    5. case
        ov |----|       |----|      |----|    |----|    |---|
        df   |----|      |-|      |---|      |-------|  |---|
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

def add_idle(acts, min_diff=pd.Timedelta('5s')):
    """ adds a dummy Idle activity for gaps between activities greater than min_diff
    Parameters
    ----------
    acts: pd.DataFrame
        the activity data with columns: start_time, end_time, activity
    Returns
    ----------
    pd.DataFrame
        activity data with columns: start_time, end_time, activity
    """
    acts = acts.copy().reset_index(drop=True)
    def func(series, df):
        """
        """
        # check if at end of the series
        if series.name == len(df)-1:
            return series
        else:
            next_entry = df.loc[series.name+1,:]
            return pd.Series({
                'start_time': series.end_time + pd.Timedelta('1ms'),
                'end_time' : next_entry.start_time - pd.Timedelta('1ms'),
                'activity' : 'idle'
            })                

     
    acts['diff'] =  acts['start_time'].shift(-1) - acts['end_time']
    tmp = acts[acts['diff'] > min_diff].copy()
    tmp = tmp.apply(func, axis=1, df=acts)
    res = pd.concat([acts.iloc[:,:-1],tmp]).sort_values(by='start_time')
    return res.reset_index(drop=True)