import pandas as pd
import numpy as np
import datetime
from pyadlml.dataset._dataset import START_TIME, END_TIME, ACTIVITY, \
                                DEVICE, NAME, TIME, VAL, device_rep2_2_rep3
from pyadlml.dataset.util import timestrs_2_timedeltas

"""
    TODO list
        - add correlation between devices by sliding window over time
          and counting 
        - time difference for each device to trigger
"""


def activities_duration_dist(df_activities, freq='minutes'):
    """
    """
    assert freq in ['minutes', 'hours', 'seconds']
    df = df_activities.copy()

    # integrate the time difference for activities
    diff = 'total_time'
    df[diff] = df[END_TIME] - df[START_TIME]
    df = df[[ACTIVITY, diff]]
    if freq == 'seconds':
        func = lambda x: x.total_seconds()
    elif freq == 'minutes':
        func = lambda x: x.total_seconds()/60
    else:
        func = lambda x: x.total_seconds()/3600
    df[freq] = df[diff].apply(func)
    return df

def activities_durations(df_activities, freq='minutes'):
    """
    returns a dataframe containing statistics about the activities durations
    """
    df = activities_duration_dist(df_activities, freq=freq)
    df = df.groupby('activity').sum()

    # compute fractions of activities to each other
    norm = df[freq].sum()
    #df[freq] = df[freq].apply(lambda x: x/norm)
    return df


def activities_count(df_activities):
    """
    Returns
    -------
    res pd.Dataframe
                   leave house  use toilet  ...  prepare Dinner  get drink
        occurence           33         111  ...              10         19

    """
    df = df_activities.copy()

    # count the occurences of the activity
    df = df.groupby(ACTIVITY).count()
    df = df.drop(columns=[END_TIME])
    df.columns = ['occurence']
    return df


def activities_dist(df_act, t_range='day', n=1000):
    """
    returns an array where for one (day | week) the activities
    are sampled over time 
    """
    return _create_dist(df_act, label=ACTIVITY, t_range=t_range, n=n)

def devices_dist(df_dev, t_range='day', n=1000):
    """
    returns an array where for one (day | week) the device
    are sampled over time 
    """
    return _create_dist(df_dev, label=DEVICE, t_range=t_range, n=n)

def _create_dist(df, label='activity', t_range='day', n=1000):
    """
        principle: ancestral sampling
            select an interval at random 
            sample uniform value of the interval
        # todo
            make gaussian Distribution centered at interval instead 
            of uniform 
            solves the problem that beyond interval limits there can't be
            any sample
    """
    assert t_range in ['day', 'week']
    df = df.copy()

    res = pd.DataFrame()
    for i in df[label].unique():
        series = _sample_ts(df[df[label] == i], n)
        res[i] = series
    return res

def _gen_rand_time(ts_min, ts_max):
    from datetime import timezone, datetime

    # convert to unix timestamp
    unx_min = ts_min.replace(tzinfo=timezone.utc).timestamp()
    unx_max = ts_max.replace(tzinfo=timezone.utc).timestamp()

    # sample uniform and convert back to timestamp
    unx_sample = np.random.randint(unx_min, unx_max)
    ts_sample = datetime.utcfromtimestamp(unx_sample).strftime('%H:%M:%S')
    
    return ts_sample

def _sample_ts(sub_df, n):
    """
    params:
        sub_df: pd.DataFrame
            dataframe with two time intervals 
            | start_time    | end_time  | .... |
            ------------------------------------
            | ts            | ts        | ...

    returns:
        series: pd.Series
            series of sampled timestamps
    """
    assert 'start_time' in sub_df.columns \
        and 'end_time' in sub_df.columns

    xs = np.empty(shape=(n), dtype=object)    
    for i in range(n):
        idx_sample = np.random.randint(0, sub_df.shape[0])
        interval = sub_df.iloc[idx_sample]
        rand_stamp = _gen_rand_time(ts_min=interval.start_time,  ts_max=interval.end_time)
        rand_stamp = '1990-01-01T'+ rand_stamp            
        tmp = pd.to_datetime(rand_stamp)
        xs[i] = tmp
    return pd.Series(xs)        

def devices_trigger_count(df):
    """
    counts the amount a device was triggered
    params: df pd.DataFrame
            a data frame in repr2 of the devices
    """
    # TODO check if the 
    df = df.groupby('device')['device'].count()
    df = pd.DataFrame(df)
    df.columns = ['trigger count']
    return df


def devices_trigger_time_diff(df):
    """
    counts the time differences between triggers of devices 
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

    # create timediff to the previous trigger
    df['time_diff'] = df['time'].diff()

    # calculate duration to next trigger
    df['row_duration'] = df.time_diff.shift(-1)

    # calculate the time differences for each device sepeartly
    df = df.sort_values(by=['device','time'])
    df['time_diff2'] = df['time'].diff()
    df.loc[df.device != df.device.shift(), 'time_diff2'] = None
    df = df.sort_values(by=['time'])
    return df

def devices_on_off_stats(df):
    """
    calculate the time and proportion a device was on vs off
    params:
        df: pd.Dataframe in shape of repr2

    returns: pd.DataFrame
        | device | td_on    | td_off   | frac on   | frac off |
        --------------------------------------------------
        | freezer| 00:10:13 | 27 days   | 0.0025    | 0.999   |
        ....

    """
    # colum names of the final dataframe
    diff = 'diff'
    td_on = 'td_on'
    td_off = 'td_off'
    frac_on = 'frac_on'
    frac_off = 'frac_off'

    df = df.sort_values(START_TIME)

    # calculate total time interval for normalization
    int_start = df.iloc[0,0]
    int_end = df.iloc[df.shape[0]-1,1]
    norm = int_end - int_start

    # calculate time deltas for online time
    df[diff] = df[END_TIME] - df[START_TIME]
    df = df.groupby(DEVICE)[diff].sum()
    df = pd.DataFrame(df)
    df.columns = [td_on]

    df[td_off] = norm - df[td_on]

    # compute percentage
    df[frac_on] = df[td_on].dt.total_seconds() \
                    / norm.total_seconds() 
    df[frac_off] = df[td_off].dt.total_seconds() \
                    / norm.total_seconds()
    return df

def device_tcorr(df, t_windows=['20s']):
    """
    computes for every time window the prevalence of device triggers
    for each device
    params: df in repr2 
            t_windows (list)
                a list of windows
            or a single window (string)

    returns: list of panda dataframes
    """
    from pyadlml.dataset.util import timestr_2_timedelta

    t_windows = timestrs_2_timedeltas(t_windows)

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

    # create timediff to the previous trigger
    df['time_diff'] = df['time'].diff()

    #knn
    #    do cumsum for row_duration 
    #    for each row mask the rows that fall into the given area
    dev_list =  df.device.unique()

    df.iloc[0,3] = pd.Timedelta(0, 's')
    df['cum_sum'] = df['time_diff'].cumsum()
    lst = []
    for t_window in t_windows:
        # create cross table with zeros
        res_df = pd.DataFrame(columns=dev_list, index=dev_list)
        for col in res_df.columns:
            res_df[col].values[:] = 0

        # this whole iterations can be done in parallel
        for row in df.iterrows():
            td = row[1].cum_sum
            dev_name = row[1].device 
            
            df['tmp'] = (td-t_window < df['cum_sum']) & (df['cum_sum'] < td+t_window)
            tmp = df.groupby('device')['tmp'].sum()
            res_df.loc[dev_name] += tmp
        lst.append(res_df)
    return lst

def device_triggers_one_day(df, t_res='1h'):
    """
    computes the amount of triggers of a device for each hour of a day summed
    over all the weeks

    params: df: pd.DataFrame
                repr2 of devices
            t_res: [0,24]h or [0,60]m for a resoltion in hours, minutes
    returns: df
            index: hours
            columsn devices
            values: the amount a device changed states 
    """

    df = device_rep2_2_rep3(df)

    # compute new table
    df['time'] = df['time'].apply(_time2int, args=[t_res])
    df = df.groupby(['time', 'device']).sum().unstack()
    df = df.fillna(0)
    df.columns = df.columns.droplevel(0)
    return df

def _time2int(ts, t_res='30m'):
    """
    rounds to the next lower min bin or hour bin
    """
    assert t_res[-1:] in ['h','m']
    val = int(t_res[:-1])

    assert (val > 0 and val <=12 and t_res[-1:] == 'h')\
        or (val > 0 and val <= 60 and t_res[-1:] == 'm')
    import datetime as dt
    zero = dt.time()

    if t_res[-1:] == 'h':
        hs = val
        h_bin = int(ts.hour/hs)*hs
        return dt.time(hour=h_bin)

    elif t_res[-1:] == 'm':
        ms = val
        m_bin = int(ts.minute/ms)*ms
        return dt.time(hour=ts.hour, minute=m_bin)
    else:
        raise ValueError

#def device_triggers_one_day(df):
#    """
#    computes the amount of triggers of a device for each hour of a day summed
#    over all the weeks
#
#    params: df: pd.DataFrame
#                repr2 of devices
#    returns: df
#            index: hours
#            columsn devices
#            values: the amount a device changed states 
#    """
#
#    # copy devices to new dfs 
#    # one with all values but start time and other way around
#    df_start = df.copy().loc[:, df.columns != END_TIME]
#    df_end = df.copy().loc[:, df.columns != START_TIME]
#
#    # set values at the end time to zero because this is the time a device turns off
#    df_start[VAL] = True
#    df_end[VAL] = False
#
#    # rename column 'End Time' and 'Start Time' to 'Time'
#    df_start.rename(columns={START_TIME: TIME}, inplace=True)
#    df_end.rename(columns={END_TIME: TIME}, inplace=True)
#
#    df = pd.concat([df_end, df_start]).sort_values(TIME)
#
#    # compute new table
#    df = df.set_index('time', drop=True)
#    df = df.groupby([df.index.hour, 'device']).sum().unstack()
#    df = df.fillna(0)
#    df.columns = df.columns.droplevel(0)
#    return df

def activities_transitions(df_act):
    """
    returns the transition matrix (a \times a) of activities
    c_ij describes how often an activity i was followed by activity j
    """
    df = df_act.copy()
    df = df[['activity']]
    df['act_after'] = df['activity'].shift(-1)

    #df = df.groupby("activity")['act_after'].value_counts()).unstack(fill_value=0)
    df = pd.crosstab(df["activity"], df['act_after'])
    return df

