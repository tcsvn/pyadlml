import pandas as pd
import numpy as np
import datetime
from pyadlml.dataset._dataset import START_TIME, END_TIME, ACTIVITY, \
                                DEVICE, NAME


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
    df = df_act.copy()

    res = pd.DataFrame()
    for act in df['activity'].unique():
        series = sample_ts(df[df['activity'] == act], n)
        res[act] = series
    return res

def gen_rand_time(ts_min, ts_max):
    from datetime import timezone, datetime

    # convert to unix timestamp
    unx_min = ts_min.replace(tzinfo=timezone.utc).timestamp()
    unx_max = ts_max.replace(tzinfo=timezone.utc).timestamp()

    # sample uniform and convert back to timestamp
    unx_sample = np.random.randint(unx_min, unx_max)
    ts_sample = datetime.utcfromtimestamp(unx_sample).strftime('%H:%M:%S')
    
    return ts_sample

def sample_ts(sub_df, n):
    xs = np.empty(shape=(n), dtype=object)    
    for i in range(n):
        idx_sample = np.random.randint(0, sub_df.shape[0])
        interval = sub_df.iloc[idx_sample]
        rand_stamp = gen_rand_time(ts_min=interval.start_time,  ts_max=interval.end_time)
        rand_stamp = '1990-01-01T'+ rand_stamp            
        tmp = pd.to_datetime(rand_stamp)
        xs[i] = tmp
    return pd.Series(xs)        


def devices_count(df):
    """
    Returns
    -------
    res pd.Dataframe
                   binary_sensor.  use toilet  ...  prepare Dinner  get drink
        occurence               33         111  ...              10         19

    """
    # transform into changepoint representation
    from pyadlml.dataset._dataset import create_changepoint
    df = create_changepoint(df)

    # count the changes for each column
    cnt = df.apply(pd.value_counts)
    cnt.drop(False, inplace=True)

    return cnt
