import pandas as pd
import numpy as np
from pyadlml.dataset import START_TIME, END_TIME, ACTIVITY

def _get_freq_func(freq):
    """ returns the correct transform function of time differences into integers
        the column on which the function is applied has to be of type timedelta
    Parameters
    """
    assert freq in ['minutes', 'hours', 'seconds', 'm', 'h', 's']
    if freq == 'seconds' or freq == 's':
        return lambda x: x.total_seconds()
    elif freq == 'minutes' or freq == 'm':
        return lambda x: x.total_seconds()/60
    else:
        return lambda x: x.total_seconds()/3600

def activities_duration_dist(df_activities, list_activities=None, freq='minutes'):
    """
    """
    df = df_activities.copy()

    # integrate the time difference for activities
    diff = 'total_time'
    df[diff] = df[END_TIME] - df[START_TIME]
    df = df[[ACTIVITY, diff]]

    # if no data is logged for a certain activity append 0.0 duration value
    if list_activities is not None:
        for activity in set(list_activities).difference(set(list(df[ACTIVITY]))):
            df = df.append(pd.DataFrame(
                data=[[activity, pd.Timedelta('0ns')]],
                columns=df.columns, index=[len(df)]))
    df[freq] = df[diff].apply(_get_freq_func(freq))
    return df.sort_values(by=ACTIVITY)

def activity_durations(df_activities, list_activities=None, freq='minutes', norm=False):
    """
    returns a dataframe containing statistics about the activities durations
    """
    df = activities_duration_dist(df_activities, list_activities=list_activities, freq=freq)
    if df.empty:
        raise ValueError("no activity was recorded")
    df = df.groupby(ACTIVITY).sum().reset_index()
    df.columns = [ACTIVITY, freq]

    # compute fractions of activities to each other
    if norm:
        norm = df[freq].sum()
        df[freq] = df[freq].apply(lambda x: x/norm)

    return df.sort_values(by=ACTIVITY)

def activities_count(df_activities, lst_activities=None):
    """ computes the count of activities within a dataframe
    Parameters
    ----------
    df_activities : pd.DataFrame
    df_activity_map : pd.DataFrame

    Returns
    -------
    res pd.Dataframe
                   leave house  use toilet  ...  prepare Dinner  get drink
        occurence           33         111  ...              10         19

    """
    res_col_name = 'occurrence'
    df = df_activities.copy()

    # count the occurences of the activity
    df = df.groupby(ACTIVITY).count().reset_index()
    df = df.drop(columns=[END_TIME])
    df.columns = [ACTIVITY, res_col_name]

    # correct for missing activities
    if lst_activities is not None:
        diff = set(lst_activities).difference(set(list(df[ACTIVITY])))
        for activity in diff:
            df = df.append(pd.DataFrame(data=[[activity, 0.0]], columns=df.columns, index=[len(df) + 1]))

    return df.sort_values(by=ACTIVITY)


def activities_transitions(df_act, lst_act=None):
    """
    returns the transition matrix (a \times a) of activities
    c_ij describes how often an activity i was followed by activity j
    """
    df = df_act.copy()
    df = df[['activity']]
    df['act_after'] = df['activity'].shift(-1)
    df = pd.crosstab(df["activity"], df['act_after'])

    if lst_act is not None:
        diff = set(lst_act).difference(set(list(df.index)))
        for activity in diff:
            df[activity] = 0
            df = df.append(pd.DataFrame(data=0.0, columns=df.columns, index=[activity]))

    # sort activities alphabetically
    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1)
    return df


def activities_dist(df_act, lst_act=None, t_range='day', n=1000):
    """ use monte-carlo to sample points from interval to approximate an activity density
        for one day
        principle: ancestral sampling
            select an interval at random
            sample uniform value of the interval
        # todo
            sample from gaussian dist centered at interval instead
            of uniform dist
            solves the problem that beyond interval limits there can't be
            any sample
    Returns
    -------
    res : pd.DataFrame
        activities a columns and n-rows. Each field is a timestamp density point
    """
    label='activity'
    df = df_act
    assert t_range in ['day', 'week']
    df = df.copy()

    res = pd.DataFrame(index=range(n))
    for i in df[label].unique():
        series = _sample_ts(df[df[label] == i], n)
        res[i] = series

    if lst_act is not None:
        for activity in set(lst_act).difference(res.columns):
            res[activity] = pd.NaT
    return res

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

def _gen_rand_time(ts_min, ts_max):
    from datetime import timezone, datetime

    # convert to unix timestamp
    unx_min = ts_min.replace(tzinfo=timezone.utc).timestamp()
    unx_max = ts_max.replace(tzinfo=timezone.utc).timestamp()

    # sample uniform and convert back to timestamp
    unx_sample = np.random.randint(unx_min, unx_max)
    ts_sample = datetime.utcfromtimestamp(unx_sample).strftime('%H:%M:%S')
    
    return ts_sample