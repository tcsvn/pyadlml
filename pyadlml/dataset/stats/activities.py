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

def activities_duration_dist(df_acts, lst_acts=None, freq='minutes'):
    """
    Computes the activity distribution.

    Parameters
    ----------
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : list, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    freq : str, optional
        Defaults to 1000.
        Can be one of 'minutes', 'seconds', 'hours', 'm', 'h', 's'.
    Returns
    -------
    df : pd.DataFrame

    """
    df = df_acts.copy()

    # integrate the time difference for activities
    diff = 'total_time'
    df[diff] = df[END_TIME] - df[START_TIME]
    df = df[[ACTIVITY, diff]]

    # if no data is logged for a certain activity append 0.0 duration value
    if lst_acts is not None:
        for activity in set(lst_acts).difference(set(list(df[ACTIVITY]))):
            df = df.append(pd.DataFrame(
                data=[[activity, pd.Timedelta('0ns')]],
                columns=df.columns, index=[len(df)]))
    df[freq] = df[diff].apply(_get_freq_func(freq))
    return df.sort_values(by=ACTIVITY)

def activity_durations(df_acts, lst_acts=None, time_unit='minutes', norm=False):
    """ Compute how much time an inhabitant spent performing an activity

    Parameters
    ----------
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : list, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    time_unit : str of {'minutes', 'seconds', 'hours', 'm', 's', 'h'}
        The unit of time the durations are returned in.
    norm : bool
        If set to *true* rather than returning the durations in a specified
        time unit, the fraction of the durations is returned.

    Returns
    -------
    df : pd.Dataframe

    Examples
    --------
    >>> from pyadlml.dataset import fetch_amsterdam
    >>> from pyadlml.stats import activity_duration
    >>> data = fetch_amsterdam()
    >>> activity_duration(data.df_activities, time_unit='m')
                activity       minutes
    0          get drink     16.700000
    1          go to bed  11070.166267
    2        leave house  22169.883333
    3  prepare Breakfast     63.500000
    4     prepare Dinner    338.899967
    5        take shower    209.566667
    6         use toilet    195.249567

    """
    df = activities_duration_dist(df_acts, lst_acts=lst_acts, freq=time_unit)
    if df.empty:
        raise ValueError("no activity was recorded")
    df = df.groupby(ACTIVITY).sum().reset_index()
    df.columns = [ACTIVITY, time_unit]

    # compute fractions of activities to each other
    if norm:
        norm = df[time_unit].sum()
        df[time_unit] = df[time_unit].apply(lambda x: x / norm)

    return df.sort_values(by=ACTIVITY)

def activities_count(df_acts, lst_acts=None):
    """ Computes how many times a certain activity occurs within the dataset.

    Parameters
    ----------
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : list, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.

    Returns
    -------
    df : pd.Dataframe

    Examples
    --------
        >>> from pyadlml.dataset import fetch_amsterdam
        >>> from pyadlml.stats import activity_count
        >>> data = fetch_amsterdam()
        >>> activity_count(data.df_activities)
                    activity  occurrence
        0          get drink         19
        1          go to bed         47
        2        leave house         33
        3  prepare Breakfast         19
        4     prepare Dinner         12
        5        take shower         22
        6         use toilet        111
    """
    res_col_name = 'occurrence'
    df = df_acts.copy()

    # count the occurences of the activity
    df = df.groupby(ACTIVITY).count().reset_index()
    df = df.drop(columns=[END_TIME])
    df.columns = [ACTIVITY, res_col_name]

    # correct for missing activities
    if lst_acts is not None:
        diff = set(lst_acts).difference(set(list(df[ACTIVITY])))
        for activity in diff:
            df = df.append(pd.DataFrame(data=[[activity, 0.0]], columns=df.columns, index=[len(df) + 1]))

    return df.sort_values(by=ACTIVITY)


def activities_transitions(df_acts, lst_acts=None):
    """  Compute a transition matrix that displays how often one
    activity was followed by another.

    Parameters
    ----------
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : list, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.

    Returns
    -------
    df : pd.Dataframe

    Notes
    -----
    The transition frequency :math:`c_{ij}` describes how often
    an activity :math:`i` was followed by activity :math:`j`.

    Examples
    --------
        >>> from pyadlml.dataset import fetch_amsterdam
        >>> from pyadlml.stats import activity_transition
        >>> data = fetch_amsterdam()
        >>> activity_transition(data.df_activities)
        act_after          get drink  go to bed  ...  use toilet
        activity
        get drink                  3          0  ...          15
        go to bed                  0          0  ...          43
        leave house                3          1  ...          22
        prepare Breakfast          1          0  ...           8
        prepare Dinner             7          0  ...           4
        take shower                0          0  ...           1
        use toilet                 5         46  ...          18
    """

    df = df_acts.copy()
    df = df[['activity']]
    df['act_after'] = df['activity'].shift(-1)
    df = pd.crosstab(df["activity"], df['act_after'])

    if lst_acts is not None:
        diff = set(lst_acts).difference(set(list(df.index)))
        for activity in diff:
            df[activity] = 0
            df = df.append(pd.DataFrame(data=0.0, columns=df.columns, index=[activity]))

    # sort activities alphabetically
    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1)
    return df


def activities_dist(df_acts, lst_acts=None, n=1000):
    """
    Approximate the activity densities for one day by
    using monte-carlo sampling from the activity intervals.

    Parameters
    ----------
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    lst_acts : list, optional
        A list of activities that are included in the statistic. The list can be a
        subset of the recorded activities or contain activities that are not recorded.
    n : int, optional
        The number of samples to draw from each activity. Defaults to 1000.

    Examples
    --------
    >>> from pyadlml.stats import activity_dist
    >>> transitions = activity_dist(data.df_activities, n=2000)
              prepare Dinner           get drink ...         leave house
    0    1990-01-01 18:12:39 1990-01-01 21:14:07 ... 1990-01-01 13:30:33
    1    1990-01-01 20:15:14 1990-01-01 20:23:31 ... 1990-01-01 12:03:13
    ..                      ...                 ...                 ...
    1999 1990-01-01 18:16:27 1990-01-01 08:49:38 ... 1990-01-01 16:18:25

    Returns
    -------
    df : pd.Dataframe
        Each row represents density point.
    """

    # todo
    #        principle: ancestral sampling
    #        select an interval at random
    #        sample uniform value of the interval
    #        sample from gaussian dist centered at interval instead
    #        of uniform dist
    #        solves the problem that beyond interval limits there can't be
    #        any sample
    # Todo include range for day or week

    label='activity'
    df = df_acts.copy()

    res = pd.DataFrame(index=range(n))
    for i in df[label].unique():
        series = _sample_ts(df[df[label] == i], n)
        res[i] = series

    if lst_acts is not None:
        for activity in set(lst_acts).difference(res.columns):
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