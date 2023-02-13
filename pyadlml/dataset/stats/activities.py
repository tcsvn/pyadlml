import pandas as pd
import numpy as np
from pyadlml.constants import START_TIME, END_TIME, ACTIVITY, TIME
from pyadlml.dataset._core.activities import add_other_activity, ActivityDict
from pyadlml.dataset.stats.util import df_density_binned
from datetime import timezone, datetime


def activity_order_by_duration(df_acts: pd.DataFrame) -> list:
    """ Computes a list of activities ordered by the cumulated time per activity

    Parameters
    ----------
    df_acts : pd.DataFrame
        An activity dataframe with columns 'start_time', 'end_time' and 'activity'.

    Returns
    -------
    list
        An ordered list where the first is the most prominent activity, followed by the second most prominent ...
    """
    if isinstance(df_acts, pd.DataFrame):
        dct_acts = ActivityDict.wrap(df_acts)
    else: 
        dct_acts = df_acts
    dfs = [activity_duration(df).set_index(ACTIVITY) for df in dct_acts.values()]

    df = pd.concat(dfs, axis=1).fillna(0)
    df['sum'] = df.sum(axis=1)
    df = df.sort_values(by='sum', ascending=False)
    return df.index.to_list()


def activity_order_by_count(df_acts: pd.DataFrame) -> list:
    """ Computes a list ordered by the most cumulated time per activity

    Parameters
    ----------
    df_acts : pd.DataFrame
        An activity dataframe with columns 'start_time', 'end_time' and 'activity'.

    Returns
    -------
    list
        An ordered list where the first is the most prominent activity, followed by the second most prominent ...
    """
    if isinstance(df_acts, pd.DataFrame):
        dct_acts = ActivityDict.wrap(df_acts)
    else: 
        dct_acts = df_acts
    dfs = [activities_count(df).set_index(ACTIVITY) for df in dct_acts.values()]

    df = pd.concat(dfs, axis=1).fillna(0)
    df['sum'] = df.sum(axis=1)
    df = df.sort_values(by='sum', ascending=False)
    return df.index.to_list()


def _get_freq_func(freq):
    """ returns the correct transform function of time differences into integers
        the column on which the function is applied has to be of type timedelta
    Parameters
    ----------


    Returns
    -------
    """
    assert freq in ['minutes', 'hours', 'seconds', 'm', 'h', 's']
    if freq == 'seconds' or freq == 's':
        return lambda x: x.total_seconds()
    elif freq == 'minutes' or freq == 'm':
        return lambda x: x.total_seconds()/60
    else:
        return lambda x: x.total_seconds()/3600

def coverage(df_activities, df_devices, datapoints=False):
    """ Computes the activity coverage for the devices.

    Parameters
    ----------
    df_activities : pd.DataFrame
        Todo
    datapoints: bool, default=False
        Determines whether the activity coverage is computed with respect
        to the datapoints covered or the total time.

    Returns
    -------
    cov : float
        The percentage covered by activities
    """
    if datapoints:
        from pyadlml.dataset._core.acts_and_devs import label_data
        df_acts = df_activities.copy()
        lbl_devs = label_data(df_devices, df_acts)
        nr_nans = lbl_devs[ACTIVITY].isna().sum()
        return 1 - nr_nans/len(lbl_devs)
    else:
        durations = activity_duration(df_activities, idle=True)
        total = durations['minutes'].sum()
        amount_idle = durations.loc[(durations[ACTIVITY] == 'idle'), 'minutes'].values[0]
        return 1 - amount_idle/total

def activities_duration_dist(df_acts, lst_acts=None, freq='minutes', idle=False):
    """ Computes the activity distribution.

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
    idle : bool, default=False
        Whether to include the idle activity. More in user guide

    Returns
    -------
    df : pd.DataFrame
        First column activity names second column total time in frequency.

    Example
    -------
    .. code python::
        TODO

    """
    df = df_acts.copy()
    if idle:
        df = add_other_activity(df)

    # integrate the time difference for activities
    diff = 'total_time'
    df[diff] = df[END_TIME] - df[START_TIME]
    df = df[[ACTIVITY, diff]]

    # if no data is logged for a certain activity append 0.0 duration value
    #if lst_acts is not None:
    #    for activity in set(lst_acts).difference(set(list(df[ACTIVITY]))):
    #        df = df.append(pd.DataFrame(
    #            data=[[activity, pd.Timedelta('0ns')]],
    #            columns=df.columns, index=[len(df)]))

    df[freq] = df[diff].apply(_get_freq_func(freq))
    return df.sort_values(by=ACTIVITY)

def activity_duration(df_acts, time_unit='minutes', normalize=False, idle=False):
    """ Compute how much time an inhabitant spent performing an activity

    Parameters
    ----------
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    time_unit : str of {'minutes', 'seconds', 'hours', 'm', 's', 'h'}
        The unit of time the durations are returned in.
    normalize : bool, default=False
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
    df = activities_duration_dist(df_acts, freq=time_unit, idle=idle)
    if df.empty:
        raise ValueError("no activity was recorded")
    df = df.groupby(ACTIVITY).sum().reset_index()
    df.columns = [ACTIVITY, time_unit]

    # compute fractions of activities to each other
    if normalize:
        normalize = df[time_unit].sum()
        df[time_unit] = df[time_unit].apply(lambda x: x / normalize)

    return df.sort_values(by=ACTIVITY)

def activities_count(df_acts: pd.DataFrame) -> pd.DataFrame:
    """ Computes how many times a certain activity occurs within the dataset.

    Parameters
    ----------
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.

    Returns
    -------
    pd.Dataframe

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

    # Count the occurences of the activity
    df = df.groupby(ACTIVITY).count().reset_index()
    df = df.drop(columns=[END_TIME])
    df.columns = [ACTIVITY, res_col_name]

    # correct for missing activities
    #if lst_acts is not None:
    #    diff = set(lst_acts).difference(set(list(df[ACTIVITY])))
    #    for activity in diff:
    #        df = df.append(pd.DataFrame(data=[[activity, 0.0]], columns=df.columns, index=[len(df) + 1]))

    return df.sort_values(by=ACTIVITY)


def activities_transitions(df_acts):
    """  Compute a transition matrix that displays how often one
    activity was followed by another.

    Parameters
    ----------
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.

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
    df = df[[ACTIVITY]]
    df['act_after'] = df[ACTIVITY].shift(-1)
    df = pd.crosstab(df[ACTIVITY], df['act_after'])

    cols = list(df.columns)
    rows = list(df.index)
    for r in rows:
        if r not in cols:
            break

    # Catches the case when either an activity is only followed once and not preceeded by another
    # activity or an activity is only preceeded once and not succeded by another activity
    if len(df.columns) != len(df.index):
        row_diff = set(df.columns).difference(set(list(df.index)))
        col_diff = set(df.index).difference(set(list(df.columns)))
        for activity in col_diff:
            # when columns are missing
            df[activity] = 0
        for activity in row_diff:
            # when rows are missing
            df = df.append(pd.DataFrame(data=0.0, columns=df.columns, index=[activity]))

    # sort activities alphabetically
    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1)

    return df


def activities_dist(df_acts, n=1000, dt=None, relative=False):
    """
    Approximate the activity densities for one day by
    using monte-carlo sampling from the activity intervals.

    Parameters
    ----------
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    n : int, optional
        The number of samples to draw from each activity. Defaults to 1000.
    dt : str of { 'xm', }, default=None
        if set

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

    df = df_acts.copy()
    activities = df_acts[ACTIVITY].unique()

    #res = pd.DataFrame(index=range(n), columns=activities, data=0)
    res = pd.DataFrame(index=range(len(activities)*n), columns=[TIME, ACTIVITY], dtype=object)

    # sample for each column separately
    for i, activity in enumerate(activities):
        series = _sample_ts(df.loc[df[ACTIVITY] == activity, [START_TIME, END_TIME]], n)
        res.loc[n*i:n*i+n-1, TIME] = series.values
        res.loc[n*i:n*i+n-1, ACTIVITY] = activity

    if relative and dt is None:
        res[TIME] = pd.to_datetime('1/1/2000 - ' + res[TIME])
        res[TIME] = res[TIME] - np.datetime64("2000-01-01 00:00:00")
        res[TIME] = res[TIME] / np.timedelta64(1, 's')

    if dt is not None:
        res[TIME] = pd.to_datetime('1/1/2000 - ' + res[TIME])
        res = df_density_binned(res, column_str=ACTIVITY, dt=dt)

    return res
    



def _sample_ts(df, n):
    """ Samples one column of activities. First chooses one activity at random and then samples a timepoint from
        within that activities interval

    Parameters
    ----------
    df: pd.DataFrame
            dataframe with two time intervals
            | start_time    | end_time  | .... |
            ------------------------------------
            | ts            | ts        | ...

    Returns
    -------
    pd.Series
        The sampled timestamps
    """
    assert START_TIME in df.columns and END_TIME in df.columns

    xs = np.empty(shape=(n,), dtype=object)
    for i in range(n):
        interval = df.iloc[np.random.randint(0, df.shape[0])]

        # convert to unix timestamp
        unx_min = interval.start_time.replace(tzinfo=timezone.utc).timestamp()
        unx_max = interval.end_time.replace(tzinfo=timezone.utc).timestamp()

        # sample uniform and convert back to timestamp
        try:
            unx_sample = np.random.randint(unx_min, unx_max)
        except ValueError:
            if unx_min == unx_max:
                unx_sample = unx_min
            else:
                raise ValueError('Sampling interval went wrong')
        xs[i] = datetime.utcfromtimestamp(unx_sample).strftime('%H:%M:%S')

    return pd.Series(xs)
