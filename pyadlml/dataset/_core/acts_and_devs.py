"""
this file is to bring datasets into specific representations
"""
import pandas as pd
from pyadlml.util import get_parallel, get_npartitions
from pyadlml.constants import OTHER, TIME, ACTIVITY, START_TIME, END_TIME

def label_data(df_devs: pd.DataFrame, df_acts: pd.DataFrame, other=False, n_jobs=1, inplace=True):
    """
    Label a dataframe with corresponding activities based on a time-index.

    Parameters
    ----------
    df_devs : pd.DataFrame
        some data representation that possesses a column 'time' including timestamps.
    df_acts : pd.DataFrame
        a datasets activities. TODO
    other : bool, optional, default=False
        if true this leads to datapoints not falling into a logged activity to be
        labeled as "other"
    n_jobs : int, optional, default=1
        the number of jobs that are run in parallel TODO look up sklearn
    inplace : bool, optional, default=True
        determines whether a new column is appended to the existing dataframe.

    Examples
    --------
    >>> raw = DiscreteEncoder()
    >>> raw
    1 time                    0   ...      13
    2 2008-03-20 00:34:38  False  ...    True
    3 2008-03-20 00:34:39  False  ...   False

    now include
    >>> label_data(raw, data.df_activities, other=True, n_jobs=10)
    1 time                    0   ...      13 activity
    2 2008-03-20 00:34:38  False  ...    True other
    3 2008-03-20 00:34:39  False  ...   False act1

    Returns
    -------
    df : pd.DataFrame
    """

    df_devs = df_devs.copy()
    df_devs[ACTIVITY] = -1

    if n_jobs == 1:
        df_devs[ACTIVITY] = df_devs[TIME].apply(
                    _map_timestamp2activity,
                    df_act=df_acts,
                    other=other)
    else:
        N = get_npartitions()
        if n_jobs == -1 or n_jobs > N:
            n_jobs = N

        import dask.dataframe as dd
        #ddf_activities = dd.from_pandas(df_activities, npartitions=get_npartitions())
        # compute with dask in parallel
        df_devs[ACTIVITY] = dd.from_pandas(df_devs[TIME], npartitions=n_jobs).\
                    map_partitions( # apply lambda functions on each partition
                        lambda df: df.apply(
                            _map_timestamp2activity,
                            df_act=df_acts,
                            other=other)).\
                    compute(scheduler='processes')
    return df_devs



def _map_timestamp2activity(timestamp, df_act, other):
    """ Map the given timestamp map to an activity in df_act

    Parameters
    ----------
    timestamp : pd.Timestamp
        E.g timestamp 2008-02-26 00:39:25
    df_act : pd.DataFrame
        An activity dataframe
    other : boolean
        Whether to map gaps to NAT or "other"

    Returns
    -------
        label of the activity
    """

    # select activity intervals that the timestamp falls into
    mask = (df_act[START_TIME] <= timestamp) & (timestamp <= df_act[END_TIME])
    matches = df_act[mask]
    match_amount = len(matches.index)

    # 1. case no activity interval matched
    if match_amount == 0 and other:
        return OTHER
    elif match_amount == 0 and not other:
        return pd.NaT

    # 2. case single row matches
    elif match_amount == 1:
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

