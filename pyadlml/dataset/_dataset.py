"""
this file is to bring datasets into specific representations
"""
import pandas as pd
import dask.dataframe as dd
from pyadlml.util import get_parallel, get_npartitions
from pyadlml.dataset import TIME, ACTIVITY, START_TIME, END_TIME

def label_data(df: pd.DataFrame, df_acts: pd.DataFrame, idle=False, n_jobs=1, inplace=True):
    """
    Label a dataframe with corresponding activities based on a time-index.

    Parameters
    ----------
    df : pd.DataFrame
        some data representation that possesses a column 'time' including timestamps.
    df_acts : pd.DataFrame
        a datasets activities. TODO
    idle : bool, optional, default=False
        if true this leads to datapoints not falling into a logged activity to be
        labeled as idle
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
    >>> label_data(raw, data.df_activities, idle=True, n_jobs=10)
    1 time                    0   ...      13 activity
    2 2008-03-20 00:34:38  False  ...    True idle
    3 2008-03-20 00:34:39  False  ...   False act1

    Returns
    -------
    df : pd.DataFrame
    """
    df = df.copy()
    df[ACTIVITY] = -1

    if n_jobs == 1:
        df[ACTIVITY] = df[TIME].apply(
                    _map_timestamp2activity,
                    df_act=df_acts,
                    idle=idle)
    else:
        N = get_npartitions()
        if n_jobs == -1 or n_jobs > N:
            n_jobs = N

        #ddf_activities = dd.from_pandas(df_activities, npartitions=get_npartitions())
        # compute with dask in parallel
        df[ACTIVITY] = dd.from_pandas(df[TIME], npartitions=n_jobs).\
                    map_partitions( # apply lambda functions on each partition
                        lambda df: df.apply(
                            _map_timestamp2activity,
                            df_act=df_acts,
                            idle=idle)).\
                    compute(scheduler='processes')
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

