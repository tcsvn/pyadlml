"""
this file is to bring datasets into specific representations
"""
import pandas as pd
import numpy as np
from pyadlml.util import get_parallel, get_npartitions
from pyadlml.constants import OTHER, OTHER_MIN_DIFF, TIME, ACTIVITY, START_TIME, END_TIME


def label_data2(df_devs: pd.DataFrame, df_acts: pd.DataFrame, other=False):
    """



    """
    df = df_acts.copy()
    df_devs = df_devs.copy()



    # A device is counted as belonging to an activity if the timestamp
    # falls into the open interval df_acts[START_TIME] <= ts <= df_acts[END_TIME]
    # For events occuring exactly at the time an activity starts or ends
    # move the activity boundary a bit in the future or past to facilitate 
    # the expected behavior
    assert not df_devs[TIME].duplicated().any() 
    eps = pd.Timedelta('1ns')
    eps_other = '1ns'

    mask_act_et_clsn = df[END_TIME].isin(df_devs[TIME])
    mask_act_st_clsn = df[START_TIME].isin(df_devs[TIME])
    df.loc[df[mask_act_et_clsn].index, END_TIME] += eps
    df.loc[df[mask_act_st_clsn].index, START_TIME] -= eps


    df = df.rename(columns={START_TIME: TIME})

    # Fill up with other
    df_other = df[[END_TIME, ACTIVITY]]
    df_other.loc[:, ACTIVITY] = OTHER
    df_other = df_other.rename(columns={END_TIME: TIME})
    df = df.drop(columns=END_TIME)
    df = pd.concat([df, df_other.iloc[:-1]], ignore_index=True, axis=0) \
        .sort_values(by=TIME) \
        .reset_index(drop=True)
    df['diff'] = df[TIME].shift(-1) - df[TIME]
    mask_invalid_others = (df['diff'] < eps_other) & (
        df[ACTIVITY] == OTHER)
    df = df[~mask_invalid_others][[TIME, ACTIVITY]]

    # CAVE replace the ending 
    df = pd.concat([df, pd.Series({
        TIME: df_acts.at[df_acts.index[-1], END_TIME],
        ACTIVITY: OTHER}
    ).to_frame().T]).reset_index(drop=True)

    df_label = df.copy()
    df_label = df_label.sort_values(by=TIME).reset_index(drop=True)
    df_devs = df_devs.sort_values(by=TIME).reset_index(drop=True)


    # if prediction is larger than ground truth activities pad gt with 'other'
    if df_devs[TIME].iat[0] < df_label[TIME].iat[0]:
        df_label = pd.concat([df_label,
            pd.Series({
                TIME: df_devs.at[0, TIME] - pd.Timedelta('1ms'), 
                ACTIVITY: OTHER
            }).to_frame().T], axis=0, ignore_index=True)

    df_dev_tmp = df_devs.copy()
    df_act_tmp = df_label.copy().sort_values(by=TIME).reset_index(drop=True)

    # Add columns
    df_dev_tmp[ACTIVITY] = np.nan
    for col in set(df_devs.columns).difference(df_act_tmp.columns):
        df_act_tmp[col] = np.nan

    df = pd.concat([df_dev_tmp, df_act_tmp], ignore_index=True, axis=0)\
        .sort_values(by=TIME)\
        .reset_index(drop=True)

    df[ACTIVITY] = df[ACTIVITY].ffill()
    df = df.dropna().reset_index(drop=True)
    assert len(df) == len(df_devs)

    if not other:
        other_mask = (df[ACTIVITY] == OTHER)
        df[other_mask] = np.nan

    return df


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
        # ddf_activities = dd.from_pandas(df_activities, npartitions=get_npartitions())
        # compute with dask in parallel
        df_devs[ACTIVITY] = dd.from_pandas(df_devs[TIME], npartitions=n_jobs).\
            map_partitions(  # apply lambda functions on each partition
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
