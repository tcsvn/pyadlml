import numpy as np
import pandas as pd
from pyadlml.dataset import START_TIME, END_TIME, TIME, NAME, VAL, DEVICE
from pyadlml.dataset.util import time2int, timestr_2_timedeltas
from pyadlml.dataset.devices import device_rep1_2_rep2, _create_devices, \
    _is_dev_rep1, _is_dev_rep2, contains_non_binary, split_devices_binary

from pyadlml.dataset.util import timestr_2_timedeltas
from pyadlml.dataset._representations.raw import create_raw
from pyadlml.util import get_npartitions
from dask import delayed
import dask.dataframe as dd

def duration_correlation(df_dev, lst_dev=None):
    """ compute the crosscorelation by comparing for every interval the binary values
    between the devices
    
    Parameters
    ----------
        df_dev: pd.DataFrame
            device representation 1 
            columns [time, device, val]
    returns
    -------
        pd.DataFrame (k x k)
        crosscorrelation between each device
    """
    if contains_non_binary(df_dev):
        df_dev, _ = split_devices_binary(df_dev)

    def func(row):
        """ gets two rows and returns a crosstab
        """        
        try:
            td = row.td.to_timedelta64()
        except:
            return None
        states = row.iloc[1:len(row)-1].values.astype(int)
        K = len(states)        
        
        for j in range(K):
            res = np.full((K), 0, dtype='timedelta64[ns]')
            tdiffs = states[j]*states*td            
            row.iloc[1+j] = tdiffs 
        return row

    def create_meta(raw):
        devices = {name : 'object' for name in raw.columns[1:-1]}
        return {**{'time': 'datetime64[ns]', 'td': 'timedelta64[ns]'}, **devices}
        
    dev_lst = df_dev[DEVICE].unique()
    df_dev = df_dev.sort_values(by='time')

    K = len(dev_lst)

    # make off to -1 and on to 1 and then calculate cross correlation between signals
    raw = create_raw(df_dev).applymap(lambda x: 1 if x else -1).reset_index()    
    raw['td'] = raw['time'].shift(-1) - raw['time']
    
    df = dd.from_pandas(raw.copy(), npartitions=get_npartitions())\
                .apply(func, axis=1).drop(columns=['time', 'td']).sum(axis=0)\
                .compute(scheduler='processes')
                #.apply(func, axis=1, meta=create_meta(raw)).drop(columns=['time', 'td']).sum(axis=0)\

    res = pd.DataFrame(data=np.vstack(df.values), columns=df.index, index=df.index)
    # normalize
    res = res/res.iloc[0,0]

    if lst_dev is not None:
        for dev in set(lst_dev).difference(set(list(res.index))):
            res[dev] = pd.NA
            res = res.append(pd.DataFrame(data=pd.NA, columns=res.columns, index=[dev]))
    return res

def devices_trigger_count(df, lst=None):
    """ counts the amount a device was triggered
    Parameters
    ----------
    df : pd.DataFrame
        Contains representation 1 of the devices
        column: [time, device, val]
    lst : list
        Optional list of device names
    Returns
    ------
    df : pd.DataFrame
        columns [device, trigger count]
    """
    assert _is_dev_rep1(df)

    col_label = 'trigger_count'

    ser = df.groupby(DEVICE)[DEVICE].count()
    df = pd.DataFrame({DEVICE:ser.index, col_label:ser.values})

    if lst is not None:
        for dev in set(lst).difference(set(list(df[DEVICE]))):
            df = df.append(pd.DataFrame(data=[[dev, 0]], columns=df.columns, index=[len(df)]))
    return df.sort_values(by=DEVICE)

#def devices_dist(df_dev, t_range='day', n=1000):
#    """
#    returns an array where for one (day | week) the device
#    are sampled over time
#    """
#    return _create_dist(df_dev, label=DEVICE, t_range=t_range, n=n)


def trigger_time_diff(df):
    """ adds a column with time in seconds between triggers of devices 
    Parameters
    ----------
    df : pd.DataFrame
        devices in representation 1
    Returns
    -------
    X : np.array of len(df)
        time deltas in seconds
    """
    # create timediff to the previous trigger
    diff_seconds = 'ds'
    df = df.copy().sort_values(by=[TIME])

    # compute the seconds to the next device
    df[diff_seconds] = df[TIME].diff().shift(-1)/pd.Timedelta(seconds=1)
    return df[diff_seconds].values[:-1]

def devices_td_on(df):
    """ computes the difference for each datapoint in device
    """
    time_difference = 'td'
    if contains_non_binary(df):
        df, _ = split_devices_binary(df)

    if not _is_dev_rep2(df):
        df, _ = device_rep1_2_rep2(df.copy(), drop=False)
    df[time_difference] = df[END_TIME] - df[START_TIME]
    return df[[DEVICE, time_difference]]


def devices_on_off_stats(df, lst=None):
    """
    calculate the time and proportion a device was on vs off
    params:
        df: pd.Dataframe in shape of repr2

    returns: pd.DataFrame
        | device | td_on    | td_off   | frac on   | frac off |
        --------------------------------------------------
        | freezer| 00:10:13 | 27 days   | 0.001    | 0.999   |
        ....

    """
    diff = 'diff'
    td_on = 'td_on'
    td_off = 'td_off'
    frac_on = 'frac_on'
    frac_off = 'frac_off'

    if contains_non_binary(df):
        df, _ = split_devices_binary(df)

    if not _is_dev_rep2(df):
        df, _ = device_rep1_2_rep2(df.copy(), drop=False)
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
    if lst is not None:
        for dev in set(lst).difference(set(list(df.index))):
            df = df.append(pd.DataFrame(data=[[pd.NaT, pd.NaT, pd.NA, pd.NA]], columns=df.columns, index=[dev]))
    return df.reset_index()\
        .rename(columns={'index':DEVICE})\
        .sort_values(by=[DEVICE])

def device_tcorr(df, lst_dev=None, t_window='20s'):
    """ computes for every time window the prevalence of device triggers
        for each device

    Parameters
    ----------
    df : pd.DataFrame
        device representation 1
        columns: [time, device, val]

    t_windows : list
        time frames or a single window (string)

    Returns 
    -------
    res : pd.DataFrame
    """

    t_window = timestr_2_timedeltas(t_window)[0]
    
    # create timediff to the previous trigger
    df = df.copy() 
    df['time_diff'] = df[TIME].diff()

    #knn
    #    do cumsum for row_duration 
    #    for each row mask the rows that fall into the given area
    if lst_dev is not None:
        dev_list = lst_dev
    else:
        dev_list =  df.device.unique()
    
    df.iloc[0,3] = pd.Timedelta(0, 's')
    df['cum_sum'] = df['time_diff'].cumsum()
    
    # create cross table with zeros
    res_df = pd.DataFrame(columns=dev_list, index=dev_list)
    for col in res_df.columns:
        res_df[col].values[:] = 0

    # this whole iterations can be done in parallel
    for row in df.iterrows():
        td = row[1].cum_sum
        dev_name = row[1].device

        df['tmp'] = (td-t_window < df['cum_sum']) & (df['cum_sum'] < td+t_window)
        tmp = df.groupby(DEVICE)['tmp'].sum()
        res_df.loc[dev_name] += tmp

    return res_df.sort_index(axis=0, ascending=True) \
        .sort_index(axis=1, ascending=True) \
        .replace(pd.NA, 0)

def device_triggers_one_day(df, lst=None, t_res='1h'):
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
    df = df.copy()
    # set devices time to their time bin
    df[TIME] = df[TIME].apply(time2int, args=[t_res])

    # every trigger should count
    df[VAL] = 1
    df = df.groupby([TIME, DEVICE]).sum().unstack()
    df = df.fillna(0)
    df.columns = df.columns.droplevel(0)

    if lst is not None:
        for device in set(lst).difference(df.columns):
           df[device] = 0
    return df
