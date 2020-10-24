import numpy as np
import pandas as pd
from pyadlml.dataset import START_TIME, END_TIME, TIME, \
    TIME, NAME, VAL, DEVICE
from pyadlml.dataset.util import time2int, timestr_2_timedeltas
from pyadlml.dataset.devices import device_rep1_2_rep2, _create_devices, \
                                    _is_dev_rep1, _is_dev_rep2

from pyadlml.dataset.util import timestr_2_timedeltas
from pyadlml.dataset._representations.raw import create_raw
from pyadlml.util import get_npartitions
from dask import delayed
import dask.dataframe as dd

def duration_correlation_parallel(df_dev):
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
        
    dev_lst = df_dev['device'].unique()
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
    return res/res.iloc[0,0]


def duration_correlation(df_dev):
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
    dev_lst = df_dev['device'].unique()
    df_dev = df_dev.sort_values(by='time')

    K = len(dev_lst)
    crosstab = np.full((K,K), 0, dtype='timedelta64[ns]')

    # make off to -1 and on to 1 and then calculate cross correlation between signals
    
    states = np.full(K, -1)
    prae_time = df_dev.iloc[0].time
    dev_idx = np.where(dev_lst == df_dev.iloc[0].device)[0][0]
    states[dev_idx] = 1
    
    
    # sweep through all 
    i=0
    for row in df_dev.iterrows():
        if i == 0: i+=1; continue

        # for every device determine cross correlation by multiplying
        # the state of the device with the vector of states in order 
        # to know if to subtract or add the time in the previous interval
        nt = row[1].time
        td = (nt - prae_time).to_timedelta64()   
        for j in range(K):
            dev_st = states[j]
            tdiffs = dev_st*states*td
            crosstab[j,:] = crosstab[j,:] + tdiffs

        # update state array with new state and set new time
        dev_idx = np.where(dev_lst == row[1].device)[0][0]
        if row[1].val:
            states[dev_idx] = 1
        else:
            states[dev_idx] = -1
        prae_time = nt
    
    # normalize by the whole time. Diagonal contains full timeframe
    crosstab = crosstab/crosstab[0,0]
    ct = pd.DataFrame(data=crosstab, index=dev_lst, columns=dev_lst)
    return ct


def devices_trigger_count(df):
    """ counts the amount a device was triggered
    Parameters
    ----------
    df : pd.DataFrame
        Contains representation 1 of the devices
        column: [time, device, val]

    Returns
    ------
    """
    assert _is_dev_rep1(df)

    df = df.groupby('device')['device'].count()
    df = pd.DataFrame(df)
    df.columns = ['trigger count']
    return df

def devices_dist(df_dev, t_range='day', n=1000):
    """
    returns an array where for one (day | week) the device
    are sampled over time 
    """
    return _create_dist(df_dev, label=DEVICE, t_range=t_range, n=n)


def trigger_time_diff(df):
    """ adds a column with time in seconds between triggers of devices 
    Parameters
    ----------
    df : pd.DataFrame
        devices in representation 1
    Returns
    -------
    X : np.array
        time deltas in seconds
    """
    # create timediff to the previous trigger
    df['time_diff'] = df['time'].diff()

    # calculate duration to next trigger
    df['row_duration'] = df.time_diff.shift(-1)

    # calculate the time differences for each device sepeartly
    #df = df.sort_values(by=['device','time'])
    #df['time_diff2'] = df['time'].diff()
    #df.loc[df.device != df.device.shift(), 'time_diff2'] = None
    df = df.sort_values(by=['time'])
    df['sec_to_next'] = df['row_duration']/pd.Timedelta(seconds=1)
    X = df['sec_to_next'].values[:-1]

    return X

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
    lst : list of panda dataframes
    """

    t_windows = timestr_2_timedeltas(t_windows)
    
    # create timediff to the previous trigger
    df = df.copy() 
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
    # compute new table
    df['time'] = df['time'].apply(time2int, args=[t_res])
    df = df.groupby(['time', 'device']).sum().unstack()
    df = df.fillna(0)
    df.columns = df.columns.droplevel(0)
    return df

