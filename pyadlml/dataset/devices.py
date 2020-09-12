import numpy as np
import pandas as pd
from pyadlml.dataset._dataset import START_TIME, END_TIME, TIME, \
    TIME, NAME, VAL, DEVICE

def _create_devices(dev_list, index=None):
    """
    creates an empty device dataframe
    """
    if index is not None:
        return pd.DataFrame(columns=dev_list, index=index)
    else:
        return pd.DataFrame(columns=dev_list)



def check_devices(df):
    """
    """
    case1 = _is_dev_rep1(df) or _is_dev_rep3(df)
    case2 = _check_devices_sequ_order(df)
    return case1 and case2 



def _check_devices_sequ_order(df):
    """
    iterate pairwise through each select device an check if the 
    sequential order in time is inconsistent

    Parameters
    ----------
    df : pd.DataFrame
        device representation 1  with columns [start_time, end_time, devices]
    """
    dev_list = df['device'].unique()
    no_errors = True
    for dev in dev_list:
        df_d = df[df['device'] == dev]
        for i in range(1,len(df_d)):
            st_j = df_d.iloc[i-1].start_time
            et_j = df_d.iloc[i-1].end_time
            st_i = df_d.iloc[i].start_time
            et_i = df_d.iloc[i].end_time
            # if the sequential order is violated return false
            if not (st_j < et_j) or not (et_j < st_i):
                print('~'*50)
                if (st_j >= et_j):                    
                    #raise ValueError('{}; st: {} >= et: {} |\n {} '.format(i-1, st_j, et_j, df_d.iloc[i-1]))
                    print('{}; st: {} >= et: {} |\n {} '.format(i-1, st_j, et_j, df_d.iloc[i-1]))
                if (et_j >= st_i):                    
                    #raise ValueError('{},{}; et: {} >= st: {} |\n{}\n\n{}'.format(i-1,i, et_j, st_i, df_d.iloc[i-1], df_d.iloc[i]))
                    print('{},{}; et: {} >= st: {} |\n{}\n\n{}'.format(i-1,i, et_j, st_i, df_d.iloc[i-1], df_d.iloc[i]))
                no_errors = False
    return no_errors



def _is_dev_rep1(df):
    """
    """
    if not START_TIME in df.columns or not END_TIME in df.columns \
    or not DEVICE in df.columns or len(df.columns) != 3:
        return False
    return True

def _is_dev_rep3(df):
    """
    """
    if DEVICE in df.columns and VAL in df.columns \
    and TIME in df.columns and len(df.columns) == 3:
        return True
    return False

def device_rep3_2_rep1(df_rep3):
    """
    transforms a device representation 3 into 2
    params: df: pd.DataFrame
                rep3: col (time, device, val)
                example row: [2008-02-25 00:20:14, Freezer, False]
    returns: df: (pd.DataFrame)
                rep: columns are (start time, end_time, device)
                example row: [2008-02-25 00:20:14, 2008-02-25 00:22:14, Freezer]         
    """
    
    df = df_rep3.copy().reset_index()
    df = df.sort_values('time')
    df['ones'] = 1
    df['val'] = df['val'].astype(bool)
    
    df_start = df[df['val']]
    df_end = df[~df['val']]

    df_end.rename(columns={'time': 'end_time'}, inplace=True)
    df_start.rename(columns={'time': 'start_time'}, inplace=True)
   
    df_end['pairs'] = df_end.groupby(['device'])['ones'].apply(lambda x: x.cumsum())
    df_start['pairs'] = df_start.groupby(['device'])['ones'].apply(lambda x: x.cumsum())        
    
    
    df = pd.merge(df_start, df_end, on=['pairs', 'device'])
    df = df.sort_values('start_time')
    
    assert int(len(df_rep3)/2) == len(df), 'input {} != {} result. Somewhere two following events of the \
    #        same device had the same starting point and end point'.format(len(df_rep3), len(df))
    return df[['start_time', 'end_time', 'device']]

def device_rep1_2_rep3(df_rep):
    """
    params: df: pd.DataFrame
                rep1: col (start time, end_time, device)
    returns: df: (pd.DataFrame)
                rep3: columns are (time, value, device)
                example row: [2008-02-25 00:20:14, Freezer, False]
    """
    # copy devices to new dfs 
    # one with all values but start time and other way around
    df_start = df_rep.copy().loc[:, df_rep.columns != END_TIME]
    df_end = df_rep.copy().loc[:, df_rep.columns != START_TIME]

    # set values at the end time to zero because this is the time a device turns off
    df_start[VAL] = True
    df_end[VAL] = False

    # rename column 'End Time' and 'Start Time' to 'Time'
    df_start.rename(columns={START_TIME: TIME}, inplace=True)
    df_end.rename(columns={END_TIME: TIME}, inplace=True)

    df = pd.concat([df_end, df_start]).sort_values(TIME)
    df = df.reset_index(drop=True)
    return df

def correct_device_rep3_ts_duplicates(df):
    """
    remove devices that went on and off at the same time. And add a microsecond
    to devices that trigger on the same time
    Parameters
    ----------
    df : pd.DataFrame
        Devices in representation 3; columns [time, device, value]
    """
    eps = pd.Timedelta('10ms')
    
    try:
        df[TIME]
    except KeyError:
        df = df.reset_index()
        
    # remove device if it went on and off at the same time    
    dup_mask = df.duplicated(subset=[TIME, DEVICE], keep=False)    
    df = df[~dup_mask]
    
    df = df.reset_index(drop=True)
    
    
    dup_mask = df.duplicated(subset=[TIME], keep=False)
    duplicates = df[dup_mask]    
    uniques = df[~dup_mask]
    
    # for every pair of duplicates add a millisecond on the second one
    duplicates = duplicates.reset_index(drop=True)
    sp = duplicates['time'] + eps   
    mask_p = (duplicates.index % 2 == 0) 
    
    duplicates['time'] = duplicates['time'].where(mask_p, sp)
    
    # concatenate and sort the dataframe 
    uniques = uniques.set_index(TIME)
    duplicates = duplicates.set_index(TIME)
    df = pd.concat([duplicates, uniques], sort=True)
    
    # set the time as index again
    df = df.sort_values(TIME)
    
    return df
    
def _has_timestamp_duplicates(df):
    """ check whether there are duplicates in timestamp present
    Parameters
    ----------
    df : pd.DataFrame
        data frame representation 3
    """
    df = df.copy()
    try:
        dup_mask = df.duplicated(subset=[TIME], keep=False)
    except KeyError:
        df = df.reset_index()
        dup_mask = df.duplicated(subset=[TIME], keep=False)
    return dup_mask.sum() > 0

def correct_devices(df):
    """
    Parameters
    ----------
    df : pd.DataFrame
        either in device representation 1 or 3
    Returns
    -------
    cor_rep1 : pd.DataFrame
    cor_rep3 : pd.DataFrame
    """
    df = df.copy()
    df = df.drop_duplicates()        
    
    # bring in correct representation
    if _is_dev_rep1(df):
        cor_rep3 = device_rep1_2_rep3(df)        
    elif _is_dev_rep3(df):
        cor_rep3 = df
    else:
        raise ValueError
    
    # correct timestamp duplicates
    while _has_timestamp_duplicates(cor_rep3):
        cor_rep3 = correct_device_rep3_ts_duplicates(cor_rep3)
    
    cor_rep1 = device_rep3_2_rep1(cor_rep3)

    assert not _has_timestamp_duplicates(cor_rep3)
    assert _check_devices_sequ_order(cor_rep1)
    
    return cor_rep1, cor_rep3