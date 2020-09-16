import numpy as np
import pandas as pd
from pyadlml.dataset._dataset import START_TIME, END_TIME, TIME, \
    TIME, NAME, VAL, DEVICE

"""
    df_devices:
        also referred to as rep1
        is used to calculate statistics for devices more easily
        and has lowest footprint in storage. Most of the computation 
        is done using this format
        Exc: 
           | start_time | end_time   | device    
        ----------------------------------------
         0 | timestamp   | timestamp | dev_name  
    
    df_dev_rep3:
        a lot of data is found in this format.
            | time      | device    | state
        ------------------------------------
         0  | timestamp | dev_name  |   1
"""

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
    
    df = df_rep3.copy().reset_index(drop=True)
    df = df.sort_values(TIME)
    df.loc[:,'ones'] = 1
    
    # seperate the 0to1 and 1to0 device changes
    df.loc[:,VAL] = df[VAL].astype(bool)
    df_start = df[df[VAL]]
    df_end = df[~df[VAL]]    
    df_end = df_end.rename(columns={TIME: END_TIME})
    df_start = df_start.rename(columns={TIME: START_TIME})
    
    # ordered in time to index them and make a correspondence
    df_end.loc[:,'pairs'] = df_end.groupby([DEVICE])['ones'].apply(lambda x: x.cumsum())
    df_start.loc[:,'pairs'] = df_start.groupby([DEVICE])['ones'].apply(lambda x: x.cumsum())        
        
    
    df = pd.merge(df_start, df_end, on=['pairs', DEVICE])
    df = df.sort_values(START_TIME)
    
    # sanity checks
    diff = int(len(df_rep3)/2) - len(df)
    assert diff == 0, 'input {} - {} == {} result. Somewhere two following events of the \
    #        same device had the same starting point and end point'.format(len(df_rep3)/2, len(df), diff)
    return df[[START_TIME, END_TIME, DEVICE]]

def device_rep1_2_rep3(df_rep):
    """
    params: df: pd.DataFrame
                rep1: columns (start time, end_time, device)
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
    df = df.sort_values(TIME).reset_index(drop=False)
    
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
    cor_rep3 = cor_rep3.sort_values(by='time')    

    # correct timestamp duplicates
    while _has_timestamp_duplicates(cor_rep3):
        cor_rep3 = correct_device_rep3_ts_duplicates(cor_rep3)

    # correct on off incosistency
    if not is_on_off_consistent(cor_rep3):
        cor_rep3 = correct_on_off_inconsistency(cor_rep3)

    assert not _has_timestamp_duplicates(cor_rep3)
    assert is_on_off_consistent(cor_rep3)

    cor_rep1 = device_rep3_2_rep1(cor_rep3)
    assert _check_devices_sequ_order(cor_rep1)
    
    return cor_rep1, cor_rep3

def is_on_off_consistent(df):
    """ devices can only go on after they are off and vice versa. check if this is true
        for every column
    """
    # check if every device starts by turning on
    for dev in df['device'].unique():
        df_dev = df[df['device'] == dev]
        assert df_dev.iloc[0].val
        
    # check if the alternating pattern of devices holds
    for dev in df['device'].unique():
        df_dev = df[df['device'] == dev].copy().sort_values(by='time').reset_index(drop=True)
        # create alternating mask
        mask = np.zeros((len(df_dev)), dtype=bool)
        mask[::2] = True
        df_dev['alternating'] = mask
        df_dev['diff'] = df_dev['val'] ^ mask
        if df_dev['diff'].sum() > 0:
            return False
    return True

def correct_on_off_inconsistency(df):
    """ 
    has multiple strategies for solving the patterns:
    e.g if the preceeding value is on and the same value is occuring now delete now
    Parameters
    ----------
    df : pd.DataFrame
        device representation 3
        
    Returns
    -------
    df : pd.DataFrame
        device representation 3
    """
    
    no_inconsistency = False # if one inconsistency is checked the loop is left
    while not no_inconsistency:
        no_inconsistency = True
        for dev in df['device'].unique():            
            df_dev = df[df['device'] == dev].copy().sort_values(by='time').reset_index()
            
            # create alternating mask
            mask = np.zeros((len(df_dev)), dtype=bool)
            mask[::2] = True
            df_dev['alternating'] = mask
            df_dev['diff'] = df_dev['val'] ^ mask
            
            if df_dev['diff'].sum() > 0:
                no_inconsistency = False
                
                # get index of rows where the previous value was the same and delete row
                df_dev['same_prec'] = ~(df_dev['val'].shift(1) ^ df_dev['val'])
                df_dev.loc[0, 'same_prec'] = False # correct shift artefact
                indices = list(df_dev[df_dev['same_prec']]['index'])
                
                df = df.drop(indices, axis=0)
            
    return df