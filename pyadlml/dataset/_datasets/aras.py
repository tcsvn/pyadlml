import numpy as np
import pandas as pd
from pyadlml.dataset.activities import _is_activity_overlapping
from pyadlml.dataset.devices import correct_devices
from pyadlml.dataset.obj import Data

def _get_device_map(data_path):
    """
    Parameters
    ----------
    data_path : string
        path to folder where aras dataset resides
    Returns
    -------
    devices : pd.DataFrame
        dataframe with information about devices 
        the index represents the column of the sensor table
    """
    cols = ['Sensor ID', 'Sensor Type', 'Place']
    devices = pd.DataFrame(columns=cols)
    with open (data_path + 'README', 'r') as file:
        lines = file.readlines()
        # create header
        for line in lines[7:27]:
            s = list(filter(None, line.split('\t')))
            s[-1:] = [s[-1:][0][:-1]]
            tmp = dict(zip(cols, s[1:]))
            devices = devices.append(tmp, ignore_index=True)

        file.close()
    return devices


def _get_activity_map(data_path):
    """
    Parameters
    ----------
    data_path : string
        path to folder where aras dataset resides
    Returns
    -------
    devices : pd.DataFrame
        Contains id matching label of activity 
    """    
    cols = ['ID', 'activity']
    activities = pd.DataFrame(columns=cols)

    with open (data_path + 'README', 'r') as file:
        lines = file.readlines()
        # create header
        for line in lines[33:60]:
            s = list(filter(None, line.split('\t')))
            s[-1:] = [s[-1:][0][:-1]]
            tmp = dict(zip(cols, s))
            activities = activities.append(tmp, ignore_index=True)

        file.close()
    activities['ID'] = activities['ID'].astype(int)
    return activities

def _read_data(data_path, activity_map, device_map):
    #DEF
    file_name = 'DAY_{}.txt'
    cols = list(device_map['Sensor ID']) + (['Resident 1', 'Resident 2'])
    res = pd.DataFrame(columns = cols)

    # read in all files
    for i in range(1,31):
        file_path = data_path + file_name.format(i)
        df = pd.read_table(file_path, sep=' ', header=None)
        df.columns = cols
        if i < 10:
            date = '2000-01-0{}'.format(i)
        else:
            date = '2000-01-{}'.format(i)
        df.index = pd.date_range(date, periods=86400, freq='s')

        # delete all rows where nothing changes
        mask = (df.diff(axis=0).sum(axis=1) == 0.0)
        mask.iloc[0] = False
        df = df[~mask]

        # append all dataframes
        res = res.append(df)

    # label activities
    act_dict = dict(zip(activity_map['ID'],activity_map['activity']))
    res['Resident 1'] = res['Resident 1'].map(act_dict)
    res['Resident 2'] = res['Resident 2'].map(act_dict)
    return res


def _create_activity_df(df, res_name):
    # bring data into representation 3 
    df = pd.DataFrame(df[res_name]).reset_index()
    
    mask_start = df[res_name] != df[res_name].shift(1)
    df_start = df[mask_start]
    df_start = df_start.rename(columns={'index' : 'start_time'})
    df_start = df_start.reset_index(drop=True)

    mask_end = df[res_name] != df[res_name].shift(-1)
    df_end = df[mask_end].reset_index(drop=True)
    
    df_start['end_time'] = df_end['index']
    
    df = df_start[['start_time', 'end_time', res_name]]
    df = df.sort_values(by='start_time')
    df = df.rename(columns={res_name : 'activity'})
    return df


def _create_device_df(df):
    """ gets a raw representation and returns devices in rep3
    Parameters
    ----------
    df : pd.DataFrame
        raw representation 
    """
    df = df.copy().iloc[:,:-2] # exclude activities
    mask = (df.diff(axis=0).sum(axis=1) == 0.0)
    mask.iloc[0] = False
    df = df[~mask]
    
    # is true where the devices change
    mask_0to1 = (df.diff(axis=0) == 1)
    mask_1to0 = (df.diff(axis=0) == -1)
    
    # for every device append the rows where the device changes state
    res = pd.DataFrame(columns=['time', 'device', 'val'])
    for device in df.columns:
        dev_0to1 = pd.DataFrame(df[device][mask_0to1[device]])
        dev_1to0 = pd.DataFrame(df[device][mask_1to0[device]])
        tmp = pd.concat([dev_0to1, dev_1to0]).reset_index()
        tmp.columns=['time', 'val']
        tmp['device'] = device
        
        res = res.append(tmp)
    
    res = res.sort_values(by='time').reset_index(drop=True)
    return res


def load(data_path):
    device_map = _get_device_map(data_path)
    activity_map = _get_activity_map(data_path)
    df = _read_data(data_path, activity_map, device_map)

    df_res1_act = _create_activity_df(df, 'Resident 1')
    df_res2_act = _create_activity_df(df, 'Resident 2')

    assert not _is_activity_overlapping(df_res1_act) \
        or not _is_activity_overlapping(df_res2_act)
    df_dev = _create_device_df(df)

    df_dev = correct_devices(df_dev)

    data = Data(df_res1_act, df_dev)

    data.df_dev_map = device_map
    data.df_act_map = activity_map        
    data.df_activities_res2 = df_res2_act
    return data