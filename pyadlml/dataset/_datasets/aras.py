import os
from pathlib import Path
import pandas as pd
from pyadlml.constants import VALUE, TIME, DEVICE, END_TIME, START_TIME, ACTIVITY
from pyadlml.dataset._core.activities import ActivityDict
from pyadlml.dataset.io import DataFetcher, MegaDownloader

ARAS_URL = 'https://mega.nz/file/hVpRADoZ#GLLZDV4Y-vgdEeEDTXnFxeG3eKllhTljMM1RK-eGyh4'
ARAS_FILENAME = 'aras.zip'

RES_1 = 'resident1'
RES_2 = 'resident2'

__all__ = ['fetch_aras']

class ArasFetcher(DataFetcher):
    def __init__(self, *args, **kwargs):

        downloader = MegaDownloader(
            url=ARAS_URL,
            fn=ARAS_FILENAME,
            url_cleaned=None,
            fn_cleaned=None,
        )

        super().__init__(
            dataset_name='aras',
            downloader=downloader,
            correct_activities=True,
            correct_devices=True
        )

    def load_data(self, folder_path: Path) -> dict:

        device_map = _get_device_map(folder_path)
        activity_map = _get_activity_map(folder_path)
        df = _read_data(folder_path, activity_map, device_map)

        df_res1_act = _create_activity_df(df, RES_1)
        df_res2_act = _create_activity_df(df, RES_2)

        df_dev = _create_device_df(df)
        lst_dev = df_dev[DEVICE].unique()
        act_dict = ActivityDict({
                RES_1:df_res1_act,
                RES_2:df_res2_act,
        })
        return dict(
            devices=df_dev,
            device_list=lst_dev,
            activities=act_dict,
            activity_list={
                RES_1: df_res1_act[ACTIVITY].unique(),
                RES_2: df_res2_act[ACTIVITY].unique(),
            }
        )

def fetch_aras(keep_original=True, cache=True, retain_corrections=False,
               folder_path=None):
    """
    Fetches the aras dataset from the internet. The original dataset or its cached version
    is stored in the data_home folder.

    Parameters
    ----------
    keep_original : bool, default=True
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool, default=True
        Determines whether the data object should be stored as a binary file for quicker access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.
    retain_corrections : bool, default=False
        When set to *true* data points that are changed or dropped during preprocessing
        are listed in the respective attributes of the data object.  Fore more information
        about the attributes refer to the :ref:`user guide <error_correction>`.

    Examples
    --------
    >>> from pyadlml.dataset import fetch_aras
    >>> data = fetch_aras()
    >>> dir(data)
    >>> [..., df_activities, df_devices, ...]

    Returns
    -------
    data : object
    """
    return ArasFetcher()(keep_original=keep_original, cache=cache, #load_cleaned=load_cleaned,
                              retain_corrections=retain_corrections, folder_path=folder_path)


    return data


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
    with open(os.path.join(data_path, 'README'), 'r') as file:
        lines = file.readlines()
        # create header
        for line in lines[7:27]:
            s = list(filter(None, line.split('\t')))
            s[-1:] = [s[-1:][0][:-1]]
            devices.loc[len(devices), :] = [a.strip() for a in s[1:]]

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
    activities = pd.DataFrame(columns=['ID', 'activity'])

    with open(os.path.join(data_path, 'README'), 'r') as file:
        lines = file.readlines()
        # create header
        for line in lines[33:60]:
            s = list(filter(None, line.split('\t')))
            s[-1:] = [s[-1:][0][:-1]]
            activities.loc[len(activities), :] = s

    activities['ID'] = activities['ID'].astype(int)
    return activities

def _read_data(data_path, activity_map, device_map):
    #DEF
    file_name = 'DAY_{}.txt'
    cols = list(device_map['Sensor ID']) + ([RES_1, RES_2])
    res = pd.DataFrame(columns = cols)

    # read in all files
    for i in range(1,31):
        file_path = os.path.join(data_path, file_name.format(i))
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
        res = pd.concat([res, df])

    # label activities
    act_dict = dict(zip(activity_map['ID'],activity_map['activity']))
    res[RES_1] = res[RES_1].map(act_dict)
    res[RES_2] = res[RES_2].map(act_dict)
    return res


def _create_activity_df(df, res_name):
    # bring data into representation 3 
    df = pd.DataFrame(df[res_name]).reset_index()
    
    mask_start = df[res_name] != df[res_name].shift(1)
    df_start = df[mask_start]
    df_start = df_start.rename(columns={'index' : START_TIME})
    df_start = df_start.reset_index(drop=True)

    mask_end = df[res_name] != df[res_name].shift(-1)
    df_end = df[mask_end].reset_index(drop=True)
    
    df_start[END_TIME] = df_end['index']
    
    df = df_start[[START_TIME, END_TIME, res_name]]
    df = df.sort_values(by=START_TIME)
    df = df.rename(columns={res_name: ACTIVITY})
    df = df[df[ACTIVITY] != 'Other'] # remove Other activity as it is the same concept as 'idle'
    return df


def _create_device_df(df):
    """ gets a raw representation and returns devices in rep1
    Parameters
    ----------
    df : pd.DataFrame
        raw representation 
    """
    df = df.copy().iloc[:, :-2] # exclude activities
    mask = (df.diff(axis=0).sum(axis=1) == 0.0)
    mask.iloc[0] = False
    df = df[~mask]
    
    # is true where the devices change
    mask_0to1 = (df.diff(axis=0) == 1)
    mask_1to0 = (df.diff(axis=0) == -1)
    
    # for every device append the rows where the device changes state
    res = pd.DataFrame(columns=[TIME, DEVICE, VALUE])
    for device in df.columns:
        dev_0to1 = pd.DataFrame(df[device][mask_0to1[device]])
        dev_1to0 = pd.DataFrame(df[device][mask_1to0[device]])
        tmp = pd.concat([dev_0to1, dev_1to0]).reset_index()
        tmp.columns = [TIME, VALUE]
        tmp[DEVICE] = device
        
        res = pd.concat([res, tmp])
    
    res = res.sort_values(by=TIME).reset_index(drop=True)
    res[VALUE] = res[VALUE].astype(bool)
    return res

