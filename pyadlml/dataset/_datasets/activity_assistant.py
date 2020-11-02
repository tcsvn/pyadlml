import pandas as pd
from pyadlml.dataset.activities import correct_activities
from pyadlml.dataset.devices import correct_devices
from pyadlml.dataset.obj import Data
from pyadlml.dataset import START_TIME, END_TIME, DEVICE, VAL, TIME

DATA_NAME = 'devices.csv'
DEV_MAP_NAME = 'device_mapping.csv'
ACT_NAME = 'activities_subject_%s.csv'

def _read_activities(path_to_file):
    activities = pd.read_csv(path_to_file)
    activities[START_TIME] = pd.to_datetime(activities[START_TIME])
    activities[END_TIME] = pd.to_datetime(activities[END_TIME])
    return activities

def _read_devices(path_to_dev_file, path_to_mapping):
    devices = pd.read_csv(path_to_dev_file)
    dev_map = pd.read_csv(path_to_mapping, index_col='id')\
                .to_dict()[DEVICE] 
    devices[DEVICE] = devices[DEVICE].map(dev_map)
    devices[VAL] = devices[VAL].astype(bool)
    devices[TIME] = pd.to_datetime(devices[TIME])
    devices = devices.reset_index(drop=True)
    return devices

def load(folder_path, subject):
    """
    """
    df_dev = _read_devices(folder_path + '/' + DATA_NAME,
                            folder_path + '/' + DEV_MAP_NAME)
    df_act = _read_activities(folder_path + '/' + ACT_NAME%(subject))

    # correct possible overlaps in activities
    df_act, cor_lst = correct_activities(df_act)
    
    # correct possible duplicates for representation 2    
    df_dev = correct_devices(df_dev)
    data = Data(df_act, df_dev)
    data.correction_activities = cor_lst
    return data
