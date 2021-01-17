import os

import pandas as pd
from pyadlml.dataset.activities import correct_activities
from pyadlml.dataset.devices import correct_devices
from pyadlml.dataset.obj import Data
from pyadlml.dataset import START_TIME, END_TIME, DEVICE, VAL, TIME, ACTIVITY

DATA_NAME = 'devices.csv'
DEV_MAP_NAME = 'device_mapping.csv'
ACT_MAP_NAME = 'activity_mapping.csv'
ACT_NAME = 'activities_subject_%s.csv'


def _read_activities(path_to_file):
    """
    """
    activities = pd.read_csv(path_to_file)
    activities[START_TIME] = pd.to_datetime(activities[START_TIME])
    activities[END_TIME] = pd.to_datetime(activities[END_TIME])
    return activities


def _read_devices(path_to_dev_file, path_to_mapping):
    devices = pd.read_csv(path_to_dev_file)
    dev_map = pd.read_csv(path_to_mapping, index_col='id')\
                .to_dict()[DEVICE] 
    devices[DEVICE] = devices[DEVICE].map(dev_map)
    devices[VAL] = devices[VAL].astype(str).replace({"0": False, "1": True})
    devices[TIME] = pd.to_datetime(devices[TIME])
    devices = devices.reset_index(drop=True)
    return devices


def _read_activity_list(path_to_file):
    lst_activities = pd.read_csv(path_to_file)
    return list(lst_activities[ACTIVITY])


def _read_device_list(path_to_file):
    lst_devices = pd.read_csv(path_to_file)
    return list(lst_devices[DEVICE])


def load(folder_path, subjects):
    """ loads the dataset from a folder
    Parameters
    ----------
    folder_path : 
    subjects : list or None
        The names of the subjects to be included

    Returns
    -------
    data: Data obj
    """
    assert isinstance(folder_path, str)
    assert isinstance(subjects, list)

    df_dev = _read_devices(os.path.join(folder_path, DATA_NAME),
                           os.path.join(folder_path, DEV_MAP_NAME))
    df_dev = correct_devices(df_dev)

    # get mappings
    lst_dev = _read_device_list(os.path.join(folder_path, DEV_MAP_NAME))
    lst_act = _read_activity_list(os.path.join(folder_path, ACT_MAP_NAME))

    data = Data(None, df_dev, activity_list=lst_act, device_list=lst_dev)

    for subject in subjects:
        df_act = _read_activities(os.path.join(folder_path, ACT_NAME%(subject)))
        # correct possible overlaps in activities
        df_act, cor_lst = correct_activities(df_act)
        setattr(data, 'df_activities_{}'.format(subject), df_act)
        setattr(data, 'correction_activities_{}'.format(subject), cor_lst)

    return data
