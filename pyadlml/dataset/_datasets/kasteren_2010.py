import pandas as pd
from pyadlml.dataset._core.obj import Data
from pyadlml.dataset import ACTIVITY, VAL, \
    START_TIME, END_TIME, TIME, NAME, DEVICE
from pyadlml.dataset._core.activities import correct_activities
from pyadlml.dataset._core.devices import correct_devices, CORRECTION_TS, CORRECTION_ONOFF_INCONS
from pyadlml.dataset.io import fetch_handler as _fetch_handler
import os

AMSTERDAM_URL = 'https://mega.nz/file/AYhzDLaS#n-CMzBO_raNAgn2Ep1GNgbhah0bHQzuA48PqO_ODEAg'
AMSTERDAM_FILENAME = 'amsterdam.zip'

def fetch_amsterdam(keep_original=False, cache=True, retain_corrections=False):
    """
    Fetches the amsterdam dataset from the internet. The original dataset or its cached version
    is stored in the :ref:`data home <storage>` folder.

    Parameters
    ----------
    keep_original : bool, default=False
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool, default=True
        Determines whether the data object should be stored as a binary file for quicker access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.
    retain_corrections : bool, default=False
        When set to *true*, data points that change or drop during preprocessing
        are listed in respective attributes of the data object. Fore more information
        about error correction refer to the :ref:`user guide <error_correction>`.

    Examples
    --------
    >>> from pyadlml.dataset import fetch_amsterdam
    >>> data = fetch_amsterdam()
    >>> dir(data)
    >>> [..., df_activities, df_devices, ...]

    Returns
    -------
    data : object
    """
    dataset_name = 'amsterdam'

    def load_amsterdam(folder_path):
        device_fp = os.path.join(folder_path, "kasterenSenseData.txt")
        activity_fp = os.path.join(folder_path, "kasterenActData.txt")

        df_act = _load_activities(activity_fp)
        df_dev = _load_devices(device_fp)
        df_act, correction_act = correct_activities(df_act, retain_corrections=retain_corrections)

        df_dev, correction_dev_dict = correct_devices(df_dev, retain_correction=retain_corrections)
        lst_act = df_act[ACTIVITY].unique()
        lst_dev = df_dev[DEVICE].unique()

        data = Data(df_act, df_dev, activity_list=lst_act, device_list=lst_dev)
        if retain_corrections:
            if correction_act is not None:
                data.correction_activities = correction_act
            if correction_dev_dict is not None:
                data.correction_devices_duplicate_timestamps = correction_dev_dict[CORRECTION_TS]
                data.correction_devices_on_off_inconsistency = correction_dev_dict[CORRECTION_ONOFF_INCONS]
        return data

    data = _fetch_handler(keep_original, cache, dataset_name,
                        AMSTERDAM_FILENAME, AMSTERDAM_URL,
                        load_amsterdam)
    return data


def _load_activities(activity_fp):
    act_map = {
        1: 'leave house',
        4: 'use toilet',
        5: 'take shower',
        10:'go to bed',
        13:'prepare Breakfast',
        15:'prepare Dinner',
        17:'get drink'
        }
    ide = 'id'
    df = pd.read_csv(activity_fp,
                     sep="\t",
                     skiprows=23,
                     skipfooter=1,
                     parse_dates=True,
                     names=[START_TIME, END_TIME, ide],
                     engine='python') 
    df[START_TIME] = pd.to_datetime(df[START_TIME])
    df[END_TIME] = pd.to_datetime(df[END_TIME])
    df[ACTIVITY] = df[ide].map(act_map)
    df = df.drop(ide, axis=1)
    return df


def _load_devices(device_fp):
    sens_labels = {
        1: 'Microwave', 
        5: 'Hall-Toilet door',
        6: 'Hall-Bathroom door',
        7: 'Cups cupboard',
        8: 'Fridge',
        9: 'Plates cupboard',
        12: 'Frontdoor',
        13: 'Dishwasher',
        14: 'ToiletFlush',
        17: 'Freezer',
        18: 'Pans Cupboard',
        20: 'Washingmachine',
        23: 'Groceries Cupboard',
        24: 'Hall-Bedroom door'
    }
    ide = 'id'
    sens_data = pd.read_csv(device_fp,
                sep="\t",
                skiprows=23,
                skipfooter=1,
                parse_dates=True,
                names=[START_TIME, END_TIME, ide, VAL],
                engine='python' #to ignore warning for fallback to python engine because skipfooter
                #dtype=[]
                )


    #k todo declare at initialization of dataframe
    sens_data[START_TIME] = pd.to_datetime(sens_data[START_TIME])
    sens_data[END_TIME] = pd.to_datetime(sens_data[END_TIME])
    sens_data[VAL] = sens_data[VAL].astype('bool')
    sens_data = sens_data.drop(VAL, axis=1)

    # replace numbers with the labels
    sens_data[DEVICE] = sens_data[ide].map(sens_labels)
    sens_data = sens_data.drop(ide, axis=1)
    sens_data = sens_data.sort_values(START_TIME)
    return sens_data