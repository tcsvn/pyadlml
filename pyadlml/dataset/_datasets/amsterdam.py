import pandas as pd
from pathlib import Path
from pyadlml.dataset.io import MegaDownloader
from pyadlml.constants import ACTIVITY, VALUE, \
    START_TIME, END_TIME, TIME, NAME, DEVICE
from pyadlml.dataset._core.devices import device_boolean_on_states_to_events
from pyadlml.dataset.io import DataFetcher

AMSTERDAM_URL = 'https://mega.nz/file/AYhzDLaS#n-CMzBO_raNAgn2Ep1GNgbhah0bHQzuA48PqO_ODEAg'
AMSTERDAM_CLEANED_URL = 'https://mega.nz/file/9R5zxKiT#Ko5NLmAWofmOTGJr0dvjHzjkX8xvKGtZw_Dy5AQ1ogY'
AMSTERDAM_FILENAME = 'amsterdam.zip'
AMSTERDAM_CLEANED_FILENAME = 'amsterdam_cleaned.joblib'
DEVICES_FN = "kasterenSenseData.txt"
ACTIVITIES_FN = "kasterenActData.txt"

__all__ = ['fetch_amsterdam']


class AmsterdamFetcher(DataFetcher):
    def __init__(self):
        downloader = MegaDownloader(
            url=AMSTERDAM_URL,
            fn=AMSTERDAM_FILENAME,
            url_cleaned=AMSTERDAM_CLEANED_URL,
            fn_cleaned=AMSTERDAM_CLEANED_FILENAME,
        )

        super().__init__(
            dataset_name='amsterdam',
            downloader=downloader,
            correct_activities=True,
            correct_devices=True,
        )

    def load_data(self, folder_path):
        device_fp = Path(folder_path).joinpath(DEVICES_FN)
        activity_fp = Path(folder_path).joinpath(ACTIVITIES_FN)

        df_act = _load_activities(activity_fp)
        df_dev = _load_devices(device_fp)
        df_dev = device_boolean_on_states_to_events(df_dev)

        lst_act = df_act[ACTIVITY].unique()
        lst_dev = df_dev[DEVICE].unique()

        return dict(
            activities=df_act,
            devices=df_dev,
            activity_list=lst_act,
            device_list=lst_dev
        )


def fetch_amsterdam(keep_original=False, cache=True, load_cleaned=False,
                    retain_corrections=False, folder_path=None):
    """
    Fetches the amsterdam dataset from the internet. The original dataset or its cached version
    is stored in the data_home folder.

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
    folder_path : str, default=""
        If set the dataset is loaded from the specified folder.

    Examples
    --------
    >>> from pyadlml.dataset import fetch_amsterdam
    >>> data = fetch_amsterdam()
    >>> data.keys()
    >>> [df_activities, df_devices, activity_list, ... ]

    Returns
    -------
    data : object
    """
    return AmsterdamFetcher()(keep_original=keep_original, cache=cache, load_cleaned=load_cleaned,
                              retain_corrections=retain_corrections, folder_path=folder_path)


def _load_activities(activity_fp):
    act_map = {
        1: 'leave house',
        4: 'use toilet',
        5: 'take shower',
        10: 'go to bed',
        13: 'prepare Breakfast',
        15: 'prepare Dinner',
        17: 'get drink'
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
                            names=[START_TIME, END_TIME, ide, VALUE],
                            engine='python'  # to ignore warning for fallback to python engine because skipfooter
                            # dtype=[]
                            )

    # k todo declare at initialization of dataframe
    sens_data[START_TIME] = pd.to_datetime(sens_data[START_TIME])
    sens_data[END_TIME] = pd.to_datetime(sens_data[END_TIME])
    sens_data[VALUE] = sens_data[VALUE].astype('bool')
    sens_data = sens_data.drop(VALUE, axis=1)

    # replace numbers with the labels
    sens_data[DEVICE] = sens_data[ide].map(sens_labels)
    sens_data = sens_data.drop(ide, axis=1)
    sens_data = sens_data.sort_values(START_TIME)
    return sens_data
