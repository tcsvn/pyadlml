import os
from pathlib import Path
import pandas as pd
from pyadlml.constants import ACTIVITY, VALUE, \
    START_TIME, END_TIME, TIME, NAME, DEVICE
import numpy as np
from pyadlml.dataset._core.devices import device_boolean_on_states_to_events
from pyadlml.dataset.io.downloader import MegaDownloader
from pyadlml.dataset.io.remote import DataFetcher

"""
Transferring Knowledge of Activity Recognition across Sensor Networks
T.L.M. van Kasteren, G. Englebienne and B.J.A. Kr�se
In Proceedings of the Eighth International Conference on Pervasive Computing (Pervasive 2010). Helsinki, Finland, 2010. 

"""


KASTEREN_2010_URL = 'https://mega.nz/file/VIhnxAxB#3UI77ZA1uh0tRiT6vHTfhYolm-uxbAXuV2TxxIyQ2AU'
KASTEREN_2010_FILENAME = 'kasteren_2010.zip'

KASTEREN_2010_A_CLEANED_URL = 'https://mega.nz/file/RJYmlKrB#9UuLiAS0yhtud98TmRc4D7uRfEYdFh5USWVMnUekWyo'
KASTEREN_2010_A_CLEANED_FN = 'cleaned_kasteren_2010_A.joblib'

KASTEREN_2010_B_CLEANED_URL = 'https://mega.nz/file/8QgFDCAT#2nQRGyelCE82VgA_N28W5edOU6hyTA8mjv_MNW-WBGk'
KASTEREN_2010_B_CLEANED_FN = 'cleaned_kasteren_2010_B.joblib'

KASTEREN_2010_C_CLEANED_URL = 'https://mega.nz/file/8Ihj0a6S#WNPJlS0eO1Gx5GkQlleR-d2t1_Ih3xC_sstnSgQXRoY'
KASTEREN_2010_C_CLEANED_FN = 'cleaned_kasteren_2010_C.joblib'



class KasterenFetcher(DataFetcher):
    def __init__(self, auto_corr_acts=True):

        downloader = MegaDownloader(
            url=KASTEREN_2010_URL,
            fn=KASTEREN_2010_FILENAME,
            url_cleaned={
                'A': KASTEREN_2010_A_CLEANED_URL,
                'B': KASTEREN_2010_B_CLEANED_URL,
                'C': KASTEREN_2010_C_CLEANED_URL,
            },
            fn_cleaned={
                'A': KASTEREN_2010_A_CLEANED_FN,
                'B': KASTEREN_2010_B_CLEANED_FN,
                'C': KASTEREN_2010_C_CLEANED_FN,
            },
        )

        super().__init__(
            dataset_name='kasteren_2010',
            downloader=downloader,
            correct_activities=auto_corr_acts,
            correct_devices=True
        )

    def load_data(self, folder_path, ident: str) -> dict:
        assert ident in ['A', 'B', 'C']

        # Load activity structure
        path = Path(folder_path).joinpath('datasets', 'house%s'%ident)
        df_act = _load_activities(path, ident)
        df_dev = _load_devices(path, ident)

        lst_act = df_act[ACTIVITY].unique()
        lst_dev = df_dev[DEVICE].unique()

        return dict(
            activities=df_act,
            devices=df_dev,
            activity_list=lst_act,
            device_list=lst_dev
        )




#@correct_acts_and_devs
def fetch_kasteren_2010(house:str ='A', keep_original=False, cache=True,
                        auto_corr_activities=True, load_cleaned=False,
                        retain_corrections=False, folder_path=None) -> dict:
    """
    Fetches the amsterdam dataset from the internet. The original dataset or its cached version
    is stored in the :ref:`data home <storage>` folder.

    Parameters
    ----------
    house : str one of {'A', 'B', 'C'}, default='A'
        The house to load from. Every house represents a different experiment at a
        different time.
    keep_original : bool, default=False
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool, default=True
        Determines whether the data object should be stored as a binary file for quicker access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.
    load_cleaned : bool, default=False
        Whether to load the 
    auto_corr_activities : bool, default=True
        Return activities and devices as is, without applying any correction. You have 
        entered the danger zone.
    retain_corrections : bool, default=False
        When set to *true*, data points that change or drop during preprocessing
        are listed in respective attributes of the data object. Fore more information
        about error correction refer to the :ref:`user guide <error_correction>`.
    folder_path : str, default=None
        If set the dataset is loaded from the specified folder.

    Examples
    --------
    >>> from pyadlml.dataset import fetch_kasteren
    >>> data = fetch_kasteren_2010(house='B')
    >>> data['activities']
        {...}

    Returns
    -------
    dict
        A dictionary containing device and activity dataframes
    """
    return KasterenFetcher(auto_corr_acts=auto_corr_activities)(keep_original=keep_original, cache=cache, #load_cleaned=load_cleaned,
                              retain_corrections=retain_corrections, folder_path=folder_path,
                              ident=house, load_cleaned=load_cleaned
    )

def _load_activities(path, house):

    from scipy.io import loadmat
    tmp = loadmat(str(path.joinpath(f'actStructHouse{house}.mat')))
    acts = tmp['activityStructure'][0][0][0]

    # load act and devs
    tmp = loadmat(str(path.joinpath(f'senseandactLabelsHouse{house}.mat')))

    # load activity labels
    act_map = {}
    act_labels = tmp['activity_labels']
    nr_acts = act_labels.shape[0]
    for i in range(nr_acts):
        try:
            # (i+1) because matlab index is from 1 to n
            act_map[i+1] = act_labels[i][0][0]
        except IndexError:
            pass
    df_act = pd.DataFrame({START_TIME: acts[:,0], END_TIME: acts[:,1], ACTIVITY: acts[:,2]})
    df_act[START_TIME] = df_act[START_TIME].apply(matlab_date_to_timestamp)
    df_act[END_TIME] = df_act[END_TIME].apply(matlab_date_to_timestamp)
    df_act[ACTIVITY] = df_act[ACTIVITY].map(act_map)

    if house == 'B':
        # the activity 73 in house c maps to a non-existing activity (16.0)
        #                   start_time                   end_time  activity
        #73 2009-07-29 10:52:57.898994 2009-07-29 10:56:45.394994      16.0
        df_act = df_act.dropna()

    # Capitalize every first letter of the activity
    df_act[ACTIVITY] = df_act[ACTIVITY].apply(lambda x: x[0].upper() + x[1:])

    return df_act.sort_values(by=START_TIME).reset_index(drop=True)




def _load_devices(path, house):

    from scipy.io import loadmat
    # load device structure
    tmp =loadmat(str(path.joinpath('sensorStructHouse%s.mat'%(house))))
    devs = tmp['sensorStructure'][0][0][0]

    # load act and devs
    tmp =loadmat(str(path.joinpath('senseandactLabelsHouse%s.mat'%(house))))

    # load device labels
    if house == 'A':
        dev_map = {}
        dev_labels = tmp['sensor_labels']
        nr_devs = dev_labels.shape[0]
        for i in range(nr_devs):
            dev_map[dev_labels[i][0][0][0]] = dev_labels[i][1][0]
        
        # The sensor names are flipped in the original dataset
        dev_map[5] = 'Hall-Bathroom door'
        dev_map[6] = 'Hall-Toilet door'
    elif house == 'B':
        dev_map = _dev_map_House_B()
    elif house == 'C':
        dev_map = _dev_map_House_C()

    df_dev = pd.DataFrame({START_TIME: devs[:, 0], END_TIME: devs[:, 1], DEVICE: devs[:, 2]})
    df_dev[START_TIME] = df_dev[START_TIME].apply(matlab_date_to_timestamp)
    df_dev[END_TIME] = df_dev[END_TIME].apply(matlab_date_to_timestamp)
    df_dev[DEVICE] = df_dev[DEVICE].map(dev_map)

    if house == 'B':
        # contains a device (23) where no corresponding label exists
        # therefore those two entries are dropped
        #                     start_time                   end_time  device
        #2003 2009-07-24 16:16:20.999997 2009-07-24 16:16:22.000001    23.0
        #2005 2009-07-24 16:22:09.000000 2009-07-24 16:22:10.999999    23.0
        df_dev = df_dev.dropna()

    df_dev = device_boolean_on_states_to_events(df_dev)
    return df_dev

from datetime import datetime, timedelta
def matlab_date_to_timestamp(matlab_datenum):
    """ converts a matlab datenum representation to pandas timestamp"""
    # convert matlab dateformat to python
    # copied from @actstruct/actstruct.m
    #     Usage Example: as = actstruct(732877.4520, 732877.4521, 20)
    #     activity 20 took place on 19-Juli-2006 from 10:50:52 till 10:51:01
    # https: // stackoverflow.com / questions / 13965740 / converting - matlabs - datenum - format - to - python
    # TODO the solution has not enough precision try other methods that are
    return pd.Timestamp(datetime.fromordinal(int(matlab_datenum))
                        + timedelta(days=matlab_datenum%1)
                        - timedelta(days=366))

assert pd.Timestamp('19-Jul-2006 10:50:52.800004') == matlab_date_to_timestamp(732877.4520)
assert pd.Timestamp('19-Jul-2006 10:51:01.44') == matlab_date_to_timestamp(732877.4521)


def _dev_map_House_C():
    """ translates dutch device names to english and drop the device type"""
    return {5: 'pressure mat bed right',        # mat bed rechts, drukmat
            6: 'pressure mat couch',            # 'mat bank, huiskamer'
            7: 'freezer reed',                  # 'vriezer, reed'
            8: 'toilet flush upstairs',         # 'toilet flush boven, flush ',
            10: 'toilet flush downstairs',      # 'toilet flush beneden. flush '
            11: 'bathroom door left',           # 'badkamer klapdeur links'
            13: 'cutlery kwik',                 # 'zelfde la als 18, kwik'
            15: 'keys',                         # 'La sleutels'
            16: 'bathroom door left',           # 'badkamer klapdeur links',
            18: 'cutlery kwik',                 # 'bestek la, kwik sensor '
            20: 'pans cupboard reed',           # 'kastje pannen, reed '
            21: 'microwave reed',               # 'magnetron, reed '
            22: 'cupboard leftovers reed',      # 'kastje restjesbakjes, reed '
            23: 'cabinet plates spices reed',   # 'kastje borden/kruiden,reed '
            25: 'toilet door downstairs',       # 'deur toilet beneden',
            27: 'cabinet cups/bowl/tuna reed',  #'astje cups/bowl/tuna, reed ',
            28: 'front door reed',              #'voordeur, reed ',
            29: 'door bedroom',                 #'deur slaapkamer',
            30: 'refrigerator',                 #'koelkast, reed ',
            35: 'bathtub pir',                  #'badkuip, pir ',
            36: 'dresser pir',
            38: 'washbasin flush upstairs',     #'wasbak boven, flush ',
            39: 'pressure mat bed left',        #'mat bed links, drukmat '
            }

def _dev_map_House_B():
    """ translates dutch device names to english and drop the device type"""
    return {1: 'toilet door',
            2: 'toaster *dead*',
            3: 'fridge',
            4: 'dead',
            5: 'cupboard groceries',
            6: 'toilet flush',
            7: 'frontdoor',
            8: 'dead',
            9: 'cupboard plates',
            10: 'Bedroom door',
            11: 'temp geyser',                  #'temp geiser',
            12: 'press bed right',              #'press bed  rechts',
            13: 'press bed left',               #'press bed links',
            14: 'cutlery drawer kwick',         #'kwik cutlary drawer',
            15: 'kwik stove lid',
            16: 'bedroom pir',                  #'PIR slaapkamer',
            17: 'temp shower',
            18: 'kwik dresser',
            19: 'bathroom pir',                 #'PIR badkamer',
            20: 'pressure mat office chair',    #'Drukmat bureaustoel (piano)',
            21: 'sink float',                   #'gootsteen float',
            22: 'pressure mat server corner',   #'drukmat stoel serverhoekje',
            24: 'balcony door',                 #'balkon deur',
            25: 'frame',                        #'raam',
            26: 'toaster',
            27: 'microwave',                    #'magnetron',
            28: 'kitchen pir',                  #'PIR keuken'
    }