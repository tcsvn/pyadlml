import pandas as pd
from pathlib import Path
from pyadlml.constants import ACTIVITY, DEVICE, START_TIME, END_TIME
from pyadlml.dataset._core.activities import create_empty_activity_df
from pyadlml.dataset._core.devices import device_boolean_on_states_to_events
from pyadlml.dataset.io.downloader import MegaDownloader
from pyadlml.dataset.io.remote import DataFetcher

MITLAB_URL = 'https://mega.nz/file/MB4BFL6S#8MjAQoS-j0Lje1UFoWUMOCay2FcdpVfla6p9MTe4SQM'
MITLAB_FILENAME = 'mitlab.zip'
SUB1 = 'subject1'
SUB2 = 'subject2'

__all__ = ['fetch_mitlab']

class MitLabFetcher(DataFetcher):
    def __init__(self, *args, **kwargs):

        downloader = MegaDownloader(
            url=MITLAB_URL,
            fn=MITLAB_FILENAME,
            url_cleaned=None,
            fn_cleaned=None,
        )

        super().__init__(
            dataset_name='mitlab',
            downloader=downloader,
            correct_activities=True,
            correct_devices=True
        )


    def load_data(self, folder_path, ident):

        assert ident in [SUB1, SUB2]

        act_path = Path(folder_path).joinpath(ident, "Activities.csv")
        dev_path = Path(folder_path).joinpath(ident, "sensors.csv")
        data_path = Path(folder_path).joinpath(ident, "activities_data.csv")

        df_dev_map = _load_device_map(dev_path)
        df_act_map = _load_activity_map(act_path)
        df_dev, df_act = _read_data(data_path, df_dev_map, df_act_map)
        df_dev = device_boolean_on_states_to_events(df_dev)

        lst_act = df_act[ACTIVITY].unique()
        lst_dev = df_dev[DEVICE].unique()

        return dict(
            activities=df_act,
            devices=df_dev,
            activity_list=lst_act,
            device_list=lst_dev
        )



def fetch_mitlab(subject='subject1', keep_original=False, cache=True,
                 retain_corrections=False, folder_path=None) -> dict:
    """
    Fetches the :ref:`mitlab <ds_mitlab>` dataset from the internet. The original dataset or its cached version
    is stored in the data_home folder.

    Parameters
    ----------
    subject : str of {'subject1', 'subject2'}
        Identifies which dataset is loaded.
    keep_original : bool, default=True
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool, default=True
        Determines whether the data object should be stored as a binary file for quicker access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.
    retain_corrections : bool, default=False
        When set to *true*, data points that change or drop during preprocessing
        are listed in respective attributes of the data object. Fore more information
        about error correction refer to the :ref:`user guide <error_correction>`.

    Returns
    -------
    dict
    """
    return MitLabFetcher()(keep_original=keep_original, cache=cache, #load_cleaned=load_cleaned,
                           retain_corrections=retain_corrections, folder_path=folder_path, 
                           ident=subject
    )




def _load_device_map(path_to_file):
    df_subx_dev = pd.read_csv(path_to_file, sep=",", header=None)
    df_subx_dev.columns = ['id', 'room', 'device']

    df_subx_dev['device'] = df_subx_dev['id'].astype(str) + ' - ' \
                            + df_subx_dev['room'] + ' ' + df_subx_dev['device']
    df_subx_dev = df_subx_dev.drop(columns='room')
    df_subx_dev = df_subx_dev.set_index('id')
    return df_subx_dev


def _load_activity_map(path_to_file):
    return pd.read_csv(path_to_file, sep=",")


def _read_data(path_to_file, df_dev, df_act):
    """ creates the device dataframe and activity dataframe
    
    The data is present in the following format:
         ACTIVITY_LABEL,DATE,START_TIME,END_TIME
         SENSOR1_ID, SENSOR2_ID, ......
         SENSOR1_OBJECT,SENSOR2_OBJECT, .....
         SENSOR1_ACTIVATION_TIME,SENSOR2_ACTIVATION_TIME, .....
         SENSOR1_DEACTIVATION_TIME,SENSOR2_DEACTIVATION_TIME, .....

         where date is in the mm/dd/yyyy format
         where time is in the hh:mm:ss format
    Parameters
    ----------
    path_to_file: str
    
    Returns
    -------
    df_devices : pd.DataFrame
    
    df_activities : pd.DataFrame
        
    """
    # create empy dataframes for devices and activities
    df_devices = pd.DataFrame(columns=[START_TIME, END_TIME, DEVICE])
    df_activities = create_empty_activity_df()

    act_list = df_act['Subcategory'].values
    
    with open(path_to_file, 'r') as f_o:
        i = 0
        read_in_device = False
        date = None
        for line in f_o.readlines():
            assert i in [0,1,2,3]
            s = line.split(',')
            # check if the line represents an activity
            if s[0] in act_list:
                assert len(s) == 4
                date = s[1]
                """
                    there is an error where the date is 4/31/2003 which doesn't exist in 
                    subject 2 data. Convert this to the next day 
                """
                if date == '4/31/2003':
                    date = '5/1/2003'
                new_row = {'activity':s[0], 
                           'start_time':pd.Timestamp(date +'T'+s[2]), 
                           'end_time':pd.Timestamp(date +'T'+s[3])
                          }
                df_activities = pd.concat([df_activities, pd.Series(data=new_row).to_frame().T],\
                                           ignore_index=True, axis=0)
                continue      
                
            # check if the line represents a device
            elif not read_in_device:
                try:
                    read_in_device = True
                    devices = s
                    # delete the '\n' for every last device 
                    devices[-1:] = [devices[-1:][0][:-1]]
                    i = 1
                    continue
                except:
                    raise ValueError
            elif i == 1:
                i = 2
                continue
            elif i == 2:
                ts_act = s                
                i = 3
            elif i == 3:
                ts_deact = s
                
                assert len(ts_act) == len(ts_deact)
                # correct timestamps by inserting a 0 where only a single digit is present
                for j in range(len(ts_act)):
                    if len(ts_act[j]) != 8:
                        ts = ts_act[j].split(':')
                        for k, digit in enumerate(ts):
                            if len(digit) == 1:
                                ts[k] = '0' + ts[k]
                        ts_act[j] = ':'.join(ts)
                    if len(ts_deact) != 8:
                        ts = ts_deact[j].split(':')
                        for k, digit in enumerate(ts):
                            if len(digit) == 1:
                                ts[k] = '0' + ts[k]
                        ts_deact[j] = ':'.join(ts)
                        
                # create rows
                for dev, ts_start, ts_end in zip(devices, ts_act, ts_deact):
                    new_row = {DEVICE:dev,
                               START_TIME:pd.Timestamp(date +'T' + ts_start),
                               END_TIME:pd.Timestamp(date +'T' + ts_end)
                              }
                    df_devices = pd.concat([df_devices, pd.Series(data=new_row).to_frame().T],\
                                           ignore_index=True, axis=0)
                i = 0
                read_in_device = False
                
        f_o.close()
        
    # map device ids to strings    
    df_devices[DEVICE] = df_devices[DEVICE].astype(int)
    df_devices[DEVICE] = df_devices[DEVICE].map(df_dev.to_dict()[DEVICE])
    df_devices = df_devices.sort_values(by=START_TIME)
    df_devices = df_devices.drop_duplicates().reset_index(drop=True)
    
    df_activities = df_activities.sort_values(by=START_TIME).reset_index(drop=True)

    return df_devices, df_activities
