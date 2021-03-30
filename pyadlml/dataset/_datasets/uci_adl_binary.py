import pandas as pd
from pyadlml.dataset.activities import START_TIME, END_TIME, ACTIVITY, correct_activities
from pyadlml.dataset.devices import DEVICE, correct_devices
from pyadlml.dataset.obj import Data
from pyadlml.dataset.io import fetch_handler as _fetch_handler
import os


UCI_ADL_BINARY_URL = 'https://mega.nz/file/AQIgDQJD#oximAQFjexTKwNP3WYzlPnOGew06YSQ2ef85vvWGN94'
UCI_ADL_BINARY_FILENAME = 'uci_adl_binary.zip'


def fetch_uci_adl_binary(keep_original=True, cache=True, retain_corrections=False, subject='OrdonezA'):
    """

    Parameters
    ----------
    keep_original : bool, default=True
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool
        Determines whether the data object should be stored as a binary file for quicker access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.
    retain_corrections : bool
        When set to *true* data points that are changed or dropped during preprocessing
        are listed in the respective attributes of the data object.  Fore more information
        about the attributes refer to the :ref:`user guide <error_correction>`.
    subject : str of {'OrdonezA', 'OrdonezB'}, default='OrdonezA'
        decides which dataset of the two houses is loaded.

    Returns
    -------
    data : object
    """
    assert subject in ['OrdonezA', 'OrdonezB']
    dataset_name = 'uci_adl_binary'

    def load_uci_adl_binary(folder_path):
        sub_dev_file = os.path.join(folder_path, '{}_Sensors.txt'.format(subject))
        if subject == 'OrdonezB':
            fix_OrdonezB_ADLS(os.path.join(folder_path, 'OrdonezB_ADLs.txt'))
            sub_act_file = os.path.join(folder_path, '{}_ADLs_corr.txt'.format(subject))
        else:
            sub_act_file = os.path.join(folder_path, '{}_ADLs.txt'.format(subject))

        return load(sub_dev_file, sub_act_file, retain_corrections, subject)

    data = _fetch_handler(keep_original, cache, dataset_name,
                        UCI_ADL_BINARY_FILENAME, UCI_ADL_BINARY_URL,
                        load_uci_adl_binary, data_postfix=subject)
    return data


def fix_OrdonezB_ADLS(path_to_file):
    """ fixes inconsistent use of tabs for delimiter in the file
    Parameters
    ----------
    path_to_file : str
        path to the file OrdonezB_ADLs.csv
    """
    
    path_corrected = path_to_file[:-17] + 'OrdonezB_ADLs_corr.txt'
    
    with open(path_to_file, 'r') as f_o, open(path_corrected, 'w') as f_t:
        for i, line in enumerate(f_o.readlines()):            
            if i in [0,1]: 
                f_t.write(line)  
                continue
            s = line.split()
            assert len(s) == 5
            new_line = s[0]+' '+s[1]+'\t\t'+s[2]+' '+s[3]+'\t\t'+s[4]                        
            f_t.write(new_line + "\n")
        f_t.close()
        f_o.close()

def _load_activities(act_path):
    df_act = pd.read_csv(act_path, delimiter='\t+', skiprows=[0,1], 
                         names=[START_TIME, END_TIME, ACTIVITY], engine='python')
    df_act[START_TIME] = pd.to_datetime(df_act[START_TIME])
    df_act[END_TIME] = pd.to_datetime(df_act[END_TIME])
    return df_act

def _load_devices(dev_path):
    df_dev = pd.read_csv(dev_path, delimiter='\t+', skiprows=[0, 1], 
                         names=[START_TIME, END_TIME, 'Location', 'Type', 'Place'], 
                         engine='python')
    df_dev[DEVICE] = df_dev['Place'] + ' ' + df_dev['Location'] + ' ' + df_dev['Type']
    
    # get room mapping devices
    df_locs = df_dev.copy().groupby([DEVICE, 'Type', 'Place', 'Location']).sum()
    df_locs = df_locs.reset_index().drop([START_TIME, END_TIME], axis=1)

    df_dev = df_dev[[START_TIME, END_TIME, DEVICE]]
    df_dev[START_TIME] = pd.to_datetime(df_dev[START_TIME])
    df_dev[END_TIME] = pd.to_datetime(df_dev[END_TIME])
    return df_dev, df_locs

def load(dev_path, act_path, retain_corrections, subject):
    """
    """
    assert subject in ['OrdonezA', 'OrdonezB']
    df_act = _load_activities(act_path)
    df_dev, df_loc = _load_devices(dev_path)

    if subject == 'OrdonezB':
        # the activity grooming is often overlapped by sleeping
        # as I deem this activity as important i make it dominant
        df_act, cor_lst = correct_activities(df_act, excepts=['Grooming'])
    elif subject == 'OrdonezA':
        df_act, cor_lst = correct_activities(df_act)

    lst_act = df_act[ACTIVITY].unique()
    lst_dev = df_dev[DEVICE].unique()

    df_dev = correct_devices(df_dev)
    data = Data(df_act, df_dev, activity_list=lst_act, device_list=lst_dev)
    data.df_dev_rooms = df_loc

    if retain_corrections:
        data.correction_activities = cor_lst

    return data