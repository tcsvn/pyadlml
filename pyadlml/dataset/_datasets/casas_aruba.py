import pandas as pd

from pyadlml.dataset.util import fill_nans_ny_inverting_first_occurence
from pyadlml.dataset.devices import correct_devices
from pyadlml.dataset.activities import correct_activities
                                        
from pyadlml.dataset.activities import correct_activity_overlap, \
    _is_activity_overlapping \

from pyadlml.dataset.obj import Data
from pyadlml.dataset import ACTIVITY, VAL, START_TIME, END_TIME, TIME, NAME, DEVICE
import os
from pyadlml.dataset.io import fetch_handler as _fetch_handler


CASAS_ARUBA_URL = 'https://mega.nz/file/QA5hEToD#V0ypxFsxiwWgVV49OzhsX8RnMNTX8MYSUM2TLL1xX6w'
CASAS_ARUBA_FILENAME = 'casas_aruba.zip'


def fetch_casas_aruba(keep_original=True, cache=True, retain_corrections=False):
    """
    Fetches the casas aruba dataset from the internet. The original dataset or its cached version
    is stored in the :ref:`data home <storage>` folder.

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
    >>> from pyadlml.dataset import fetch_casas_aruba
    >>> data = fetch_casas_aruba()
    >>> dir(data)
    >>> [..., df_activities, df_devices, ...]

    Returns
    -------
    data : object
    """
    dataset_name = 'casas_aruba'

    def load_casas_aruba(folder_path):
        _fix_data(os.path.join(folder_path, "data"))

        data_path = os.path.join(folder_path, 'corrected_data.csv')

        df = _load_df(data_path)
        df_dev = _get_devices_df(df)
        df_act = _get_activity_df(df)

        df_dev = correct_devices(df_dev)
        df_act, cor_lst = correct_activities(df_act)

        lst_act = df_act[ACTIVITY].unique()
        lst_dev = df_dev[DEVICE].unique()

        data = Data(df_act, df_dev, activity_list=lst_act, device_list=lst_dev)
        return data

    data = _fetch_handler(keep_original, cache, dataset_name,
                        CASAS_ARUBA_FILENAME, CASAS_ARUBA_URL,
                        load_casas_aruba)
    return data


def _fix_data(path):
    """
    as the data is very inconsistent with tabs and spaces this is to make it alright again
    produces: 
        date time,id,value,activity 
    """
    data_path_tf = path[:-4] + 'corrected_data.csv'
    with open(path, 'r') as f_o, open(data_path_tf, 'w') as f_t:
        i= 1
        for line in f_o.readlines():
            s = line.split()
            # weird enterhome and leave home devices that appear inconsistently and are omitted
            if s[2] in ['ENTERHOME', 'LEAVEHOME']:
                continue
            # there is an error in line 1476694 where M014 is replaced with a 'c'
            if s[2] == 'c':
                s[2] = 'M014'
            # 'c' and '5' are randomly added onto values - remove them
            if 'c' in s[3]:
                s[3] = s[3].replace('c', '')                
            if '5' in s[3] and s[2][0] == 'M':
                s[3] = s[3].replace('5', '')
            if s[3] in ['ONM026', 'ONM009', 'ONM024']:         
                s[3] = 'ON'                
            # line 886912 error should be on
            if s[2][0] == 'M' and len(s[3]) == 1:
                s[3] = 'ON'                
            # line 900915 error should be off 
            if s[2][0] == 'M' and len(s[3]) == 2 and s[1] == '18:13:47.291404':
                s[3] = 'OFF'
            # add new line
            new_line = " ".join(s[:2]) + "," + ",".join(s[2:4])
            try:
                s[4] # test if there is an activity
                new_line += "," + " ".join(s[4:])
            except:
                pass
                
            f_t.write(new_line + "\n")
        f_t.close()
        f_o.close()


def _load_df(data_path):
    df = pd.read_csv(data_path,
                    sep=",",
                    #parse_dates=True,
                    infer_datetime_format=True,
                    na_values=True,
                    names=[START_TIME, 'id', VAL, ACTIVITY],
                    engine='python' #to ignore warning for fallback to python engine because skipfooter
                    #dtyp
                    )
    df[START_TIME] = pd.to_datetime(df[START_TIME])
    return df

def _val_activity_count(df_act):
    # confirm data assumptions from readme
    assert len(df_act[df_act[ACTIVITY] == 'Meal_Preparation']) == 1606
    # observed 2919 times line below
    len(df_act[df_act[ACTIVITY] == 'Relax'])# == 2910 # does not correspond to authors reported values
    assert len(df_act[df_act[ACTIVITY] == 'Eating']) == 257
    assert len(df_act[df_act[ACTIVITY] == 'Work'])== 171
    assert len(df_act[df_act[ACTIVITY] == 'Sleeping']) == 401
    assert len(df_act[df_act[ACTIVITY] == 'Wash_Dishes']) == 65
    assert len(df_act[df_act[ACTIVITY] == 'Bed_to_Toilet']) == 157
    assert len(df_act[df_act[ACTIVITY] == 'Enter_Home']) == 431
    assert len(df_act[df_act[ACTIVITY] == 'Leave_Home']) == 431
    assert len(df_act[df_act[ACTIVITY] == 'Housekeeping']) == 33
    assert len(df_act[df_act[ACTIVITY] == 'Respirate']) == 6

def _get_devices_df(df):
    df = df.drop(ACTIVITY, axis=1)
    bin_mask = (df['val'] == 'ON') | (df['val'] == 'OFF')

    # preprocess only binary devices to ON-OFF--> False True
    df_binary = df[bin_mask]
    df_binary[VAL] = (df_binary[VAL] == 'ON')
    #tmp = {VAL: (df_binary[VAL] == 'ON')} # TODO suppress warnings
    #df_binary = df_binary.assign(tmp)

    # preprocess only numeric devices
    num_mask = pd.to_numeric(df[VAL], errors='coerce').notnull()
    df_num = df[num_mask]
    df_num[VAL] = df_num[VAL].astype(float)
    #df_num = df_num.assign({VAL: df_num[VAL].astype(float)}) # TODO suppress warnings

    # preprocess categorical devices
    df_cat = df[~num_mask & ~bin_mask]

    # join datasets
    df = pd.concat([df_cat, df_binary, df_num], axis=0, ignore_index=True)
    df.columns = [TIME, DEVICE, VAL]
    df = df.sort_values(by=TIME).reset_index(drop=True)

    return df


def _get_activity_df(df):
    # get all rows containing activities
    df = df[~df[ACTIVITY].isnull()][[START_TIME, ACTIVITY]].copy()
    
    act_list = pd.unique(df[ACTIVITY])
    
    new_df_lst = []
    for i in range(1, len(act_list), 2):
        activity = act_list[i][:-4]
        act_begin = act_list[i-1]
        act_end = act_list[i]
           
        # create subsets for begin and end of chosen activity
        df_res = df[df[ACTIVITY] == act_begin].reset_index(drop=True)
        df_end = df[df[ACTIVITY] == act_end].reset_index(drop=True)
        
        # append sorted end_time to start_time as they should be
        # pairwise together
        df_res[ACTIVITY] = activity
        df_res[END_TIME] = df_end[START_TIME]
        new_df_lst.append(df_res)
    
    # data preparation
    res = pd.concat(new_df_lst)
    res = res.reindex(columns=[START_TIME, END_TIME, ACTIVITY])
    res = res.sort_values(START_TIME)
    res = res.reset_index(drop=True)
    return res