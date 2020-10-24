import pandas as pd
import numpy as np
from pyadlml.dataset.devices import _create_devices
from pyadlml.dataset.obj import Data
from pyadlml.dataset.devices import correct_devices
from pyadlml.dataset.activities import correct_activities, _create_activity_df



def _load_device_map(path_to_file):
    df_subx_dev = pd.read_csv(path_to_file, sep=",", header=None)
    df_subx_dev.columns = ['id', 'room', 'device']

    df_subx_dev['device'] = df_subx_dev['id'].astype(str) + ' - ' \
                    + df_subx_dev['room'] + ' ' + df_subx_dev['device']
    df_subx_dev = df_subx_dev.drop(columns='room')
    df_subx_dev =  df_subx_dev.set_index('id')
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
    df_devices = pd.DataFrame(columns=['start_time', 'end_time', 'device'])
    df_activities = _create_activity_df()

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
                df_activities = df_activities.append(new_row, ignore_index=True)
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
                    #print('dev: ', dev, ' ts_start: ', ts_start, ' ts_end: ', ts_end)
                    new_row = {'device':dev, 
                               'start_time':pd.Timestamp(date +'T' + ts_start), 
                               'end_time':pd.Timestamp(date +'T' + ts_end)
                              }
                    df_devices = df_devices.append(new_row, ignore_index=True)                
                i = 0
                read_in_device = False
                
        f_o.close()
        
    # map device ids to strings    
    df_devices['device'] = df_devices['device'].astype(int)
    df_devices['device'] = df_devices['device'].map(df_dev.to_dict()['device'])
    df_devices = df_devices.sort_values(by='start_time')
    df_devices = df_devices.drop_duplicates().reset_index(drop=True)
    
    
    df_activities = df_activities.sort_values(by='start_time').reset_index(drop=True)


    return df_devices, df_activities


def load(dev_path, act_path, data_path):
    df_dev_map = _load_device_map(dev_path)
    df_act_map = _load_activity_map(act_path)
    df_dev, df_act = _read_data(data_path, df_dev_map, df_act_map)    

    df_act, cor_lst = correct_activities(df_act)
    df_dev_rep1 = correct_devices(df_dev)
        
    data = Data(df_act, df_dev_rep1)
    
    data.df_dev_map = df_dev_map
    data.df_act_map = df_act_map        
    return data