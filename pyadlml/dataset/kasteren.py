import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas._libs.index import timedelta


START_TIME = 'start_time'
END_TIME = 'end_time'
VAL = 'val'
NAME = 'name'
DEVICE = 'device'
"""
    df_activities:
        start_time | end_time   | activity
        ---------------------------------
        timestamp   | timestamp | act_name

    df_devices:
        toggle_time | device_1 | ...| device_n
        ----------------------------------------
        timestamp   | state_0  | ...| state_n
"""
class Data():
    def __init__(self, activities, devices):
        self.df_activities = activities
        self.df_devices = devices

    def create(self, data_dep='iid'):
        if data_dep == 'iid':
            pass
        elif data_dep == 'sequential':
            pass

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
    df['activities'] = df[ide].map(act_map)
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

    # replace numbers with the labels
    sens_data[DEVICE] = sens_data[ide].map(sens_labels)
    sens_data = sens_data.drop(ide, axis=1)
    sens_data = sens_data.sort_values(START_TIME)
    return sens_data


def load(device_fp, activity_fp):
    df_activities = _load_activities(activity_fp)
    df_devices = _load_devices(device_fp)

    # transform into proper shape
    print(df_devices.index)
    for i in range(0,len(df_devices)):
        pass

    return Data(df_activities, df_devices)