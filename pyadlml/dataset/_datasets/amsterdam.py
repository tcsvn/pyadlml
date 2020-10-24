import numpy as np
import pandas as pd

from pyadlml.dataset.util import fill_nans_ny_inverting_first_occurence
from pyadlml.dataset.obj import Data

from pyadlml.dataset.activities import correct_activity_overlap,\
    _is_activity_overlapping

from pyadlml.dataset import  ACTIVITY, VAL, \
    START_TIME, END_TIME, TIME, NAME, DEVICE

from pyadlml.dataset.activities import correct_activities
from pyadlml.dataset.devices import correct_devices



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


def load(device_fp, activity_fp):
    df_act = _load_activities(activity_fp)
    df_dev = _load_devices(device_fp)
    df_act, cor_lst = correct_activities(df_act)
    
    df_dev = correct_devices(df_dev)

    data = Data(df_act, df_dev)
    data.correction_activities = cor_lst
    
    return data

