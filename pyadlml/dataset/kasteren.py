import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas._libs.index import timedelta
from pyadlml.dataset.util import fill_nans_ny_inverting_first_occurence
from pyadlml.dataset._dataset import Data
from pyadlml.dataset._dataset import ACTIVITY, VAL, START_TIME, END_TIME, TIME, NAME, DEVICE

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

    # replace numbers with the labels
    sens_data[DEVICE] = sens_data[ide].map(sens_labels)
    sens_data = sens_data.drop(ide, axis=1)
    sens_data = sens_data.sort_values(START_TIME)
    return sens_data

def load(device_fp, activity_fp):
    df_activities, df_devices = _load(device_fp, activity_fp)
    return Data(df_activities, df_devices)



def _load(device_fp, activity_fp):
    from pyadlml.dataset._dataset import correct_activity_overlap, \
        correct_device_ts_duplicates, _is_activity_overlapping
    from pyadlml.dataset.util import print_df
    df_activities = _load_activities(activity_fp)

    # correct overlapping activities as going to toilet is done in parallel
    # for this dataset >:/
    while _is_activity_overlapping(df_activities):
        df_activities = correct_activity_overlap(df_activities)

    df_devices = _load_devices(device_fp)

    # copy devices to new dfs 
    #   one with all values but start time and other way around
    df_start = df_devices.copy().loc[:, df_devices.columns != END_TIME]
    df_end = df_devices.copy().loc[:, df_devices.columns != START_TIME]

    # set values at the end time to zero because this is the time a device turns off
    df_start[VAL] = True
    df_end[VAL] = False

    # rename column 'End Time' and 'Start Time' to 'Time'
    df_start.rename(columns={START_TIME: TIME}, inplace=True)
    df_end.rename(columns={END_TIME: TIME}, inplace=True)

    df_devices = pd.concat([df_end, df_start]).sort_values(TIME)
    df_devices = df_devices.reset_index(drop=True)

    # check if all timestamps have no duplicate
    df_devices = correct_device_ts_duplicates(df_devices)
    assert df_devices[TIME].is_unique

    # transpose the dataframe
    df_devices = df_devices.pivot(index=TIME, columns=DEVICE, values=VAL)

    lower = 50
    #print_df(df_devices.iloc[ 0:lower, 0:2])
    df_devices = fill_nans_ny_inverting_first_occurence(df_devices)
    df_devices = df_devices.fillna(method='ffill')

    return df_activities, df_devices