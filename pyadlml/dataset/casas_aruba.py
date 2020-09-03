import pandas as pd

from pyadlml.dataset.util import fill_nans_ny_inverting_first_occurence

from pyadlml.dataset._dataset import Data, correct_activity_overlap, \
    correct_device_ts_duplicates, \
    _is_activity_overlapping, correct_device_rep3_ts_duplicates, \
    device_rep1_2_rep3, device_rep3_2_rep1, \
    ACTIVITY, VAL, START_TIME, END_TIME, TIME, NAME, DEVICE


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
            new_line = " ".join(s[:2]) + "," + ",".join(s[2:4])
            try:
                s[4] # test if there is an activity
                new_line += "," + " ".join(s[4:])
            except:
                new_line += "," + 'NaN'
            f_t.write(new_line + "\n")
        f_t.close()
        f_o.close()

def _load_df(data_path):
    df = pd.read_csv(data_path,
                    sep=",",
                    #parse_dates=True,
                    infer_datetime_format=True,
                    na_values=None,
                    names=[START_TIME, 'id', VAL, ACTIVITY],
                    engine='python' #to ignore warning for fallback to python engine because skipfooter
                    #dtyp
                    )
    df[START_TIME] = pd.to_datetime(df[START_TIME])
    return df

def _get_activity_df(df):
    # get all rows containing activities
    df_act = df[~df[ACTIVITY].isnull()][[START_TIME, ACTIVITY]]
    
    act_list = pd.unique(df_act[ACTIVITY])
    
    new_df_lst = []
    for i in range(1, len(act_list), 2):
        activity = act_list[i][:-4]
    
        # select the df subset of the activity
        mask_begin = act_list[i-1] == df_act[ACTIVITY]
        mask_end = act_list[i] == df_act[ACTIVITY]
        df_tmp = df_act[mask_begin | mask_end]
        df_tmp = df_tmp.sort_values(START_TIME)
    
        # create subsets for begin and end of chosen activity
        df_res = df_tmp[df_tmp[ACTIVITY] == act_list[i-1]]
        df_end = df_tmp[df_tmp[ACTIVITY] == act_list[i]]
        df_res = df_res.reset_index(drop=True)
        df_end = df_end.reset_index(drop=True)
        
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



def _val_activity_count(df_act):
    # confirm data assumptions from readme
    assert len(df_act[df_act['activity'] == 'Meal_Preparation']) == 1606
    # observed 2919 times line below
    len(df_act[df_act['activity'] == 'Relax'])# == 2910 # does not confirm 
    assert len(df_act[df_act['activity'] == 'Eating']) == 257
    assert len(df_act[df_act['activity'] == 'Work'])== 171
    assert len(df_act[df_act['activity'] == 'Sleeping']) == 401
    assert len(df_act[df_act['activity'] == 'Wash_Dishes']) == 65
    assert len(df_act[df_act['activity'] == 'Bed_to_Toilet']) == 157
    assert len(df_act[df_act['activity'] == 'Enter_Home']) == 431
    assert len(df_act[df_act['activity'] == 'Leave_Home']) == 431
    assert len(df_act[df_act['activity'] == 'Housekeeping']) == 33
    assert len(df_act[df_act['activity'] == 'Respirate']) == 6

def _get_devices_df(df):
    df_dev = df[(df['val'] == 'ON') | (df['val'] == 'OFF')]
    df_dev = df_dev.drop('activity', axis=1)
    df_dev['val'] = df_dev['val'] == 'ON'
    df_dev.columns = ['time', 'device', 'val']
    return df_dev

def load(data_path):
    # load data and separate into activities and devices
    df = _load_df(data_path)
    df_act = _get_activity_df(df)
    df_dev = _get_devices_df(df)

    # validate or correct activity data
    _val_activity_count(df_act)
    print(_is_activity_overlapping(df_act))

    # validate or correct device data
    # TODO

    # create data object 
    data = Data(df_act, df_dev)
    data.df_dev_rep3 = cor_rep3
    return data
