import pandas as pd

from pyadlml.dataset.util import fill_nans_ny_inverting_first_occurence
from pyadlml.dataset.devices import correct_devices
from pyadlml.dataset.activities import correct_activities
                                        
from pyadlml.dataset.activities import correct_activity_overlap, \
    _is_activity_overlapping \

from pyadlml.dataset.obj import Data
from pyadlml.dataset import ACTIVITY, VAL, START_TIME, END_TIME, TIME, NAME, DEVICE

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
    df = _load_df(data_path)
    df_dev = _get_devices_df(df)
    df_act = _get_activity_df(df)
    
    dev_rep1 = correct_devices(df_dev)
    df_act, cor_lst = correct_activities(df_act)
    
    data = Data(df_act, dev_rep1)
    return data

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