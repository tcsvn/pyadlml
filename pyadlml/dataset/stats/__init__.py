import pandas as pd 
from pyadlml.dataset._dataset import label_data

def contingency_table_triggers_01(df_devs, df_acts, idle=False):

    """
    output: 
        read like this: dev 1 turned 123 times from 1 to 0 while act 1 was present
        example
        ---------------------------
                | act 1 | .... | act n|
        ------------------------------- 
        dev 1 0 | 123   |      | 123  | 
        dev 1 1 | 122   |      | 141  |
        ... 
        dev n 0 | 123   |      | 123  | 
        dev n 1 | 122   |      | 141  |
    """
    df = label_data(df_devs, df_acts, idle=idle)
    df['val2'] = df['val'].astype(int)
    return pd.pivot_table(df, 
               columns='activity',
               index=['device', 'val'],
               values='val2',
               aggfunc=len,
               fill_value=0)


def contingency_table_triggers(df_devs, df_acts, idle=False):
    """
    output: 
        read like this: dev 1 was 123 times triggered while act 1 was present
        example
        ---------------------------
                | act 1 | .... | act n|
        ------------------------------- 
        dev 1 | 123   |      | 123  |
        ... 
        dev n | 123   |      | 123  | 
    """
    df = label_data(df_devs, df_acts, idle=idle)    
    df['val'] = 1
    return pd.pivot_table(df, 
               columns='activity',
               index='device',
               values='val',
               aggfunc=len,
               fill_value=0)

def _join_to_interval(x):
    #print('x: ', x)
    return pd.Interval(x[0], x[1])

def _create_cont_df():
    """
    returns: empty pd Dataframe 
        | i | device | val |  activity | time_diff
        -----------------------------
          1 | dev1   | 0   | act1      | td
    """
    df = pd.DataFrame(columns=['device', 'val', 'activity', 'time_diff'])
    df['time_diff'] = pd.to_timedelta(df['time_diff'])
    return df 

def _calc_intervall_diff(i1, i2):
    """
    computes the difference of time between two overlapping intervals
        form new intervall by taking 
        #  |...| 
        #  ~  ~
        # |...|
    
    returns
    -------
        time difference 
    """
    assert i1.overlaps(i2)
    mi = max([i1.left, i2.left])
    ma = min([i1.right, i2.right])
    #return pd.Interval(mi, ma)
    return ma-mi

def _get_activity_device_interval_overlaps(df_acts, df_devs):
    """
    returns: empty pd Dataframe 
        | i | device | val | activity | time_diff
        -----------------------------
          1 | dev1   | 0   | act1     | td
    """

    # add the "OFF" time intervals
    dev_list = df_devs['device'].unique()
    df_final_res = _create_cont_df()
    for dev in dev_list:
        df_cur_dev = df_devs[df_devs['device'] == dev]
        tmp = df_cur_dev.copy()
        tmp['val'] = False
        tmp['start_time'] = tmp['start_time'].shift(-1)

        # swap columns and delete last invalid row
        tmp[['start_time', 'end_time']] = tmp[['end_time', 'start_time']]
        tmp = tmp[:-1]

        df_cur_dev = df_cur_dev.append(tmp).sort_values('start_time')
        df_cur_dev['time_int'] = df_cur_dev[['start_time', 'end_time']].apply(_join_to_interval, axis=1)
        #fridge

        # add interval to df activity 
        df_acts['time_int'] = df_acts[['start_time', 'end_time']].apply(_join_to_interval, axis=1)

        df_res = _create_cont_df()

        for row in df_cur_dev.iterrows():
            # get overlapping activities
            t_int = row[1].time_int
            val = row[1].val
            
            mask = df_acts['time_int'].apply(lambda x: x.overlaps(t_int))
            overlapping_acts = df_acts[mask]

            # if no activity overlaps 
            if len(overlapping_acts) == 0:
                #print('emtpy')
                pass

            else:          
                # for all overlapping activities compute the interval overlaps and 
                df_tmp = _create_cont_df()
                for i, row_a in enumerate(overlapping_acts.iterrows()):
                    td = _calc_intervall_diff(t_int, row_a[1].time_int)
                    df_tmp.loc[i] = [dev, val, row_a[1].activity, td]                   
                df_res = df_res.append(df_tmp, ignore_index=True)
                
        df_final_res = df_final_res.append(df_res, ignore_index=True)
    return df_final_res

def contingency_table_interval_overlaps(df_act, df_dev): 
    """
    output: 
        read like this: dev 1 turned 123 times from 1 to 0 while act 1 was present
        example
        ---------------------------
                | act 1 | .... | act n|
        ------------------------------- 
        dev 1 0 | 123   |      | 123  | 
        dev 1 1 | 122   |      | 141  |
        ... 
        dev n 0 | 123   |      | 123  | 
        dev n 1 | 122   |      | 141  |
    """
    df = _get_activity_device_interval_overlaps(df_act, df_dev)
    con = pd.crosstab(
               index=[df['device'], df['val']],
               columns=df['activity'],
               values=df['time_diff'],
               aggfunc=sum)
    con = con.fillna(pd.Timedelta(milliseconds=0))
    return con