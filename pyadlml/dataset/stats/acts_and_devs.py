import pandas as pd 
from pyadlml.dataset._dataset import label_data
from pyadlml.util import get_npartitions, get_parallel
import dask.dataframe as dd
from pyadlml.dataset import START_TIME, END_TIME, TIME, DEVICE
from pyadlml.dataset._representations.raw import create_raw
#import __logger__


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

def contingency_intervals(df_dev, df_act):
    """ compute the crosscorelation by comparing for every interval the binary values
    between the devices
    
    Parameters
    ----------
    df_dev: pd.DataFrame
        device representation 1 
        columns [time, device, val]
    df_act : pd.DataFrame
        contains activities
        columns [start_time, end_time, activity]
        
    Returns
    -------
    output : pd.DataFrame
        read like this: dev 1 turned 123 times from 1 to 0 while act 1 was present
        example
        ---------------------------
                | act 1 | .... | act n|
        ------------------------------- 
        dev 1 0 | 123   |      | 123  | 
        dev 1 1 | 122   |      | 141  |
        ... 
        dev n 0 | 123   |      | 123  | 
        dev n 1 | 122   |      | 141  
    """
    TD = 'time_difference_to_succ'
    
    def func(row, raw, dev_lst):
        """ determines for each activity row the totol time that was spent in either on or off state for each device
        Parameters
        ----------
        row : pd.Series
            a row of the activity dataframe contatining the start and end time for one acitivity
        """        
        # get selection of relevant devices
        act_start_time = row.start_time
        act_end_time = row.end_time
        raw_sel = raw[(act_start_time <= raw['time']) & (raw['time'] <= act_end_time)].copy()

        if raw_sel.empty:
            # the case when no device activation fell into the recorded activity timeframe
            return pd.Series(index=row.index, name=row.name, dtype=row.dtype)



        # determine end and start time and correct for the intervals before/after
        # the first/last state vector s0,sn
        #     s0 ---------I --activity --sn--------I
        #     | ~~~tds~~~ |              | ~~tde~~ |
        #    rs          as             re        ae

        # try to get the preceding state vector of devices before the activity starts
        idx_first = raw_sel.index[0] - 1
        if idx_first == -1:
            # edge case when the first activity starts before the first recording
            # this case isn't solvable. So a heurstic that doesn't skew the statistic
            # to much is to assume the same state at the start of the activity
            raw_sel = raw_sel.append(raw_sel.iloc[0].copy()).sort_values(by=[TIME])
            raw_sel.iat[0, raw_sel.columns.get_loc(TD)] = raw_sel.iloc[0].time - act_start_time
        else:
            raw_sel = raw_sel.append(raw.iloc[idx_first]).sort_values(by=[TIME])
            raw_start = raw_sel.iloc[0]
            t_diff_start = act_start_time - raw_start.time
            raw_sel.at[raw_sel.iloc[0].name, TD] -= t_diff_start

        # set time difference for last state vector until activity ends
        raw_sel.at[raw_sel.iloc[-1].name, TD] = act_end_time - raw_sel.iloc[-1].time

        for dev in dev_lst:
            ser = raw_sel.groupby(by=[dev])[TD].sum()
            # the tries are for the cases when a device is on/off the whole time
            try:
                dev_on_time = ser.ON
            except AttributeError:
                dev_on_time = pd.Timedelta('0ns')
            try:
                dev_off_time = ser.OFF
            except AttributeError:
                dev_off_time = pd.Timedelta('0ns')

            row.at[ser.index.name + " On"] = dev_on_time
            row.at[ser.index.name + " Off"] = dev_off_time        
        return row
    
    def create_meta(raw):
        devices = {name : 'object' for name in raw.columns[1:-1]}
        return {**{'time': 'datetime64[ns]', 'td': 'timedelta64[ns]'}, **devices}
        
    dev_lst = df_dev['device'].unique()
    df_dev = df_dev.sort_values(by='time')
    raw = create_raw(df_dev).applymap(lambda x: 'ON' if x else 'OFF').reset_index(drop=False)
    raw[TD] = raw['time'].shift(-1) - raw['time']
    
    y = [(d1 + ' Off',d2 + ' On') for d1,d2 in zip(dev_lst, dev_lst)]
    new_cols = [d for tup in y for d in tup] 
            
    
    df_act = df_act.copy().join(pd.DataFrame(index=df_act.index, columns=new_cols))
    if True: # TODO parallel is not working
    #if not get_parallel():
        df = df_act.apply(func, args=[raw, dev_lst], axis=1)
        df = df.drop(columns=['start_time', 'end_time'])
        df = df.groupby('activity').sum()
        return df.T
    else:
        df = dd.from_pandas(df_act.copy(), npartitions=get_npartitions())\
                .apply(func, args=[raw, dev_lst], axis=1)\
                .drop(columns=['start_time', 'end_time'])\
                .groupby('activity').sum()\
                .compute(scheduler='processes')
                
        return df.T
    
