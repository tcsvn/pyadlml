import pandas as pd 
from pyadlml.dataset._dataset import label_data
from pyadlml.util import get_npartitions, get_parallel
import dask.dataframe as dd
from pyadlml.dataset import START_TIME, END_TIME, TIME, DEVICE, VAL, ACTIVITY
from pyadlml.dataset._representations.raw import create_raw
#import __logger__


def contingency_table_triggers_01(df_devs, df_acts, idle=False):
    """
    Compute the amount a device turns "on" or "off" respectively
    during the different activities.

    Parameters
    ----------
    df_devs : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    idle : bool
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.

    Examples
    --------
    >>> from pyadlml.stats import contingency_triggers_01
    >>> contingency_triggers_01(data.df_devices, data.df_activities)
    activity            get drink  go to bed  ... use toilet
    devices                                   ...
    Cups cupboard Off   18          0         ...          0
    Cups cupboard On    18          0         ...          0
    Dishwasher Off       1          0         ...          0
    Dishwasher On        1          0         ...          0
                   ... ...        ...         ...        ...
    Washingmachine Off   0          0         ...          0
    Washingmachine On    0          0         ...          0
    [14 rows x 7 columns]

    Results
    -------
    df : pd.DataFrame
    """
    dev_index = 'devices'
    ON = 'On'
    OFF = 'Off'
    df = label_data(df_devs, df_acts, idle=idle)
    
    df['val2'] = df[VAL].astype(int)
    df = pd.pivot_table(df,
               columns=ACTIVITY,
               index=[DEVICE, VAL],
               values='val2',
               aggfunc=len,
               fill_value=0)

    # format text strings
    def func(x):
        if "False" in x:
            return x[:-len("False")] + " Off"
        else:
            return x[:-len("True")] + " On"

    df = df.reset_index()
    df[dev_index] = df[DEVICE] + df[VAL].astype(str)
    df[dev_index] = df[dev_index].apply(func)
    df = df.set_index(dev_index)
    df = df.drop([DEVICE, VAL], axis=1)
    return df


def contingency_table_triggers(df_devs, df_acts, idle=False):
    """
    Compute the amount of device triggers occuring during the different activities.

    Parameters
    ----------
    df_devs : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    idle : bool
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.

    Examples
    --------
    >>> from pyadlml.stats import contingency_triggers
    >>> contingency_triggers(data.df_devices, data.df_activities)
    activity            get drink  go to bed  ...  use toilet
    device
    Cups cupboard              36          0  ...           0
    Dishwasher                  2          0  ...           0
               ...            ...        ...  ...         ...
    Washingmachine              0          0  ...           0
    [7 rows x 7 columns]

    Results
    -------
    df : pd.DataFrame
    """
    df = label_data(df_devs, df_acts, idle=idle)    
    df[VAL] = 1
    return pd.pivot_table(df, 
               columns=ACTIVITY,
               index=DEVICE,
               values=VAL,
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
    td = 'time_diff'
    df = pd.DataFrame(columns=[DEVICE, VAL, ACTIVITY, td])
    df[td] = pd.to_timedelta(df[td])
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


def contingency_intervals(df_devs, df_acts, idle=False):
    """
    Compute the time a device is "on" or "off" respectively
    during the different activities.

    Parameters
    ----------
    df_devs : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    idle : bool
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.

    Examples
    --------
    >>> from pyadlml.stats import contingency_duration
    >>> contingency_duration(data.df_devices, data.df_activities)
    activity                     get drink ...             use toilet
    Hall-Bedroom door Off  0 days 00:01:54 ... 0 days 00:12:24.990000
    Hall-Bedroom door On   0 days 00:14:48 ... 0 days 03:02:49.984000
    ...                                ...
    Washingmachine On      0 days 00:00:00 ...        0 days 00:00:00
    [14 rows x 7 columns]

    Returns
    -------
    df : pd.DataFrame
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
        raw_sel = raw[(act_start_time <= raw[TIME]) & (raw[TIME] <= act_end_time)].copy()

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
        return {**{TIME: 'datetime64[ns]', 'td': 'timedelta64[ns]'}, **devices}
        
    dev_lst = df_devs[DEVICE].unique()
    df_devs = df_devs.sort_values(by=TIME)
    raw = create_raw(df_devs).applymap(lambda x: 'ON' if x else 'OFF').reset_index(drop=False)
    raw[TD] = raw[TIME].shift(-1) - raw[TIME]
    
    y = [(d1 + ' Off',d2 + ' On') for d1,d2 in zip(dev_lst, dev_lst)]
    new_cols = [d for tup in y for d in tup] 
            
    
    df_acts = df_acts.copy().join(pd.DataFrame(index=df_acts.index, columns=new_cols))
    if True: # TODO parallel is not working
    #if not get_parallel():
        df = df_acts.apply(func, args=[raw, dev_lst], axis=1)
        df = df.drop(columns=[START_TIME, END_TIME])
        df = df.groupby(ACTIVITY).sum()
        return df.T
    else:
        df = dd.from_pandas(df_acts.copy(), npartitions=get_npartitions())\
                .apply(func, args=[raw, dev_lst], axis=1)\
                .drop(columns=[START_TIME, END_TIME])\
                .groupby(ACTIVITY).sum()\
                .compute(scheduler='processes')
        return df.T
