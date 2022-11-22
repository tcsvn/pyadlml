import functools
import inspect
from copy import copy

import pandas as pd
import numpy as np

from pyadlml.constants import TIME, END_TIME, START_TIME, DEVICE, VALUE, BOOL, \
    CAT, ACTIVITY
from pandas.api.types import infer_dtype
from pyadlml.dataset._core.activities import ActivityDict, _is_activity_overlapping, \
    correct_activity_overlap

"""
    includes generic methods for manpulating dataframes
"""

def print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

def unitsfromdaystart(ts, unit='s'):
    """ Computes units passed from the start of the day until the timestamp
    """
    ts = pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts
    seconds = ts.hour*3600 + ts.minute*60 + ts.second
    if unit == 's':
        return int(seconds)
    elif unit == 'm':
        return int(seconds/60)
    elif unit == 'h':
        return int(seconds/3600)
    else:
        raise ValueError





def timestr_2_timedeltas(t_strs):
    """
        gets either a string or a list of strings to convert to a list of 
        timedeltas
    """
    if isinstance(t_strs, list):
        return [timestr_2_timedelta(t_str) for t_str in t_strs]
    else:
        return [timestr_2_timedelta(t_strs)]


def timestr_2_timedelta(t_str):
    """
        t_str (string)
        of form 30s, 30m
    """
    ttype = t_str[-1:]
    val = int(t_str[:-1])

    assert ttype in ['h','m','s']
    assert (val > 0 and val <=12 and ttype == 'h')\
        or (val > 0 and val <= 60 and ttype == 'm')\
        or (val > 0 and val <= 60 and ttype == 's')
    import datetime as dt
    if ttype == 's':
        return pd.Timedelta(seconds=val)
    if ttype == 'm':
        return pd.Timedelta(seconds=val*60)
    if ttype == 'h':
        return pd.Timedelta(seconds=val*3600)


def time2int(ts, t_res='30m'):
    """
    rounds to the next lower min bin or hour bin
    """
    assert t_res[-1:] in ['h','m']
    val = int(t_res[:-1])

    assert (val > 0 and val <=12 and t_res[-1:] == 'h')\
        or (val > 0 and val <= 60 and t_res[-1:] == 'm')

    import datetime as dt
    zero = dt.time()

    if t_res[-1:] == 'h':
        hs = val
        h_bin = int(ts.hour/hs)*hs
        return dt.time(hour=h_bin)

    elif t_res[-1:] == 'm':
        ms = val
        m_bin = int(ts.minute/ms)*ms
        return dt.time(hour=ts.hour, minute=m_bin)
    else:
        raise ValueError



def fill_nans_ny_inverting_first_occurence(df):
    """
    fills up true or false values
    :param df:
                Name           0      1         10
        time                                     
        2008-02-25 00:20:14    NaN    NaN   ...    True
        2008-02-25 00:22:57    NaN    NaN   ...    False
        2008-02-25 09:33:41    NaN    True  ...    True
        2008-02-25 09:33:42    False    NaN   ...    False
    :return:
                Name           0      1        10   
        Time                                       
        2008-02-25 00:20:14    True   False ... True 
        2008-02-25 00:22:57    True   False ... False 
        2008-02-25 09:33:41    True   True  ... True 
        2008-02-25 09:33:42    False   NaN  ... False 
    """
    for col_label in df.columns:
        col = df[col_label]

        # get timestamp of first valid index and replace previous Nans 
        #   by opposite
        ts = col.first_valid_index()
        idx = df.index.get_loc(ts)
        col.iloc[0:idx] = not col[ts]
    return df


def categorical_2_binary(df_devices, cat_list):
    """
    Transforms all categorical devices within the device dataframe into
    binary by creating a new device 'cat:cat_value' that reports a 1
    when the category is activated and 0 when it ends

    Parameters
    ----------
    df_devices : pd.DataFrame
        A device dataframe
    cat_list : lst
        Categorical devices from the dataframe

    Returns
    -------
    res : pd.DataFrame
        A device dataframe with newly created devices

    """
    mask_cat = df_devices[DEVICE].isin(cat_list)
    df_cat = df_devices[mask_cat].copy()
    df_cat.loc[:, 'new_device'] = df_cat[DEVICE] + ':' + df_cat[VALUE]
    df_cat.loc[:, VALUE] = True
    for dev in df_cat[DEVICE].unique():
        df = df_cat[df_cat[DEVICE] == dev]
        df_new = df.copy()
        df_new.loc[:, DEVICE] = df['new_device'].shift(1)
        df_new = df_new[1:]
        df_new.loc[:, VALUE] = False
        df_new.loc[:, TIME] += pd.Timedelta('1ns')
        df_cat = pd.concat([df_cat, df_new])

    mask = df_cat[DEVICE].isin(cat_list)
    df_cat.loc[mask, DEVICE] = df_cat.loc[mask, 'new_device']
    df_cat = df_cat.drop(columns='new_device')

    df_devices = pd.concat([df_devices[~mask_cat], df_cat])\
                    .sort_values(by=TIME)\
                    .reset_index(drop=True)
    return df_devices

def infer_dtypes(df_devices):
    """ Infers automatically the datatypes for each device of a device dataframe
    and returns a dictionary containing data types mapped to device names

    Parameters
    ----------
    df_devices : pd.DataFrame
        A device dataframe

    Returns
    -------
    res : dict
        Dictionary with keys: {'categorical' : [...], 'boolean': [...] , 'numerical' : [...]}

    """
    dev_cat = []
    dev_bool = []
    dev_num = []

    dev_lst = df_devices[DEVICE].unique()
    for dev in dev_lst:
        vals = df_devices[df_devices[DEVICE] == dev][VALUE]
        inf = infer_dtype(vals, skipna=True)
        if inf == 'string' or inf == 'object' or inf == 'mixed':
            try:
                pd.to_numeric(vals.dropna().unique())
                dev_num.append(dev)
            except:
                dev_cat.append(dev)
        elif inf == 'boolean':
            dev_bool.append(dev)
        elif inf == 'floating':
            dev_num.append(dev)
        else:
            raise ValueError('could not infer correct dtype for device {}'.format(dev))

    return {'categorical': dev_cat, 'boolean': dev_bool, 'numerical': dev_num}


def select_timespan(df_devs=None, df_acts=None, start_time=None, end_time=None, clip_activities=False):
    """ Selects a subset of a device and an activity dataframe based on a time span given
    Parameters
    ----------
    df_devices : pd.DataFrame or None
        A device dataframe
    df_activities : pd.DataFrame, dict containing dataframes or None
        An activity dataframe
    start_time : str or None
        The start time from
    end_time : str or None
    
    clip_activities : bool, default=False
        If set then the activities are clipped to the start and end time

    Returns
    -------
    df_d, df_a : the two subsets
    """
    # Cast to pandas timestamp
    start_time = str_to_timestamp(start_time) if isinstance(start_time, str) else start_time
    end_time = str_to_timestamp(end_time) if isinstance(end_time, str) else end_time

    if df_devs is not None:
        if start_time is not None:
            dev_start_sel = (df_devs[TIME] >= start_time)
        else:
            dev_start_sel = np.full(len(df_devs), True)
        if end_time is not None:
            dev_end_sel = (df_devs[TIME] < end_time)
        else:
            dev_end_sel = np.full(len(df_devs), True)
        df_devs = df_devs[dev_start_sel & dev_end_sel]

    if df_acts is not None:
        df_acts_inst_type = type(df_acts)
        df_acts = ActivityDict.wrap(df_acts)

        for k in df_acts.keys():
            df_activity = df_acts[k]
            if start_time is not None:
                act_start_sel = ~(df_activity[END_TIME] < start_time)
            else:
                act_start_sel = np.full(len(df_activity), True)
            if end_time is not None:
                act_end_sel = ~(df_activity[START_TIME] > end_time)
            else:
                act_end_sel = np.full(len(df_activity), True)

            df_activity = df_activity[act_start_sel & act_end_sel]

            # clip activities if they extend in regions that are not in the timespan
            if not df_activity.empty and start_time is not None \
                    and (start_time < df_activity[END_TIME].iat[0] and start_time > df_activity[START_TIME].iat[0]):
                df_activity[START_TIME].iat[0] = start_time
            if not df_activity.empty and clip_activities and end_time is not None \
            and end_time < df_activity[END_TIME].iat[-1]:
                df_activity[END_TIME].iat[-1] = end_time
                
            df_acts[k] = df_activity

        df_acts = df_acts.unwrap(df_acts_inst_type) 

    if df_acts is None:
        return df_devs
    elif df_devs is None:
        return df_acts
    else:
        return df_devs, df_acts





def str_to_timestamp(val):
    """ Converts a datetime string to a panda Timestamp.
    the day-first format is used.

    Parameters
    ----------
    val : list or string

    Returns
    -------
    pd.Timestamp

    """
    return pd.to_datetime(val, dayfirst=True)

def remove_days(df_devices, df_activities, days=[], offsets=[], retain_corrections=False):
    """ Removes the given days from activities and devices and shifts the succeeding days by that amount
        forward.

    Parameters
    ----------
    df_devices : pd.DataFrame
    df_activities : pd.DataFrame
    days : list
        List of strings
    offsets : list
        Offsets that are added to the corresponding days.

    Returns
        df_devices : pd.DataFrame
        df_activities : pd.DataFrame
    """
    nr_days = len(days)
    df_devs = df_devices.copy()
    df_acts = df_activities.copy()

    # Add offsets to the days specified
    for i in range(len(days)):
        days[i] = str_to_timestamp(days[i])
        if i < len(offsets):
            days[i] = days[i] + pd.Timedelta(offsets[i])

    # sort days from last to first
    days = np.array(days)[np.flip(np.argsort(days))]
    dtypes = infer_dtypes(df_devs)


    # 1. remove iteratively the latest day and shift the succeeding part accordingly
    for day in days:
        # when day is e.g 2008-03.23 00:00:00 then day after will be 2008-03-24 00:00:00
        # these variables have to be used as only timepoints can be compared as seen below
        day_after = day + pd.Timedelta('1D')

        # Remove devices events within the selected day
        mask = (day < df_devs[TIME]) & (df_devs[TIME] < day_after)
        removed_devs = df_devs[mask].copy()
        df_devs = df_devs[~mask]

        # shift the succeeding days timeindex one day in the past
        succeeding_days = (day_after < df_devs[TIME])
        df_devs.loc[succeeding_days, TIME] = df_devs[TIME] - pd.Timedelta('1D')

        # Binary devices that change an odd amount of states in that day will have
        # a wrong state for the succeeding days until the next event
        nr_dev_triggers = removed_devs.groupby(by=[DEVICE]).count()
        for dev in nr_dev_triggers.index:
            nr = nr_dev_triggers.loc[dev, 'time']
            if nr % 2 != 0 and (dev in dtypes[BOOL] or dev in dtypes[CAT]):
                print(f'Warning: Removed odd #events: device {dev} => inconsistent device states. Correction advised!')


        # Remove activities in that day that do not extend from the previous day into the selected
        # day or extend from the selected day into the next day
        mask_act_within_day = (day < df_acts[START_TIME]) & (df_acts[END_TIME] < day_after)
        df_acts = df_acts[~mask_act_within_day]

        # Special case where one activity starts before the selected day and ends after the selected day
        mask_special = (df_acts[START_TIME] < day) & (day_after < df_acts[END_TIME])
        if mask_special.any():
            # Adjust the ending by removing one day
            df_acts.loc[mask_special, END_TIME] = df_acts[END_TIME] - pd.Timedelta('1D')

        # Shift Activities that start in or after the selected day by one day
        succeeding_days = (day <= df_acts[START_TIME]) & (day_after < df_acts[END_TIME])
        df_acts.loc[succeeding_days, START_TIME] = df_acts[START_TIME] - pd.Timedelta('1D')
        df_acts.loc[succeeding_days, END_TIME] = df_acts[END_TIME] - pd.Timedelta('1D')
        df_acts['shifted'] = False
        df_acts.loc[succeeding_days, 'shifted'] = True

        # Special case where one activity starts before the selected day and ends inside the selected day
        # and there is no activity after the selected day
        #  | day_before | Sel day      | day after
        #     |-------------|
        #    I can't just move the ending one day before as this would reverse START_TIME and END_TIME order
        # -> The last activity ending is clipped to the start of the selected day
        # TODO has to be done before any activity is shifted
        #mask_last_true = pd.Series(np.zeros(len(df_acts), dtype=np.bool_))
        #mask_last_true.iat[-1] = True
        #mask_special = (df_acts[START_TIME] < day) & (df_acts[END_TIME] <= day_after) & mask_last_true
        #if mask_special.any():
        #    assert mask_special.sum() == 1
        #    df_acts.loc[mask_special, END_TIME] = day

        df_acts = df_acts.sort_values(by=START_TIME).reset_index(drop=True)
        # Merge activities from the day_before that overlap with the shifted activities from day after
        # there are 4 cases where overlaps into the respective days have to be handled
        #                           |   db  |   sd  |   da  |=>|   db  |   sd  |   da  |=>|   db  |   sd  |   da  |
        # 1. db overlaps da            |--------|    |~|          |--------|                 |-----|~|
        #                                                              |~|
        # 2. db intersect into  da      |------|     |~~~~|         |------|                   |----|~~~~|
        #                                                               |~~~~|
        # 3. da overlaps db             |-|   |~~~~~~~|            |-|                       |-|~~~|
        #                                                       |~~~~~~~|
        # 4. da intersect into db       |---|  |~~~~~|          |---|                        |---|~~~|
        #                                                         |~~~~~|
        # 5. da intersect into db         |----||~~~~~|            |-----|                 |---|~~~|
        #                                                      |~~~~~~|
        # case 1:
        # select activities that cross the boundary between days
        mask_db_into_sd = (df_acts[START_TIME] < day) & (df_acts[END_TIME] > day) & (df_acts[END_TIME] < day_after)
        idxs = np.where(mask_db_into_sd)[0]
        assert len(idxs) <= 2
        if len(idxs) == 2:
            # case 5: case when both activities extend into each days
            # clip both to midnight
            df_acts.iat[idxs[0], 2] = day
            df_acts.iat[idxs[1], 1] = day + pd.Timedelta('1ms')
        if len(idxs) == 1:
            idx_overlapping = idxs[0]

            # Check if the overlapping activity is part of the shifted or not
            if df_acts.iat[idx_overlapping, 3]:
                # Case when the shifted activities extend into the day before the selected day
                last_unshifted_act = df_acts.loc[(df_acts[END_TIME] < day), :]\
                                .copy().sort_values(by=END_TIME, ascending=True)\
                                .iloc[-1, :]

                # clip extending activities start_time to the end_time of the first shifted activity
                df_acts.iat[idx_overlapping, 0] = last_unshifted_act[END_TIME] + pd.Timedelta('1ms')
            else:
                # Case when the previous activities extends into the selected day
                first_shifted_act = df_acts.loc[(df_acts[START_TIME] > day) & (df_acts[END_TIME] < day_after), :]\
                                            .copy().sort_values(by=START_TIME, ascending=True)\
                                            .iloc[0, :]

                # clip extending activities end_time to the start of the first shifted activity
                df_acts.iat[idx_overlapping, 1] = first_shifted_act[START_TIME] - pd.Timedelta('1ms')

        df_acts = df_acts.drop(columns='shifted')

        from pyadlml.dataset._core.devices import correct_devices
        df_devs, corrections_dev = correct_devices(df_devs, retain_corrections)

        #assert not _is_activity_overlapping(df_acts)

    if retain_corrections:
        try:
            corrections_act
        except:
            corrections_act = []
        try:
            corrections_dev
        except:
            corrections_dev = []
        return df_devs, df_acts, corrections_act, corrections_dev
    else:
        return df_devs, df_acts


def df_difference(df1: pd.DataFrame, df2: pd.DataFrame, which=None):
    """Find rows which are different between two DataFrames.

    Parameters
    ----------
    df1 : pd.DataFrame
        TODO
    df2 : pd.DataFrame
        TODO
    which : None or str, default=None
        TODO


    Returns
    -------

    """
    from pyadlml.dataset._core.devices import is_device_df
    from pyadlml.dataset._core.activities import is_activity_df

    comparison_df = df1.copy().merge(
        df2.copy(),
        indicator=True,
        how='outer'
    )
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]

    if is_activity_df(df1):
        return diff_df[[START_TIME, END_TIME, ACTIVITY]]
    elif is_device_df(df1):
        return diff_df[[TIME, DEVICE, VALUE]]
    else:
        return diff_df




def event_times(df_devices, start_time=None, end_time=None):
    """
    Parameters
    ----------
    time_array : nd.array or pd.Series or pd.DataFrame
    start_time : pd.Timestamp

    end_time : pd.Timestamp


    Returns
    -------
    res : nd.array
        Array of transformed timestamps
    start_time : pd.Timestamp

    end_time : pd.Timestamp
    """
    if isinstance(time_array, pd.DataFrame):
        time_array = time_array[TIME].values
    if isinstance(time_array, pd.Series):
        time_array = time_array.values

    # get start and end_time
    if start_time is None:
        start_time = time_array[0]
    if end_time is None:
        end_time = time_array[-1]

    # map to values between [0,1]
    res = (time_array - start_time)/(end_time - start_time)

    return res, start_time, end_time


def num_to_timestamp(val, start_time, end_time):
    """Converts value [0,1] into timestamp between start_time and end_time"""
    return start_time + val*(end_time - start_time)

def timestamp_to_num(ts, start_time, end_time):
    """Converts timestamp between start_time and end_time into value in [0,1]"""
    return float((ts - start_time)/(end_time - start_time))

def get_sorted_index(df: pd.DataFrame, rule='alphabetical', area: pd.DataFrame = None) -> np.ndarray:
    """ Returns a new dataframes index that is sorted after a specific rule

    Parameters
    ----------
    df : pd.DataFrame
        Has to contain the column 'activity' or 'device'

    rule : one of {list, ndarray, 'alphabetical', 'area', str}
        The rule on how to order the array.

        - alphabetical : the

    Returns
    -------
    new_order : ndarray
        An array with the new ordered indices for the dataframe
    """
    rule_is_iter = (isinstance(rule, list) or isinstance(rule, np.ndarray))

    # If df is a list the enumeration for that list is returned
    if (isinstance(df, list) or isinstance(df, np.ndarray)) and rule_is_iter:
        mapping = {v: k for k, v in enumerate(rule)}
        return np.vectorize(mapping.get)(df)

    df = df.copy()
    df['order'] = np.arange(len(df))
    new_order = []

    if rule_is_iter:
        mapping = {v: k for k, v in enumerate(rule)}
        order_changed = False
        try:
            df['order'] = df[ACTIVITY].map(mapping)
            order_changed = True
        except:
            pass
        try:
            df['order'] = df[DEVICE].map(mapping)
            order_changed = True
        except:
            pass
        assert order_changed, '"activity" or "device" was not present in dataframe'
    elif rule == 'alphabetical':
        if ACTIVITY in df.columns:
            df = df.sort_values(by=ACTIVITY)
        elif DEVICE in df.columns:
            df = df.sort_values(by=DEVICE)
        else:
            raise KeyError(f"Tried to sort alphabetical but no activity or device column was found, only {rule}")
    elif rule == 'value':
        # The case when the column other than ACTIVITY or order should be used
        cs = df.columns
        col = [i for i in cs if i not in [ACTIVITY, DEVICE, 'order']]
        assert len(col) == 1, 'When value is specified there must be only one remaining column.'
        if ACTIVITY in df.columns:
            df = df.sort_values(by=col[0])
        elif DEVICE in df.columns:
            df = df.sort_values(by=col[0])
        else:
            raise KeyError(f"Tried to sort alphabetical but no activity or device column was found, only {rule}")
    elif rule == 'areas':
        raise NotImplementedError
    elif isinstance(rule, str):
        assert rule in df.columns, f'The rule {rule} was not in the dataframes columns.'
        df = df.sort_values(by=rule)
    else:
        raise KeyError()
    new_order = df['order'].values

    return new_order


def extract_kwargs(func):
    """ Gets a function end sets all args to kwargs and all default values from
        the signature.
    """
    @functools.wraps(func)
    def wrapper_extract(*args, **kwargs):
        sign = inspect.signature(func).parameters

        # assign the args to the respective kwargs
        for arg_name, arg in zip(sign, args):
            kwargs[arg_name] = arg

        # set the rest of the kwargs to the default values
        for param in sign.values():
            if param.name not in kwargs.keys():
                kwargs[param.name] = param.default

        return func(**kwargs)
    return wrapper_extract


def check_order(func):
    @extract_kwargs
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        o = kwargs['order']
        assert o in ['alphabetical', 'occurence'] or isinstance(o, list) \
               or isinstance(o, np.ndarray), f'Sort rule is either "alphabetical", a custom string or a iterable. Found {o}'
        tmp = func(*args, **kwargs)
        return tmp
    return wrapper

def check_scale(func):
    @extract_kwargs
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert kwargs['scale'] in ['linear', 'log']
        return func(*args, **kwargs)
    return wrapper


def device_order_by(df_devs, rule='alphabetical'):
    """Return a list with ordered devices
    """
    is_iterable = isinstance(rule, list) or isinstance(rule, np.ndarray)
    if is_iterable:
        return rule

    assert rule in ['alphabetical', 'count', 'area'], f'rule{rule} not in assertion.'

    if rule == 'alphabetical':
        dev_order = df_devs[DEVICE].unique()
        dev_order.sort()
    elif rule == 'count':
        from pyadlml.dataset.stats.devices import device_order_by_count
        dev_order = device_order_by_count(df_devs)
    else:
        raise NotImplementedError('The order has still to be implemented')
    return dev_order

def activity_order_by(dct_acts, rule='alphabetical'):
    """ Return a list with ordered activities
    """
    is_iterable = isinstance(rule, list) or isinstance(rule, np.ndarray)
    if is_iterable:
        # TODO refactor, negate: any not int and not dct is none than below 
        if all(isinstance(item, int) for item in rule) or dct_acts is None:
            return rule

        # Filter out activities that are not in df_acts
        rule = copy(list(rule))
        for act_not_in_df in set(rule) - set(dct_acts[ACTIVITY].unique()):
            rule.remove(act_not_in_df)
        return rule

    assert rule in ['alphabetical', 'duration', 'count', 'area'], f'rule{rule} not in assertion.'

    if isinstance(dct_acts, pd.DataFrame):
        dct_acts = ActivityDict.wrap(dct_acts)

    if rule == 'alphabetical':
        act_order = dct_acts.get_activity_union()
        act_order.sort()
    elif rule == 'duration':
        from pyadlml.dataset.stats.activities import activity_order_by_duration
        act_order = activity_order_by_duration(dct_acts)
    elif rule == 'count':
        from pyadlml.dataset.stats.activities import activity_order_by_count
        act_order = activity_order_by_count(dct_acts)
    else:
        raise NotImplementedError('Area has still to be implemented.')

    return act_order

DATASET_STRINGS = [
    'casas_aruba',
    'amsterdam',
    'mitlab_1',
    'mitlab_2',
    'aras',
    'kasteren_A',
    'kasteren_B',
    'kasteren_C',
    'tuebingen_2019',
    'uci_ordonezA',
    'uci_ordonezB',
    'act_assist',
    'dump'
]
def fetch_data_by_string(dataset: str, cache=True, identifier=None):
    """ Fetches data based on a given string representation

    Parameters
    ----------
    dataset: str

    identifier: str, default=None
        Identifies a custom dataset in the data home. In the case
        of an act_assist dataset this is the folder name. In the case
        of a dumped dataframe with pyadlml.io.dump this is the name.

    Returns
    -------
    data : pyadlml.dataset.obj
    """

    from pyadlml.dataset import fetch_amsterdam, \
                                fetch_mitlab, fetch_aras, fetch_kasteren_2010, \
                                fetch_casas_aruba, fetch_tuebingen_2019, \
                                fetch_uci_adl_binary, load_act_assist
    if dataset == DATASET_STRINGS[0]:
        data = fetch_casas_aruba(cache=cache)
    elif dataset == DATASET_STRINGS[1]:
        data = fetch_amsterdam(cache=cache)
    elif dataset == DATASET_STRINGS[2]:
        data = fetch_mitlab(subject='subject1', cache=cache)
    elif dataset == DATASET_STRINGS[3]:
        data = fetch_mitlab(subject='subject2', cache=cache)
    elif dataset == DATASET_STRINGS[4]:
        data = fetch_aras(cache=cache)
    elif dataset == DATASET_STRINGS[5]:
        data = fetch_kasteren_2010(house='A', cache=cache)
    elif dataset == DATASET_STRINGS[6]:
        data = fetch_kasteren_2010(house='B', cache=cache)
    elif dataset == DATASET_STRINGS[7]:
        data = fetch_kasteren_2010(house='C', cache=cache)
    elif dataset == DATASET_STRINGS[8]:
        data = fetch_tuebingen_2019(cache=cache)
    elif dataset == DATASET_STRINGS[9]:
        data = fetch_uci_adl_binary(subject='OrdonezA', cache=cache)
    elif dataset == DATASET_STRINGS[10]:
        data = fetch_uci_adl_binary(subject='OrdonezB', cache=cache)
    elif dataset == DATASET_STRINGS[11]:
        data = load_act_assist(identifier)
    elif dataset == DATASET_STRINGS[12]:
        from pyadlml.dataset._core.devices import is_device_df
        from pyadlml.dataset.io import load
        from pyadlml.dataset._core.activities import is_activity_df
        df_lst = load(identifier)

        df_devs = None
        df_acts = None
        for df in df_lst:
            if is_device_df(df):
                df_devs = df
            elif is_activity_df(df):
                df_acts = df
            else:
                raise ValueError('At least one of the dataframes has to be a device dataframe')
        assert df_devs is not None and df_acts is not None
        data = Data(df_acts, df_devs)

    else:
        raise ValueError(f'No suitable option specified: {dataset}.\nAvailable are {DATASET_STRINGS}')

    return data


def get_dev_row_where(df, time, dev, state):
    time = pd.Timestamp(time)
    df = df.copy().reset_index().set_index(TIME)
    mask = (df[DEVICE] == dev) \
           & (df.index == time) \
           & (df[VALUE] == state)
    df = df.reset_index().set_index('index')
    return df[mask.values].copy()

def get_dev_rows_where(df_devs, rows):
    res = [get_dev_row_where(df_devs, r[0], r[1], r[2]) for r in rows]
    return pd.concat(res)

def append_devices(df_devs, rows):
    df = pd.DataFrame(data=rows, columns=[TIME, DEVICE, VALUE])
    df[TIME] = pd.to_datetime(df[TIME])
    return pd.concat([df_devs, df])

def remove_devices(df_devs, rows):
    idx_to_drop = [get_dev_row_where(df_devs, r[0], r[1], r[2]).index[0] for r in rows]
    return df_devs.drop(index=idx_to_drop)
