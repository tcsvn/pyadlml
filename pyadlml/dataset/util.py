import functools
import inspect
import joblib
from copy import copy

import pandas as pd
import numpy as np

from pyadlml.constants import DATASET_STRINGS, TIME, END_TIME, START_TIME, DEVICE, VALUE, BOOL, \
    CAT, ACTIVITY
from pandas.api.types import infer_dtype
from pyadlml.dataset._core.activities import ActivityDict, _is_activity_overlapping, \
    correct_activity_overlap, is_activity_df

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
        elif inf == 'floating' or 'integer':
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
        df_devs = df_devs[dev_start_sel & dev_end_sel].copy()

    if df_acts is not None:
        df_acts_inst_type = type(df_acts)
        df_acts = ActivityDict.wrap(df_acts).copy()

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
                df_activity.at[df_activity.index[0], START_TIME] = start_time
            if not df_activity.empty and clip_activities and end_time is not None \
            and end_time < df_activity[END_TIME].iat[-1]:
                df_activity.at[df_activity.index[-1], END_TIME] = end_time
                
            df_acts[k] = df_activity

        df_acts = df_acts.unwrap(df_acts_inst_type) 

    if df_acts is None:
        return df_devs
    elif df_devs is None:
        return df_acts
    else:
        return df_devs, df_acts


def get_last_states(df_devs: pd.DataFrame, left_bound=-1) -> dict:
    """ Creates a dictionary where every device maps to its last known value.

    Parameters
    ----------
    df_devs : pd.DataFrame
        TODO add description 
    left_bound : int
        The index of the first event from which the states are determined directed
        at the past 

    Returns
    -------
    res : dict
        A device mapping to initial values
    """
    left_bound = len(df_devs) if left_bound == -1 else left_bound
    return df_devs.iloc[:left_bound, :]\
            .sort_values(by=TIME)\
            .groupby(DEVICE)\
            .last()[VALUE]\
            .to_dict()

def get_first_states(df_devs: pd.DataFrame) -> dict:
    """ Creates a dictionary where every device maps to its first known value.

    Parameters
    ----------
    df_devs : pd.DataFrame
        TODO add description 

    Returns
    -------
    res : dict
        A device mapping to initial values
    """
    return df_devs.copy()\
        .sort_values(by=TIME)\
        .groupby(DEVICE)\
        .first()[VALUE]\
        .to_dict()



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


def df_difference(df1: pd.DataFrame, df2: pd.DataFrame, which=None, return_mask=False):
    """Find rows which are different between two DataFrames.

    Parameters
    ----------
    df1 : pd.DataFrame
        TODO
    df2 : pd.DataFrame
        TODO
    which : None or str, default=None
        When set to 'left' returns the elements where the 
        first dataframe differs from the second
    return_mask: book, default=False

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
        mask = (comparison_df['_merge'] != 'both')
    else:
        mask = (comparison_df['_merge'] == which)

    if return_mask:
        return mask
    else:
        diff_df = comparison_df[mask]

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


def device_order_by(df_devs: pd.DataFrame, rule='alphabetical', dev2area=None) -> list:
    """Create a list of devices ordered by some rule

    Returns
    -------
    list of strings
    """
    if isinstance(rule, list) or isinstance(rule, np.ndarray):
        return rule

    allowed_combis =['alphabetical', 'count', 'area', 'area+alphabetical', 'area+count'] 
    assert rule in allowed_combis, f'rule{rule} not in assertion.'

    if rule == 'alphabetical':
        dev_order = df_devs[DEVICE].unique().tolist()
        dev_order.sort() # CAVE returns None, do not change ^^
    elif rule == 'count':
        from pyadlml.dataset.stats.devices import device_order_by_count
        dev_order = device_order_by_count(df_devs)
    elif 'area' in rule:
        assert dev2area is not None, 'When selecting area as sorting measure, an area mapping must be given as parameter.'
        split = rule.split('+')
        dev2area['area'] = dev2area['area'].fillna('-na')
        if len(split) == 1 or split[1] == 'alphabetical' :
            dev_order = dev2area.groupby('area')\
                                .apply(lambda x: x.sort_values(by=DEVICE))\
                                .reset_index(drop=True)[DEVICE].to_list()
        elif split[1] == 'count':
            from pyadlml.dataset.stats.devices import device_order_by_count
            ordered_by_count = device_order_by_count(df_devs)
            dev_order = pd.DataFrame(zip(range(len(ordered_by_count), ordered_by_count)), columns=['count', DEVICE])
            dev_order = dev_order.set_index(DEVICE)
            tmp = dev2area.set_index(DEVICE)
            tmp['count'] = dev_order['count']
            dev_order = dev2area.groupby('area')\
                                .apply(lambda x: x.sort_values(by='count'))\
                                .reset_index(drop=True)[DEVICE].to_list()

    return dev_order

def activity_order_by(dct_acts, rule: str='alphabetical') -> list:
    """ Return a list with ordered activities

    Parameters
    ----------
    dct_acts: pd.DataFrame or ActivityDict
    rule: str
        The rule one of:
            - alphabetical
            - duration
            - count
            - area
    Returns
    -------
    list of str
        The ordered activity names
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


def fetch_by_name(dataset: str, identifier=None, **kwargs) -> dict:
    """ Fetches a dataset based on a given string. 

    Parameters
    ----------
    dataset: str


    Returns
    -------
    dict
    """
    error_msg = f'No suitable option specified: {dataset}.\nAvailable are {DATASET_STRINGS}'
    assert dataset in DATASET_STRINGS + ['joblib'], error_msg

    from pyadlml.dataset import fetch_amsterdam, \
                                fetch_mitlab, fetch_aras, fetch_kasteren_2010, \
                                fetch_tuebingen_2019, fetch_casas,\
                                fetch_uci_adl_binary, load_act_assist
    if dataset == DATASET_STRINGS[0]:
        return fetch_casas(testbed='aruba', **kwargs)
    if dataset == DATASET_STRINGS[1]:
        return fetch_casas(testbed='kyoto_2010', **kwargs)
    if dataset == DATASET_STRINGS[2]:
        return fetch_casas(testbed='milan', **kwargs)
    if dataset == DATASET_STRINGS[3]:
        return fetch_casas(testbed='cairo', **kwargs)
    if dataset == DATASET_STRINGS[4]:
        return fetch_casas(testbed='tulum', **kwargs)
    elif dataset == DATASET_STRINGS[5]:
        return fetch_amsterdam(**kwargs)
    elif dataset == DATASET_STRINGS[6]:
        return fetch_mitlab(subject='subject1', **kwargs)
    elif dataset == DATASET_STRINGS[7]:
        return fetch_mitlab(subject='subject2', **kwargs)
    elif dataset == DATASET_STRINGS[8]:
        return fetch_aras(**kwargs)
    elif dataset == DATASET_STRINGS[9]:
        return fetch_kasteren_2010(house='A', **kwargs)
    elif dataset == DATASET_STRINGS[10]:
        return fetch_kasteren_2010(house='B', **kwargs)
    elif dataset == DATASET_STRINGS[11]:
        return fetch_kasteren_2010(house='C', **kwargs)
    elif dataset == DATASET_STRINGS[12]:
        return fetch_tuebingen_2019(**kwargs)
    elif dataset == DATASET_STRINGS[13]:
        return fetch_uci_adl_binary(subject='OrdonezA', **kwargs)
    elif dataset == DATASET_STRINGS[14]:
        return fetch_uci_adl_binary(subject='OrdonezB', **kwargs)
    elif dataset == DATASET_STRINGS[15]:
        return load_act_assist(identifier)
    elif dataset == 'joblib':
        return joblib.load(identifier)


def get_dev_row_where(df, time, dev, state):
    raise ValueError('deprecated: use get_index_matchingrows instead')
    time = pd.Timestamp(time)
    mask = (df[DEVICE] == dev) \
           & (df[TIME] == time) \
           & (df[VALUE] == state)
    df = df.reset_index().set_index('index')
    return df[mask.values].copy()

def get_dev_rows_where(df_devs, rows):
    raise ValueError('deprecated: use get_index_matchingrows instead')
    res = [get_dev_row_where(df_devs, r[0], r[1], r[2]) for r in rows]
    return pd.concat(res)

def append_devices(df_devs, rows):
    raise ValueError('deprecated: use get_index_matchingrows instead')
    df = pd.DataFrame(data=rows, columns=[TIME, DEVICE, VALUE])
    df[TIME] = pd.to_datetime(df[TIME])
    return pd.concat([df_devs, df])

def remove_devices(df_devs, rows):
    raise ValueError('deprecated: use get_index_matchingrows instead')
    idx_to_drop = [get_dev_row_where(df_devs, r[0], r[1], r[2]).index[0] for r in rows]
    return df_devs.drop(index=idx_to_drop)




def to_sktime(df_devs: pd.DataFrame, df_acts: pd.DataFrame = None, return_type: str=None, 
              return_X_y: bool=False):
    """

    Parameters
    ----------
    df_devs: pd.DataFrame
        TODO 

    df_acts: pd.DataFrame, optional
        TODO

    return_type: str or None, default=None
        Memory data format specification to return X in, None = “nested_univ” type. str can be any supported sktime Panel mtype,
        for list of mtypes, see datatypes.MTYPE_REGISTER for specifications, see examples/AA_datatypes_and_datasets.ipynb
        commonly used specifications:
            - “nested_univ: nested pd.DataFrame, pd.Series in cells 
            - “numpy3D”/”numpy3d”/”np3D”: 3D np.ndarray (instance, variable, time index) 

    return_X_y: bool, default=False
        it returns two objects, if False, it appends the class labels to the dataframe.


    
    Returns 
    -------
    X: pd.DataFrame
        The time series data for the problem with n_cases rows and either n_dimensions or n_dimensions+1 columns. Columns 1 to n_dimensions are the series associated with each case. If return_X_y is False, column n_dimensions+1 contains the class labels/target variable.
    y: numpy array, optional
        The class labels for each case in X, returned separately if return_X_y is True, or appended to X if False

    """
    from pyadlml.preprocessing.preprocessing import LabelMatcher

    return_type = 'nested_univ' if return_type is None else return_type
    return_type = 'numpy3d' if return_type in ['numpy3D', 'np3D', 'np3d'] else return_type

    if return_type in ["pd-multiindex", "numpy2d"]:
        err_txt = "BasicMotions loader: Error, attempting to load into a numpy2d array, but cannot because it is a multivariate problem. Use numpy3d instead"
        raise ValueError(err_txt)

    assert return_type in ["numpy3d","nested_univ"]


    if df_acts is not None:
        y = LabelMatcher(other=False).fit_transform(df_acts, df_devs).values
        y = y[:,1].astype(np.dtype('U'))
    else: 
        y = None

    if return_type == "nested_univ":
        """
        nested dataframe
        , TIME, DEVICE, VALUE 
        0, [S_T, S_D, S_V]
        """
        columns = [TIME, DEVICE, VALUE]
        index = 'RangeIndex'
        data = {TIME:[df_devs[TIME]], DEVICE:[df_devs[DEVICE]], VALUE: [df_devs[VALUE]]}
        X = pd.DataFrame(columns=columns, index=[0], data=data)
        if y is not None:
            if return_X_y:
                return X, y 
            else:
                X.loc[0, 'class_val'] = y
                return X
        else:
            return X

    elif return_type == "numpy3d":
        """ 
        Create array [S, F, T] with S being # sequences/recordings, F being
        the number of features (possible F + 1 for y) and T being the sequence length
        """
        X = np.swapaxes(np.expand_dims(df_devs.values, 0), 1, 2)
        if y is not None:
            if return_X_y:
                return X, y
            else:
                return np.append(X, y.reshape(1, 1, -1), axis=1)
        else:
            return X

    else:
        raise

def to_sktime2(df_X: pd.DataFrame, df_y: pd.DataFrame, return_type: str=None, 
              return_X_y: bool=False):
    """

    Parameters
    ----------
    df_X: pd.DataFrame
        A dataframe where the columns are the features. There must be at least 
        one column named 'time' in order to correclty match the labels. For 
        example a device dataframe with features [TIME, DEVICE, VALUE]

    df_acts: pd.DataFrame, optional
        TODO

    return_type: str or None, default=None
        Memory data format specification to return X in, None = “nested_univ” type. str can be any supported sktime Panel mtype,
        for list of mtypes, see datatypes.MTYPE_REGISTER for specifications, see examples/AA_datatypes_and_datasets.ipynb
        commonly used specifications:
            - “nested_univ: nested pd.DataFrame, pd.Series in cells 
            - “numpy3D”/”numpy3d”/”np3D”: 3D np.ndarray (instance, variable, time index) 

    return_X_y: bool, default=False
        it returns two objects, if False, it appends the class labels to the dataframe.


    
    Returns 
    -------
    X: pd.DataFrame
        The time series data for the problem with n_cases rows and either n_dimensions or n_dimensions+1 columns. Columns 1 to n_dimensions are the series associated with each case. If return_X_y is False, column n_dimensions+1 contains the class labels/target variable.
    y: numpy array, optional
        The class labels for each case in X, returned separately if return_X_y is True, or appended to X if False

    """

    from pyadlml.dataset._core.devices import is_device_df
    return_type = 'nested_univ' if return_type is None else return_type
    return_type = 'numpy3d' if return_type in ['numpy3D', 'np3D', 'np3d'] else return_type

    if return_type in ["pd-multiindex", "numpy2d"]:
        err_txt = "BasicMotions loader: Error, attempting to load into a numpy2d array, but cannot because it is a multivariate problem. Use numpy3d instead"
        raise ValueError(err_txt)

    assert return_type in ["numpy3d","nested_univ"]


    """
    Create appropriate targets depending on given input
    """
    if is_activity_df(df_y):
        from pyadlml.preprocessing.preprocessing import LabelMatcher
        y = LabelMatcher(other=False).fit_transform(df_y, df_X).values
        y = y[:,1].astype(np.dtype('U'))
    else:
        # Ensure there are no other things such as timestamps choose the last dimension as 
        y = df_y[:,:,-1] if len(df_y.shape) == 3 else df_y

        if len(y.shape) == 2 and y.shape[1] > 1:
            # When the windows were produces with many-to-many relationship reduce to many-to-one 
            # since 
            y = y[:,-1]
            print('Warning!!! Reducing many-to-many into many-to-one. Check if this was the intended measure.')
        y = y.squeeze() 
        assert len(y.shape) == 1


    X = df_X.copy()

    assert y.shape[0] == X.shape[0], f"Shape mismatch between X and y: {y.shape[0]} vs. {X.shape[0]}"

    if return_type == "nested_univ" and len(X.shape) == 2:
        """ Convert a 2d dataframe or numpy array 
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'dim_{i}'for i in range(X.shape[1])])

        if not is_device_df(X):
            # TODO do sth. here??
            pass

        """
        nested dataframe
        , TIME, DEVICE, VALUE 
        0, [S_T, S_D, S_V]
        """
        F = len(X.columns)

        X['target'] = y
        X['tmp'] = (X['target'].shift(1) != X['target']).astype(int)
        y = X[X['tmp'] == 1].loc[:, 'target'].copy().reset_index(drop=True)
        X['tmp'] = X['tmp'].cumsum()

        """
        Produce [N, F, s]
        where N is the number of each subsequence
        """
        data_X = []
        for yi, (_, Xi) in zip(y, X.groupby(by='tmp')):
            datum = {col: Xi[col] for col in Xi.columns[:-2]}
            if not return_X_y:
                datum['class_val'] = yi
            data_X.append(datum)

        X = pd.DataFrame(data_X)
        if return_X_y:
            return X, y 
        else:
            return X
    elif return_type == "nested_univ" and len(X.shape) == 3:
        """ Convert a 3d ndarray to nested_univ. This is the case when a windowing
            approach is used before Sktime.

            X has the shape (S, T, F) where S is the number of sequences, T the sequence length 
            and F the number of features
        """
        data_X = []
        for i in range(len(y)):
            # Xi of shape (T, F) where T is the sequence length and F are the number of features
            Xi, yi = X[i], y[i]
            datum = {f'dim_{f}': pd.Series(Xi[:, f]) for f in range(Xi.shape[1])}
            if not return_X_y:
                datum['class_val'] = yi
            data_X.append(datum)

        X = pd.DataFrame(data_X)
        if return_X_y:
            return X, y 
        else:
            return X
    elif return_type == "numpy3d":
        """ 
        Create array [S, F, T] with S being # sequences/recordings, F being
        the number of features (possible F + 1 for y) and T being the sequence length
        """
        raise NotImplementedError
        X = np.swapaxes(np.expand_dims(df_devs.values, 0), 1, 2)
        if y is not None:
            if return_X_y:
                return X, y
            else:
                return np.append(X, y.reshape(1, 1, -1), axis=1)
        else:
            return X

    else:
        raise
