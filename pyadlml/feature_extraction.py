import pandas as pd
from pyadlml.dataset import TIME


def _tres_2_pd_res(res):
    if _is_hour(res) or _is_sec(res):
        return res
    if _is_min(res):
        return res[:-1] + 'min'


def _tres_2_discrete(resolution):
    """ create resolution for one day
    """
    res = []
    range = pd.date_range(start='1/1/2000', end='1/2/2000', freq=resolution).time
    range = [str(d) for d in range]
    for t1, t2 in zip(range[:-1], range[1:]):
        res.append(t1 + '-' + t2)
    return res


def _tres_2_vals(resolution):
    """ create resolution for one day
    """
    ser = pd.date_range(start='1/1/2000', end='1/2/2000', freq=resolution).to_series()
    if _is_min(resolution):
        return ser.dt.floor(resolution).dt.minute
    if _is_hour(resolution): return ser.dt.floor(resolution).dt.hour
    if _is_sec(resolution):
        return ser.dt.floor(resolution).dt.sec


def _is_hour(res) -> bool:
    return res[-1:] == 'h'


def _is_min(res) -> bool:
    return res[-1:] == 'm'


def _is_sec(res) -> bool:
    return res[-1:] == 's'


# TODO flag for deletion
#def _time_extend_int(ts, t_res='30m'):
#    """
#    gets
#    """
#    assert _valid_tres(t_res)
#    val = int(t_res[:-1])
#    res = t_res[-1:]
#    import datetime as dt
#    diff = pd.Timedelta(t_res)
#    if res == 'h':
#        tmp1 = str(ts)
#        tmp3 = pd.Timestamp(ts) + diff
#        tmp2 = str(ts + diff)
#        return str(dt.time) + ' - ' + str(dt.time + diff)
#
#    elif res == 'm':
#        ms = val
#        m_bin = int(ts.minute/ms)*ms
#        return dt.time(hour=ts.hour, minute=m_bin)

def _valid_tres(t_res):
    val = int(t_res[:-1])
    res = t_res[-1:]
    assert t_res[-1:] in ['h','m', 's']
    assert (val > 0 and val <=12 and  res == 'h') \
        or (val > 0 and val <= 60 and res == 'm') \
        or (val > 0 and val <= 60 and res == 's')
    return True


def _time2intervall(ts, t_res='30m'):
    """
    rounds to the next lower min bin or hour bin """
    ts_ceil = ts.ceil(freq=t_res).time()
    ts_floor = ts.floor(freq=t_res).time()
    return str(ts_floor) + '-' + str(ts_ceil)


def extract_time_bins(df, resolution='2h', inplace=True):
    """
    Create one-hot encoding for times of the day

    Parameters
    ----------
    df: pd.DataFrame
        A device dataframe. The dataframe has to include a column with
        the name 'time' containing the timestamps of the representation.
    resolution: str
        The frequency that the day is divided. Different resolutions are
        accepted, like minutes '[x]m' seconds '[x]s] or hours '[x]h'
    inplace : bool, default=True
        Determines whether to append the one-hot encoded time_bins as columns
        to the existing dataframe or return only the one-hot encoding.

    Returns
    -------
    df : pd.DataFrame
        One-hot encoding of the devices.

    """
    assert resolution[-1:] in ['m', 's', 'h']
    RES = 'res'

    df = df.copy()
    df[RES] = df[TIME].apply(_time2intervall, args=[resolution])
    one_hot = pd.get_dummies(df[RES]).astype(bool)
    df = df.join(one_hot, on=df.index)
    del(df[RES])

    # add columns that don't exist in the dataset
    cols = _tres_2_discrete(resolution)
    for v in cols:
        if v not in df.columns:
            df[v] = False

    if inplace:
        return df
    else:
        return df[[cols]]


def extract_day_of_week(df, one_hot_encoding=True, inplace=True):
    """
    Appends seven columns one-hot encoded for week days to a dataframe based
    on the dataframes timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        A device dataframe with a column named 'time' containing timestamps.
    one_hot_encoding : bool, optional, default=False
        Determines
    inplace : bool, default=True
        Determines whether to append the one-hot encoded time_bins as columns
        to the existing dataframe or return only the one-hot encoding.

    Returns
    ----------
    df : pd.DataFrame
    """
    TMP_COL = 'weekday'

    df = df.copy()
    df[TMP_COL] = df[TIME].dt.day_name()
    if not one_hot_encoding:
        return df
    one_hot = pd.get_dummies(df[TMP_COL])
    df = df.join(one_hot, on=df.index)

    # add weekdays that didn't occur in the column
    # TODO get the names from pandas instead of hardcoded
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for wd in weekdays:
        if wd not in df.columns:
            df[wd] = False

    if inplace:
        return df
    else:
        raise NotImplementedError

def extract_time_difference(df, normalize=False, inplace=True):
    """
    Adds one column with time difference between two succeeding rows.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing a column named 'time' including valid pandas timestamps.
    normalize : bool, optional, default=False
        Whether to normalize the time differences to the interval [0,1]
    inplace : bool, optional , default=True
        Determines whether to add the column to and return the existing dataframe or
        return a dataframe containing only the time differences.
    Returns
    -------
    df : pd.DataFrame
    """
    pass