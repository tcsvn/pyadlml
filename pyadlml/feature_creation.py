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
    if _is_hour(resolution):
        return ser.dt.floor(resolution).dt.hour
    if _is_sec(resolution):
        return ser.dt.floor(resolution).dt.sec


def _is_hour(res):
    return res[-1:] == 'h'
def _is_min(res):
    return res[-1:] == 'm'
def _is_sec(res):
    return res[-1:] == 's'

def _time_extend_int(ts, t_res='30m'):
    """
    gets
    """
    assert _valid_tres(t_res)
    val = int(t_res[:-1])
    res = t_res[-1:]
    import datetime as dt
    diff = pd.Timedelta(t_res)
    if res == 'h':
        tmp1 = str(ts)
        tmp3 = pd.Timestamp(ts) + diff
        tmp2 = str(ts + diff)
        return str(dt.time) + ' - ' + str(dt.time + diff)

    elif res == 'm':
        ms = val
        m_bin = int(ts.minute/ms)*ms
        return dt.time(hour=ts.hour, minute=m_bin)

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
    rounds to the next lower min bin or hour bin
    """
    ts_ceil = ts.ceil(freq=t_res).time()
    ts_floor = ts.floor(freq=t_res).time()
    return str(ts_floor) + '-' + str(ts_ceil)


def extract_time_bins(df, resolution='2h'):
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

    Returns
    -------
    df : pd.DataFrame
        One-hot encoding of the devices.

    """
    assert resolution[-1:] in ['m', 's', 'h']
    RES = 'res'
    TMP = 'tmp'
    df = df.copy()
    df[RES] = df[TIME].apply(_time2intervall, args=[resolution])
    df[TMP] = True
    df = df[[TIME, RES, TMP]]

    df = pd.pivot(df, values=TMP, index=TIME, columns=RES).reset_index()
    cols = _tres_2_discrete(resolution)

    # addee columns that don't exist in the dataset
    for v in cols:
        if v not in df.columns:
            df[v] = False
    return df


def extract_day_of_week(df):
    """
    Create one-hot encoding for week days

    Parameters
    ----------
    df: pd.DataFrame
        A device dataframe with a column named 'time' containing timestamps.

    Returns
    ----------
    df : pd.DataFrame
    """
    df = df.copy()
    TMP_COL = 'asdfasdfasdf'
    df[TMP_COL]  = df['time'].dt.day_name()
    df = df[['time', TMP_COL]]


    return res.reset_index(drop=True)

def extract_time_difference(df, normalize=False):
    """
    Creates a new feature that

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing a timestamp column named 'time'. The 'time' column
        is used to apply the changes
    normalize : bool, optional
        Whether to normalize the time differences to the interval [0,1]

    Returns
    -------
    df : pd.DataFrame
    """
    pass