import pandas as pd
import numpy as np
"""
    includes generic methods for manpulating dataframes
"""

def print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

def concatenate(df1, df2, df1_rev_hm, df2_rev_hm, df1_hm):
    """ concatenates df1 with df2 
        with columns matching according to hashmaps
    Parameters
    ----------
    df1 : pd.Dataframe
    df2 : pd.Dataframe
    df1_rev_hm : dict
    df2_rev_hm : dict

    Returns
    -------
    df : pd.Dataframe
    """
    # set column
    df1_lbld = _label_df_cols(df1, df1_rev_hm)
    df2_lbld = _label_df_cols(df2, df2_rev_hm)
    df_res = pd.concat([df1, df2])

    # todo create crosstab
    df_resunlbld = _label_df_cols(df_res, df1_hm)
    return df_res

def timestrs_2_timedeltas(t_strs):
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



def _label_df_cols(df, hm):
    cols = df.columns.values
    for col in cols:
        lbl = hm[col]
        df.rename(columns={col:lbl}, inplace=True)
    return df

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


def resample_data(df, freq):
    """ splits data into time bins and a new frequency of
     occurences either upsample or downsample data

    Parameters
    ----------
    df
    freq (string)
        the frequency or the timeslice delta
        e.g
            '0.5min'
            '30min'

    Returns
    -------

    """
    assert isinstance(freq[0],int) and freq[0] > 0
    assert freq[0] in ['ms', 's', 'm']

    resampler = df.resample(freq)
    rs_df = resampler.apply(self._aggr)
    rs_df = self._ffill_bfill_inv_first_dev(rs_df)
    return rs_df



def _aggr(series: pd.Series) -> pd.Series:
    """ for every value in a the dataframe to be resampled decide which
    value stands in the new field of the resampled dataframe. If a new
    value would be generated write np.nan into it. If the value matches the time
    bin write the value into it. If there is a multitude of device changes falling
    into one time bin take the last one to represent the time bin.
    Parameters
    ----------
    series
        a number of elements that have to be set into one time bin
    Returns
    -------

    """
    if series.empty:
        return np.nan
    elif series.size == 1:
        return series
    else:
        #return self._aggr_multi_items_last(series)
        return self._aggr_multi_items_sample(series)

def _aggr_multi_items_sample(series: pd.Series):
    """
    as take last has been shown to not yield good performance sample one
    entry of the series and return its value instead
    Returns
    -------
        True or False
    """
    rand_idx = np.random.randint(0, series.size)
    val = series.values[rand_idx]
    return val

def _aggr_multi_items_last(series: pd.Series):
    """

    Parameters
    ----------
    series

    Returns
    -------
        True or False

    """
    tmp = series.values[-1:][0]
    return tmp

