import pandas as pd
import numpy as np
"""
    includes generic methods for manpulating dataframes
"""

def print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


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


