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

def _label_df_cols(df, hm):
    cols = df.columns.values
    for col in cols:
        lbl = hm[col]
        df.rename(columns={col:lbl}, inplace=True)
    return df

def first_row_df2dict(df):
    """
    creates a dictionary by taking the column names as keys and as the
    values the first items of the first row
    Parameters
    ----------
    df  pd.DataFrame
        with only one row
        e.g
              leave house  use toilet  ...  prepare Dinner  get drink
        perc     0.650096    0.005725  ...        0.010038    0.00049

    Returns
    -------
    res dict
    """
    res = {}
    for i, col_name in enumerate(df.columns):
        res[col_name] = df.iloc[0][col_name]
    return res


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


