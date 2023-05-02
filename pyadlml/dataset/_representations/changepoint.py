import pandas as pd
from pyadlml.constants import TIME, DEVICE, VALUE

def create_changepoint(df_devs):
    """

    Parameters
    ----------
    df_devs : pd.DataFrame
        A device dataframe, refer to guide

    Returns
    -------
    df : pd.DataFrame
    """

    # create binary vectors and set all changepoints to true
    df = df_devs.copy()
    df[VALUE] = True
    df = df.pivot(index=TIME, columns=DEVICE, values=VALUE)\
        .fillna(False)\
        .astype(int)\
        .reset_index()
    return df


def resample_changepoint(cp: pd.DataFrame, dt:str, use_dask=False) -> pd.DataFrame:
    """
    Resamples the changepoint representation with a given resolution

    Parameters
    ----------
    cp : pd.DataFrame
        A device dataframe in changepoint representation [TIME, dev1, dev2, ..., devn]

    dt : str
        

    Returns
    -------
    pd.DataFrame
        Resampled dataframe in changepoint representation
    """
    df = cp.copy() 
    df = df.sort_values(by=TIME).set_index(TIME)
    df = df.resample('1s', kind='timestamp').count()
    df[df > 1] = 1
    return df.reset_index()
