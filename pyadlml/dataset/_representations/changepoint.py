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


def resample_changepoint(cp, dt):
    """
    Resamples the changepoint representation with a given resolution

    Parameters
    ----------
    cp : pd.DataFrame
        A device dataframe in changepoint representation

    dt : str

    Returns
    -------
    cp : pd.DataFrame
        Resampled dataframe in changepoint representation
    """
    cp = cp.set_index(TIME)
    resampler = cp.resample(dt, kind='timestamp')
    cp = resampler.apply(_cp_evaluator)\
        .reset_index()
    return cp


def _cp_evaluator(series: pd.Series):
    """ for each element return 1 if there was a change in the given interval or 0 otherwise
    Parameters
    ----------
        series: pd.Series
            contains name of the column the evaluator operates on
            contains the timestamps for that change in set frame and the value 
            that the specified column has at set timestamp.

            is empty if all columns that have no changing entity

        df: pd.DataFrame
            contains device representation 3. Is for determining the states of other
            device/columns for the timestamps

    Returns: 0|1
    """

    if series.empty or series.sum() == 0:
        return 0
    else:
        return 1