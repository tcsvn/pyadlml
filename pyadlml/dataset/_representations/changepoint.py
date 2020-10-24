import pandas as pd
from pyadlml.dataset import TIME, DEVICE, VAL
from pyadlml.dataset._dataset import label_data


def create_changepoint(df_devices, t_res=None, idle=False):
    dev = df_devices.copy()
    cp = _apply_changepoint(df_devices.copy())
    
    if t_res is not None:
        resampler = cp.resample(t_res, kind='timestamp')
        cp = resampler.apply(_cp_evaluator, dev=dev)
        
    return cp

def _cp_evaluator(series: pd.Series, dev):
    """ for each element return 1 if there was a change in the given interval or 0 otherwise
    Parameters
    ----------
        series: pd.Series
            conatins name of the column the evaluator operates on
            contains the timestamps for that change in set frame and the value 
            that the specified column has at set timestamp.

            is empty if all columns that have no changing entity

        df: pd.DataFrame
            contains device representation 3. Is for determining the states of other
            device/columns for the timestamps

    Returns: 0|1
    """
    if series.empty:
        return 0
    else:
        if series.sum() == 0:
            return 0
        else:
            return 1

def _apply_changepoint(df):
    """
    Parameters
    ----------
        df: pd.DataFrame
        device representation 3
    """
    # create binary vectors and set all changepoints to true
    df = df.pivot(index=TIME, columns=DEVICE, values=VAL)
    df[(df == False)] = True
    
    # fix rest
    df = df.fillna(False)
    df = df.astype(int)
    return df
