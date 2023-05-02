import pandas as pd
from pyadlml.constants import TIME, DEVICE, VALUE
from pyadlml.dataset._representations.changepoint import create_changepoint


def create_lastfired(df_devs):
    """
    creates the last fired representation
    """
    return create_changepoint(df_devs)


def resample_last_fired(lf, t_res):
    """

    Parameters
    ----------
    lf : pd.DataFrame
        last fired representation

    """
    lf = lf.set_index(TIME)
    resampler = lf.resample(t_res, kind='timestamp')
    lf = resampler.apply(_lf_evaluator, df=lf.copy())
    lf = lf.fillna(method='ffill')\
        .reset_index()
    return lf


def _lf_evaluator(series: pd.Series, df:pd.DataFrame) -> int:
    """
    Is called column-wise for each element of the
    
    Parameters
    ----------
    series : pd.Series
        conatins name of the column the evaluator operates on
        contains the timestamps for that change in that frame and the value
        that the specified column has at that timestamp.

        is empty if all columns that have no changing entity

    df : pd.DataFrame
        contains device representation 3. Is for determining the states of other
        device/columns for the timestamps

    Returns: 0|1
    """
    #print('series: ', series, '\n')
    if series.empty:
        return None
    else:         
        # check if this device is triggered during the timeslice
        # this is okay because other devices where triggered -> it has to be zero
        if series.sum() == 0:
            return 0
        
        # get dataframe for timeslice
        ts = series.index[0]
        if len(series) == 1:
            te = ts
        else:
            te = series.index[-1]
        df_slice = df.loc[ts:te]
        
        # if the device changed the state check if it was the 
        # last device to do so in the interval. When true 
        # the timeslice should be 1 for device last fired and 0 otherwise
        for i in range(len(df_slice)-1, -1, -1):
            
            # check if any device triggered
            if df_slice.iloc[i].values.sum() == 0:
                continue
                
            # check if our device was the one who triggered
            elif df_slice.loc[df_slice.index[i], series.name] == 1:
                return 1
            else:
                return 0