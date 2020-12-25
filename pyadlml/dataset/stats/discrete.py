import pandas as pd
import numpy as np

ON_APPENDIX = " On"
OFF_APPENDIX = " Off"
def contingency_table_01(X, y):
    """ creates a contingency table for the matrix X and labels y
    Parameters
    ----------
    X : pd.Series or DataFrame
        matrix of 0-1 vectors as rows representing a state of 
        the smart home
    y : np array of string
        labels for X
    
    Returns
    ---------
    pd.DataFrame
    """

    X = X.copy()
    if isinstance(y, pd.DataFrame):
        y = y['activity']

    X = X.reset_index(drop=True)
    first_dev = X.columns[0]
    index = X[first_dev]
    res = pd.crosstab(index=index, columns=y)
    res.index = _set_onoffappendix(res.index, first_dev)

    for dev_name in X.columns[1:]:
        tmp = pd.crosstab(index=X[dev_name], columns=y)
        tmp.index = _set_onoffappendix(tmp.index, dev_name)
        res = pd.concat([res, tmp])
    return res

def _set_onoffappendix(index, dev_name):
    """ Using a dataframe index with [true, false] to create a new index with [dev_name + 'on', ...]
    """
    if not index.values[0] and index.values[1]:
        return [dev_name + OFF_APPENDIX, dev_name + ON_APPENDIX]
    elif index.values[0] and not index.values[1]:
        return [dev_name + ON_APPENDIX, dev_name + OFF_APPENDIX]

def cross_correlation(rep):
    """ compute the crosscorelation by comparing for every binary state
    of the smarthome between the devices
    
    Parameters
    ----------
        rep: pd.DataFrame
            device of either raw, lastfired or changepoint representation
            01 vectors of rows representing a state in the smart home
    returns
    -------
        pd.DataFrame (k x k)
        crosscorrelation between each device
    """
    
    dev_lst = rep.columns
    rep = rep.copy().sort_values(by='time')
    rep = rep.where((rep == 1), -1)
    rep = rep.astype('object')
    
    # for every field in a row multiply the each item with the state of the row
    # yields 1 for similar states w.r.t to other devices and -1 for dissimlar
    # values
    def func(series):
        state = series.copy().values
        for i, val in enumerate(state):
            series.iloc[i] = val*state        
        return series
    
    tmp = rep.apply(func, axis=1)
    
    # create pandas dataframe
    arr2d = np.stack(tmp.sum(axis=0).values)
    idx = tmp.columns
    crosstab = pd.DataFrame(data=arr2d, index = tmp.columns, columns=tmp.columns)
    
    # normalize
    crosstab = crosstab/crosstab.iloc[0,0]
    return crosstab