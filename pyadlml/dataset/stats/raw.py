import pandas as pd
import numpy as np

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
    if isinstance(y, pd.DataFrame):
        y = y['activity']
    res = pd.crosstab(index=X[X.columns[0]], columns=y)
    res.index = [X.columns[0] + ' Off', X.columns[0] + ' On']
    
    for dev_name in X.columns[1:]:
        tmp = pd.crosstab(index=X[dev_name], columns=y)
        tmp.index = [dev_name + ' Off', dev_name + ' On']
        res = pd.concat([res, tmp])
    return res



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