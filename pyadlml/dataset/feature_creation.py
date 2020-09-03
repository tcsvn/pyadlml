import pandas as pd

def add_devices_timeint(df_dev, freq='2h'):
    """ creates devices for the specified frequency with value on during
        that frequency
        
    Parameters
    ----------
    df_dev: pd.DataFrame
        devices dataframe, columns start_time, end_time, val, device name
        
    Returns
    ----------
    pd.DataFrame
    """
    df_dev = df_dev.copy()
    # specify 
    st = df_dev['start_time'].iloc[0].ceil('h')
    et = df_dev['end_time'].iloc[-1].floor('h')
    ts=pd.date_range(st, et, freq=freq)
    
    # create Dataframe
    res = pd.DataFrame({'start_time' : ts})
    res['end_time'] = res['start_time'].shift(-1)
    res = res.iloc[:-1,:]
    res['val'] = True
    
    # correct so that no overlap in time is there
    res['start_time'] = res['start_time'] + pd.Timedelta('1ms')
    res['end_time'] = res['end_time'] - pd.Timedelta('1ms')
    
    # create activity label
    res['device'] = res['start_time'].dt.floor('h').dt.strftime("%H:%M:%S")
    res['device2'] = res['end_time'].dt.ceil('h').dt.strftime("%H:%M:%S")
    res['device'] = 'TI: ' + res['device'] + ' - ' + res['device2']
    res = res.iloc[:, :-1]
    
    # combine with dataframe
    res = pd.concat([df_dev, res]).sort_values(by=['start_time'])
    return res.reset_index(drop=True)


def add_devices_dayofweek(df_dev):
    """ creates devices for the specified frequency with value on during
        that frequency
        
    Parameters
    ----------
    df_dev: pd.DataFrame
        devices dataframe, columns start_time, end_time, val, device name
        
    Returns
    ----------
    pd.DataFrame
    """
    df_dev = df_dev.copy()
    # specify 
    
    st = df_dev['start_time'].iloc[0].ceil('d')
    et = df_dev['end_time'].iloc[-1].floor('d')
    ts=pd.date_range(st, et, freq='D').to_series()
    
    #ts.dt.
    
    # create DataFrame
    res = pd.DataFrame({'start_time' : ts})
    res['end_time'] = res['start_time'].shift(-1)
    res = res.iloc[:-1,:]
    res['val'] = True
    
    # correct so that no overlap in time is there
    res['start_time'] = res['start_time'] + pd.Timedelta('1ms')
    res['end_time'] = res['end_time'] - pd.Timedelta('1ms')
    
    # create activity label
    #res['device'] = res['start_time'].dt.floor('h').dt.dayofweek
    res['device'] = res['start_time'].dt.floor('h').dt.day_name()
    #res['device'] = 'WD: ' + res['device'].astype(str)
    
    
    # combine with dataframe
    res = pd.concat([df_dev, res]).sort_values(by=['start_time'])
    return res.reset_index(drop=True)