import pandas as pd
from pyadlml.dataset import TIME, DEVICE, VAL
import datetime

def load_homeassistant(db_url, limit=5000000, start_time=None, end_time=None):
    """ returns a dataframe representation of the homeassistant database
    Parameters
    ----------
    db_url : String
        filepath to homeassistant database
    limit : int
        the limit TODO don't know ^^
    start_time : str
        Datetime string to select the rows later that start_time
    Returns
    -------
    df : pd.DataFrame
    """
    if start_time == None and end_time == None:
        query = f"""
        SELECT entity_id, state, last_changed
        FROM states
        WHERE
            state NOT IN ('unknown', 'unavailable')
        ORDER BY last_changed DESC
        LIMIT {limit}
        """
    else:
        if end_time == None:
            end_time = datetime.datetime.now()
        query = f"""
        SELECT entity_id, state, last_changed
        FROM states
        WHERE
            state NOT IN ('unknown', 'unavailable') AND
            last_changed BETWEEN 
            '{start_time}' AND '{end_time}'
        ORDER BY last_changed ASC
        LIMIT {limit}
        """
    df = pd.read_sql_query(query, db_url)
    return df

def load_homeassistant_devices(db_url, device_list, start_time=None, end_time=None):
    """ creates as dataframe in representation 1 from homeassistant database
    
    Parameters
    ----------
    db_url : String
        filepath to homeassistant database        
    device_list : lst
        device selection to filter
    start_time : pd.Timestamp
        the start time from when to filter
    end_time : pd.Timestamp
        the end time from when to filter
    Returns
    -------
    df : pd.DataFraem
    """
    
    df = load_homeassistant(db_url, start_time=start_time, end_time=end_time)
    df = df[df['entity_id'].isin(device_list)]
    df[TIME] = pd.to_datetime(df['last_changed'])
    df[VAL] = (df['state'] == 'on').astype(int)
    df[DEVICE] = df['entity_id']
    df = df[[TIME, DEVICE, VAL ]]
    return df