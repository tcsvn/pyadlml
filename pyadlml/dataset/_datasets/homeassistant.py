import pandas as pd
from pyadlml.constants import TIME, DEVICE, VALUE
import datetime

def load_homeassistant(db_url:str, start_time=None, end_time=None) -> pd.DataFrame:
    """
    Loads the Home Assistant database into a pandas dataframe.

    Parameters
    ----------
    db_url : str
        A valid Home Assistant database url. Is used to establish the connection..
    start_time : str, optional
        Datetime string to exclude all rows before the specified start time.
    end_time : str, optional
        Datetime string to exclude all rows after the specified end time.

    Examples
    --------
    >>> from pyadlml.dataset import load_homeassistant
    >>> db_url = "sqlite:///config/homeassistant-v2.db"
    >>> load_homeassistant(db_url)
                    entity_id         state                 last_changed
    0        binary_sensor.p1            on   2019-07-23 07:21:59.954940
    1        binary_sensor.p1           off   2019-07-23 07:21:47.404236
    ...                   ...            ...                         ...
    1199320           sun.sun  above_horizon  2019-05-05 10:35:41.198544
    [1199321 rows x 3 columns]

    Returns
    -------
    df : pd.DataFrame
    """
    limit=5000000

    """ weird new requirement 
    """
    if 'sqlite' in db_url:
        lc_null_fix = ",iif(last_changed_ts is NULL,last_updated_ts,last_changed_ts) as last_changed_ts"
    elif 'mariadb' in db_url:
        lc_null_fix = ",if(last_changed_ts is NULL,last_updated_ts,last_changed_ts) as last_changed_ts"
    elif 'postgresql' in db_url:
        lc_null_fix = ',(case when last_changed_ts is NULL then last_updated_ts else last_changed_ts end)'
    else:
        print("WARNING: LAST_CHANGED may be null not substituted")
        lc_null_fix = ''

    # Convert timestamps to seconds
    start_time_nx = pd.Timestamp(start_time).timestamp()
    end_time_nx = pd.Timestamp(end_time).timestamp()

    if start_time is None and end_time is None:
        cond_time = ''
    if start_time is None and end_time is not None:
        raise NotImplementedError
    if start_time is not None and end_time is None:
        raise NotImplementedError
    if start_time is not None and end_time is not None:
        cond_time = f"AND last_updated_ts BETWEEN '{start_time_nx}' AND '{end_time_nx}'"
    
    query = f"""
    SELECT entity_id, state, last_updated_ts
    {lc_null_fix}
    FROM states
    WHERE
        state NOT IN ('unknown', 'unavailable')
        {cond_time}
    LIMIT {limit}
    """
    df = pd.read_sql_query(query, db_url)
    df['last_changed_ts'] = pd.to_datetime(df['last_changed_ts'], unit='s', origin='unix')
    df = df.drop(columns='last_updated_ts')
    return df

def load_homeassistant_devices(db_url, device_list, start_time=None, end_time=None):
    """
    Creates a device dataframe for selected devices within a certain timeframe.

    Parameters
    ----------
    db_url : str
        A valid Home Assistant database url. Is used to establish the connection.
    device_list : lst
        device selection to filter
    start_time : str, optional
        Datetime string to exclude all rows before the specified start time.
    end_time : str, optional
        Datetime string to exclude all rows after the specified end time.

    Examples
    --------
    >>> from pyadlml.dataset import load_homeassistant_devices
    >>> db_url = "sqlite:///config/homeassistant-v2.db"
    >>> lst_dev = ['binary_sensor.b1','switch.computer','light.l1','light.l2']
    >>> df_devices = load_homeassistant(db_url, device_list=lst_dev)

    Returns
    -------
    df_devs : pd.DataFrame
        All recorded devices from the Home Assistant dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    """
    
    df = load_homeassistant(db_url, start_time=start_time, end_time=end_time)


    # Bring into pyadlml structure
    df = df[df['entity_id'].isin(device_list)]
    df = df.rename(columns={'entity_id': DEVICE, 'state': VALUE, 'last_changed_ts': TIME})
    df = df[[TIME, DEVICE, VALUE]]

    # Convert to proper datatypes
    df[TIME] = pd.to_datetime(df[TIME])

    # Map binary device values to -> True, False
    #      is on or off or if binary-device is not on or off
    df[VALUE] = df[VALUE].replace(to_replace='on', value=True)
    df[VALUE] = df[VALUE].replace(to_replace='off', value=False)

    return df