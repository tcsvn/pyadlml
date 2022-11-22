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
        lc_null_fix = ",iif(last_changed is NULL,last_updated,last_changed) as last_changed"
    elif 'mariadb' in db_url:
        lc_null_fix = ",if(last_changed is NULL,last_updated,last_changed) as last_changed"
    elif 'postgresql' in db_url:
        lc_null_fix = ',(case when last_changed is NULL then last_updated else last_changed end)'
    else:
        print("WARNING: LAST_CHANGED may be null not substituted")
        lc_null_fix = ''


    if start_time == None and end_time == None:
        query = f"""
        SELECT entity_id, state, last_updated
        {lc_null_fix}
        FROM states
        WHERE
            state NOT IN ('unknown', 'unavailable')
        ORDER BY last_updated DESC
        LIMIT {limit}
        """
    else:
        if end_time == None:
            end_time = datetime.datetime.utcnow()
        query = f"""
        SELECT entity_id, state, last_updated
        {lc_null_fix}
        FROM states
        WHERE
            state NOT IN ('unknown', 'unavailable') AND
            last_updated BETWEEN 
            '{start_time}' AND '{end_time}'
        ORDER BY last_updated ASC
        LIMIT {limit}
        """
    df = pd.read_sql_query(query, db_url)
    df = df.drop(columns='last_updated')
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
    df = df.rename(columns={'entity_id': DEVICE, 'state': VALUE, 'last_changed': TIME})
    df = df[[TIME, DEVICE, VALUE]]

    # Convert to proper datatypes
    df[TIME] = pd.to_datetime(df[TIME])

    # Map binary device values to -> True, False
    # TODO critical, what happens if a non-binary-device 
    #      is on or off or if binary-device is not on or off
    df[VALUE] = df[VALUE].replace(to_replace='on', value=1)
    df[VALUE] = df[VALUE].replace(to_replace='off', value=0)

    return df