import pandas as pd

def hass_db_2_df(db_url, limit=5000000):
    """
    Parameters
    ----------
    db_url : String
        url to the database 
    """
    query = f"""
    SELECT entity_id, state, last_changed
    FROM states
    WHERE
        state NOT IN ('unknown', 'unavailable')
    ORDER BY last_changed DESC
    LIMIT {limit}
    """
    df = pd.read_sql_query(query, db_url)
    return df

def hass_db_2_data(db_url, device_list, start_time=None):
  """ gets a dataframe with devices in devices list from a start_time up 
  Parameters
  ----------
  Returns
  -------
  df : pd.DataFrame
  df2 : pd.DataFrame
      
  """
  device_dict = {device_list[i]:i for i in range(len(device_list))}
  
  df = hass_db_2_df(db_url)
  df = df[df['entity_id'].isin(device_list)]
  df['time'] = pd.to_datetime(df['last_changed'])
  df['val'] = (df['state'] == 'on').astype(int)
  df['device'] = df['entity_id']
  df = df[[ 'time', 'device','val']]
  
  # encode data
  df['device'] = df["device"].replace(device_dict)
  
  # create feature map
  df2 = pd.DataFrame(device_list, columns=['devices'])
  df2 = df2.reset_index()
  df2.columns = ['id', 'devices']
  return df, df2
