import pandas as pd



def get_rel_act_duration(df_activities):
    """
    counts the timedeltas of the activites and calculates the
    percentage of activities in relation to each other
    Returns
    -------
    res pd.Dataframe
                   leave house  use toilet  ...  prepare Dinner  get drink
        perc               0.3         ...                 0.2        0.01

    """
    label = 'tads'
    df = _get_act_dur_helper(df_activities, label, freq='sec')

    norm = df[label].sum()
    df[label] = df[label]/norm
    df = df.transpose()
    df.columns = self._df_idcol2lblcol(df.columns)
    return df

def _get_act_dur_helper(df_act, label, freq='sec'):
    df = df_act.copy()
    df['DIFF'] = df[END_TIME] - df[START_TIME]
    df = df.drop(columns=[END_TIME, START_TIME])
    df = self._act_data.groupby('Idea').sum()     # type: pd.DataFrame
    df.columns = [label]
    if freq == 'sec':
        df[label] = df[label].apply(lambda x: x.total_seconds())
    elif freq == 'min':
        df[label] = df[label].apply(lambda x: x.total_minutes())
    else:
        df[label] = df[label].apply(lambda x: x.total_hours())
    return df



def get_activities_count(df_activities):
    """
    Returns
    -------
    res pd.Dataframe
                   leave house  use toilet  ...  prepare Dinner  get drink
        occurence           33         111  ...              10         19

    """
    df = df_activities.groupby('Idea').count()
    df = df.drop(columns=[END_TIME])
    df.columns = ['occurence']
    df = df.transpose()
    print('asdf')
    lst = df.columns
    new_lst = []
    for item in lst:
        new_lst.append(
            self._activity_label_reverse_hashmap[item]
        )
    df.columns = new_lst
    return df

def get_total_act_durations(freq='sec'):
    """
    Returns
    -------
        time_dist (pd.DataFrame)
        the total amount of time distirbutions
    """
    dfpc = self._acts.get_total_act_duration()
    amount_freq = self._freqstr2factor(freq)
    dfpctmp = dfpc.apply(lambda x: x/amount_freq)
    return dfpctmp

def _freqstr2factor(freq):
    """
    Parameters
    ----------
    freq : str
        the frequency at which
        e.g
            '3sec' -> 3
    Returns
    -------
    int
        the
    """
    assert 'min' in freq or 'sec' in freq or 'hour' in freq
    if len(freq) == 3:
        return 1
    if 'sec' in freq or 'min' in freq:
        return int(freq[:-3])
    if 'hour' in freq:
        return int(freq[:-4])



def get_devices_count(df_devices):
    """
    Returns
    -------
    res pd.Dataframe
                   binary_sensor.  use toilet  ...  prepare Dinner  get drink
        occurence               33         111  ...              10         19

    """
    df_cp = self._apply_change_point(self._dev_data.copy())
    cnt = df_cp.apply(pd.value_counts)
    cnt.drop(False, inplace=True)
    new_columns = []
    for item in cnt.columns:
        new_columns.append(
            self._sensor_label_reverse_hashmap[item]
        )
    cnt.columns = new_columns
    return cnt
