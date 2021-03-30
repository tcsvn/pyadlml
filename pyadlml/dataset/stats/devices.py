import numpy as np
import pandas as pd
from pyadlml.dataset import START_TIME, END_TIME, TIME, NAME, VAL, DEVICE
from pyadlml.dataset.util import time2int, timestr_2_timedeltas
from pyadlml.dataset.devices import device_rep1_2_rep2, _create_devices, \
    _is_dev_rep1, _is_dev_rep2, contains_non_binary, split_devices_binary

from pyadlml.dataset.util import timestr_2_timedeltas
from pyadlml.dataset._representations.raw import create_raw
from pyadlml.util import get_npartitions
from dask import delayed
import dask.dataframe as dd


def duration_correlation(df_devs, lst_devs=None):
    """
    Compute the similarity between devices by comparing the binary values
    for every interval.

    Parameters
    ----------
    df_devs : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    lst_devs: list of str, optional
        A list of devices that are included in the statistic. The list can be a
        subset of the recorded devices or contain devices that are not recorded.

    Examples
    --------
    >>> from pyadlml.stats import device_duration_corr
    >>> device_duration_corr(data.df_devs)
    device              Cups cupboard  Dishwasher  ...  Washingmachine
    device                                         ...
    Cups cupboard            1.000000    0.997571  ...        0.999083
    Dishwasher               0.997571    1.000000  ...        0.996842
    ...
    Washingmachine           0.999083    0.996842  ...        1.000000
    [14 rows x 14 columns]

    Returns
    -------
    df : pd.DataFrame
        A dataframe of every device against another device. The values range from -1 to 1
        where higher values represent more similarity.
    """
    TD = 'td'


    if contains_non_binary(df_devs):
        df_devs, _ = split_devices_binary(df_devs)

    def func(row):
        """ gets two rows and returns a crosstab
        """        
        try:
            td = row.td.to_timedelta64()
        except:
            return None
        states = row.iloc[1:len(row)-1].values.astype(int)
        K = len(states)        
        
        for j in range(K):
            res = np.full((K), 0, dtype='timedelta64[ns]')
            tdiffs = states[j]*states*td            
            row.iloc[1+j] = tdiffs 
        return row

    def create_meta(raw):
        devices = {name : 'object' for name in raw.columns[1:-1]}
        return {**{TIME: 'datetime64[ns]', TD: 'timedelta64[ns]'}, **devices}
        
    dev_lst = df_devs[DEVICE].unique()
    df_devs = df_devs.sort_values(by=TIME)

    K = len(dev_lst)

    # make off to -1 and on to 1 and then calculate cross correlation between signals
    raw = create_raw(df_devs).applymap(lambda x: 1 if x else -1).reset_index()
    raw[TD] = raw[TIME].shift(-1) - raw[TIME]
    
    df = dd.from_pandas(raw.copy(), npartitions=get_npartitions())\
                .apply(func, axis=1).drop(columns=[TIME, TD]).sum(axis=0)\
                .compute(scheduler='processes')
                #.apply(func, axis=1, meta=create_meta(raw)).drop(columns=['time', 'td']).sum(axis=0)\

    res = pd.DataFrame(data=np.vstack(df.values), columns=df.index, index=df.index)
    # normalize
    res = res/res.iloc[0, 0]

    if lst_devs is not None:
        for dev in set(lst_devs).difference(set(list(res.index))):
            res[dev] = pd.NA
            res = res.append(pd.DataFrame(data=pd.NA, columns=res.columns, index=[dev]))
    return res


def devices_trigger_count(df_devs, lst_devs=None):
    """
    Compute the amount a device is triggered throughout a dataset.

    Parameters
    ----------
    df_devs : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    lst_devs: lst of str, optional
        A list of devices that are included in the statistic. The list can be a
        subset of the recorded devices or contain devices that are not recorded.

    Examples
    --------
    >>> from pyadlml.stats import device_trigger_count
    >>> device_trigger_count(data.df_devs)
                    device  trigger_count
    0        Cups cupboard             98
    1           Dishwasher             42
    ..                 ...            ...
    13      Washingmachine             34

    Returns
    -------
    df : pd.DataFrame
        A dataframe with devices and their respective triggercounts as columns.
    """
    assert _is_dev_rep1(df_devs)

    col_label = 'trigger_count'

    ser = df_devs.groupby(DEVICE)[DEVICE].count()
    df_devs = pd.DataFrame({DEVICE: ser.index, col_label:ser.values})

    if lst_devs is not None:
        for dev in set(lst_devs).difference(set(list(df_devs[DEVICE]))):
            df_devs = df_devs.append(pd.DataFrame(data=[[dev, 0]],
                                                  columns=df_devs.columns,
                                                  index=[len(df_devs)]))
    return df_devs.sort_values(by=DEVICE)

def trigger_time_diff(df_devs):
    """
    Compute the time difference between sucessive device triggers.

    Parameters
    ----------
    df_devs : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.

    Examples
    --------
    >>> from pyadlml.stats import device_time_diff
    >>> device_time_diff(data.df_devs)
    array([1.63000e+02, 3.30440e+04, 1.00000e+00, ..., 4.00000e+00,
           1.72412e+05, 1.00000e+00])

    Returns
    -------
    X : ndarray
        Array of time deltas in seconds.
    """

    # create timediff to the previous trigger
    diff_seconds = 'ds'
    df_devs = df_devs.copy().sort_values(by=[TIME])

    # compute the seconds to the next device
    df_devs[diff_seconds] = df_devs[TIME].diff().shift(-1) / pd.Timedelta(seconds=1)
    return df_devs[diff_seconds].values[:-1]


def devices_td_on(df_devs):
    """
    Compute the amount of time a device was in the "on" state for each datapoint.

    Parameters
    ----------
    df_devs : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.

    Examples
    --------
    >>> from pyadlml.stats import device_on_time
    >>> device_on_time(data.df_devs)
                      device              td
    0      Hall-Bedroom door 0 days 00:02:43
    1      Hall-Bedroom door 0 days 00:00:01
    ...                  ...             ...
    1309           Frontdoor 0 days 00:00:01
    [1310 rows x 2 columns]

    Returns
    -------
    df : pd.DataFrame
        A dataframe with two columns, the devices and the time differences.
    """
    time_difference = 'td'
    df = df_devs.copy()
    if contains_non_binary(df):
        df, _ = split_devices_binary(df)

    if not _is_dev_rep2(df):
        df, _ = device_rep1_2_rep2(df.copy(), drop=False)
    df[time_difference] = df[END_TIME] - df[START_TIME]
    return df[[DEVICE, time_difference]]


def devices_on_off_stats(df_devs, lst_devs=None):
    """
    Calculate the time and proportion a device was in the "on"
    versus the "off" state.

    Parameters
    ----------
    df_devs : pd.DataFrame
        A datasets device dataframe. The columns are ['time', 'device', 'val'].
    lst_devs : list, optional
        An optional list of all device names. Use this if there exist devices
        that are not present in the recorded dataset but should be included in the statistic.

    Examples
    --------
    >>> from pyadlml.stats import device_on_off
    >>> device_on_off(data.df_devs)
                    device                  td_on                  td_off   frac_on  frac_off
    0        Cups cupboard 0 days 00:10:13.010000 27 days 18:34:19.990000  0.000255  0.999745
    1           Dishwasher        0 days 00:55:02        27 days 17:49:31  0.001376  0.998624
    ...                ...                    ...                     ...        ...      ...
    13      Washingmachine        0 days 00:08:08        27 days 18:36:25  0.000203  0.999797

    Returns
    -------
    df : pd.DataFrame
    """

    diff = 'diff'
    td_on = 'td_on'
    td_off = 'td_off'
    frac_on = 'frac_on'
    frac_off = 'frac_off'

    if contains_non_binary(df_devs):
        df_devs, _ = split_devices_binary(df_devs)

    if not _is_dev_rep2(df_devs):
        df_devs, _ = device_rep1_2_rep2(df_devs.copy(), drop=False)
    df_devs = df_devs.sort_values(START_TIME)

    # calculate total time interval for normalization
    int_start = df_devs.iloc[0, 0]
    int_end = df_devs.iloc[df_devs.shape[0] - 1, 1]
    norm = int_end - int_start

    # calculate time deltas for online time
    df_devs[diff] = df_devs[END_TIME] - df_devs[START_TIME]
    df_devs = df_devs.groupby(DEVICE)[diff].sum()
    df_devs = pd.DataFrame(df_devs)
    df_devs.columns = [td_on]

    df_devs[td_off] = norm - df_devs[td_on]

    # compute percentage
    df_devs[frac_on] = df_devs[td_on].dt.total_seconds() \
                       / norm.total_seconds()
    df_devs[frac_off] = df_devs[td_off].dt.total_seconds() \
                        / norm.total_seconds()
    if lst_devs is not None:
        for dev in set(lst_devs).difference(set(list(df_devs.index))):
            df_devs = df_devs.append(pd.DataFrame(data=[[pd.NaT, pd.NaT, pd.NA, pd.NA]], columns=df_devs.columns, index=[dev]))
    return df_devs.reset_index()\
        .rename(columns={'index':DEVICE})\
        .sort_values(by=[DEVICE])

def device_tcorr(df_devs, lst_devs=None, t_window='20s'):
    """
    Count the prevalence of devices that trigger within the same time frame.

    Parameters
    ----------
    df_devs : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    lst_devs: list of str, optional
        A list of devices that are included in the statistic. The list can be a
        subset of the recorded devices or contain devices that are not recorded.
    t_window : str of {'[1-12]h', '[1-60]s'}, optional
        Size of the time frame for a single window. The window size
        is given in either n seconds 'ns' or n hours 'nh'. Defaults to 20 seconds '20s'.

    Examples
    --------
    >>> from pyadlml.stats import device_trigger_sliding_window
    >>> device_trigger_sliding_window(data.df_devs)
                       Cups cupboard Dishwasher  ...  Washingmachine
    Cups cupboard                332         10  ...               0
    Dishwasher                    10         90  ...
    ...                          ...        ...  ...             ...
    Washingmachine                 0          0  ...              86

    Returns
    -------
    df : pd.DataFrame
    """

    t_window = timestr_2_timedeltas(t_window)[0]
    
    # create timediff to the previous trigger
    df_devs = df_devs.copy()
    df_devs['time_diff'] = df_devs[TIME].diff()

    #knn
    #    do cumsum for row_duration 
    #    for each row mask the rows that fall into the given area
    if lst_devs is not None:
        dev_list = lst_devs
    else:
        dev_list =  df_devs.device.unique()
    
    df_devs.iloc[0, 3] = pd.Timedelta(0, 's')
    df_devs['cum_sum'] = df_devs['time_diff'].cumsum()
    
    # create cross table with zeros
    res_df = pd.DataFrame(columns=dev_list, index=dev_list)
    for col in res_df.columns:
        res_df[col].values[:] = 0

    # this whole iterations can be done in parallel
    for row in df_devs.iterrows():
        td = row[1].cum_sum
        dev_name = row[1].device

        df_devs['tmp'] = (td - t_window < df_devs['cum_sum']) & (df_devs['cum_sum'] < td + t_window)
        tmp = df_devs.groupby(DEVICE)['tmp'].sum()
        res_df.loc[dev_name] += tmp

    return res_df.sort_index(axis=0, ascending=True) \
        .sort_index(axis=1, ascending=True) \
        .replace(pd.NA, 0)


def device_triggers_one_day(df_devs, lst_devs=None, t_res='1h'):
    """
    Divide a day into time bins and compute how many device triggers fall into
    each bin.

    Parameters
    ----------
    df_devs : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    lst_devs: lst of str, optional
        A list of devices that are included in the statistic. The list can be a
        subset of the recorded devices or contain devices that are not recorded.
    t_res : str of {'[1,24]h', '[1,60]m'}, default='1h'
        The resolution or binsize the day is divided into. The default value is
        one hour '1h'.

    Examples
    --------
    >>> from pyadlml.stats import device_trigger_one_day
    >>> device_trigger_one_day(data.df_devs, t_res='1h')
    device    Cups cupboard  Dishwasher   ...  Washingmachine
    time                                  ...
    00:00:00            0.0         0.0   ...             0.0
    01:00:00           16.0         0.0   ...             0.0
    ...
    23:00:00            6.0         8.0   ...             2.0

    Returns
    -------
    df : pd.DataFrame
        A dataframe where the columns are the devices and the rows bin the day.
    """
    df_devs = df_devs.copy()
    # set devices time to their time bin
    df_devs[TIME] = df_devs[TIME].apply(time2int, args=[t_res])

    # every trigger should count
    df_devs[VAL] = 1
    df_devs = df_devs.groupby([TIME, DEVICE]).sum().unstack()
    df_devs = df_devs.fillna(0)
    df_devs.columns = df_devs.columns.droplevel(0)

    if lst_devs is not None:
        for device in set(lst_devs).difference(df_devs.columns):
           df_devs[device] = 0
    return df_devs