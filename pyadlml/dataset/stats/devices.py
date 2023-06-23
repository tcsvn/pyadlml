import numpy as np
import pandas as pd
from pyadlml.constants import START_TIME, END_TIME, TIME, NAME, VALUE, DEVICE, NUM, CAT, BOOL
from pyadlml.dataset.stats.util import df_density_binned
from pyadlml.dataset.util import time2int, timestr_2_timedeltas, infer_dtypes, categorical_2_binary
from pyadlml.dataset._core.devices import device_events_to_states, _create_devices, \
    is_device_df, _is_dev_rep2, contains_non_binary, split_devices_binary, create_device_info_dict

from pyadlml.dataset.util import timestr_2_timedeltas
from pyadlml.dataset._representations.state import create_state
from pyadlml.util import get_npartitions
from dask import delayed

def device_order_by_count(df_devices):
    return event_count(df_devices)\
        .sort_values(by='event_count', ascending=False)[DEVICE].tolist()


def state_cross_correlation(df_devices,  n_jobs=-1):
    """
    Compute the similarity between devices by comparing the binary values
    for every interval.

    Parameters
    ----------
    df_devices : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    n_jobs : int, default=-1


    Examples
    --------
    >>> from pyadlml.stats import device_state_cross_correlation
    >>> device_state_cross_correlation(data.df_devices)
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
    td = 'td'

    df_devs = df_devices.copy()
    dtypes = infer_dtypes(df_devs)

    # drop numerical devices
    df_devs = df_devs[~df_devs[DEVICE].isin(dtypes[NUM])]
    if dtypes[CAT]:
        df_devs = categorical_2_binary(df_devs, dtypes[CAT])

    def func(row):
        """ gets two rows and returns a crosstab
        """
        try:
            td = row.td.to_timedelta64()
        except:
            return None
        states = row.iloc[1:len(row)-1].values.astype(int)

        for j in range(len(states)):
            row.iloc[1+j] = states[j]*states*td
        return row

    def create_meta(raw):
        return {TIME: 'datetime64[ns]',
                **{name: object for name in raw.columns[1:-1]},
                td: 'timedelta64[ns]'}

    dev_lst = df_devs[DEVICE].unique()
    df_devs = df_devs.sort_values(by=TIME)
    info_dict = create_device_info_dict(df_devs)
    # make off to -1 and on to 1 and then calculate cross correlation between signals
    raw = create_state(df_devs, info_dict)
    raw.loc[:, dev_lst] = raw.loc[:, dev_lst].applymap(lambda x: 1 if x else -1)
    raw[td] = raw[TIME].shift(-1) - raw[TIME]
    n = get_npartitions()

    # TODO apply leads to to COPY on slice warnings. check out why this is happenign
    with pd.option_context('mode.chained_assignment', None):

        import dask.dataframe as dd
        df = dd.from_pandas(raw.copy(), npartitions=n)\
               .apply(func, axis=1, meta=create_meta(raw))\
               .drop(columns=[TIME, td])\
               .sum(axis=0)\
               .compute(scheduler='threads')

    res = pd.DataFrame(data=np.vstack(df.values), columns=df.index, index=df.index)
    # normalize
    res = res/res.iloc[0, 0]

    #if lst_devs is not None:
    #    for dev in set(lst_devs).difference(set(list(res.index))):
    #        res[dev] = pd.NA
    #        res = res.append(pd.DataFrame(data=pd.NA, columns=res.columns, index=[dev]))
    return res


def state_times(df_devices: pd.DataFrame, binary_state: str = 'on', categorical: bool = True) -> pd.DataFrame:
    """
    Compute times a device is in a certain state.

    Parameters
    ----------
    df_devices : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    binary_state : str one of {'on', 'off'},  default='on'
        pass
    categorical : bool, default=True
        If set, categorical devices are also considered

    Examples
    --------
    >>> from pyadlml.stats import device_on_time
    >>> device_on_time(data.df_devs)
           time          device              td
    0                    Hall-Bedroom door 0 days 00:02:43
    1                    Hall-Bedroom door 0 days 00:00:01
    ...                  ...             ...
    1309                 Frontdoor 0 days 00:00:01
    [1310 rows x 2 columns]

    Returns
    -------
    pd.DataFrame
        A dataframe with two columns, the devices and the time differences.
    """
    assert binary_state in ['on', 'off']
    td = 'td'
    df = df_devices.copy()
    dtypes = infer_dtypes(df)

    # drop numerical devices
    df = df[~df[DEVICE].isin(dtypes[NUM])]

    if categorical and dtypes[CAT]:
        df = categorical_2_binary(df, dtypes[CAT])
    elif not categorical and dtypes[CAT]:
        df = df[~df[DEVICE].isin(dtypes[CAT])]

    # invert binary devices values if the 'off' state is desired
    if binary_state == 'off':
        mask_true = (df[DEVICE].isin(dtypes[BOOL]) & df[VALUE] == True)
        mask_false = (df[DEVICE].isin(dtypes[BOOL]) & df[VALUE] == False)
        df.loc[mask_true, VALUE] = False
        df.loc[mask_false, VALUE] = True

    df = df.sort_values(by=TIME)
    df[td] = pd.NaT
    last_ts = df[TIME].iloc[-1]
    for dev in df[DEVICE].unique():
        diff = df.loc[df[DEVICE] == dev, TIME].shift(-1).copy() \
               - df.loc[df[DEVICE] == dev, TIME].copy()
        diff.iloc[-1] = last_ts - df.loc[df[DEVICE] == dev, TIME].iloc[-1]
        df.loc[(df[DEVICE] == dev), 'td'] = diff
    df[td] = pd.to_timedelta(df[td])
    return df.iloc[:-1, :].loc[(df[VALUE] == True), [TIME, DEVICE, td]]


def state_fractions(df_devices: pd.DataFrame) -> pd.DataFrame:
    """ Computes the fraction a device is in a certain state.
    Categorical devices

    Parameters
    ----------
    df_devices : pd.DataFrame
        A device dataframe

    Examples
    --------
    ..code python:
        >>> from pyadlml.stats import state_fractions
        >>> state_fractions(data.df_devices)
                    device                     td_on  ...   frac_on  frac_off
        0           bool_1 0 days 00:00:20.039999994  ...  0.334056  0.665944
        1           bool_2 0 days 00:00:39.999999995  ...  0.666778  0.333222
        ...
        5  cat_2:sub_cat_1 0 days 00:00:09.970000001  ...  0.166194  0.833806
        6  cat_2:sub_cat_2 0 days 00:00:30.020000001  ...  0.500417  0.499583
        7  cat_2:sub_cat_3 0 days 00:00:19.979999992  ...  0.333056  0.666945

    Returns
    -------
    pd.DataFrame
        The fraction a device is in a certain state.

    """
    from pyadlml.dataset.util import infer_dtypes, categorical_2_binary
    df_devs = df_devices.copy()

    dtypes = infer_dtypes(df_devs)
    diff = 'diff'
    td = 'td'
    frac = 'frac'

    # drop numerical devices
    df_devs = df_devs[~df_devs[DEVICE].isin(dtypes[NUM])]

    df_devs = device_events_to_states(df_devs, extrapolate_states=True)
    df_devs = df_devs.sort_values(START_TIME)

    # calculate total time interval for normalization
    int_start = df_devs.at[df_devs.index[0], START_TIME]
    int_end = df_devs.at[df_devs.index[-1], END_TIME]
    total_time = int_end - int_start

    # Calculate time deltas for online time
    df_devs[diff] = df_devs[END_TIME] - df_devs[START_TIME]
    df_devs = df_devs.groupby(by=[DEVICE, VALUE], observed=True)[diff].sum()
    df_devs = pd.DataFrame(df_devs)
    df_devs.columns = [td]

    # Calculate fraction
    df_devs[frac] = df_devs[td].dt.total_seconds()/total_time.total_seconds()

    return df_devs.reset_index()\
        .rename(columns={'index': DEVICE})\
        .sort_values(by=[DEVICE])


def event_count(df_devices: pd.DataFrame) -> pd.DataFrame:
    """ Compute the amount a device is triggered throughout a dataset.

    Parameters
    ----------
    df_devices : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.

    Examples
    --------
    >>> from pyadlml.stats import device_event_count
    >>> device_event_count(data.df_devices)
                    device    event_count
    0        Cups cupboard             98
    1           Dishwasher             42
    ..                 ...            ...
    13      Washingmachine             34

    Returns
    -------
    pd.DataFrame
        The devices and their respective triggercounts.
    """
    assert is_device_df(df_devices)

    col_label = 'event_count'

    ser = df_devices.groupby(DEVICE, observed=True)[DEVICE].count()
    df_devices = pd.DataFrame({DEVICE: ser.index, col_label: ser.values})

    #if lst_devs is not None:
    #    for dev in set(lst_devs).difference(set(list(df_devices[DEVICE]))):
    #        df_devices = df_devices.append(pd.DataFrame(data=[[dev, 0]],
    #                                        columns=df_devices.columns,
    #                                                    index=[len(df_devices)]))
    return df_devices.sort_values(by=DEVICE)


def inter_event_intervals(df_devices: pd.DataFrame) -> np.ndarray:
    """ Compute the time difference between sucessive device events
        in seconds.

    Parameters
    ----------
    df_devices : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.

    Examples
    --------
    >>> from pyadlml.stats import device_iei
    >>> device_iei(data.df_devices)
    array([1.63000e+02, 3.30440e+04, 1.00000e+00, ..., 4.00000e+00,
           1.72412e+05, 1.00000e+00])

    Returns
    -------
    np.ndarray
        Array of time deltas in seconds.
    """

    # Create timediff to the previous event
    diff_seconds = 'ds'
    df_devices = df_devices.copy().sort_values(by=[TIME])

    # Compute the seconds to the next device
    df_devices[diff_seconds] = df_devices[TIME].diff().shift(-1) / pd.Timedelta(seconds=1)
    return df_devices[diff_seconds].values[:-1]


def resampling_imputation_loss(df_devices: pd.DataFrame, dt: str, return_fraction: bool = False):
    """
    Computes the amount of events neglected when resampling the device dataframe with
    a certain bin size (dt)

    Parameters
    ----------
    df_devices : pd.DataFrame
        A device dataframe
    dt : str, one of ['m','h','s','ms']
        The resolution at which  the data is resampled.
    return_fraction : boolean, default=False
        Whether the fraction of imputed events is returned or the total number

    Returns
    -------
    float or int
        Either the fraction or the total amount of lost events depending on the `return_fraction` parameter
    """
    df_devs = df_devices.copy()\
                        .drop(columns=[VALUE])\
                        .set_index(TIME)
    nr_events_disregarded = 0
    
    # For each device count the #events that excessively fall into one bin
    for dev in df_devs[DEVICE].unique():
        # Resample and count the number of events that fall into one bin
        tmp = df_devs[df_devs[DEVICE] == dev].copy()\
                          .resample(dt, kind='timestamp')\
                          .count()

        # Subtract one event for that no imputation woudl be necessary
        val_imp = tmp[(tmp[DEVICE] >= 1)] - 1
        nr_events_disregarded += val_imp.sum()

    if return_fraction:
        return nr_events_disregarded/len(df_devices)
    else:
        return nr_events_disregarded


def event_cross_correlogram_slice(df_devs: pd.DataFrame, lst_devs=None, t_window: str ='20s') -> pd.DataFrame:
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
    pd.DataFrame
        TODO
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
        tmp = df_devs.groupby(DEVICE, observed=True)['tmp'].sum()
        res_df.loc[dev_name] += tmp

    return res_df.sort_index(axis=0, ascending=True) \
        .sort_index(axis=1, ascending=True) \
        .replace(pd.NA, 0)


def events_one_day(df_devices: pd.DataFrame, dt: str=None) -> pd.DataFrame:
    """
    Divide a day into time bins and compute how many device triggers fall into
    each bin.

    Parameters
    ----------
    df_devices : pd.DataFrame
        A datasets recorded devices. For more information refer to
        :ref:`user guide<device_dataframe>`.
    dt : str of {'[1,24]h', '[1,60]m'}, default=None
        The resolution or binsize the day is divided into.

    Examples
    --------
    >>> from pyadlml.stats import device_trigger_one_day
    >>> device_trigger_one_day(data.df_devices, dt='1h')
    device    Cups cupboard  Dishwasher   ...  Washingmachine
    time                                  ...
    00:00:00            0.0         0.0   ...             0.0
    01:00:00           16.0         0.0   ...             0.0
    ...
    23:00:00            6.0         8.0   ...             2.0

    Returns
    -------
    pd.DataFrame
        A dataframe where the columns are the devices and the rows bin the day.
    """
    df_devices = df_devices.copy()

    if dt is None:
        raise NotImplementedError
    else:
        return df_density_binned(df_devices, column_str=DEVICE, dt=dt)


def event_cross_correlogram(df_devices: pd.DataFrame, binsize: str = '1s', maxlag: str = '2m') -> pd.DataFrame:
    """  Calculate cross correlogram

      ccg = correlogram(t, assignment, binsize, maxlag) calculates the
      cross- and autocorrelograms for all device pairs.
      For device on y axis, the events of device on x-axis happen around that device 

    Parameters
    ----------
    df_devices :
        device dataframe
    binsize : str, default='1s'
        The size of one bin in the correlogram
    maxlag : str, default='2m'
        The size of the window for which spikes should be considered.

    Returns
    -------
    ccg             computed correlograms   #device x #device x #bins 
    bins            bin times relative to center    #bins x 1
    """

    import dask
    ET = 'event_times'
    df_dev = df_devices.copy()

    devices = df_devices[DEVICE].unique()
    n_devs = len(devices)
    maxlag = pd.Timedelta(maxlag).seconds
    binsize = pd.Timedelta(binsize).seconds

    n_bins = int((maxlag / binsize) * 2) + 1    # add +1 for symmetric histogram

    ccg = np.zeros((n_devs, n_devs, n_bins))
    bins = np.linspace(-maxlag, maxlag, n_bins)

    # get times in milliseconds relative to the first event
    start_time = df_dev[TIME].iloc[0]
    df_dev[ET] = (df_dev[TIME] - start_time).dt.seconds

    def calc_hist(df_dev, ref, tar):
        # select reference and target device
        t_ref = df_dev[df_dev[DEVICE] == ref][ET].values
        t_tar = df_dev[df_dev[DEVICE] == tar][ET].values

        hist_sum = np.zeros(n_bins)

        # Fix event of reference and do histogram of target events around that fixation
        for k, event in enumerate(t_ref):

            # In case of auto-correlogram delete the reference event from the target array
            # otherwise the 0-bin will have additional #len(tar) events 
            t_tar_prime = np.delete(t_tar, k) if ref == tar else t_tar

            # create histograms
            hist, _ = np.histogram(t_tar_prime, bins=n_bins, range=(event - maxlag,  event + maxlag), density=False)
            hist_sum += hist
        return hist_sum

    # make dask delayed computation for each field
    result = []
    for i, ref in enumerate(devices):
        for j, tar in enumerate(devices):
            result.append(calc_hist(df_dev, ref, tar))
            #result.append(dask.delayed(calc_hist)(df_dev, ref, tar))
    #result = dask.compute(result, scheduler='threads')[0]

    # reintegrate into array
    for i, ref in enumerate(devices):
        for j, tar in enumerate(devices):
            ccg[i, j, :] = result[i*len(devices)+j] # (n_bins,) -> (n_bins, dev, dev)

    return ccg, bins


def _calc_hist(df_dev, ref, tar, n_bins, maxlag):

    # Select reference and target device
    t_ref = df_dev.loc[df_dev[DEVICE] == ref, 'event_times'].copy().values
    t_tar = df_dev.loc[df_dev[DEVICE] == tar, 'event_times'].copy().values
    # Duplicate target #ref times on 0-axis
    tar_matrix = np.tile(t_tar, (len(t_ref), 1))                # (TA,) -> (RE, TA)
    # Duplicate ref #target times on 1-axis
    shift = np.repeat(t_ref.reshape(-1, 1), len(t_tar), axis=1) # (RE, ) -> (RE, TA)

    # Shift every target (row) by reference fix point => zero-centering
    # i.e. first target tm[0,:] is shifted by 0, since shift[0,:] = 0 pointwise, loop
    shifted_target = tar_matrix - shift     # (RE, TA) -> (RE, TA)

    # Compute histograms around 0, (RE, TA) -> (RE, n_bins, 2) containing hist and edges
    if True:
        tmp2 = np.apply_along_axis(lambda *x, **y: np.histogram(*x, **y)[0], axis=1, arr=shifted_target, 
                                bins=n_bins, range=(-maxlag, maxlag), density=False)
        hist_sum2 = tmp2.sum(axis=0)
        return hist_sum2

    tmp = np.apply_along_axis(np.histogram, axis=1, arr=shifted_target,
                            bins=n_bins, range=(-maxlag, maxlag), density=False)

    # Count over hists (0)
    hist_sum = tmp[:, 0].sum(axis=0)
    return hist_sum



def event_cross_correlogram2(df_devices, binsize='1s', maxlag='2m', v1=True, use_dask=False):
    """  Calculate cross correlogram

      ccg = correlogram(t, assignment, binsize, maxlag) calculates the
      cross- and autocorrelograms for all pairs of clusters with input

    Parameters
    ----------
    df_devices :
        device dataframe
    binsize : str, default='1s'
        The size of one bin in the correlogram
    maxlag : str, default='2m'
        The size of the window for which spikes should be considered.

    Returns
    -------
    ccg             computed correlograms   #bins x #device x #device
    bins            bin times relative to center    #bins x 1
    """
    ET = 'event_times'
    df_dev = df_devices.copy()

    devices = df_devices[DEVICE].unique()
    n_devs = len(devices)
    maxlag = pd.Timedelta(maxlag).seconds
    binsize = pd.Timedelta(binsize).seconds

    n_bins = int((maxlag / binsize) * 2) + 1    # add +1 for symmetric histogram

    ccg = np.zeros((n_devs, n_devs, n_bins))
    bins = np.linspace(-maxlag, maxlag, n_bins)

    # get times in milliseconds relative to the first event
    start_time = df_dev[TIME].iloc[0]
    df_dev[ET] = (df_dev[TIME] - start_time).dt.seconds
    """
    Normally a correlogram between two devices is computed by fixing one event of device 1, computing 
    the histogram of the device 2 eventrain, zero-centered around the fixed event and repeat the procedure 
    for every event of device 1.
    
    The fast implementation repeats the event stream of device 1. Then subtract for devices 2 event stream
    is computed. By subtracting each event  from devices 1 vectorized event stream  all events are zero centered
    
    
    """


    # make dask delayed computation for each field
    result = []
    import itertools
    import dask
    if use_dask:
        df_dev = dask.delayed(df_dev)

    k = 0
    for i, j in itertools.combinations_with_replacement(range(0, len(devices)), 2):
        #print(f'({i},{j}) -> {k}')
        if use_dask:
            result.append(dask.delayed(_calc_hist)(df_dev, devices[i], devices[j], n_bins, maxlag))
        else:
            result.append(_calc_hist(df_dev, devices[i], devices[j], n_bins, maxlag))
        k+=1

    if use_dask:
        result = dask.compute(*result)

    # reintegrate into array
    for k, (i, j) in enumerate(itertools.combinations_with_replacement(range(0, len(devices)), 2)):
        #print(f'{k} -> ({i}, {j})')
        if i != j:
            ccg[i, j, :] = result[k]
            ccg[j, i, :] = np.flip(result[k])
        else:
            ccg[i, i, :] = result[k]

    return ccg, bins, devices




def event_cross_correlogram3(df_devices, binsize='1s', maxlag='2m', fix=[], to=[], use_dask=False):
    """  Calculate event cross correlogram. 
        Device A event is fixed and the events of another device B are counted happening in the range of
        maxlag around the fixed events of device A. 
        Therefore, the histgoram c_ij displays the events of device i happening around the device j. 
        Thus a bar in the histogram c_ij means x amound of device i events happen before/after an event
        of device j

    Parameters
    ----------
    df_devices :
        device dataframe
    fix : list

    to: list

    binsize : str, default='1s'
        The size of one bin in the correlogram
    maxlag : str, default='2m'
        The size of the window for which spikes should be considered.

    Returns
    -------
    ccg             computed correlograms   #bins x #device x #device
    bins            bin times relative to center    #bins x 1
    """
    ET = 'event_times'
    df_dev = df_devices.copy()

    devices = df_devices[DEVICE].unique()
    n_devs = len(devices)

    if not fix and not to:
        n_rows, n_cols = [n_devs]*2
        rows, cols = [devices]*2
    if fix:
        n_cols, cols = len(fix), fix
    if to:
        n_rows, rows = len(to), to


    maxlag = pd.Timedelta(maxlag).seconds

    binsize = pd.Timedelta(binsize).seconds

    n_bins = int((maxlag / binsize) * 2) + 1    # add +1 for symmetric histogram

    ccg = np.zeros((n_rows, n_cols, n_bins))
    bins = np.linspace(-maxlag, maxlag, n_bins)

    # get times in milliseconds relative to the first event
    start_time = df_dev[TIME].iloc[0]
    df_dev[ET] = (df_dev[TIME] - start_time).dt.seconds
    """
    Normally a correlogram between two devices is computed by fixing one event of device 1, computing 
    the histogram of the device 2 eventrain, zero-centered around the fixed event and repeat the procedure 
    for every event of device 1.
    
    The fast implementation repeats the event stream of device 1. Then subtract for devices 2 event stream
    is computed. By subtracting each event  from devices 1 vectorized event stream  all events are zero centered
    
    
    """
    def combination_with_replacement(l1, l2, r=2):
        lst = []
        hm = set()
        for e1 in l1:
            for e2 in l2:
                p = frozenset([e1, e2])
                if p not in hm:
                    hm.add(p)
                    lst.append((e1, e2))
        return lst
    class CombHM():
        def __init__(self, rows, cols):
            self.iter_lst = []
            self.value_dict = dict()
            k=0
            for e1 in rows:
                for e2 in cols:
                    key = (e1, e2)
                    if not self._has_item(key):
                        self.value_dict[key] = None
                        self.iter_lst.append(key)
                        k+=1

        def _has_item(self, key):
            if key in self.value_dict.keys():
                return True
            elif (key[1], key[0]) in self.value_dict.keys():
                return True
            else:
                return False

        def _is_rev_key(self, key):
            if key in self.value_dict.keys():
                return False
            elif (key[1], key[0]) in self.value_dict.keys():
                return True
            else:
                raise AttributeError

        def __iter__(self): 
            return self.iter_lst.__iter__()

        def __next__(self):
            return self.iter_lst.__next__()

        def __setitem__(self, idxs, value):
            if isinstance(idxs, int):
                idxs = self.iter_lst[idxs]
            else:
                assert len(idxs) == 2
                idxs = (idxs[0], idxs[1])

            self.value_dict[idxs] = value

        def __getitem__(self, idxs):
            """
            For a single index return the order in which the elements were added. For
            multiple indices 
            """
            if isinstance(idxs, int):
                idxs = self.iter_lst[idxs]
            else:
                assert len(idxs) == 2
                idxs = (idxs[0], idxs[1])

            if self._has_item(idxs):
                if self._is_rev_key(idxs):
                    return np.flip(self.value_dict[(idxs[1], idxs[0])])
                else:
                    return self.value_dict[idxs]
            else:
                raise AttributeError


    # make dask delayed computation for each field
    result = []
    import itertools
    import dask
    if use_dask:
        df_dev = dask.delayed(df_dev)

    combis = CombHM(rows, cols)

    #if rows != cols:
    if True:
        for r, c in combis:
            # TODO REMOVE Compute when the column events happen w.r.t. to a fixed row event
            # Compute when the row events happen w.r.t. to a fixed column event
            if use_dask:
                combis[r,c] = dask.delayed(_calc_hist)(df_dev, c, r, n_bins, maxlag)
            else:
                combis[r,c] = _calc_hist(df_dev, c, r, n_bins, maxlag)

        if use_dask:
            keys = list(combis)
            for k, v in zip(keys, dask.compute(*[combis[k] for k in keys])):
                combis[k] = v 

        for r, c in itertools.product(range(len(rows)), range(len(cols))):
            ccg[r, c, :] = combis[rows[r], cols[c]]

        return ccg, bins, rows, cols 

    else:
        k = 0
        for i, j in itertools.combinations_with_replacement(range(0, len(devices)), 2):
            #print(f'({i},{j}) -> {k}')
            if use_dask:
                result.append(dask.delayed(_calc_hist)(df_dev, devices[i], devices[j], n_bins, maxlag))
            else:
                result.append(_calc_hist(df_dev, devices[i], devices[j], n_bins, maxlag))
            k+=1

        if use_dask:
            result = dask.compute(*result)

        # reintegrate into array
        for k, (i, j) in enumerate(itertools.combinations_with_replacement(range(0, len(devices)), 2)):
            #print(f'{k} -> ({i}, {j})')
            if i != j:
                ccg[i, j, :] = result[k]
                ccg[j, i, :] = np.flip(result[k])
            else:
                ccg[i, i, :] = result[k]

    return ccg, bins, rows, cols 



def fano_factor(df_devs: pd.DataFrame, dt=None, inplace=False) -> float:
    """
    Repeat experiment serveral times -> measured spke count varies between one trial and the next.
    The fano factor F is the variance of the spike count divided by its mean. 

    Parameters
    ----------



    Returns
    -------
    float
        the fano factor 

    """
    mean, variance = firing_rate_moments(df_devs, dt, inplace=True)
    return variance / mean



def firing_rate_moments(df_devs: pd.DataFrame, dt=None, times=None, inplace=False):
    """ Calculates mean and variance 

    Takes the temporal average
        Each timeslice length dt is seen as an repeated experiment K
            v_k = n_k^{sp}/T
        The mean <n^sp> is estimated by averaging over all repeated timeslices


    """
    df = firing_rate(df, dt, inplace=False)
    mean = df['rate'].average()

    # \delta n_k^sp
    deviations = (df['rate'] - mean)**2
    variance = deviations.average()

    return mean, variance



def firing_rate(df_devs: pd.DataFrame, dt=None, times=None, inplace=False):
    """
    Compute the mean firing rate 
        v = n^{sp}/T



        file:///home/chris/Downloads/journal.pcbi.1010792.pdf

        https://neuronaldynamics.epfl.ch/online/Ch7.S2.html


    Parameters
    ----------
    df_devs : pd.DataFrame
        The device readings in the format ['time', 'device', 'value']. 

    dt : str, default=None



    Returns
    -------
    """

    df = df_devs.copy()\
                .sort_values(by=TIME)\
                .reset_index(drop=True)

    if dt is None:
        dt =df[TIME].last() - df[TIME].first()

    df['rate'] = 1

    df = df.set_index(TIME)
    # Compute spike counts per window with wiundow anchored at the right edge
    df['rate'] = df['rate'].rolling(dt, center=False).sum()

    df = df.reset_index()

    # Do not need this since window size equals rate
    #amount, unit = _split_unit(dt)
    # Convert to rate 
    #df['rate'] = df['rate'] / amount

    if inplace:
        return df
    else:
        return df['rate'].values 


def _split_unit(dt):

    tail = dt.lstrip('0123456789')
    head = dt[:-len(tail)]
    return int(head), tail