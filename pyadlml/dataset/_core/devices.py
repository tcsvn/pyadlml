from pathlib import Path
import numpy as np
from scipy import signal as sc_signal
import pandas as pd

from pyadlml.constants import START_TIME, END_TIME, TIME, VALUE, DEVICE, BOOL, NUM, CAT
from pyadlml.dataset.util import df_difference, infer_dtypes

CORRECTION_TS = 'correction_ts'
CORRECTION_ONOFF_INCONS = 'correction_on_off_inconsistency'
COLS_DEV = [TIME, DEVICE, VALUE]

"""
    df_devices:
        Standard representation for device dataframes.
        Exc: 
           | time      | device    | state
        ------------------------------------
         0  | timestamp | dev_name  |   1

    Representation 2:
        is used to calculate statistics for binary devices more easily. 
        Some computation is easier using this format
           | start_time | end_time   | device    | state
        --------------------------------------------------
         0 | timestamp   | timestamp | dev_name  | state_0
 
 """


def _create_devices(dev_list: list, index=None):
    """
    creates an empty device dataframe
    """
    if index is not None:
        return pd.DataFrame(columns=dev_list, index=index)
    else:
        return pd.DataFrame(columns=dev_list)


def _check_devices_sequ_order(df: pd.DataFrame):
    """
    iterate pairwise through each select device an check if the 
    sequential order in time is inconsistent

    Parameters
    ----------
    df : pd.DataFrame
        device representation 1  with columns [start_time, end_time, devices]
    """
    dev_list = df[DEVICE].unique()
    no_errors = True
    for dev in dev_list:
        df_d = df[df[DEVICE] == dev]
        for i in range(1, len(df_d)):
            st_j = df_d.iloc[i - 1].start_time
            et_j = df_d.iloc[i - 1].end_time
            st_i = df_d.iloc[i].start_time
            # et_i = df_d.iloc[i].end_time
            # if the sequential order is violated return false
            if not (st_j < et_j) or not (et_j < st_i):
                print('~' * 50)
                if st_j >= et_j:
                    # raise ValueError('{}; st: {} >= et: {} |\n {} '.format(i-1, st_j, et_j, df_d.iloc[i-1]))
                    print('{}; st: {} >= et: {} |\n {} '.format(i - 1, st_j, et_j, df_d.iloc[i - 1]))
                if et_j >= st_i:
                    # raise ValueError('{},{}; et: {} >= st: {} |\n{}\n\n{}'.format(i-1,i, et_j, st_i, df_d.iloc[i-1], df_d.iloc[i]))
                    print('{},{}; et: {} >= st: {} |\n{}\n\n{}'.format(i - 1, i, et_j, st_i, df_d.iloc[i - 1],
                                                                       df_d.iloc[i]))
                no_errors = False
    return no_errors


def _is_dev_rep2(df: pd.DataFrame):
    """
    """
    if not START_TIME in df.columns or not END_TIME in df.columns \
            or not DEVICE in df.columns or len(df.columns) != 3:
        return False
    return True


def is_device_df(df: pd.DataFrame) -> bool:
    """ Checks if a dataframe is a valid device dataframe.
    """
    return DEVICE in df.columns \
           and VALUE in df.columns \
           and TIME in df.columns \
           and len(df.columns) == 3


def get_index_matching_rows(df_devs, rows, tolerance='1ms'):

    if isinstance(rows, list):
        df = pd.DataFrame(rows, columns=COLS_DEV)
        df[TIME] = pd.to_datetime(df[TIME], errors='coerce', dayfirst=True)
    else:
        print('went here')
        df = rows

    assert isinstance(df, pd.DataFrame)

    tol = pd.Timedelta(tolerance)
    idxs = []
    for _, row in df.iterrows():
        mask_st = (row[TIME]-tol < df_devs[TIME])\
                & (df_devs[TIME] < row[TIME]+tol)
        mask_dev = (df_devs[DEVICE] == row[DEVICE])
        res = df_devs[mask_st & mask_dev].index.values
        assert len(res) <= 1
        if len(res) == 1:
            idxs.append(*res)
        if len(res) == 0:
            print('Warning!!! Tried to delete device but could not')
    return idxs






def device_events_to_states(df_devs: pd.DataFrame, start_time=None, end_time=None, extrapolate_states=False):
    """ Transforms device dataframe from an event representation into a state representation,
        
    Parameters
    ----------
    df_devs : pd.DataFrame
        In event representation, a dataframe with columns (time, device, value)
        example row: [2008-02-25 00:20:14, Freezer, False]
    extrapolate_states : Boolean, default=False
        Whether boolean devices should add extra states for the first occurring events
    start_time : pd.Timestamp, str or default=None
        The start time from which to
    end_time : pd.Timestamp, str or default=None
        The start time from which to

    Returns
    -------
    df : pd.DataFrame
        In state representation with columns (start time, end_time, device, value)
        example row: [2008-02-25 00:20:14, 2008-02-25 00:22:14, Freezer, True]

    df, lst
    """
    epsilon = '1ns'

    df = df_devs.copy() \
        .reset_index(drop=True) \
        .sort_values(TIME)



    dtypes = infer_dtypes(df)

    first_timestamp = df[TIME].iloc[0] if start_time is None else pd.Timestamp(start_time)
    last_timestamp  = df[TIME].iloc[-1] if end_time is None else pd.Timestamp(end_time)

    if extrapolate_states:
        # create additional rows in order to compensate for the state duration of the first event
        # to the selected event for boolean devices
        eps = pd.Timedelta(epsilon)

        # Prevents boolean cast warning when extrapolatiing states
        df[VALUE] = df[VALUE].astype(object)

        for dev in dtypes[BOOL]:
            df_dev = df[df[DEVICE] == dev]
            first_row = df_dev.iloc[0].copy()
            if first_row[TIME] != first_timestamp and first_row[TIME] - first_timestamp > pd.Timedelta('1s'):
                first_row[VALUE] = not first_row[VALUE]
                first_row[TIME] = first_timestamp + eps
                df = pd.concat([df, first_row.to_frame().T], axis=0, ignore_index=True)
            eps += pd.Timedelta(epsilon)

    df = df.sort_values(TIME).reset_index(drop=True)

    lst_cat_or_bool = dtypes[CAT] + dtypes[BOOL]
    res = pd.DataFrame(columns=[START_TIME, END_TIME, DEVICE, VALUE])

    if lst_cat_or_bool:
        df_cat_bool = df[df[DEVICE].isin(lst_cat_or_bool)].copy()
        df_cat_bool = df_cat_bool.rename(columns={TIME: START_TIME})
        df_cat_bool[END_TIME] = pd.NaT
        for dev in lst_cat_or_bool:
            mask = (df_cat_bool[DEVICE] == dev)
            df_cat_bool.loc[mask, END_TIME] = df_cat_bool.loc[mask, START_TIME].shift(-1)
        if extrapolate_states:
            df_cat_bool = df_cat_bool.fillna(last_timestamp)
        else:
            df_cat_bool = df_cat_bool.dropna()
        res = pd.concat([res, df_cat_bool])

    if dtypes[NUM]:
        df_num = df[df[DEVICE].isin(dtypes[NUM])].copy()
        df_num = df_num.rename(columns={TIME: START_TIME})
        df_num[END_TIME] = df_num[START_TIME].copy()
        res = res.append(df_num)

    res[START_TIME] = pd.to_datetime(res[START_TIME])
    res[END_TIME] = pd.to_datetime(res[END_TIME])

    return res


def device_boolean_on_states_to_events(df_devs_states: pd.DataFrame):
    """
    Parameters
    ----------
    df_devs_states : pd.DataFrame
        A table with the columns (start time, end_time, device)
        All states are assumed to be 'on'/True

    Returns
    -------
    df_devs : (pd.DataFrame)
        rep1: columns are (time, device, val)
        example row: [2008-02-25 00:20:14, Freezer, False]
    """
    # copy devices to new dfs
    # one with all values but start time and other way around
    df_start = df_devs_states.copy().drop(columns=END_TIME)
    df_end = df_devs_states.copy().drop(columns=START_TIME)

    # Set values at the end time to zero because this is the time a device turns off
    df_start[VALUE] = True
    df_end[VALUE] = False

    # Rename column 'End Time' and 'Start Time' to 'Time'
    df_start.rename(columns={START_TIME: TIME}, inplace=True)
    df_end.rename(columns={END_TIME: TIME}, inplace=True)

    df = pd.concat([df_end, df_start]).sort_values(TIME) \
        .reset_index(drop=True)
    return df


def device_states_to_events(df_devs_states: pd.DataFrame):
    """
    Parameters
    ----------
    df_devs_states : pd.DataFrame
        A table with the columns (start time, end_time, device, val).

    Returns
    -------
    df_devs : (pd.DataFrame)
        The table columns are (time, device, val)

    Example
    -------
    >>> device_states_to_events(df_devs_states)
        time,               device,   value
        2008-02-25 00:20:14, Freezer, False
    """

    # copy devices to new dfs
    # one with all values but start time and other way around
    # tmp = df_devs_states.copy().rename(columns={START_TIME: TIME})
    # res = []
    # if dtypes[BOOL]:
    #    mask_bool_true = df_devs_states[DEVICE].isin(dtypes[BOOL]) \
    #                    & (df_devs_states[VAL] == True)
    #    df = df_devs_states.loc[mask_bool_true, [START_TIME, END_TIME, VAL]]
    #    df_devs_boolean = device_boolean_on_states_to_events(df.copy())
    #    res.append(df_devs_boolean)
    df = df_devs_states.copy()
    df.rename(columns={START_TIME: TIME}, inplace=True)
    df.drop(columns=END_TIME, inplace=True)
    return df


def correct_device_ts_duplicates(df: pd.DataFrame):
    """
    remove devices that went on and off at the same time. And add a microsecond
    to devices that trigger on the same time
    Parameters
    ----------
    df : pd.DataFrame
        Devices in representation 1; columns [time, device, value]
    """
    eps = pd.Timedelta('10ms')

    try:
        df[TIME]
    except KeyError:
        df = df.reset_index()

    # remove device if it went on and off at the same time    
    dup_mask = df.duplicated(subset=[TIME, DEVICE], keep=False)
    df = df[~dup_mask]

    df = df.reset_index(drop=True)

    dup_mask = df.duplicated(subset=[TIME], keep=False)
    duplicates = df[dup_mask]
    uniques = df[~dup_mask]

    # for every pair of duplicates add a millisecond on the second one
    duplicates = duplicates.reset_index(drop=True)
    sp = duplicates[TIME] + eps
    mask_p = (duplicates.index % 2 == 0)

    duplicates[TIME] = duplicates[TIME].where(mask_p, sp)

    # concatenate and sort the dataframe 
    uniques = uniques.set_index(TIME)
    duplicates = duplicates.set_index(TIME)
    df = pd.concat([duplicates, uniques], sort=True)

    # set the time as index again
    df = df.sort_values(TIME).reset_index(drop=False)

    return df


def _has_timestamp_duplicates(df: pd.DataFrame):
    """ check whether there are duplicates in timestamp present
    Parameters
    ----------
    df : pd.DataFrame
        data frame representation 1: [time, device, val]
    """
    df = df.copy()
    try:
        dup_mask = df.duplicated(subset=[TIME], keep=False)
    except KeyError:
        df = df.reset_index()
        dup_mask = df.duplicated(subset=[TIME], keep=False)
    return dup_mask.sum() > 0


def split_devices_binary(df: pd.DataFrame):
    """ separate binary devices and non-binary devices
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe in device representation 1
    Returns
    -------
    df_binary, df_non_binary : pd.DataFrames
        Dataframe with binary devices and dataframe without binary devices
    """
    mask_binary = df[VALUE].apply(lambda x: isinstance(x, bool))
    return df[mask_binary], df[~mask_binary]


def contains_non_binary(df: pd.DataFrame) -> bool:
    """ determines whether the the dataframes values contain non-boolean values
    These can be continuous values of categorical.
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe in device representation 1
    Returns
    -------
    boolean
    """
    return not df[VALUE].apply(lambda x: isinstance(x, bool)).all()


def correct_devices(df: pd.DataFrame, retain_correction=False) -> pd.DataFrame:
    """
    Applies correction to devices.
    1. drops all duplicates
    2. timestamps that are identical are separated by an offset
    3. for binary devices redundant reportings of the same value are dropped

    Parameters
    ----------
    df : pd.DataFrame
        either in device representation 1 or 2
    retain_correction: bool, default=False
        decides whether corrections are computed and returned or not.

    Returns
    -------
    cor_rep1 : pd.DataFrame
        Device dataframe in representation 1
    corrections : dict or None

    """
    if retain_correction:
        corrections = {}
    else:
        corrections = None

    df = df.copy()
    df = df.drop_duplicates()

    if df.empty:
        return df

    df = df.sort_values(by=TIME).reset_index(drop=True)

    df_cor = df.copy()
    # correct timestamp duplicates
    while _has_timestamp_duplicates(df_cor):
        df_cor = correct_device_ts_duplicates(df_cor)
    assert not _has_timestamp_duplicates(df_cor)

    if retain_correction:
        corrections[CORRECTION_TS] = df_difference(df, df_cor)

    df = df_cor.copy()

    if contains_non_binary(df):
        df_binary, df_non_binary = split_devices_binary(df)
        non_binary_exist = True
    else:
        df_binary = df
        non_binary_exist = False

    # correct on/off inconsistency
    if not is_on_off_consistent(df_binary):
        df_binary = correct_on_off_inconsistency(df_binary)
    assert is_on_off_consistent(df_binary)

    # join dataframes
    if non_binary_exist:
        df = pd.concat([df_binary, df_non_binary], axis=0, ignore_index=True)
    else:
        df = df_binary

    if retain_correction:
        corrections[CORRECTION_ONOFF_INCONS] = df_difference(df_cor, df)

    df = df.sort_values(by=TIME).reset_index(drop=True)

    return df, corrections


def on_off_consistent_func(df: pd.DataFrame, dev: list):
    """ compute for each device if it is on/off consistent
    Parameters
    ----------
    df : pd.DataFrame
        the whole activity dataframe
    dev : str
        a device that occurs in the dataframe
    Returns
    -------
    tupel [arg1, arg2]
        first argument is a boolean whether this device is consistent
        second is
    """
    df_dev = df[df[DEVICE] == dev].sort_values(by=TIME).reset_index(drop=True)
    first_val = df_dev[VALUE].iloc[0]
    if first_val:
        mask = np.zeros((len(df_dev)), dtype=bool)
        mask[::2] = True
    else:
        mask = np.ones((len(df_dev)), dtype=bool)
        mask[::2] = False
    return [not (df_dev[VALUE] ^ mask).sum() > 0, df_dev[[TIME, DEVICE, VALUE]]]


def is_on_off_consistent(df: pd.DataFrame):
    """ devices can only go on after they are off and vice versa. check if this is true
        for every device.
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe in representation 1.
        The dataframe must not include timestamp duplicates! When it does they can be arbitrarily reordered
        when sorting for time thus destroying the consistency.
    """

    import dask
    from dask import delayed
    lazy_results = []
    for dev in df[DEVICE].unique():
        res = dask.delayed(on_off_consistent_func)(df.copy(), dev)
        lazy_results.append(res)

    results = np.array(list(dask.compute(*lazy_results)), dtype=object)
    return results[:, 0].all()


def correct_on_off_inconsistency(df: pd.DataFrame):
    """ 
    has multiple strategies for solving the patterns:
    e.g if the preceeding value is on and the same value is occuring now delete now
    Parameters
    ----------
    df : pd.DataFrame
        device representation 3
        
    Returns
    -------
    df : pd.DataFrame
        device representation 3
    """
    from dask import delayed
    import dask

    def correct_part(df_dev):
        """ get index of rows where the previous on/off value was the same and delete row
        Parameters
        ----------
        df_dev : pd.DataFrame
            subset of the big dataframe consisting only of events for a fixed device
        """
        df_dev['same_prec'] = ~(df_dev[VALUE].shift(1) ^ df_dev[VALUE])
        df_dev.loc[0, 'same_prec'] = False  # correct shift artefact
        indices = list(df_dev[df_dev['same_prec']].index)
        df_dev = df_dev.drop(indices, axis=0)
        return df_dev[[TIME, DEVICE, VALUE]]

    df = df.copy()
    # create list of tuples e.g [(True, df_dev1), (False, df_dev2), ...]
    dev_list = df[DEVICE].unique()
    dev_df_list = []
    for devs in dev_list:
        dev_df_list.append(delayed(on_off_consistent_func)(df, devs))
    results = np.array(list(dask.compute(*dev_df_list)), dtype=object)

    # filter inconsistent devices
    incons = results[np.where(np.logical_not(results[:, 0]))[0], :][:, 1]

    corrected = []
    for part in incons:
        corrected.append(delayed(correct_part)(part))
    corr_dfs = delayed(lambda x: x)(corrected).compute()
    corr_devs = [df[DEVICE].iloc[0] for df in corr_dfs]
    tmp = [df[~df[DEVICE].isin(corr_devs)], *corr_dfs]

    return pd.concat(tmp, ignore_index=True) \
        .sort_values(by=TIME).reset_index(drop=True)


def _create_empty_dev_dataframe():
    df = pd.DataFrame(data=[], columns=[TIME, DEVICE, VALUE])
    df[TIME] = pd.to_datetime(df[TIME])
    return df


def most_prominent_categorical_values(df: pd.DataFrame):
    tmp = df
    tmp['conc'] = tmp[DEVICE] + ',' + tmp[VALUE]
    dev_list = tmp[DEVICE].unique()
    tmp = pd.concat([tmp[TIME], pd.get_dummies(tmp['conc'], dtype=int)], axis=1)
    tmp2 = tmp.copy()
    for dev in dev_list:
        dev_col_cats = []
        for col in tmp.columns:
            if dev not in col:
                continue
            dev_col_cats.append(col)

        for di in dev_col_cats:
            other = dev_col_cats[:]
            other.remove(di)
            for o in other:
                tmp2[di] = tmp2[di] - tmp[o]
    tmp2 = tmp2.set_index(TIME)
    tmp2 = tmp2.where(tmp2 != 0, np.nan)
    tmp2 = tmp2.ffill(axis=0)
    tmp2 = tmp2.where(tmp2 != -1, 0)
    tmp2 = tmp2.reset_index()
    tmp2['td'] = tmp2[TIME].shift(-1) - tmp2[TIME]

    lst = []
    for dev in dev_list:
        dev_col_cats = []
        for col in tmp.columns:
            if dev not in col:
                continue
            dev_col_cats.append(col)
        asdf = []
        max_td = pd.Timedelta('0s')
        max_cat = None
        for di in dev_col_cats:
            abc = tmp2[[di, 'td']].groupby(di).sum().reset_index()
            td_on = abc.iloc[1, 1]
            if max_td < td_on:
                max_td = td_on
                max_cat = di
            asdf.append(td_on)
        lst.append([dev, max_cat.split(',')[1]])
    ML_STATE = 'ml_state'
    df = pd.DataFrame(data=lst, columns=[DEVICE, ML_STATE])
    return df


def _get_bool_mask(df: pd.DataFrame, col: list):
    """ returns a mask where the columns are booleans"""
    if df[col].dtype == 'bool':
        # return df[col].na # TODO hack for returning
        return (df[col] == True) | (df[col] == False)
    mask = (df[col].astype(str) == 'False') | (df[col].astype(str) == 'True')
    return mask


def _get_num_mask(df, col):
    df = df.copy()
    df[col] = df[col].astype(str)
    df[col] = df[col].where(df[col] != 'True', 'stub')
    df[col] = df[col].where(df[col] != 'False', 'stub')
    num_mask = pd.to_numeric(df[col], errors='coerce').notnull()
    return num_mask


def _get_bool(df):
    def func(x):
        # check if every is one of
        if x[VALUE].dtype == 'bool':
            return True
        else:
            return ((x[VALUE].astype(str) == 'False') | (x[VALUE].astype(str) == 'True')).all()

    return df.groupby(by=[DEVICE]).filter(func)


def create_device_info_dict(df_dev: pd.DataFrame) -> dict:
    """
    Infers for each device the most likely state

    Parameters
    ----------
    df_dev : pd.DataFrame
        A device dataframe
    dtypes: pd.DataFrame
        a dataframe containing every device and its corresponding datattype

    Returns
    -------
    df : pd.DataFrame
        the result
    """
    ML_STATE = 'ml_state'
    DTYPE = 'dtype'
    df_dev = df_dev.copy()

    # extract devices of differing dtypes
    dtypes = infer_dtypes(df_dev)

    res = {}
    for key in dtypes.keys():
        for dev in dtypes[key]:
            res[dev] = {}
            res[dev][DTYPE] = key

    # get most likely binary states
    if dtypes[BOOL]:
        from pyadlml.dataset.stats.devices import state_fractions
        dsf = state_fractions(df_dev[df_dev[DEVICE].isin(dtypes[BOOL])].copy())
        for dev in dtypes[BOOL]:
            true_has_more = dsf.loc[(dsf[DEVICE] == dev) & (dsf[VALUE] == True), 'frac'].values[0] \
                            > dsf.loc[(dsf[DEVICE] == dev) & (dsf[VALUE] == False), 'frac'].values[0]
            res[dev][ML_STATE] = true_has_more

    # get most likely numerical states
    # use median for the most likely numerical state
    if dtypes[NUM]:
        df_num = df_dev.loc[df_dev[DEVICE].isin(dtypes[NUM]), [DEVICE, VALUE]].copy()
        df_num[VALUE] = pd.to_numeric(df_num[VALUE])
        res_num = df_num.groupby(by=[DEVICE]).median()
        for dev in dtypes[NUM]:
            res[dev][ML_STATE] = res_num.at[dev, VALUE]

    # get most likely categorical states
    if dtypes[CAT]:
        df_cat = df_dev[df_dev[DEVICE].isin(dtypes[CAT])].copy()
        res_cat = most_prominent_categorical_values(df_cat)
        res_cat.set_index(DEVICE, inplace=True)
        for dev in dtypes[CAT]:
            res[dev][ML_STATE] = res_cat.at[dev, ML_STATE]

    return res


def device_remove_state(df_devs, state, td, eps='0.2s'):
    """ Remove the events corresponding to a device that is a certain time
        in the specified state.

    Parameters
    ----------
    df_devs : pd.DataFrame
    state : bool
    td : str
    eps : str, default='0.2s'

    Returns
    ------
    pd.DataFrame
        The table with the specified events missing
    """
    eps = pd.Timedelta(eps)
    td = pd.Timedelta(td)

    # Note that due to extrapolate_states the very last events
    # for each device are disregarded
    df = device_events_to_states(df_devs.copy(), extrapolate_states=False)

    # Mark states that fall match the criterion
    df['diff'] = df[END_TIME] - df[START_TIME]
    df['target'] = (td - eps < df['diff']) & (df['diff'] < td + eps) \
                   & (df[VALUE] == state)

    # Remove states
    df = df[df['target'] == False]

    df = device_states_to_events(df[[START_TIME, END_TIME, DEVICE, VALUE]])
    df = correct_on_off_inconsistency(df)

    # Since the transformation above lost the last device events
    # those events are readded
    idx_last_occ = [df_devs[df_devs[DEVICE] == d].iloc[-1].name
                    for d in df_devs[DEVICE].unique()]
    last_occ = df_devs.iloc[idx_last_occ, :]
    df = pd.concat([df, last_occ]) \
        .sort_values(by=TIME) \
        .reset_index(drop=True)

    return df


def _generate_signal(signal, dt='250ms'):
    # Convert strings to timedelta
    dt = pd.Timedelta(dt)
    max_len = pd.Timedelta('0s')
    for i in range(len(signal)):
        signal[i] = (signal[i][0], pd.Timedelta(signal[i][1]))
        max_len += signal[i][1]

    nr_bins = np.floor(max_len / dt).astype(int)
    discretized_sig = np.zeros(nr_bins)
    current_step = 0
    for i in range(len(signal)):
        state = signal[i][0]
        steps = np.floor((signal[i][1] / max_len) * nr_bins).astype(int)
        discretized_sig[current_step:current_step + steps] = 1 if state else -1
        current_step += steps

    # TODO refactor, why does the extrapolation of states do not work out???
    # Eliminate Zeropadding
    last_state = discretized_sig[0]
    for i in range(1, len(discretized_sig)):
        if discretized_sig[i] == 0:
            discretized_sig[i] = last_state
        last_state = discretized_sig[i] 

    return discretized_sig



def create_sig_and_corr(df: pd.DataFrame, hit_idx: int, dt_prae_hit: pd.Timedelta, dt_post_hit: pd.Timedelta,
                        ss_discrete: np.ndarray):
    """ 

    Parameters
    ----------
    df : pd.DataFrame
        

    """
    idx, first, last = get_idxs(df, hit_idx, dt_prae_hit, dt_post_hit)
    tmp = df.loc[idx, [VALUE, 'diff']].values
    tmp = np.insert(tmp, 0, [[not tmp[0, 0], first]], axis=0)
    tmp = np.append(tmp, [[not tmp[-1, 0], last]], axis=0)
    signal = _generate_signal(tmp, dt='250ms')
    corr = sc_signal.correlate(signal, ss_discrete, mode='full')
    return signal, corr


def get_idxs(df, hit_idx, dt_prae_hit, dt_post_hit):
    """ Gets device indices of events before and after to the hit 
        to later on process in a cross correlation.

    Parameters
    ----------
    df : pd.DataFrame
        device dataframe of the selected device
    hit_idx: int
        Indice of the hit
    dt_prae_hit: pd.Timedelta
        How much time to consider prea hit
    dt_prae_hit: pd.Timedelta
        How much time to consider post hit

    Returns
    -------
    list, 
        The involved indices for the signal
    """
    get_df_idx = lambda x: df[df['index'] == x].index.values[0]

    res_idxs = [hit_idx]
    df = df.reset_index()

    if get_df_idx(hit_idx) == 0:
        first = dt_prae_hit
    else:
        # Iterate the events backward and finish if iterated time difference
        # is higher than the origs. signals 
        idx_lower = get_df_idx(hit_idx) - 1
        dt_iter = dt_prae_hit
        dt_prev_tmp = dt_iter - df.at[idx_lower, 'diff']
        while dt_prev_tmp > pd.Timedelta('0s') and idx_lower >= 0:
            dt_iter -= df.at[idx_lower, 'diff']
            res_idxs.insert(0, df.at[idx_lower, 'index'])
            idx_lower -= 1
            dt_prev_tmp -= df.at[idx_lower, 'diff']
        first = dt_iter

    if get_df_idx(hit_idx) == len(df) - 1:
        last = dt_post_hit
    else:
        # Iterate the events forward and finish if iterated time difference
        # is higher than the origs. signals 
        idx_upper = get_df_idx(hit_idx) + 1
        dt_iter = dt_post_hit
        dt_next_tmp = dt_iter - df.at[idx_upper, 'diff']
        while dt_next_tmp > pd.Timedelta('0s') and idx_upper < len(df) - 1:
            dt_iter -= df.at[idx_upper, 'diff']
            res_idxs.append(df.at[idx_upper, 'index'])
            idx_upper += 1
            dt_next_tmp -= df.at[idx_upper, 'diff']
        last = dt_iter

    df = df.set_index('index')
    return res_idxs, first, last



def device_remove_state_matching_signal(df_devs: pd.DataFrame,
                                        #signal: list[tuple[bool, str], ... ],
                                        device: str,
                                        signal: list,
                                        matching_state: int,
                                        corr_threshold=0.2, 
                                        eps_state: str = '0.2s') -> pd.DataFrame:
    """
    Removes states from a device dataframe that match a signal.

    Parameters
    ----------
    df_devs : pd.DataFrame
        A device dataframe
    state_indicator : int
        A number from 0 to length of signal -1 that indicates the state that is going
        to be removed when the signal matches
    eps_state : str, default='0.2s'
        The tolerance a target state may deviate from the given duration in the signal.
    eps_corr : float, default=0.1
        Determines the allowed deviation for the maximium correlation to the correlation of a perfect
        matching signal in percent.

    Example
    -------
    >>> from pyadlml.dataset import fetch_amsterdam
    >>> from pyadlml.dataset.devices import device_remove_state_matching_signal
    >>> data = fetch_amsterdam()
    >>> df = data.df_devices.copy()
    >>> sig_original = [ \
            (True, '5s'), \
            (False, '4s'), \
            (True, '1s'), \
            (False, '5s') \
        ] \
    >>> df = device_remove_state_matching_signal(df, sig_original, eps_corr=2)
    """


    matching_state_idx = matching_state
    eps_state = '0.2s'
    # Tolerance for cross correlation w.r.t. auto correlation to highlight as a match
    sig_search = [(s[0], pd.Timedelta(s[1])) for s in signal]


    # Find 
    td = pd.Timedelta(sig_search[matching_state_idx][1])
    state = sig_search[matching_state_idx][0]
    eps_state = pd.Timedelta(eps_state)

    from scipy import signal as sc_signal
    # Create auto correlation and retrieve a proper threshold
    ss_discrete = _generate_signal(sig_search, dt='250ms')
    auto_corr = sc_signal.correlate(ss_discrete, ss_discrete, mode='full')
    auto_corr_max = auto_corr.max()

    # Compute the number of counts that the signal may deviate
    # but still counts as a match
    # For example if a window is 72 units long and the maximum
    # correlation would also be 72 for a signal with the same length. Therefore
    # the eps_corr count would be 14 if the signal is allowed to differ 20%
    if isinstance(corr_threshold, float):
        corr_threshold = int(len(ss_discrete)*corr_threshold)
        corr_threshold = auto_corr_max - corr_threshold

    # Count total length and length up and after matching state of signal
    ss_total_dt, ss_dt_prae_match, ss_dt_post_match = [pd.Timedelta('0s')]*3
    for i, (s, dt) in enumerate(sig_search):
        ss_total_dt += dt
        if i < matching_state_idx:
            ss_dt_prae_match += dt
        if i > matching_state_idx:
            ss_dt_post_match += dt

    plot_dt_around_sig = ss_total_dt*0.25



    def create_hit_list(df_devs):
        df = df_devs[df_devs[DEVICE] == device].copy()
        df['to_convert'] = False
        df['diff'] = pd.Timedelta('0ns')
        df['target'] = False

        df['diff'] = df[TIME].shift(-1) - df[TIME]
        df['target'] = (td - eps_state < df['diff'])\
                    & (df['diff'] < td + eps_state)\
                    & (df[VALUE] == state)

        # Correct the case where the first occurence is already a match
        df.at[df.index[-1], 'diff'] = ss_total_dt

        # Get indices of hits and select first match for display
        hits = df[(df['target'] == True)].index.to_list()
        return hits, df

    hits, df = create_hit_list(df_devs)

    df_sel_dev = df_devs.copy()[df_devs[DEVICE] == device].reset_index()

    for h in hits:
        sig, corr = create_sig_and_corr(df, h, ss_dt_prae_match, ss_dt_post_match, ss_discrete)

        if corr.max() > corr_threshold:
            # Get succeding dev index
            mask = (df_sel_dev['index'] == h)
            idxs_to_drop = df_sel_dev[mask | mask.shift(1)]['index'].values.tolist()
            df_devs = df_devs.drop(idxs_to_drop)

    return df_devs.reset_index(drop=True)




    # Old

    from scipy import signal as sc_signal
    sig_original = signal
    td = pd.Timedelta(sig_original[matching_state][1])
    state = sig_original[matching_state][0]
    eps_state = pd.Timedelta(eps_state)

    # Create perfect correlation yielding a proper threshold
    win = _generate_signal(sig_original, dt='250ms')
    perfect_corr = sc_signal.correlate(win, win, mode='full')
    perfect_corr_max = perfect_corr.max()

    # Compute the number of counts that the signal may deviate
    # but still counts as a match
    # For example if a window is 72 units long and the maximum
    # correlation would also be 72 for a signal with the same length. Therefore
    # the eps_corr count would be 14 if the signal is allowed to differ 20%
    eps_corr = int(len(win)*eps_corr)

    # Create a timelength to append to
    tmp1 = [pd.Timedelta(s[1]) for s in sig_original]
    max_sig = sum(tmp1, pd.Timedelta('0s'))
    states_to_past = [pd.Timedelta(sig_original[s][1]) for s in range(0, matching_state)]
    dt_states_to_past = sum(states_to_past, pd.Timedelta('0s'))
    states_to_future = [pd.Timedelta(sig_original[s][1]) for s in range(matching_state+1, len(sig_original))]
    dt_states_to_future = sum(states_to_future, pd.Timedelta('0s'))

    df = df.copy().reset_index(drop=True)
    df['to_convert'] = False
    df['diff'] = pd.Timedelta('0ns')
    df['target'] = False

    for dev in df[DEVICE].unique():
        dev_mask = (df[DEVICE] == dev)
        max_idx = df.loc[dev_mask].index[-1]
        df.loc[dev_mask, 'diff'] = df.loc[dev_mask, TIME].shift(-1) - df.loc[dev_mask, TIME]
        df.loc[dev_mask, 'target'] = (td - eps_state < df.loc[dev_mask, 'diff']) \
                                     & (df.loc[dev_mask, 'diff'] < td + eps_state) \
                                     & (df.loc[dev_mask, VALUE] == state)
        df.at[max_idx, 'diff'] = max_sig

    for dev in df[DEVICE].unique():
        dev_mask = (df[DEVICE] == dev)
        hits = df[(df['target'] == True) & dev_mask].index.to_list()
        df_dev = df[dev_mask].copy().reset_index()
        for j, h in enumerate(hits):
            idx = [h]
            get_df_idx = lambda i: df_dev.at[i, 'index']

            # Retrieve index of preceeding and succeeding event of the same device
            dev_idx = df_dev[(df_dev['index'] == h)].index[0]
            if dev_idx == 0:
                first = dt_states_to_past
            else:
                dev_lower = dev_idx - 1
                dt_tmp = dt_states_to_past
                dt_prev_tmp = dt_tmp - df_dev.at[dev_lower, 'diff']
                while dt_prev_tmp > pd.Timedelta('0s') and dev_lower >= 0:
                    dt_tmp -= df_dev.at[dev_lower, 'diff']
                    idx.insert(0, get_df_idx(dev_lower))
                    dev_lower -= 1
                    dt_prev_tmp -= df_dev.at[dev_lower, 'diff']
                first = dt_tmp
            if dev_idx == len(df_dev) - 1:
                last = dt_states_to_future
            else:
                dev_upper = dev_idx + 1
                dt_tmp = dt_states_to_future
                dt_next_tmp = dt_tmp - df_dev.at[dev_upper, 'diff']
                while dt_next_tmp > pd.Timedelta('0s') and dev_upper < len(df_dev) - 1:
                    dt_tmp -= df_dev.at[dev_upper, 'diff']
                    idx.append(get_df_idx(dev_upper))
                    dev_upper += 1
                    dt_next_tmp -= df_dev.at[dev_upper, 'diff']
                last = dt_tmp


            # Aasdf
            tmp = df.loc[idx, [VALUE, 'diff']].values
            tmp = np.insert(tmp, 0, [[not tmp[0, 0], first]], axis=0)
            tmp = np.append(tmp, [[not tmp[-1, 0], last]], axis=0)
            signal = _generate_signal(tmp, dt='250ms')

            corr = sc_signal.correlate(signal, win, mode='full')


            is_match = perfect_corr_max - corr.max() < eps_corr
            if is_match:
                df.at[h, 'to_convert'] = True
            #if not is_match:
            #   is_match
            #print(f'match: {is_match}')

    df = df[df['to_convert'] == False].copy()
    df = df[[TIME, DEVICE, VALUE]]
    res = correct_on_off_inconsistency(df)
    return res
