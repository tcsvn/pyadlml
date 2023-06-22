import numpy as np
import pandas as pd

from pyadlml.constants import DEVICE, TIME, VALUE, CAT, NUM, BOOL
from pyadlml.dataset._core.devices import _create_devices, create_device_info_dict, most_prominent_categorical_values
from pyadlml.dataset.util import infer_dtypes
import dask.dataframe as dd

ST_FFILL = 'ffill'
ST_INT_COV = 'interval_coverage'


def create_state(df_dev, dataset_info=None, dev_pre_values={}, n_jobs=None):
    """

    Parameters
    ----------
    df_dev : pd.DataFrame

    dataset_info : dict
        first key: devices (DEVICE)
        per dev key: most likely value ('ml_state')
        per dev key: datatype ('dtype')
    dev_pre_values : dict
        a dictionary a mapping from device to values. This mapping should be
        used for values where the preceeding value is not known.

    Returns
    -------
        return df:
        | time  | dev_1 | ....  | dev_n |
        --------------------------------
        | ts1   |   1   | ....  | open |
    """
    df_dev = df_dev.copy()

    if dataset_info is None:
        dataset_info = create_device_info_dict(df_dev)

    df = df_dev.pivot(index=TIME, columns=DEVICE, values=VALUE)
    df = df.reset_index()

    # get all learned devices by data type
    dev_cat = [dev for dev in dataset_info.keys() if dataset_info[dev]
               ['dtype'] == CAT]
    dev_bool = [dev for dev in dataset_info.keys() if dataset_info[dev]
                ['dtype'] == BOOL]
    dev_num = [dev for dev in dataset_info.keys() if dataset_info[dev]
               ['dtype'] == NUM]

    # filter for devices that appear in given dataset
    devs = set(df_dev[DEVICE].unique())
    dev_cat = list(set(dev_cat).intersection(devs))
    dev_bool = list(set(dev_bool).intersection(devs))
    dev_num = list(set(dev_num).intersection(devs))

    # set the first element for each boolean device to the opposite value of the
    # first occurrence
    for dev in dev_bool:
        fvi = df[dev].first_valid_index()
        if fvi != 0:
            if dev_pre_values:
                df.loc[0, dev] = dev_pre_values[dev]
            else:
                value = df[dev].iloc[fvi]
                df.loc[0, dev] = not value

    # set the first element of each categorical device to the most likely value
    for dev in dev_cat:
        if dev_pre_values:
            new_val = dev_pre_values[dev]
        else:
            new_val = dataset_info[dev]['ml_state']
        df.loc[0, dev] = new_val

    # set the first element of numerical values to the given value if dev_pre_values
    # dict is given
    for dev in dev_num:
        if dev_pre_values:
            new_val = dev_pre_values[dev]
            df.loc[0, dev] = new_val

    # fill from start to end NaNs with the preceding correct value
    df_cat_bool = df[list(dev_bool) + list(dev_cat)].ffill()

    # join all dataframes
    df = pd.concat([df[TIME], df[dev_num], df_cat_bool], axis=1)

    # for all devices that are present in the info but not in the current dataframe infer value
    for dev in (set(dataset_info.keys()) - set(df.columns)):
        if dev_pre_values:
            df[dev] = dev_pre_values[dev]
        else:
            df[dev] = dataset_info[dev]['ml_state']
    return df


def resample_state(df_dev, dt, most_likely_values=None):
    df_dev = df_dev.copy()\
                   .sort_values(by=TIME)\
                   .reset_index(drop=True)

    origin = df_dev.at[0, TIME].floor(freq=dt)
    dtypes = infer_dtypes(df_dev)

    # Compute the last state value for each timebin
    df_last = df_dev.copy()
    df_last['bin'] = df_last.groupby(pd.Grouper(key=TIME, freq=dt, origin=origin))\
                            .ngroup()

    df_last = df_last.groupby(['bin', DEVICE], observed=True)\
                     .last()\
                     .reset_index()
    df_last = df_last.drop(columns='bin')\
                     .sort_values(by=TIME)\
                     .reset_index(drop=True)\
                     [[TIME, DEVICE, VALUE]]

    # Calculate additional infos for each event, such as
    # if it is a collision, its bin_start and end time
    df = df_dev.copy()
    df[DEVICE] = df[DEVICE].astype(object)
    df['bin'] = df.groupby(pd.Grouper(key=TIME, freq=dt, origin=origin)).ngroup()
    #df['coll'] = (df.groupby(['bin', 'device']).transform('size') > 1).astype(bool)

    df2 = df.copy().set_index(TIME)
    df2 = df2.resample(dt, origin=origin).count().reset_index()
    df2[df2.columns[1:]] = np.nan
    df_devs = pd.concat([df, df2], axis=0).sort_values(by=TIME).reset_index(drop=True)

    # Add start time bin to each device
    orig_values = ~df_devs.isna().any(axis=1)
    df_devs['bin_time_start'] = df_devs[TIME]
    df_devs.loc[orig_values, 'bin_time_start'] = np.nan
    df_devs['bin_time_start'] = df_devs['bin_time_start'].ffill()

    # Add bin-start and end times to each device
    df_devs['bin_time_end'] = df_devs[TIME]
    df_devs.loc[orig_values, 'bin_time_end'] = np.nan
    df_devs['bin_time_end'] = df_devs['bin_time_end'].bfill()
    last_time_bin = df_devs.at[df_devs.index[-1], 'bin_time_start'] 
    df_devs['bin_time_end'] = df_devs['bin_time_end'].fillna(last_time_bin+ pd.Timedelta(dt))

    df_devs = df_devs.dropna()

    # Add an additional device for each device in time bin at the start of the bin
    # with correct[sic] state. Add eps to first device such that it is in a bin if it turns
    # out to be the prominent device
    df_devs_first_per_bin = df_devs.copy().groupby(['bin', DEVICE], observed=True)\
                        .first()\
                        .reset_index()
    df_devs_first_per_bin[TIME] = df_devs_first_per_bin['bin_time_start'] + pd.Timedelta('1ns')
    bool_mask = df_devs_first_per_bin[DEVICE].isin(dtypes['boolean'])
    df_devs_first_per_bin.loc[bool_mask, VALUE] = ~df_devs_first_per_bin.loc[bool_mask, VALUE].astype(bool)
    cat_mask = df_devs_first_per_bin[DEVICE].isin(dtypes['categorical'])
    # If first category is the most prominent -> mark for correction at end of function
    df_devs_first_per_bin.loc[cat_mask, VALUE] = np.inf

    # For each device get the timedelta to following device and replace last with end of time bin
    df_devs = pd.concat([df_devs, df_devs_first_per_bin], axis=0)\
                      .sort_values(by=TIME).reset_index(drop=True)
    df_devs['time_diff'] = df_devs.groupby([DEVICE, 'bin'], observed=True)[TIME].diff(-1).abs()
    df_devs['time_diff_to_end'] = df_devs['bin_time_end'] - df_devs[TIME]
    df_devs['time_diff'] = df_devs['time_diff'].fillna(df_devs['time_diff_to_end'])

    # Get the most promintent value per time bin 
    # -> only one device of a certain type per bin
    idx = df_devs.groupby([DEVICE, 'bin'], observed=True)['time_diff'].idxmax()
    df_most_prom = df_devs.loc[idx].sort_values(by=TIME).reset_index(drop=True)

    # Create state representation that has correct ffill and bbfill but not 
    # necessarily the correct values for the collisions
    first_values = df_dev.groupby(DEVICE, observed=True)[VALUE].first().to_dict()
    for dev in dtypes[BOOL]:
        first_values[dev] = not first_values[dev] 
    for dev in dtypes[CAT]:
        first_values[dev] = most_likely_values[dev]['ml_state']
    
    # Create resampled state representation where in each 
    # bin the value corresponds to the last known device state
    df_state_last = create_state(df_last, dev_pre_values=first_values).set_index(TIME)
    df_state_last = df_state_last.resample(dt, kind='timestamp', origin=origin).ffill()

    # There is no last element for the first time bin. Since all device states in the next 
    # timebin are correct copy those, except for 
    first_dev = df_dev.iloc[0, df_dev.columns.tolist().index(DEVICE)]
    first_dev_col_idx = df_state_last.columns.tolist().index(first_dev)
    df_state_last.iat[0, first_dev_col_idx] = np.inf
    df_state_last.iloc[0] = df_state_last.iloc[0].where(df_state_last.iloc[0] == np.inf, df_state_last.iloc[1])
    df_state_last.iat[0, first_dev_col_idx] = first_values[first_dev]

    # Create resampled state representation where in each 
    # bin the value corresponds to the most prominent device state or is nan
    df_most_prom = df_most_prom.pivot(index=TIME, columns=DEVICE, values=VALUE)
    df_most_prom = df_most_prom.resample(dt, kind='timestamp', origin=origin).first()
    df_most_prom.index.name = TIME

    assert len(df_most_prom) == len(df_state_last) \
        and set(df_most_prom.columns) == set(df_state_last.columns) \
        and (df_most_prom.index == df_state_last.index).all()

    df_most_prom = df_most_prom.replace({None: np.nan})
    df_most_prom = df_most_prom[df_state_last.columns]

    df = df_most_prom.where(df_most_prom.notna(), df_state_last)\
           .reset_index(names=TIME)

    # For timeslices where the most prominent category was the one that extended
    # from the previous timeslice, correct the states
    for dev in dtypes[CAT]:
        if np.inf in df[dev].unique():
            cat_coll_idxs = df[df[dev] == np.inf].index
            times = df.loc[cat_coll_idxs, TIME]
            times = pd.DataFrame({TIME: times})
            times[TIME] = pd.to_datetime(times[TIME])
            prec_records = pd.merge_asof(times, df_last[df_last[DEVICE] == dev], on=TIME, direction='backward')
            df.loc[cat_coll_idxs, dev] = prec_records[VALUE].values
        # The case when for the first occurence of a category the cat is not the 
        # prominent in the timeslice. Just use first value
        if df[dev].isna().sum() == 1:
            df[dev] = df[dev].fillna(first_values[dev])
        assert np.inf not in df[dev].unique() \
            and np.nan not in df[dev].unique()

    assert df.notna().all().all() \
       and (df != np.inf).all().all()
    return df