import numpy as np
import pandas as pd

from pyadlml.constants import DEVICE, TIME, VALUE, CAT, NUM, BOOL
from pyadlml.dataset._core.devices import _create_devices, most_prominent_categorical_values

ST_FFILL = 'ffill'
ST_INT_COV = 'interval_coverage'

def create_raw(df_dev, dataset_info, dev_pre_values={}):
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

    df = df_dev.pivot(index=TIME, columns=DEVICE, values=VALUE)
    df = df.reset_index()

    # get all learned devices by data type
    dev_cat = [dev for dev in dataset_info.keys() if dataset_info[dev]['dtype'] == CAT]
    dev_bool = [dev for dev in dataset_info.keys() if dataset_info[dev]['dtype'] == BOOL]
    dev_num = [dev for dev in dataset_info.keys() if dataset_info[dev]['dtype'] == NUM]

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


def resample_raw(df_raw, df_dev, dt, most_likely_values=None, n_jobs=1):
    """
    Resamples a raw representation
    """

    # get dtypes in order for choosing different collision behavior
    dev_dtypes = _infer_types(df_raw)
    if most_likely_values is None:
        from pyadlml.dataset.devices import get_most_likely_value
        most_likely_values = get_most_likely_value(df_dev)
    most_likely_values = most_likely_values.set_index(DEVICE)

    variant = 'a'
    if variant == 'a':
        return resample_pandas(df_raw, dt, df_dev, dev_dtypes, most_likely_values)
    elif variant == 'c':
        return resample_pandas_and_pandarell_applymap(df_raw, dt, df_dev, dev_dtypes, most_likely_value)
    elif variant == 'd':
        return resample_dask_pandarell(df_raw, dt, dev_dtypes, df_dev, most_likely_values)
    elif variant == 'b':
        return resample_pure_dask(df_raw, dt, dev_dtypes, df_dev, most_likely_values)


"""
FROM HERE ON ARE THE DIFFERENT resample techniques. As of now pandas and pandarell applymap 
is the fastest.
"""


def parallel_resample_apply(resampler, func):
    """
    https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
    # TODO write manual parallelism
    """
    import pandas as pd
    from joblib import Parallel, delayed
    import multiprocessing
    jobs = []
    for name, group in resampler:
        jobs.append(delayed(func)(group))
    ret_lst = Parallel(n_jobs=multiprocessing.cpu_count())(jobs)
    #retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in grouped)
    return pd.concat(retLst)

def resample_pandas(df_raw, t_res, df_dev, dev_dtypes, most_likely_values):
        df_raw = df_raw.set_index(TIME)
        resampler = df_raw.resample(t_res, kind='timestamp')

        # first do a forward fill to correctly represent intervals where no observation falls into
        # ffill takes a devices last known value and assigns it to the timeslice
        raw_ff = resampler.ffill()

        # then for intervals where multiple sensors trigger choose the most prevalent for categorical and
        # boolean values. For numerical drop the most unlikely one (determined on difference to mean)
        raw_int = resampler.apply(_assign_timeslices,
                                  t_res=t_res,
                                  dev=df_dev,
                                  dev_dtypes=dev_dtypes,
                                  most_likely_values=most_likely_values
                                  )
        # combine both by filling gaps with the forward fills
        raw = raw_int.where(~raw_int.isnull(), raw_ff)

        raw = raw.reset_index()
        return raw


def resample_pandas_and_pandarell_applymap(df_raw, t_res, df_dev, dev_dtypes, most_likely_value):
    """ use pandas with and pandarallel applymap """
    from pandarallel import pandarallel
    pandarallel.initialize(verbose=1)

    df_raw = df_raw.set_index(TIME)
    raw_ff = df_raw.resample(t_res, kind='timestamp').ffill()

    # first make resampling and safe lists of choices inside each bin
    raw_int = df_raw.resample(t_res).apply(lambda x: [x.name, x.tolist(), x.index.tolist()])

    def _assign_timeslicesC(lst: list) -> pd.Series:
        """
        cc. kasteren

        Parameters
        ----------
        series: pd.Series
            the datapoints that are to be assigned for a timeslice for one column/device
        t_res: the resolution of the intervals
            e.g 30s ->  09:38:30, 09:39:00
        devs: pd.DataFrame
            used to identify when values changes

        """
        name = lst[0]
        values = lst[1]
        timestamps = lst[2]
        series = pd.Series(values, index=timestamps)
        series.name = lst[0]

        # return nan if no element matchs, happens till first occurence of data
        series = series.dropna()
        if series.empty:
            return np.nan

        # if one element matches the interval slot, assign the one
        elif series.size == 1:
            return series

        # if there are multiple elements falling into the same timeslice
        else:
            if series.name in dev_dtypes['boolean']:
                tmp = resolve_collision_boolean(series, t_res, df_dev)

            elif series.name in dev_dtypes['categorical']:
                tmp = resolve_collision_categorical(series, t_res, df_dev)

            elif series.name in dev_dtypes['numerical']:
                tmp = resolve_collision_numerical(series, t_res, df_dev, most_likely_values)
            else:
                raise ValueError("The column/device didn't match either boolean, categorical or numerical type.")
            return tmp

    # on each field apply the collision resolvment
    #raw_int = raw_int.applymap(_assign_timeslicesC)
    #raw_int = raw_int.parallel_applymap(_assign_timeslicesC)

    # combine both by filling gaps with the forward fills
    raw = raw_int.where(~raw_int.isnull(), raw_ff)

    raw = raw.reset_index()
    return raw

def resample_pure_dask(df_raw, t_res, dev_dtypes, df_dev, most_likely_values):
        import dask.dataframe as dd
        dd_raw = dd.from_pandas(df_raw, npartitions=n_jobs)

        # TODO problem dask resampler doesn't implement function ffill
        # solution 1
        resampler_ffill = dd_raw.groupby(pd.Grouper(key=TIME, freq=t_res, origin='start'))
        def func(x):
            """ returns nan if list is empty and """
            x = x.dropna()
            if x.empty:
                return np.nan
            else:
                return x.iloc[-1]

        raw_ff = resampler_ffill.apply(func)
        df = raw_ff.compute()
        # solution 2: use rolling in combination with apply and a lambda
        # function that mimics ffill to achieve parallelnes

        resampler = dd_raw.resample(t_res)
        raw_int = resampler.apply(_assign_timeslices, t_res=t_res, dev=df_dev,
                              dev_dtypes=dev_dtypes, most_likely_values=most_likely_values)
        raw = raw_int.where(~raw_int.isnull(), raw_ff)
        raw = raw.reset_index().compute()
        return raw


def resample_dask_pandarell(df_raw, t_res, dev_dtypes, df_dev, most_likely_values):
        import dask.dataframe as dd
        n_jobs = 8
        df_raw = df_raw.set_index(TIME)
        raw_ff = df_raw.resample(t_res, kind='timestamp').ffill()

        # first make resampling and safe lists of choices inside each bin
        dd_raw_int = dd.from_pandas(df_raw, npartitions=n_jobs)
        raw_count = dd_raw_int.resample(t_res).count().compute()
        row_mask = (raw_count > 0).any(axis=1)
        # first make resampling and safe lists of choices inside each bin
        #raw_int = df_raw.resample(t_res).apply(lambda x: [x.name, x.tolist(), x.index.tolist()])
        raw_int = parallel_resample_apply(df_raw.resample(t_res), lambda x: [x.name, x.tolist(), x.index.tolist()])
        raw_int_subset = raw_int.loc[row_mask]

        def _assign_timeslicesC(lst: list) -> pd.Series:
            name = lst[0]
            values = lst[1]
            timestamps = lst[2]
            series = pd.Series(values, index=timestamps)
            series.name = name
            # return nan if no element matchs, happens till first occurence of data
            series = series.dropna()
            if series.empty:
                return np.nan

            # if one element matches the interval slot, assign the one
            elif series.size == 1:
                return series

            # if there are multiple elements falling into the same timeslice
            else:
                if series.name in dev_dtypes['boolean']:
                    tmp = resolve_collision_boolean(series, t_res, df_dev)

                elif series.name in dev_dtypes['categorical']:
                    tmp = resolve_collision_categorical(series, t_res, df_dev)

                elif series.name in dev_dtypes['numerical']:
                    tmp = resolve_collision_numerical(series, t_res, df_dev, most_likely_values)
                else:
                    raise ValueError("The column/device didn't match either boolean, categorical or numerical type.")
                return tmp

        from pandarallel import pandarallel
        pandarallel.initialize(verbose=2)
        # on each field apply the collision resolvment
        raw_int_subset = raw_int_subset.parallel_apply(_assign_timeslicesC)
        raw_int = pd.concat(raw_int_subset, raw_int[~row_mask]).sort_values(by=TIME)

        # combine both by filling gaps with the forward fills
        raw = raw_int.where(~raw_int.isnull(), raw_ff)

        raw = raw.reset_index()
        return raw

def _assign_timeslices(series: pd.Series, t_res, dev, dev_dtypes, most_likely_values) -> pd.Series:
    """
    cc. kasteren
    
    Parameters
    ----------
    series: pd.Series
        the datapoints that are to be assigned for a timeslice for one column/device
    t_res: the resolution of the intervals
        e.g 30s ->  09:38:30, 09:39:00
    devs: pd.DataFrame
        used to identify when values changes

    """
    # return nan if no element matchs, happens till first occurence of data
    series = series.dropna()
    if series.empty:
        return np.nan

    # if one element matches the interval slot, assign the one
    elif series.size == 1:
        return series

    # if there are multiple elements falling into the same timeslice
    else:
        if series.name in dev_dtypes['boolean']:
            tmp = resolve_collision_boolean(series, t_res, dev)

        elif series.name in dev_dtypes['categorical']:
            tmp = resolve_collision_categorical(series, t_res, dev)

        elif series.name in dev_dtypes['numerical']:
            tmp = resolve_collision_numerical(series, t_res, dev, most_likely_values)
        else:
            raise ValueError("The column/device didn't match either boolean, categorical or numerical type.")
        return tmp


def resolve_collision_numerical(series: pd.Series, t_res, dev, most_likely_values: pd.DataFrame):
    """
    The value that is nearer to the mean value is treated as more likely

    Discussion: maybe the value that is further away from the mean carries more information and
    should therefore be kept ???
    """
    dev_mean = most_likely_values.at[series.name, 'ml_state']
    # get elements position with minimal distance to most likely value (mean/median)
    min_idx = abs(series - dev_mean).values.argmin()
    val = series.iloc[min_idx]
    return val


def resolve_collision_categorical(series, t_res, df_dev):
    """
    choose the element with the maximal overlap
    """
    dev_name = series.name

    # create a category timedelta score dictionary
    df_subdev = df_dev[df_dev[DEVICE] == dev_name]
    cats = list(df_subdev[VALUE].unique())
    tds = [pd.Timedelta(seconds=0) for i in range(len(cats))]
    categories = dict(zip(cats,tds))

    # deduce the beginning of the interval and the timeslice
    start_time = series.index[0].floor(t_res)
    end_time = start_time + pd.Timedelta(t_res)

    # get preceding categorical value
    prec_values = df_subdev[df_subdev[TIME] < start_time]
    if prec_values.empty:
        # if this is the first slice there are no preceding values so take the first of the series
        prae_val = series[0]
    else:
        # take the last known category as
        prae_val = prec_values.iat[-1, 2]
    prae_ts = start_time

    # compute cumulative states in each sensor
    for entry in series.iteritems():
        cur_ts = entry[0]
        categories[prae_val] += cur_ts - prae_ts
        prae_val = entry[1]
        prae_ts = cur_ts

    # add time to end of timeslice to the last known category
    categories[prae_val] += end_time - prae_ts

    # select category with most overlap
    max_cat = max(categories, key=lambda key: categories[key])
    return max_cat

def resolve_collision_boolean(series, t_res, dev):
    """
    choose the element with the maximal overlap
    """
    cum_time_0 = pd.Timedelta(seconds=0)
    cum_time_1 = pd.Timedelta(seconds=0)

    # deduce the beginning of the interval and the interval
    start_time = series.index[0].floor(t_res)
    end_time = start_time + pd.Timedelta(t_res)

    # deduce value at start of the timeslice
    df_subdev = dev[dev[DEVICE] == series.name]
    prec_values = df_subdev[df_subdev[TIME] < start_time]
    if prec_values.empty:
        # if this is the first slice there are no preceding values so take the first of the series
        prae_val = series[0]
    else:
        # take the last known category as
        prae_val = prec_values.iat[-1, 2]

    prae_time = start_time
    # compute cumulative states in each sensor
    for entry in series.iteritems():
        ts = entry[0]
        cur_val = entry[1]
        if prae_val:
            cum_time_1 += ts-prae_time
        else:
            cum_time_0 += ts-prae_time
        prae_val = cur_val
        prae_time = ts

    # compute cum time to remaining end of interval
    if prae_val:
        cum_time_1 += end_time-prae_time
    else:
        cum_time_0 += end_time-prae_time

    assert cum_time_0 + cum_time_1 == pd.Timedelta(t_res)

    # return the state that was more present in the interval
    return int(cum_time_0 < cum_time_1)