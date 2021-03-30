import numpy as np
import pandas as pd

from pyadlml.dataset import DEVICE, TIME, VAL
from pyadlml.dataset.devices import _create_devices
from pyadlml.dataset._dataset import label_data

from pandas.api.types import infer_dtype

ST_FFILL = 'ffill'
ST_INT_COV = 'interval_coverage'

def _infer_types(df):
    """
    dataframe in raw representation where the columns correspond to devices
    """
    dev_cat = []
    dev_bool = []
    dev_num = []

    dev_lst = df.columns[1:]
    for dev in dev_lst:
        inf = infer_dtype(df[dev], skipna=True)
        if inf == 'string' or inf == 'object':
            dev_cat.append(dev)
        elif inf == 'boolean':
            dev_bool.append(dev)
        elif inf == 'floating':
            dev_num.append(dev)
        else:
            raise ValueError('could not infer correct dtype for device {}'.format(dev))

    return {'categorical': dev_cat, 'boolean': dev_bool, 'numerical': dev_num}

def create_raw(df_dev, most_likely_values=None):
    """
        return df:
        | time  | dev_1 | ....  | dev_n |
        --------------------------------
        | ts1   |   1   | ....  |  0    |
    """


    df_dev = df_dev.copy()

    df = df_dev.pivot(index=TIME, columns=DEVICE, values=VAL)
    df = df.reset_index()

    dev_dtypes = _infer_types(df)
    dev_cat = dev_dtypes['categorical']
    dev_bool = dev_dtypes['boolean']
    dev_num = dev_dtypes['numerical']

    # set the first element for each boolean device to the opposite value of the
    # first occurrence
    for dev in dev_bool:
        fvi = df[dev].first_valid_index()
        if fvi != 0:
            value = df[dev].iloc[fvi]
            df.loc[0, dev] = not value

    # set the first element of each categorical device to the most likely value
    if len(dev_cat) != 0:
        if most_likely_values is None:
            from pyadlml.dataset.devices import most_prominent_categorical_values
            tmp = df_dev[df_dev[DEVICE].isin(dev_cat)]
            most_likely_values = most_prominent_categorical_values(tmp)
        mlv = most_likely_values.set_index(DEVICE)
        for dev in dev_cat:
            new_val = mlv.loc[dev]['ml_state']
            df.loc[0,dev] = new_val

    df_num = df[dev_num]
    df_cat_bool = df[dev_bool + dev_cat]
    # fill from start to end NaNs with the preceeding correct value
    df_cat_bool = df_cat_bool.ffill()
    df = pd.concat([df[TIME], df_num, df_cat_bool], axis=1)
    return df

def resample_raw(df_raw, df_dev, t_res, most_likely_values=None):
    """
    Resamples a raw representation
    """

    # get dtypes in order for choosing different collision behavior
    dev_dtypes = _infer_types(df_raw)
    if most_likely_values is None:
        from pyadlml.dataset.devices import get_most_likely_value
        most_likely_values = get_most_likely_value(df_dev)
    most_likely_values = most_likely_values.set_index(DEVICE)

    df_raw = df_raw.set_index(TIME)
    resampler = df_raw.resample(t_res, kind='timestamp')

    # first do a forward fill to correctly represent intervals where no observation falls into
    # ffill takes a devices last known value and assigns it to the timeslice
    raw_ff = resampler.ffill()

    # then for intervals where multiple sensors trigger choose the most prevalent for categorical and
    # boolean values. For numerical drop the most unlikely one (determined on difference to mean)
    raw_int = resampler.apply(_assign_timeslices, t_res=t_res, dev=df_dev,
                              dev_dtypes=dev_dtypes, most_likely_values=most_likely_values)

    print()
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
    cats = list(df_subdev[VAL].unique())
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