from pyadlml.constants import TIME, VALUE
import numpy as np
import pandas as pd

from pyadlml.preprocessing import SequenceSlicer


def get_tds_per_batch(X, window_size, stride, seq_type):
    """ returns the time deltas for each batch

    Parameters
    ----------
    X : pd.DataFrame
        A dataframe where at least one column has the name TIME. Can be either a `StateVector`
        or a `data.df_devices`.
    window_size : int
        The number of datapoints that are grouped together for a sequence
    stride : int
        The amount of steps that sliding window takes

    Returns
    -------
    df : pd.DataFrame
        A dataframe of the format (N) where for each batch the time differences
        are given
    """
    # extract time to index mapping
    time = X[TIME]
    index = X.index.values[:, np.newaxis]

    # print(index.shape)
    # slice and dice data
    ss = SequenceSlicer(rep=seq_type, window_size=window_size, stride=stride)
    tmp, _ = ss.fit_transform(index)

    diffs = []
    for i in range(tmp.shape[0]):
        batch = time[tmp[i].squeeze()]

        # compute td difference to successor
        batch_shifted = batch.shift(-1)
        batch_diff = (batch_shifted - batch)[:-1]

        diffs.append(batch_diff)
    return diffs


def comp_tds_sums(X, window_size, stride, seq_type):
    sums = []
    diffs = get_tds_per_batch(X, window_size, stride, seq_type)
    for d in diffs:
        sums.append(d.sum())
    sums = np.array(sums)
    return sums


def comp_tds_sums_mean(X, window_size, stride, seq_type):
    sums = comp_tds_sums(X, window_size, stride, seq_type)
    return sums.mean()


def comp_tds_sums_median(X, window_size, stride):
    sums = comp_tds_sums(X, window_size, stride)
    return np.median(sums)


def df_density_binned(df, column_str, dt):
    """ Compute the binned density of samples

    Parameters
    ----------
    df : pd.DataFrame
        df of with columns [time, column_str]
    """

    df[TIME] = df[TIME].dt.floor(freq=dt).dt.time

    df[VALUE] = 1
    df = df.groupby([TIME, column_str]).sum().unstack()
    df = df.fillna(0)
    df.columns = df.columns.droplevel(0)

    # Add times bins where no sample hits anything
    range = pd.date_range(start='1/1/2000', end='1/2/2000', freq=dt).time
    df2 = pd.DataFrame(data=0, dtype=float, columns=df.columns, index=[d for d in range[:-1]])
    df2.update(df)
    df2 = df2.reset_index(drop=False).rename(columns={'index': TIME})
    return df2