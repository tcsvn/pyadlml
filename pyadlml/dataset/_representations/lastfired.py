import pandas as pd
from pyadlml.constants import TIME, DEVICE, VALUE
from pyadlml.dataset._representations.changepoint import create_changepoint
import dask.dataframe as dd


def create_lastfired(df_devs, n_jobs=None, no_compute=False):
    """ Creates the last fired representation
    """
    return create_changepoint(df_devs, n_jobs=n_jobs, no_compute=no_compute)


def resample_last_fired(df_devs, dt=None, n_jobs=None):
    """

    Parameters
    ----------
    lf : pd.DataFrame
        last fired representation

    """
    use_dask = n_jobs is not None
    df = df_devs.sort_values(by=TIME).copy()
    origin = df.at[0, TIME].floor(freq=dt)

    # Only keep last device to have fired in a bin
    df['bin'] = df.groupby(
        pd.Grouper(key=TIME, freq=dt, origin=origin))\
        .ngroup()

    if use_dask:
        n = 100 # MB
        df = dd.from_pandas(df, npartitions=n_jobs)

    df = df.groupby(['bin', DEVICE], observed=True)\
                    .last()\
                    .reset_index()
    df = df.drop(columns='bin')\
                    .sort_values(by=TIME)\
                    .reset_index(drop=True)\
                    [[TIME, DEVICE, VALUE]]    
    df[VALUE] = True
    df = df.pivot_table(index=TIME, columns=DEVICE, values=VALUE)\
        .fillna(False)\
        .astype(int)\
        .reset_index()
    df = df.set_index(TIME)
    if use_dask:
        df = df.compute()

    df = df.resample(dt).ffill()
    first_dev = df_devs.iloc[0, df_devs.columns.tolist().index(DEVICE)]
    first_dev_col_idx = df.columns.tolist().index(first_dev)
    df.iat[0, first_dev_col_idx] = 1.0
    df = df.fillna(0.0)
    df = df.reset_index()

    return df
