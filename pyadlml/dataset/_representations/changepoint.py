import pandas as pd
from pyadlml.constants import TIME, DEVICE, VALUE
import dask.dataframe as dd
from pyadlml.dataset.util import memory_usage

def create_changepoint(df_devs, n_jobs=None, no_compute=False):
    """ Create binary vectors and set all changepoints to true

    Parameters
    ----------
    df_devs : pd.DataFrame
        A device dataframe, refer to guide
    
    n_jobs : None
        The number of jobs

    Returns
    -------
    df : pd.DataFrame
    """

    if n_jobs is not None:
        n = 100 # MB
        ddf = dd.from_pandas(df_devs, npartitions=n_jobs)
        #ddf.repartition(npartitions=int(1+memory_usage(ddf, 'MB').compute() // n))
        ddf[VALUE] = True
        ddf = ddf.pivot_table(index=TIME, columns=DEVICE, values=VALUE)\
            .fillna(False)\
            .astype(int)\
            .reset_index()
        if no_compute:
            return ddf
        else:
            return ddf.compute()
    else:
        df = df_devs.copy()
        df[VALUE] = True
        df = df.pivot(index=TIME, columns=DEVICE, values=VALUE)\
            .fillna(False)\
            .astype(int)\
            .reset_index()
        return df


def resample_changepoint(df_devs: pd.DataFrame, dt:str, n_jobs=None) -> pd.DataFrame:
    """
    Resamples the changepoint representation with a given resolution

    Parameters
    ----------
    cp : pd.DataFrame
        A device dataframe in changepoint representation [TIME, dev1, dev2, ..., devn]

    dt : str
        

    Returns
    -------
    pd.DataFrame
        Resampled dataframe in changepoint representation
    """
    if n_jobs is not None:
        ddf = create_changepoint(df_devs, n_jobs=n_jobs, no_compute=True)
        ddf = ddf.sort_values(by=TIME).set_index(TIME)
        ddf = ddf.resample(dt).count()
        ddf[ddf > 1] = 1
        df = ddf.reset_index().compute()
        return df
    else:
        df = create_changepoint(df_devs)
        df = df.sort_values(by=TIME).set_index(TIME)
        df = df.resample(dt, kind='timestamp').count()
        df[df > 1] = 1
        return df.reset_index()
