import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from pyadlml.dataset import TIME


def _tres_2_pd_res(res):
    if _is_hour(res) or _is_sec(res):
        return res
    if _is_min(res):
        return res[:-1] + 'min'


def _tres_2_discrete(resolution):
    """ create resolution for one day
    """
    res = []
    range = pd.date_range(start='1/1/2000', end='1/2/2000', freq=resolution).time
    range = [str(d) for d in range]
    for t1, t2 in zip(range[:-1], range[1:]):
        res.append(t1 + '-' + t2)
    return res


def _tres_2_vals(resolution):
    """ create resolution for one day
    """
    ser = pd.date_range(start='1/1/2000', end='1/2/2000', freq=resolution).to_series()
    if _is_min(resolution):
        return ser.dt.floor(resolution).dt.minute
    if _is_hour(resolution): return ser.dt.floor(resolution).dt.hour
    if _is_sec(resolution):
        return ser.dt.floor(resolution).dt.sec


def _is_hour(res) -> bool:
    return res[-1:] == 'h'


def _is_min(res) -> bool:
    return res[-1:] == 'm'


def _is_sec(res) -> bool:
    return res[-1:] == 's'

def _valid_tres(t_res):
    val = int(t_res[:-1])
    res = t_res[-1:]
    assert t_res[-1:] in ['h','m', 's']
    assert (val > 0 and val <=12 and  res == 'h') \
        or (val > 0 and val <= 60 and res == 'm') \
        or (val > 0 and val <= 60 and res == 's')
    return True


def _time2intervall(ts, t_res='30m'):
    """
    rounds to the next lower min bin or hour bin """
    ts_ceil = ts.ceil(freq=t_res).time()
    ts_floor = ts.floor(freq=t_res).time()
    return str(ts_floor) + '-' + str(ts_ceil)


def extract_time_difference(df, normalize=False, inplace=True):

    return df

class DayOfWeekExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, one_hot_encoding=False, inplace=False):
        self.one_hot_encoding = one_hot_encoding
        self.inplace = inplace

    def fit(self, X):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        """
        Appends seven columns one-hot encoded for week days to a dataframe based
        on the dataframes timestamps.

        Parameters
        ----------
        df : pd.DataFrame
            A device dataframe with a column named 'time' containing timestamps.
        one_hot_encoding : bool, optional, default=False
            Determines
        inplace : bool, default=True
            Determines whether to append the one-hot encoded time_bins as columns
            to the existing dataframe or return only the one-hot encoding.

        Returns
        ----------
        df : pd.DataFrame
        """
        WD_COL = 'weekday'

        df = X.copy()
        df[WD_COL] = df[TIME].dt.day_name()
        if not self.one_hot_encoding:
            if self.inplace:
                return df
            else:
                return df[WD_COL]

        one_hot = pd.get_dummies(df[WD_COL])
        df = df.join(one_hot, on=df.index)

        # add weekdays that didn't occur in the column
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for wd in weekdays:
            if wd not in df.columns:
                df[wd] = False
        del(df[WD_COL])
        if self.inplace:
            return df
        else:
            return df[weekdays]

class TimeBinExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, resolution='2h', one_hot_encoding=False, inplace=False):
        assert resolution[-1:] in ['m', 's', 'h']
        self.resolution = resolution
        self.inplace = inplace
        self.one_hot_encoding=one_hot_encoding

    def fit(self, X):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        """
        Create one-hot encoding for times of the day

        Parameters
        ----------
        df: pd.DataFrame
            A device dataframe. The dataframe has to include a column with
            the name 'time' containing the timestamps of the representation.
        resolution: str
            The frequency that the day is divided. Different resolutions are
            accepted, like minutes '[x]m' seconds '[x]s] or hours '[x]h'
        inplace : bool, default=True
            Determines whether to append the one-hot encoded time_bins as columns
            to the existing dataframe or return only the one-hot encoding.

        Returns
        -------
        df : pd.DataFrame
            One-hot encoding of the devices.
        """
        TIME_BINS = 'time_bins'
        df = X.copy()
        df[TIME_BINS] = df[TIME].apply(_time2intervall, args=[self.resolution])

        if not self.one_hot_encoding:
            if self.inplace:
                return df
            else:
                return df[TIME_BINS]

        one_hot = pd.get_dummies(df[TIME_BINS]).astype(bool)
        df = df.join(one_hot, on=df.index)
        del(df[TIME_BINS])

        # add columns that don't exist in the dataset
        cols = _tres_2_discrete(self.resolution)
        for v in cols:
            if v not in df.columns:
                df[v] = False

        if self.inplace:
            return df
        else:
            return df[cols]

class TimeDifferenceExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, normalize=False, inplace=False):
        self.normalize = normalize
        self.inplace = inplace

    def fit(self, X):
        self.high_ = None
        self.low_ = None
        return self


    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        """
        Adds one column with time difference between two succeeding rows.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe containing a column named 'time' including valid pandas timestamps.
        normalize : bool, optional, default=False
            Whether to normalize the time differences to the interval [0,1]
        inplace : bool, optional , default=True
            Determines whether to add the column to and return the existing dataframe or
            return a dataframe containing only the time differences.
        Returns
        -------
        df : pd.DataFrame
        """
        self.fit(X)
        return tmp
