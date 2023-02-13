import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from pyadlml.constants import TIME, DEVICE

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
    assert t_res[-1:] in ['h', 'm', 's']
    assert (val > 0 and val <= 12 and res == 'h') \
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
    """
    Appends seven columns one-hot encoded for week days to a dataframe based
        on the dataframes timestamps.

    Attributes
    ----------
    one_hot_encoding : bool, optional, default=False
        Determines

    inplace : bool, default=True
        Determines whether to append the one-hot encoded time_bins as columns
        to the existing dataframe or return only the one-hot encoding.

    """

    WD_COL = 'weekday'
    WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

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
        X : pd.DataFrame
            A device dataframe with a column named 'time' containing timestamps.

        Returns
        ------
        df : pd.DataFrame
        """

        df = X.copy()
        df[self.WD_COL] = df[TIME].dt.day_name()
        if not self.one_hot_encoding:
            if self.inplace:
                return df
            else:
                return df[self.WD_COL]

        one_hot = pd.get_dummies(df[self.WD_COL])
        df = df.join(one_hot, on=df.index)

        # add weekdays that didn't occur in the column
        for wd in self.WEEKDAYS:
            if wd not in df.columns:
                df[wd] = False
        del (df[self.WD_COL])
        if self.inplace:
            return df
        else:
            return df[self.WEEKDAYS]


class TimeBinExtractor(TransformerMixin, BaseEstimator):
    """
    Divides a day into equal-length time bins and assigns time bin as features to
    a row that falls into that bin.

    Attributes
    ----------
    dt: str, default='2h'
        The frequency that the day is divided. Different resolutions are
        accepted, like minutes '[x]m' seconds '[x]s] or hours '[x]h'
    inplace : bool, default=False
        Determines whether to append the one-hot encoded time_bins as columns
        to the existing dataframe or return only the one-hot encoding.

    """
    TIME_BINS = 'time_bins'
    ALLOWED_RESOLUTIONS = ['m', 's', 'h']

    def __init__(self, dt='2h', one_hot_encoding=False, inplace=False):
        self.dt = dt
        self.inplace = inplace
        self.one_hot_encoding = one_hot_encoding

    def fit(self, X, y=None):
        assert self.dt[-1:] in self.ALLOWED_RESOLUTIONS
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """

        Parameters
        ----------
        X: pd.DataFrame
           A device dataframe. The dataframe has to include a column with
           the name 'time' containing the timestamps of the representation.

        Returns
        -------
        df : pd.DataFrame
            One-hot encoding of the devices.
        """
        self.fit(X,y)

        return self.transform(X)

    def transform(self, X, y=None):
        """

        Parameters
        ----------
        X: pd.DataFrame
           A device dataframe. The dataframe has to include a column
           named 'time' containing timestamps of the representation.

        Returns
        -------
        df : pd.DataFrame
            One-hot encoding of the devices.
        """
        assert self.dt[-1:] in self.ALLOWED_RESOLUTIONS

        df = X.copy()
        df[self.TIME_BINS] = df[TIME].apply(_time2intervall, args=[self.dt])

        if not self.one_hot_encoding:
            if self.inplace:
                return df
            else:
                return df[self.TIME_BINS]

        one_hot = pd.get_dummies(df[self.TIME_BINS]).astype(bool)
        df = df.join(one_hot, on=df.index)
        del (df[self.TIME_BINS])

        # add columns that don't exist in the dataset
        cols = _tres_2_discrete(self.dt)
        for v in cols:
            if v not in df.columns:
                df[v] = False

        if self.inplace:
            return df
        else:
            return df[cols]


class EventTimeExtractor(TransformerMixin, BaseEstimator):
    """
    Extracts the time difference to either the next or previous event

    Attributes
    ----------
    direction : str, optional, default='to_predecessor'
        The direction to which the timedifference is calculated.
    normalize : bool, optional, default=False
        Whether to apply min-max scaling to normalize the time differences to the interval [0,1]
    inplace : bool, optional , default=False
        Determines whether to add the column to and return the existing dataframe or
        return a dataframe containing only the time differences.
    unit : one of {None, 'm', 's'}, default=None
        The unit in which the time difference is expressed. When nothing is specified
        transform returns pandas timedeltas otherwise integers
    """
    BACKWARD = 'to_predecessor'
    FORWARD = 'to_successor'
    COL_NAME = 'td'

    def __init__(self, direction='to_predecessor', unit=None, normalize=False, inplace=False):
        self.normalize = normalize
        self.inplace = inplace
        self.direction = direction
        self.unit = unit

    def _compute_td(self, df):
        """
        compute the time difference to successor and predecessor
        """
        df[self.FORWARD] = df[TIME].shift(-1) - df[TIME]
        df[self.BACKWARD] = df[TIME] - df[TIME].shift(1)
        return df

    def fit(self, X: pd.DataFrame, y=None):
        assert TIME in X.columns

        # compute a dataframe with mean time diff for each device
        df = self._compute_td(X.copy())

        # compute average value
        self.td_avg = df[[DEVICE, self.BACKWARD, self.FORWARD]]\
            .iloc[1:-1]\
            .groupby(DEVICE)\
            .mean(numeric_only=False)\
            .reset_index()

        # compute the highest and lowest value
        if self.normalize:
            self.min_ = df[self.direction].max()
            self.max_ = df[self.direction].min()

        return self

    def fit_transform(self, X, y=None, **fit_params):
        assert TIME in X.columns

        self.fit(X)
        return self.transform(X)

    def transform(self, X, y=None):
        """
        Adds one column with time difference between two succeeding rows.

        Parameters
        ----------
        X : pd.DataFrame
            A dataframe containing a column named 'time' including valid pandas timestamps.

        Returns
        -------
        df : pd.DataFrame
        """
        assert TIME in X.columns

        df = self._compute_td(X.copy())

        # impute either last or first value with the average of the value
        if self.direction == self.BACKWARD:
            del(df[self.FORWARD])
            first_device = df.iloc[0][DEVICE]
            df.at[0, self.BACKWARD] = self.td_avg[self.td_avg[DEVICE] == first_device][self.BACKWARD].iloc[0]
            df.rename(columns={self.BACKWARD: self.COL_NAME}, inplace=True)

        elif self.direction == self.FORWARD:
            del(df[self.BACKWARD])
            last_device = df.iloc[-1][DEVICE]
            df.at[0, self.FORWARD] = self.td_avg[self.td_avg[DEVICE] == last_device][self.FORWARD].iloc[0]
            df.rename(columns={self.FORWARD: self.COL_NAME}, inplace=True)
        else:
            raise ValueError

        if self.normalize:
            # apply min max normalization
            df[self.COL_NAME] = (df[self.COL_NAME] - self.min_)/(self.max_ - self.min_)

            # clip values that don't fit
            df[self.COL_NAME] = df[self.COL_NAME].where(df[self.COL_NAME] < 1, 1)
            df[self.COL_NAME] = df[self.COL_NAME].where(df[self.COL_NAME] > 0, 0)

        elif self.unit is not None:
            if self.unit == 's':
                df[self.COL_NAME] = df[self.COL_NAME].dt.seconds
            elif self.unit == 'ms':
                df[self.COL_NAME] = df[self.COL_NAME].dt.microseconds
            elif self.unit == 'h':
                df[self.COL_NAME] = df[self.COL_NAME].dt.hours
            else:
                raise ValueError

        if self.inplace:
            return df
        else:
            return df[self.COL_NAME]