from pyadlml.dataset._representations.raw import create_raw, resample_raw
from pyadlml.dataset._representations.changepoint import create_changepoint, resample_changepoint
from pyadlml.dataset._representations.lastfired import create_lastfired, resample_last_fired
from pyadlml.dataset._representations.image import create_lagged_raw, create_lagged_lastfired, \
                                            create_lagged_changepoint
from pyadlml.dataset._dataset import label_data
from pyadlml.dataset import ACTIVITY, TIME, DEVICE, VAL, END_TIME, START_TIME
from pyadlml.dataset.devices import device_rep1_2_rep2
import sklearn.preprocessing as preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

ENC_RAW = 'raw'
ENC_LF = 'lastfired'
ENC_CP = 'changepoint'
REPS = [ENC_RAW, ENC_LF, ENC_CP]


from pyadlml.pipeline import XOrYTransformer, XAndYTransformer, YTransformer


class DropTimeIndex(BaseEstimator, TransformerMixin, XOrYTransformer):
    def __init__(self):
        XOrYTransformer.__init__(self)

    def fit(self, X=None, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        drops columns that are not time dependent
        """
        if X is not None:
            X = X.loc[:, X.columns != TIME]
        if y is not None:
            y = y.loc[:, y.columns != TIME]
        return X, y

    def transform(self, A):
        assert A is not None
        return A.loc[:, A.columns != TIME]

class DataFrame2NdArray(BaseEstimator, TransformerMixin, XOrYTransformer):
    def __init__(self):
        XOrYTransformer.__init__(self)

    def fit(self, X=None, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        drops columns that are not time dependent
        """
        if X is not None:
            X = X.values
        if y is not None:
            y = y.values
        return X, y

    def transform(self, X):
        raise ValueError

class TestSubset(TransformerMixin, XOrYTransformer):
    def __init__(self, date_range=[], y=False):
        assert isinstance(date_range, list)
        self.date_range = date_range
        XAndYTransformer.__init__(self)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x_mask = _create_mask(X, self.date_range)
        return X[x_mask]

    def fit_transform(self, X, y=None):
        x_mask = _create_mask(X, self.date_range)
        if y is not None:
            assert len(X) == len(y)
            y_mask = _create_mask(y, self.date_range)
            return X[x_mask], y[y_mask]
        else:
            return X[x_mask]

class TrainSubset(TransformerMixin, XOrYTransformer):
    def __init__(self, date_range=[], y=False):
        assert isinstance(date_range, list)
        self.date_range = date_range
        XAndYTransformer.__init__(self)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x_mask = _create_mask(X, self.date_range)
        return X[x_mask]

    def fit_transform(self, X, y=None):
        x_mask = _create_mask(X, self.date_range)
        if y is not None:
            assert len(X) == len(y)
            y_mask = _create_mask(y, self.date_range)
            return X[x_mask], y[y_mask]
        else:
            return X[x_mask]

def _create_mask(X, date_range):
    x_mask = (X[TIME] == 'false_init')
    for time_pair in date_range:
        x_mask = x_mask | ((time_pair[0] < X[TIME]) & (X[TIME] < time_pair[1]))
    return x_mask

class KeepSubset(TransformerMixin, XAndYTransformer):
    def __init__(self, date_range=[], y=False):
        assert isinstance(date_range, list)
        self.date_range = date_range
        XAndYTransformer.__init__(self)

    def fit(self, X, y):
        return self

    def transform(self, X):
        raise ValueError

    def fit_transform(self, X, y):
        assert len(X) == len(y)
        x_mask = (X[TIME] == 'false_init')
        y_mask = (X[TIME] == 'false_init')
        for time_pair in self.date_range:
            x_mask = x_mask | ((time_pair[0] < X[TIME]) & (X[TIME] < time_pair[1]))
            y_mask = y_mask | ((time_pair[0] < y[TIME]) & (y[TIME] < time_pair[1]))
        return X[x_mask], y[y_mask]

class DropSubset(TransformerMixin, XAndYTransformer):
    def __init__(self, date_range=[], y=False):
        assert isinstance(date_range, list)
        self.date_range = date_range
        XAndYTransformer.__init__(self)

    def fit(self, X, y):
        return self

    def transform(self, X):
        raise ValueError

    def fit_transform(self, X, y):
        assert len(X) == len(y)
        x_mask = (X[TIME] == 'false_init')
        y_mask = (X[TIME] == 'false_init')
        for time_pair in self.date_range:
            x_mask = x_mask | ((time_pair[0] < X[TIME]) & (X[TIME] < time_pair[1]))
            y_mask = y_mask | ((time_pair[0] < y[TIME]) & (y[TIME] < time_pair[1]))
        return X[~x_mask], y[~y_mask]

class BinaryEncoder(BaseEstimator, TransformerMixin):
    """
    Create a sequence of binary state vectors from a device event stream. Read more in the :ref:`User Guide <preprocessing_discretization>`.

    Parameters
    ----------
    encode : {'raw', 'lastfired', 'changepoint'}, default='raw'
        Method used to encode the transformed result.

        raw
            Encode the transformed result with a binary vector
            corresponding to a state in the smart home.
        lastfired
            Encode the transformed result with binary vector. A devices
            field is 1, if the device triggered last. Otherwise the field
            is 0.
        changepoint
            Encode the transformed result with binary vector. A devices
            field is 1, if the device triggers at that time. Otherwise the field
            is 0.
    t_res : str, optional, default=None
        The timeslices resolution for discretizing the event stream. If
        set to None the event stream is not discretized.
    sample_strategy : {'ffill', 'on_time'}, optional, default='ffill'
        Strategy used to assign statevectors to timeslices if
        multiple events fall into the same timeslice.
        ffill
            A timeslice assumes the last known value of the device
        on_time
            A timeslice is assigned the most prominent state of the device
            within the timeslice
        random
            A timeslice is assigned a random state of the device
        .. versionadded:: 0.24

    Attributes
    ----------
    encode : string
        The binary representation for the data.
    t_res : str or None
        Determines the timeslice size.
    sample_strategy : str or None
        The sample strategy.
    data : pd.DataFrame or None
        The fitted data.

    Examples
    --------
    >>> from pyadlml.preprocessing import BinaryEncoder
    >>> enc = BinaryEncoder(encode='raw')
    >>> raw = enc.fit_transform(data.df_devices)
    >>> len(raw)
    10000
    """

    def __init__(self, encode=ENC_RAW, t_res=None, sample_strategy='ffill'):
        assert encode in REPS
        self.encode = encode
        self.data = None
        self.t_res=t_res
        self.sample_strategy=sample_strategy


    def fit(self, df_devs, y=None):
        """
        Fit the estimator.

        Parameters
        ----------
        df_devs : pd.DataFrame, optional
            recorded devices from a dataset. For more information refer to
            :ref:`user guide<device_dataframe>`.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self
        """
        self.data = self._transform(df_devs)
        return self


    def _transform(self, df_devs):
        if self.encode == ENC_RAW:
            data = create_raw(df_devs)
            if self.t_res is not None:
                data = resample_raw(data,
                                     self.t_res,
                                     df_devs,
                                     sample_strat=self.sample_strategy)

        elif self.encode == ENC_CP:
            data = create_changepoint(df_devs)
            if self.t_res is not None:
                data = resample_changepoint(data, self.t_res)

        elif self.encode == ENC_LF:
            data = create_lastfired(df_devs)
            if self.t_res is not None:
                data = resample_last_fired(data, self.t_res)
        else:
            raise ValueError

        data = data.reset_index(drop=False)
        return data


    def fit_transform(self, df_devs, y=None):
        """

        Parameters
        ----------
        df_devs : pd.DataFrame, optional
            recorded devices from a dataset. For more information refer to
            :ref:`user guide<device_dataframe>`.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        """
        self.fit(df_devs)
        return self.data

    def inverse_transform(self, raw):
        """
        Transform discretized data back to original feature space.
        Note that this function does not regenerate the original data
        due to discretization rounding.

        Parameters
        ----------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data in the binned space.

        Returns
        -------
        Xinv : ndarray, dtype={np.float32, np.float64}
            Data in the original feature space.
        """
        raise NotImplementedError

    def transform(self, df_devs=None):
        """
        Discretize the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.

        Returns
        -------
        Xt : {ndarray, sparse matrix}, dtype={np.float32, np.float64}
            Data in the binned space. Will be a sparse matrix if
            `self.encode='onehot'` and ndarray otherwise.
        """
        return self._transform(df_devs)


class LabelEncoder(TransformerMixin, YTransformer):
    """
        wrapper around labelencoder to handle time series data
    """
    def __init__(self, idle=False):
        """
        Parameters
        ----------
        idle : bool, default=False
            todo

        Returns
        -------
        self
        """
        self.df_acts = None
        self.idle = idle
        self._lbl_enc = preprocessing.LabelEncoder()


    def fit(self, df_acts):
        """
        Fit label encoder.

        Parameters
        ----------
        df_acts : pd.DataFrame
            recorded activities from a dataset. Fore more information refer to the
            :ref:`user guide<activity_dataframe>`.

        Returns
        -------
        self : returns an instance of self
        """
        self.df_acts = df_acts
        self._lbl_enc.fit(self.df_acts[ACTIVITY].values)
        return self


    def fit_transform(self, df_acts, X):
        """
        Fit label encoder and return encoded labels.

        Parameters
        ----------
        df_acts : pd.DataFrame, optional
            recorded activities from a dataset. Fore more information refer to the
            :ref:`user guide<activity_dataframe>`.

        Returns
        ------
        df : pd.DataFrame
        """
        self.df_acts = df_acts
        df = label_data(X, self.df_acts, self.idle)
        encoded_labels = self._lbl_enc.fit_transform(df[ACTIVITY].values)
        return pd.DataFrame(data={TIME: df[TIME].values, ACTIVITY: encoded_labels})

    def inverse_transform(self, x, retain_index=False):
        """
        Transform labels back to original encoding.

        Parameters
        ----------
        x : array like or pd.DataFrame or pd.Series
            array of numbers that are transformed to labels
        retain_index : bool, default=False
            TODO

        Returns
        -------
            TODO
        """
        raise NotImplementedError
        #if isinstance(x, np.ndarray):
        #    res = self._lbl_enc.inverse_transform(x)
        #    if retain_index:
        #        return pd.Series(data=res, index=self.df_devs.index[:len(res)])
        #    else:
        #        return res
        #
        #elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        #    tmp_index = x.index
        #    res = self._lbl_enc.inverse_transform(x.values)
        #    return pd.Series(data=res, index=tmp_index)
        #else:
        #    raise ValueError

    def transform(self, X):
        """

        Parameters
        ----------
        X :

        """
        #if isinstance(X, pd.DataFrame) and set(X.columns) == {START_TIME, END_TIME, ACTIVITY}:
        #    df = label_data(X, self.df_acts, self.idle)
        #    encoded_labels = self._lbl_enc.fit_transform(df[ACTIVITY].values)
        #    return pd.DataFrame(index=df[TIME], data=encoded_labels, columns=[ACTIVITY])
        if isinstance(X, pd.DataFrame) and TIME in X.columns:
            df = label_data(X, self.df_acts, self.idle)
            encoded_labels = self._lbl_enc.transform(df[ACTIVITY].values)
            return pd.DataFrame(data={TIME: df[TIME].values, ACTIVITY: encoded_labels})

        # return only the labels for a nd array 
        elif isinstance(X, np.ndarray):
            return self._lbl_enc.transform(X)

        else:
            raise ValueError


class ImageEncoder():
    def __init__(self, rep, window_size, t_res=None, sample_strat='ffill'):
        self.data = None
        self.rep = rep
        self.t_res = t_res
        self.window_size = window_size
        self.sample_strat = sample_strat

    def fit(self, df_devices):
        if self.rep == 'raw':
            self.data = create_lagged_raw(
                df_devices, 
                window_size=self.window_size, 
                t_res=self.t_res, 
                sample_strat=self.sample_strat)

        elif self.rep == 'changepoint':
            self.data = create_lagged_changepoint(
                df_devices, 
                window_size=self.window_size, 
                t_res=self.t_res)

        elif self.rep == 'lastfired':
            return create_lagged_lastfired(
            df_devices, 
            window_size=self.window_size, 
            t_res=self.t_res)

    def fit_transform(self, df_devices):
        self.fit(df_devices)
        return self.data

    def inverse_transform(self, lgd_raw):
        """
        """
        raise NotImplementedError

    def set_params(self, t_res=None, window_size=10, sample_strat='ffill'):
        if t_res is not None:
            self.t_res = t_res
        raise NotImplementedError

    def transform(self, df_devices):
        if self.rep == 'raw':
            return create_lagged_raw(
                df_devices, 
                window_size=self.window_size, 
                t_res=self.t_res, 
                sample_strat=self.sample_strat)
        elif self.rep == 'changepoint':
            return create_lagged_changepoint(
                df_devices, 
                window_size=self.window_size, 
                t_res=self.t_res)
        elif self.rep == 'lastfired':
            return create_lagged_lastfired(
            df_devices, 
            window_size=self.window_size, 
            t_res=self.t_res)
        else:
            raise ValueError


class ImageLabelEncoder():
    """
    wrapper around labelencoder to handle time series data
    """
    def __init__(self, df_devices, window_size, t_res=None, idle=False):
        self.window_size = window_size
        self.t_res = t_res
        self.idle = idle
        self._lbl_enc = preprocessing.LabelEncoder()
        self.df_devices = df_devices
        self.df_index = self._create_index(df_devices, t_res)
        
    def _create_index(self, df_devices, t_res):
        """
        create the dummy dataframe for the index from the devices
        index | val
        """
        df = df_devices.copy()
        df = df.pivot(index=TIME, columns=DEVICE, values=VAL).iloc[:,:1]
        df = df.astype(bool) # just to have a lower memory footprint
        
        # resample with frequency
        resampler = df.resample(t_res, kind='timestamp')
        df_index = resampler.sum()
        df_index.columns = [VAL]
        df_index[VAL] = 1
        return df_index

        
    def fit(self, df_activities):
        df = label_data(self.df_index, df_activities, self.idle)
        # start where the labeling begins
        df = df.iloc[self.window_size:,:]
        self._lbl_enc.fit(df[ACTIVITY].values)

    def fit_transform(self, df_activities):
        df = label_data(self.df_index, df_activities, self.idle)
        df = df.iloc[self.window_size:,:]
        encoded_labels = self._lbl_enc.fit_transform(df[ACTIVITY].values)
        return pd.DataFrame(index=df.index, data=encoded_labels, columns=[ACTIVITY])

    def inverse_transform(self, x, retain_index=False):
        """
        Parameters
        ----------
        x: array like or pd.DataFrame or pd.Series
            array of numbers that are transformed to labels
        """
        if isinstance(x, np.ndarray):
            res = self._lbl_enc.inverse_transform(x)
            if retain_index:
                return pd.Series(data=res, index=self.df_index.index[:len(res)])
            else:
                return res
                
        elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series): 
            tmp_index = x.index
            res = self._lbl_enc.inverse_transform(x.values)
            return pd.Series(data=res, index=tmp_index)
        else:
            raise ValueError

    def set_params(self, t_res=None, idle=None, window_size=None):
        if t_res is not None:
            self.t_res = t_res
            self.df_index = self._create_index(self.df_devices, t_res)
        if idle is not None:
            self.idle = idle
        if window_size is not None:
            self.window_size = window_size

    def get_params(self):
        return {'window_size': self.window_size, 'idle': self.idle, 't_res':self.t_res}

    def transform(self, df_activities):
        df = label_data(self.df_index, df_activities, self.idle)
        df = df.iloc[self.window_size:,:]
        encoded_labels = self._lbl_enc.transform(df[ACTIVITY].values)
        return pd.DataFrame(index=df.index, data=encoded_labels, columns=[ACTIVITY])

