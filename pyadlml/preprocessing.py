from pyadlml.dataset._representations.raw import create_raw
from pyadlml.dataset._representations.changepoint import create_changepoint
from pyadlml.dataset._representations.lastfired import create_lastfired
from pyadlml.dataset._representations.image import create_lagged_raw, create_lagged_lastfired, \
                                            create_lagged_changepoint
from pyadlml.dataset._dataset import label_data
from pyadlml.dataset import ACTIVITY
from pyadlml.dataset.devices import device_rep1_2_rep2
import sklearn.preprocessing as preprocessing
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np

ENC_RAW = 'raw'
ENC_LF = 'lastfired'
ENC_CP = 'changepoint'
REPS = [ENC_RAW, ENC_LF, ENC_CP]


class DiscreteEncoder(BaseEstimator):
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
    t_res : str or None, default=None
        The timeslices resolution for discretizing the event stream. If
        set to None the event stream is not discretized.

    sample_strat : {'ffill', 'on_time'}, default='ffill'
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
        The representation the event stream is in
    t_res : str or None
    sample_strat : str
    data : pd.DataFrame or None

    Notes
    -----
    Maybe do sth. here

    Examples
    --------
    >>> X = [[-2, 1, -4,   -1],
    ...      [-1, 2, -3, -0.5],
    ...      [ 0, 3, -2,  0.5],
    ...      [ 1, 4, -1,    2]]
    >>> est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    >>> est.fit(X)
    KBinsDiscretizer(...)
    >>> Xt = est.transform(X)
    >>> Xt  # doctest: +SKIP
    array([[ 0., 0., 0., 0.],
           [ 1., 1., 1., 0.],
           [ 2., 2., 2., 1.],
           [ 2., 2., 2., 2.]])
    Sometimes it may be useful to convert the data back into the original
    feature space. The ``inverse_transform`` function converts the binned
    data into the original feature space. Each value will be equal to the mean
    of the two bin edges.
    >>> est.bin_edges_[0]
    array([-2., -1.,  0.,  1.])
    >>> est.inverse_transform(Xt)
    array([[-1.5,  1.5, -3.5, -0.5],
           [-0.5,  2.5, -2.5, -0.5],
           [ 0.5,  3.5, -1.5,  0.5],
           [ 0.5,  3.5, -1.5,  1.5]])
    """

    def __init__(self, encode=ENC_RAW, t_res=None, sample_strat='ffill'):
        assert encode in REPS
        self.encode = encode
        self.t_res = t_res
        self.sample_strat = sample_strat
        self.data = None

    def fit(self, df_devices, y=None):
        """
        Fit the estimator.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be discretized.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        Returns
        -------
        self
        """
        if self.encode == ENC_RAW:
            self.data = create_raw(
                df_devices,
                t_res=self.t_res,
                sample_strat=self.sample_strat
            )
        elif self.encode == ENC_CP:
            self.data = create_changepoint(
                df_devices,
                t_res=self.t_res
            )
        elif self.encode == ENC_LF:
            self.data = create_lastfired(
                df_devices,
                t_res=self.t_res
            )
        else:
            raise ValueError
        return self


    def fit_transform(self, df_devices):
        self.fit(df_devices)
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

    def transform(self, df_devices=None):
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
        if self.encode == 'raw':
            return create_raw(
                df_devices,
                t_res=self.t_res,
                sample_strat=self.sample_strat
            )
        elif self.encode == 'changepoint':
            return create_changepoint(
                df_devices,
                t_res=self.t_res
            )
        elif self.encode == 'lastfired':
            return create_lastfired(
                df_devices,
                t_res=self.t_res
            )

class LabelEncoder():
    """
        wrapper around labelencoder to handle time series data
    """
    def __init__(self, df_devices, idle=False):
        self.labels = None
        self.df_devices = df_devices
        self.idle = idle
        self._lbl_enc = preprocessing.LabelEncoder()

    def fit(self, df_activities, y=None):
        """ labels data and creates the numeric representations 
        Parameters
        ----------
        df_activities : pd.DataFrame
            Columns are end_time, start_time, activity. 
        """
        df = label_data(self.df_devices, df_activities, self.idle)
        self._lbl_enc.fit(df[ACTIVITY].values)

    def fit_transform(self, df_activities):
        df = label_data(self.df_devices, df_activities, self.idle)
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
                return pd.Series(data=res, index=self.df_devices.index[:len(res)])
            else:
                return res
                
        elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series): 
            tmp_index = x.index
            res = self._lbl_enc.inverse_transform(x.values)
            return pd.Series(data=res, index=tmp_index)
        else:
            raise ValueError

    def set_params(self, t_res=None, sample_strat=None):
        if t_res is not None:
            self.t_res = t_res
        if sample_strat is not None:
            self.sample_strat = sample_strat

    def get_params(self):
        return self.labels, self.idle, self.df_devices

    def transform(self, x):
        """
        
        """
        # if the input is a dataframe of activities, than fit and
        # transform the data accordingly 
        col_names = {'start_time', 'end_time', ACTIVITY}
        if isinstance(x, pd.DataFrame) and set(x.columns) == col_names:
            df = label_data(self.df_devices, x, self.idle)
            encoded_labels = self._lbl_enc.transform(df[ACTIVITY].values)
            return pd.DataFrame(index=df.index, data=encoded_labels, columns=[ACTIVITY])
        
        # return only the labels for a nd array 
        elif isinstance(x , np.ndarray):
            return self._lbl_enc.transform(x)
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
        df = df.pivot(index='time', columns='device', values='val').iloc[:,:1]
        df = df.astype(bool) # just to have a lower memory footprint
        
        # resample with frequency
        resampler = df.resample(t_res, kind='timestamp')
        df_index = resampler.sum()
        df_index.columns = ['val']
        df_index['val'] = 1
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

