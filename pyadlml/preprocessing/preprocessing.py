import pandas as pd
import numpy as np
from copy import copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


from sklearn.utils.metaestimators import _safe_split
import sklearn.preprocessing as preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from pyadlml.dataset._representations.state import create_state, resample_state
from pyadlml.dataset._representations.changepoint import create_changepoint, resample_changepoint
from pyadlml.dataset._representations.lastfired import create_lastfired, resample_last_fired
from pyadlml.dataset._core.acts_and_devs import label_data, label_data2
from pyadlml.constants import ACTIVITY, TIME, DEVICE, VALUE, END_TIME, START_TIME, \
    ENC_STATE, ENC_LF, ENC_CP, REPS
from pyadlml.dataset._core.devices import create_device_info_dict
from pyadlml.pipeline import XOrYTransformer, XAndYTransformer, YTransformer
from pyadlml.constants import OTHER
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from plotly import graph_objects as go
from pyadlml.constants import *
import matplotlib.pyplot as plt

__all__ = []


class Df2Numpy(TransformerMixin, XOrYTransformer):
    """ Transforms dataframes to numpy arrays by respecting
    the column order
    """
    _x_columns = []
    _y_columns = []

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self._x_columns = X.columns
        if y is not None:
            self._y_columns = y.columns

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y):
        # reorder columns
        X = X[self._x_columns]
        y = y[self._y_columns]

        #
        X_np = X.values
        y_np = y.values
        return X_np, y_np


class DfCaster(TransformerMixin, XOrYTransformer):
    """ Transforms dataframes to torch tensors by respecting
    the column order
    """
    _x_columns = []
    _y_columns = []
    Df2Numpy = 'df->np'
    Df2Torch = 'df->torch'

    def __init__(self, x_conv, y_conv):
        self.y_conv = y_conv
        self.x_conv = x_conv

    def fit(self, X, y=None):
        assert self.y_conv in [self.Df2Numpy, self.Df2Torch]
        assert self.x_conv in [self.Df2Numpy, self.Df2Torch]

        self._x_columns = X.columns
        if y is not None:
            self._y_columns = y.columns

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X : pd
        """

        # reorder columns
        X = X[self._x_columns]

        # cast to respective format
        if self.x_conv == self.Df2Torch:
            import torch
            X_prime = torch.tensor(X.values, dtype=torch.float32)
        elif self.x_conv == self.Df2Numpy:
            X_prime = X.values.astype('float32')

        if y is not None:
            y = y[self._y_columns]
            if self.y_conv == self.Df2Torch:
                import torch
                y_prime = torch.tensor(y.values, dtype=torch.int64)
            elif self.y_conv == self.Df2Numpy:
                # ensure that y has dim (N,) and is of correct type
                y_prime = y.values.squeeze().astype('int64')
            return X_prime, y_prime
        else:
            return X_prime


class Df2Torch(TransformerMixin, XOrYTransformer):
    """ Transforms dataframes to torch tensors by respecting
    the column order
    """
    _x_columns = []
    _y_columns = []

    def __init__(self, only_y=False, only_X=False):
        self.only_y = only_y
        self.only_X = only_X

    def fit(self, X, y=None):
        self._x_columns = X.columns
        if y is not None:
            self._y_columns = y.columns

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y):
        """
        Parameters
        ----------
        X : pd
        """
        import torch

        # reorder columns
        X = X[self._x_columns]
        y = y[self._y_columns]

        # cast to tensors
        y_tensor = torch.tensor(y.values, dtype=torch.int64)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)

        return X_tensor, y_tensor


class Identity(TransformerMixin, XOrYTransformer):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    @XOrYTransformer.x_or_y_transform
    def transform(self, X=None, y=None):
        return X, y


class RandomUnderSampler(TransformerMixin, XAndYTransformer):

    def __init__(self, sampling_strategy):
        self.sampling_strategy = sampling_strategy

    def fit(self, X, y, *args):
        from imblearn.under_sampling import RandomUnderSampler as ImbRUS
        self.rus_ = ImbRUS(sampling_strategy=self.sampling_strategy)
        return self

    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y):
        return self.rus_.fit_resample(X, y)


class DropNanRows(XOrYTransformer):
    """
    Drops all rows where the label or the data has a nan-value
    """

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """
        If X is given transform y
        """
        if y is not None:
            if isinstance(y, pd.DataFrame):
                nan_mask = y[ACTIVITY].isna().values
                return X[~nan_mask], y[~nan_mask]

            elif isinstance(y, np.ndarray):
                raise NotImplementedError
        else:
            nan_mask = X.isna().values
            return X[~nan_mask]


class DropDuplicates(TransformerMixin, XOrYTransformer):
    def __init__(self, ignore_columns=[], merge_on='time'):
        BaseEstimator.__init__(self)
        XOrYTransformer.__init__(self)

        self.ignore_columns = [ignore_columns] if not isinstance(
            ignore_columns, list) else ignore_columns
        self.merge_on = merge_on

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)

    @XOrYTransformer.x_or_y_transform
    def transform(self, X=None, y=None):
        """  Depending if X or y is given drop either all Nans in y or all nans in X or
        all Nans in the union of X and y.

        """
        if y is not None and X is not None:
            assert isinstance(y, pd.DataFrame)
            tmp = X.copy().reset_index(drop=True)
            y = y.copy().reset_index(drop=True)

            y_cols = y.columns.to_list()
            if len(y.columns) > 1:
                y_cols.remove(self.merge_on)
                tmp = tmp.merge(y, on=self.merge_on)
            else:
                tmp[y_cols] = y
            comp_cols = list(set(tmp.columns) - set(self.ignore_columns))
            dup_mask = tmp[comp_cols].duplicated()

            y = y[~dup_mask].reset_index(drop=True)
            X = X[~dup_mask].reset_index(drop=True)

        if y is not None and X is None:
            y = y.drop_duplicates()
        if X is not None and y is None:
            X = X.drop_duplicates()
        return X, y


class FinalTimeTransformer():
    pass


class DropTimeIndex(TransformerMixin, XOrYTransformer, FinalTimeTransformer):
    def __init__(self, only_y=False):
        XOrYTransformer.__init__(self)
        self.only_y = only_y

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def fit(self, X=None, y=None):
        return self

    def fit_transform(self, X=None, y=None, **fit_params):
        """ Drops columns that are not time dependent
        """
        return self.transform(X, y)

    @XOrYTransformer.x_or_y_transform
    def transform(self, X=None, y=None):
        if X is not None and not self.only_y:
            self.times_ = X[TIME]
            X = X.loc[:, X.columns != TIME]
        if y is not None:
            y = y.loc[:, y.columns != TIME]
        return X, y


class Encoder():
    features_ = None

    def to_dataframe(self, X, onehot=False):
        print()
        raise NotImplementedError


class IndexEncoder():

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def fit(self, X, y=None):
        self.n_features_in_ = 3
        self.lbl_enc = preprocessing.LabelEncoder()
        self.lbl_enc.fit(X[DEVICE])

    def transform(self, X, y=None):
        X[DEVICE] = self.lbl_enc.transform(X[DEVICE])
        return X

    def inverse_transform(self, X):
        return self.lbl_enc.inverse_transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y=None)
        return self.transform(X, y)

    def onehot(self, X: np.ndarray):
        """ Create one-hot encoding for given array of indices with
            correct feature labels assigned
        """
        n_values = np.max(X) + 1
        values = list(X)
        onehot = np.eye(n_values)[values]
        columns = self.inverse_transform(np.arange(n_values))
        onehot = pd.DataFrame(onehot, columns=columns)
        return onehot


class DropColumn():

    def __repr__(self):
        return f"{self.__class__.__name__}(column_name='{self.column_name}')"

    def __init__(self, column_name):
        self.column_name = column_name

    def transform(self, X, y=None):
        X = X.drop(columns=self.column_name)
        return X

    def fit_transform(self, X, y=None):
        return self.transform(X, y)


class Event2Vec(Encoder, BaseEstimator, TransformerMixin):
    """
    Create a sequence of vectors from a device event stream.
    Read more in the :ref:`User Guide <preprocessing_discretization>`.

    Parameters
    ----------
    encode : {'state', 'changepoint', 'last_fired'}, default='raw'
        Determines the state-vectors encoding.

        state 
            Encode the event stream as vector, where each field
            represents a device. Each entry represents the current
            state that the device is in.

        changepoint
            Encode the event stream as one-hot binary vector, where
            each field represents a device. A one indicates the
            device that produced the event.

        last_fired
            Encode the event stream as one-hot binary vector,
            where each field represents a device. The device that produced
            the last event is indicated with one.

    dt : str, optional, default=None
        The timeslices resolution for discretizing the event stream. If
        set to None the event stream is not discretized.
    
    out_names: 

    Attributes
    ----------
    encode : string
        The encoding for the vectors
    dt : str or None
        If not None, determines the binsize the event stream is discretized into
    data_info_ : dict or None
        Dictionary containing further device information such as most likely value
        or device datatype. This is used to speed up internal computation.
    classes_ : list
        A list of all devices. The lists order corresponds to the device order of the
        transformed state-vectors.

    Examples
    --------
    >>> from pyadlml.dataset import fetch_amsterdam
    >>> data = fetch_amsterdam()
    >>> from pyadlml.preprocessing import Event2Vec
    >>> e2v = Event2Vec(encode='state')
    >>> X = e2v.fit_transform(data['devices'])
    >>> e2v_p = Event2Vec(encode='changepoint+raw', dt='6s')
    >>> X_resampled = e2v_p.fit_transform(data['devices])

    """

    def __init__(self, encode='state', dt=None, out_features=[], n_jobs=None):
        super().__init__()
        self.encode = encode
        self.dt = dt
        self.out_features = out_features
        self.n_jobs = n_jobs

    @classmethod
    def _is_valid_encoding(cls, encoding):
        from itertools import chain, permutations

        def helper(iterable):
            s = list(iterable)
            return chain.from_iterable(permutations(s, r) for r in range(len(s)+1))
        # generate all possible valid combinations
        valid_combos = ['+'.join(combo) for combo in helper(REPS)]
        return encoding in valid_combos

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
        assert Event2Vec._is_valid_encoding(self.encode)

        # Create dict off all input features with mean for numerical
        # and most common value for binary or categorical features
        self.data_info_ = create_device_info_dict(df_devs)
        self.n_features_in_ = 3
        self.features_in_ = [TIME, DEVICE, VALUE]
        if self.out_features:
            feature_names_out = self.out_features
            del (self.out_features)
        else:
            feature_names_out = list(self.data_info_.keys())

        feature_names_out.sort()
        self.feature_names_out_ = [TIME] + feature_names_out

        self.n_features_out = len(self.feature_names_out_)

        return self

    def transform(self, df_devs=None, y=None, initial_states={}):
        """
        Discretize the data.

        Parameters
        ----------
        df_devs : array-like of shape (n_samples, n_features)
            The device dataframe to be transformed.
        y : None
            todo copy standard scipy sentence
        initial_states : dict
            A dictionary with device state mapping containing information about
            the last state the device held before the first event. If no dictionary 
            is given the states are infered by most likely values for 
            categorical and boolean devices.

        Returns
        -------
        Xt : {ndarray, sparse matrix}, dtype={np.float32, np.float64}
            Data in the binned space. Will be a sparse matrix if
            `self.encode='onehot'` and ndarray otherwise.
        """
        PRAEFIX_LF = 'lf_'
        PRAEFIX_CP = 'cp_'

        df_lst = []
        iters = self.encode.split('+')
        df_devs[DEVICE] = df_devs[DEVICE].astype('category')
        for enc in iters:
            if enc == ENC_STATE:
                if self.dt is None:
                    data = create_state(df_devs, self.data_info_,
                                    dev_pre_values=initial_states)
                else:
                    data = resample_state(
                            df_dev=df_devs, 
                            dt=self.dt,
                            most_likely_values=self.data_info_
                    )

                # convert boolean data into integers (1,0)
                dev_bool = [dev for dev in self.data_info_.keys() if self.data_info_[
                    dev]['dtype'] == BOOL]
                data[dev_bool] = data[dev_bool].applymap(
                    lambda x: 1 if x == True else 0)

            elif enc == ENC_CP:
                if self.dt is None:
                    data = create_changepoint(df_devs, self.n_jobs)
                else:
                    data = resample_changepoint(df_devs, self.dt, self.n_jobs)

                # set values that are missing in transform but were present when fitting to 0
                dev_diff = set(self.feature_names_out_) - set(data.columns)
                if len(dev_diff) > 0:
                    for dev in dev_diff:
                        data[dev] = 0

                # add prefix to make column names unique
                if len(iters) > 1:
                    data.columns = [
                        TIME] + list(map(PRAEFIX_CP.__add__, data.columns[1:]))

            elif enc == ENC_LF:
                if self.dt is None:
                    data = create_lastfired(df_devs)
                else:
                    data = resample_last_fired(df_devs, self.dt, self.n_jobs)

                # set values that are missing in transform but were present when fitting to 0
                dev_diff = set(self.feature_names_out_) - set(data.columns)
                if len(dev_diff) > 0:
                    for dev in dev_diff:
                        data[dev] = 0

                # add prefix to make column names unique
                if len(iters) > 1:
                    data.columns = [
                        TIME] + list(map(PRAEFIX_LF.__add__, data.columns[1:]))

            else:
                raise ValueError
            data = data.set_index(TIME)
            df_lst.append(data)

        df_res = pd.concat(df_lst, axis=1).reset_index()
        # Ensure to always return feature columns in order
        return df_res[self.feature_names_out_]

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
        return self.transform(df_devs)


class LabelMatcher(BaseEstimator, TransformerMixin, YTransformer):
    """
    Labels a dataframe with corresponding activities

    Attributes
    ----------
    other : bool, default=False
        If true items that are not

    """

    def __init__(self, other: bool = False, encode_labels=False, use_dask=False, classes=[]):
        """
        Initialize the object of the class with the given parameters.

        Parameters
        ----------
        other : bool, optional
            A boolean parameter to indicate whether to include the 'other' activity in the processing,
            the functionality depends on the specifics of the use case (default is False).

        encode_labels : bool, optional
            A flag to specify if labels are encoded or not. If true, labels will be encoded
            as integers based on classes defined. (default is False).

        use_dask : bool, optional
            A flag to indicate whether to use Dask for parallel computations. If true, computations 
            will be carried out in parallel to improve efficiency (default is False).

        classes : list of str, optional
            A list of predefined class labels for the data. This is particularly useful in scenarios 
            such as cross-validation where some classes may not be present in the training data, 
            but may exist in the validation data (default is []).

        Returns
        -------
        None
        """
        self.other = other
        self.use_dask = use_dask
        self.encode_labels = encode_labels
        self.classes = classes

    def fit(self, y, X):
        """
        Fit label encoder.

        Parameters
        ----------
        y : pd.DataFrame
            recorded activities from a dataset. Fore more information refer to the
            :ref:`user guide<activity_dataframe>`.

        Returns
        -------
        self : returns an instance of self
        """
        self.df_acts_ = y
        if self.classes:
            self.classes_ = self.classes
            del (self.classes)
        else:
            self.classes_ = list(y[ACTIVITY].unique())

        if self.other and OTHER not in self.classes_:
            self.classes_.append(OTHER)
        self.classes_.sort()

        return self

    def class2int_(self, x=None):
        f = {v: k for k, v in enumerate(self.classes_)}.get
        if x is None:
            return f
        else:
            return f(x)

    def int2class_(self, x=None):
        f = {k: v for k, v in enumerate(self.classes_)}.get
        if x is None:
            return f
        else:
            return f(x)

    def fit_transform(self, y, X):
        """
        Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : pd.DataFrame
            An activity dataframe recorded activities from a dataset. Fore more information refer to the
            :ref:`user guide<activity_dataframe>`.
        X : pd.DataFrame
            A dataframe containing a column with name 'time' of timestamps.

        Returns
        ------
        df : pd.DataFrame
            The X dataframe with an additional column containing the corresponding activity labels
        """
        self.fit(y, X)
        return self.transform(y, X)

    def _is_iterable(self, z):
        try:
            iter(z)
            return True
        except:
            return False

    def _wrap_iterable(self, lst, iter):
        """ list 
        """
        if isinstance(iter, np.ndarray):
            return np.array(lst)
        elif isinstance(iter, list):
            return lst
        elif isinstance(iter, list):
            return lst
        elif isinstance(iter, pd.Series):
            tmp = pd.Series(lst)
            tmp.index = iter.index
            return tmp
        else:
            raise NotImplementedError

    def transform(self, y: pd.DataFrame, X: pd.DataFrame = None) -> pd.DataFrame:
        """

        Parameters
        ----------
        y : pd.DataFrame
            An activity dataframe
        X : pd.DataFrame
            A device dataframe

        Returns
        -------
        pd.DataFrame
            A dataframe with column TIME, ACTIVITY matchin the rows of X
            TODO asdf
        """
        # normal case where X an y are provided
        n_jobs = 1000 if self.use_dask else 1
        df = label_data2(X, y, self.other)[
            [TIME, ACTIVITY]].copy()
        if self.encode_labels:
            df[ACTIVITY] = df[ACTIVITY].map(self.class2int_())
        return df


class DropDevices(BaseEstimator, XAndYTransformer, TransformerMixin):
    def __init__(self, devices=[]):
        self.devices = devices

    def fit_transform(self, X, y):
        return self.transform(X, y)

    def transform(self, X, y):
        idxs = X[X[DEVICE].isin(self.devices)].index.values
        return X[idxs], y[idxs]


class KeepOnlyDevices(BaseEstimator, XAndYTransformer, TransformerMixin):
    def __init__(self, devices=[], ignore_y=False):
        self.devices = devices
        self.ignore_y = ignore_y

    def fit_transform(self, X, y):
        return self.transform(X, y)

    def transform(self, X, y):
        idxs = X[X[DEVICE].isin(self.devices)].index.values
        y = y[idxs] if not self.ignore_y else y
        return X.loc[idxs, :], y


class Timestamp2Seqtime(BaseEstimator, TransformerMixin, XOrYTransformer, FinalTimeTransformer):
    def __init__(self, dt='s'):
        super().__init__()
        self.dt = dt

    @XOrYTransformer.x_or_y_transform
    def fit_transform(self, X, y=None):
        X, y = self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X, y=None):
        assert self.dt in ['ms', 's', 'm', 'h']
        return X, y

    def transform(self, X, y=None):
        """ Change every timestamp into unit i.e. seconds relative
            to the first timestamp in the sequence.
        """
        self.time = X[TIME]
        X[TIME] -= X[TIME].iloc[0]
        if self.dt == 'ms':
            X[TIME] = X[TIME]/pd.Timedelta('1milli')
        elif self.dt == 's':
            X[TIME] = X[TIME]/pd.Timedelta('1sec')
        elif self.dt == 'm' or self.dt == 'min':
            X[TIME] = X[TIME]/pd.Timedelta('1min')
        elif self.dt == 'h':
            X[TIME] = X[TIME]/pd.Timedelta('1hr')
        return X, y


class SkTime(BaseEstimator, XAndYTransformer):

    def __init__(self, rep='nested_univ'):
        self.rep = rep

    def fit(self, X, y):
        assert self.rep in ['nested_univ', 'numpy3d']

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y):
        from pyadlml.dataset.util import to_sktime2
        Xt, yt = to_sktime2(X, y, return_X_y=True)
        return Xt, yt


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, inplace=True, as_bool=False):
        self.column_name = column_name
        self.inplace = inplace
        self.as_bool = as_bool

    def fit(self, X, y=None):
        self.one_hot_encoder_ = SkOneHotEncoder()
        self.one_hot_encoder_.fit(
            X[self.column_name].values.reshape(-1, 1))

        self.categories = self.column_name + ':' + \
            self.one_hot_encoder_.categories_[0]
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        is_df = isinstance(X, pd.DataFrame)
        assert is_df or isinstance(X, pd.Series)

        if is_df:
            ser = X[self.column_name]
        else:
            ser = X

        encoded = self.one_hot_encoder_.transform(
            ser.values.reshape(-1, 1)).toarray()
        df_encoded = pd.DataFrame(encoded, columns=self.categories,
                                  dtype=bool if self.as_bool else int)

        if self.inplace and is_df:
            X = X.drop(columns=self.column_name)
            X = pd.concat([X, df_encoded], axis=1)
        return X


class TimeEncoder(BaseEstimator, TransformerMixin):

    @classmethod
    def period_length(cls, res):
        return int(pd.Timedelta('24h')/pd.Timedelta(res))

    @classmethod
    def time2res(cls, Xt, res):
        if isinstance(Xt, pd.DataFrame):
            ser = Xt[TIME]
        else:
            ser = Xt
        return (ser - ser.dt.floor('D'))/pd.Timedelta(res)


class SineCosEncoder(TimeEncoder):
    def __init__(self, res='100ms'):
        super().__init__()
        self.res = res

    def fit(self, X=None, y=None):
        self.res = pd.Timedelta(self.res)
        self.lmbd_ = int(pd.Timedelta('24h')/self.res)
        # Calc angular frequency
        self.w_ = (2*np.pi)/self.lmbd_
        return self

    def transform(self, X, y=None):
        assert TIME in X.columns, "The DataFrame must include a 'time' column"

        # Create a copy of the DataFrame to avoid changes to the original one
        Xt = X.copy()

        # We assume the time range represents a full day
        Xt['time_in_res'] = (Xt[TIME] - Xt[TIME].dt.floor('D'))/self.res

        # Add sin_emb and cos_emb columns to the DataFrame
        Xt['time_sin'] = np.sin(self.w_*Xt['time_in_res'])
        Xt['time_cos'] = np.cos(self.w_*Xt['time_in_res'])

        # Drop the temporary 'x' column
        Xt = Xt.drop(columns=['time_in_res'])

        return Xt

    def plotly(self, X):
        assert hasattr(self, 'w_'), 'Fit the transformer before plotting.'
        df = self.transform(X)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[TIME], y=df['time_sin'], name='sin'))
        fig.add_trace(go.Scatter(x=df[TIME], y=df['time_cos'], name='cos'))
        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
        return fig

    def plotly_sin_vs_cos(self, X):
        df = self.transform(X)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['sin_emb'].values,
            y=df['cos_emb'].values,
            hovertemplate='Sin: %{x}<br>Cos %{y}<br>Time: %{customdata}',
            mode='markers',
            customdata=df.index.values)
        )
        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        fig.show()


class HalfSineEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, res='100ms'):
        self.res = res

    def fit(self, X=None, y=None):
        self.res = pd.Timedelta(self.res)
        self.lmbd_ = int(pd.Timedelta('24h')/self.res)
        # Calc angular frequency
        self.w_ = (np.pi)/self.lmbd_
        return self

    def transform(self, X, y=None):
        assert 'time' in X.columns, "The DataFrame must include a 'time' column"
        # Create a copy of the DataFrame to avoid changes to the original one
        Xt = X.copy()

        # We assume the time range represents a full day
        Xt['time_in_res'] = (Xt[TIME] - Xt[TIME].dt.floor('D'))/self.res

        # Add sin_emb and cos_emb columns to the DataFrame
        Xt['time_sin'] = np.sin(self.w_*Xt['time_in_res'])

        # Drop the temporary 'x' column
        Xt = Xt.drop(columns=['time_in_res'])

        return Xt

    def plotly(self, X):
        assert hasattr(self, 'w_'), 'Fit the transformer before plotting.'
        df = self.transform(X)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[TIME], y=df['time_sin'], name='sin'))
        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
        return fig


class PositionalEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, d_dim, max_period=1e4, inplace=True):
        self.d_dim = d_dim
        self.max_period = max_period
        self.inplace = inplace

    def fit(self, X=None, y=None):
        self.min_freq = 1/self.max_period
        return self

    def get_angular_freqs(self):
        i = np.arange(self.d_dim)//2
        ws = np.power(self.min_freq, (2*i/self.d_dim))
        return ws

    def get_periods(self):
        return 2*np.pi*self.get_angular_freqs()

    def plot_angular_freq(self, scale='linear'):
        assert scale in ['linear', 'log']

        ws = self.get_angular_freqs()
        fig, ax = plt.subplots()
        ax.set_yscale(scale)
        ax.plot(np.arange(len(ws)), ws,
                label=f'min{min(ws):.3f}, max{max(ws):.3f}')
        ax.set_xlabel('d - dimension')
        ax.set_ylabel('$\omega(d)$')
        ax.grid(True)
        ax.legend()
        return fig

    def plot_waves(self, seq_length):
        freqs = self.get_angular_freqs()

        n_dim = len(freqs)
        # Create a figure and an axis
        fig, ax = plt.subplots(figsize=(10, 10))

        # Loop over the range of dimensions
        x = np.linspace(0, seq_length, 1000)
        for d in range(n_dim):
            oscil = np.sin if d % 2 == 0 else np.cos
            ax.plot(x, oscil(freqs[d]*x)+2*d, label=f'pe dim={d}')
        # Set the axis scales to be equal
        # ax.set_aspect('equal', 'box')

        # Show the legend
        ax.legend()
        return fig

    def transform(self, X, y=None, inplace=None):
        """ 

        sin(w*t) 
        """
        inplace = inplace if inplace is not None else self.inplace

        freqs = self.get_angular_freqs()
        if isinstance(X, pd.DataFrame):
            x = X[TIME].values
        else:
            x = X
        assert isinstance(x, np.ndarray), 'DEBUG'

        xt = self._pos_enc(freqs, x)
        if inplace:
            xt = pd.DataFrame(data=xt, columns=[
                              f'pe_dim_{i}' for i in range(0, self.d_dim)])
            X = pd.concat([X, xt], axis=1)
        else:
            X = xt
        return X

    def _pos_enc(self, freqs, pos):
        pos_enc = pos.reshape(-1, 1)*freqs.reshape(1, -1)
        pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
        pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
        return pos_enc

    def plot_pe_mat(self, X):
        ### Plotting ####

        mat = self.transform(X, inplace=False)
        fig, ax = plt.subplots()
        im = ax.pcolormesh(mat.T, cmap='RdBu')
        ax.set_ylabel('Depth')
        ax.set_ylim((0, self.d_dim))
        ax.set_xlabel('Position')
        ax.set_title("PE matrix heat map")
        fig.colorbar(im)
        return fig

    def plot_dotproduct_similarity(self, X, pos):
        # let's choose the first vector

        pe = self.transform(X, inplace=False)
        first_vector = pe[pos, :]

        # compute the dot products
        dot_products = np.dot(pe, first_vector)

        # visualize with matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dot_products)
        ax.set_title(f'Dot products of the {pos} vector with rest')
        ax.set_xlabel('Vector index')
        ax.set_ylabel('Dot product')
        ax.grid(True)
        return fig


class TimePositionalEncoding(PositionalEncoding):

    def __init__(self, d_dim, max_period=pd.Timedelta('1D'), res=pd.Timedelta('8.64s'), inplace=True):
        super().__init__(d_dim, max_period)  # Add super call to the parent class
        self.res = res
        self.inplace = inplace

    def fit(self, X=None, y=None):
        # Map timestamp values to a sequence
        self.res = pd.Timedelta(self.res)
        self.max_period = self._td2num(self.max_period)
        self.min_freq = 1/self.max_period
        return self

    def _td2num(self, x):
        if isinstance(x, str):
            x = pd.Timedelta(x)

        x = x/pd.Timedelta(self.res)
        return x

    def _num2td(self, x):
        return pd.Timedelta(self.res)*x

    def transform(self, X, y=None, inplace=None):
        """ 

        sin(w*t) 
        """
        inplace = self.inplace if inplace is None else inplace

        if isinstance(X, pd.DataFrame):
            x = X[TIME]

        assert isinstance(x, pd.Series), 'TimePosEnc: Sth. went wrong'
        td = (x - pd.to_datetime(x.dt.date))
        time_scaled = self._td2num(td).to_numpy()
        freqs = self.get_angular_freqs()

        # calc pos enc (T, d_dim)
        xt = self._pos_enc(freqs, time_scaled)

        if inplace:
            xt = pd.DataFrame(data=xt, columns=[
                              f'pe_dim_{i}' for i in range(0, self.d_dim)])
            xt = pd.concat([X, xt], axis=1)

        return xt


class CyclicTimePositionalEncoding(TimePositionalEncoding):

    def __init__(self, d_dim, max_period=pd.Timedelta('1D'), res=pd.Timedelta('8.64s'), base=2):
        super().__init__(d_dim, max_period, res)  # Add super call to the parent class
        self.base = base

    def get_angular_freqs(self):
        """period of length 1"""
        i = np.arange(self.d_dim)//2
        lmbd = self.max_period
        f = 1/(lmbd/(np.minimum((self.base**i).astype(int), lmbd)))
        ws = 2*np.pi*f
        return ws

    def get_max_base(self):
        return np.floor(np.log10(self.max_period)/np.log10(self.base)).astype(int)

    def plotly_wave_length_per_dim(self):
        from plotly import graph_objects as go
        fig = go.Figure()
        x = self.get_periods()*self.res
        x += pd.Timestamp('01.01.2000 00:00:00')
        fig.add_trace(go.Scatter(y=np.arange(0, self.d_dim), x=x))
        return fig
