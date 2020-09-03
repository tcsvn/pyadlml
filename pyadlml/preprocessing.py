from pyadlml.dataset.representations.raw import create_raw 
from pyadlml.dataset.representations.changepoint import create_changepoint
from pyadlml.dataset.representations.lastfired import create_lastfired
from pyadlml.dataset.representations.image import create_lagged_raw
from pyadlml.dataset._dataset import label_data
import sklearn.preprocessing as preprocessing
import pandas as pd
import numpy as np

class RawEncoder():
    def __init__(self, t_res=None, sample_strat='ffill'):
        self.raw = None
        self.t_res = t_res
        self.sample_strat = sample_strat

    def fit(self, df_devices):
        self.raw = create_raw(
            df_devices,
            t_res=self.t_res,
            sample_strat=self.sample_strat
        )

    def fit_transform(self, df_devices):
        self.fit(df_devices)
        return self.raw

    def inverse_transform(self, raw):
        """
        """
        raise NotImplementedError

    def set_params(self, t_res=None, sample_strat=None):
        if t_res is not None:
            self.t_res = t_res
        if sample_strat is not None:
            self.sample_strat = sample_strat

    def transform(self):
        return self.raw



class ChangepointEncoder():
    def __init__(self, t_res=None):
        self.cp = None
        self.t_res = t_res

    def fit(self, df_devices):
        self.cp = create_changepoint(
            df_devices,
            t_res=self.t_res
        )

    def fit_transform(self, df_devices):
        self.fit(df_devices)
        return self.cp

    def inverse_transform(self, cp):
        """
        """
        raise NotImplementedError

    def set_params(self, t_res=None):
        if t_res is not None:
            self.t_res = t_res

    def transform(self, df_devices):
        return create_changepoint(
            df_devices,
            t_res=self.t_res
        )


class LastFiredEncoder():
    def __init__(self, t_res=None):
        self.lf = None
        self.t_res = t_res

    def fit(self, df_devices):
        self.lf = create_lastfired(
            df_devices,
            t_res=self.t_res
        )

    def fit_transform(self, df_devices):
        self.fit(df_devices)
        return self.lf

    def inverse_transform(self, lf):
        """
        """
        raise NotImplementedError

    def set_params(self, t_res=None):
        if t_res is not None:
            self.t_res = t_res

    def transform(self, df_devices):
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

    def fit(self, df_activities):
        df = label_data(self.df_devices, df_activities, self.idle)
        self._lbl_enc.fit(df['activity'].values)

    def fit_transform(self, df_activities):
        df = label_data(self.df_devices, df_activities, self.idle)
        encoded_labels = self._lbl_enc.fit_transform(df['activity'].values)
        return pd.DataFrame(index=df.index, data=encoded_labels, columns=['activity'])

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

    def transform(self, df_activities):
        df = label_data(self.df_devices, df_activities, self.idle)
        encoded_labels = self._lbl_enc.transform(df['activity'].values)
        return pd.DataFrame(index=df.index, data=encoded_labels, columns=['activity'])



class LaggedRawEncoder():
    def __init__(self, window_size, t_res=None, sample_strat='ffill'):
        self.lgd_raw = None
        self.t_res = t_res
        self.window_size = window_size
        self.sample_strat = sample_strat

    def fit(self, df_devices):
        self.lgd_raw = create_lagged_raw(
            df_devices, 
            window_size=self.window_size, 
            t_res=self.t_res, 
            sample_strat=self.sample_strat)

    def fit_transform(self, df_devices):
        self.fit(df_devices)
        return self.lgd_raw

    def inverse_transform(self, lgd_raw):
        """
        """
        raise NotImplementedError

    def set_params(self, t_res=None, window_size=10, sample_strat='ffill'):
        if t_res is not None:
            self.t_res = t_res
        raise NotImplementedError

    def transform(self, df_devices):
        return create_lagged_raw(
            df_devices, 
            window_size=self.window_size, 
            t_res=self.t_res, 
            sample_strat=self.sample_strat)

    
class LaggedRawLabelEncoder():
    """
    wrapper around labelencoder to handle time series data
    """
    def __init__(self, df_devices, window_size, t_res=None, sample_strat='ffill', idle=False):
        self.labels = None
        self.window_size = window_size
        self.t_res = t_res
        self.raw = create_raw(df_devices, t_res=t_res, sample_strat=sample_strat)
        self.idle = idle
        self._lbl_enc = preprocessing.LabelEncoder()

    def fit(self, df_activities):
        df = label_data(self.raw, df_activities, self.idle)
        # start where the labeling begins
        df = df.iloc[self.window_size:,:]
        self._lbl_enc.fit(df['activity'].values)

    def fit_transform(self, df_activities):
        df = label_data(self.raw, df_activities, self.idle)
        df = df.iloc[self.window_size:,:]
        encoded_labels = self._lbl_enc.fit_transform(df['activity'].values)
        return encoded_labels

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
        return self.labels, self.idle, self.raw

    def transform(self, df_activities):
        df = label_data(self.raw, df_activities, self.idle)
        df = df.iloc[self.window_size:,:]
        encoded_labels = self._lbl_enc.transform(df['activity'].values)
        return encoded_labels