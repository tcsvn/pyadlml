from pyadlml.dataset._representations.raw import create_raw 
from pyadlml.dataset._representations.changepoint import create_changepoint
from pyadlml.dataset._representations.lastfired import create_lastfired
from pyadlml.dataset._representations.image import create_lagged_raw, create_lagged_lastfired, \
                                            create_lagged_changepoint
from pyadlml.dataset._dataset import label_data
from pyadlml.dataset.devices import device_rep1_2_rep2
import sklearn.preprocessing as preprocessing
import pandas as pd
import numpy as np

REPS = ['raw', 'lastfired', 'changepoint']

class DiscreteEncoder():
    def __init__(self, rep, t_res=None, sample_strat='ffill'):
        assert rep in REPS

        self.rep = rep
        self.t_res = t_res
        self.sample_strat = sample_strat
        self.data = None

    def fit(self, df_devices):
        if self.rep == 'raw':
            self.data = create_raw(
                df_devices,
                t_res=self.t_res,
                sample_strat=self.sample_strat
            )
        elif self.rep == 'changepoint':
            self.data = create_changepoint(
                df_devices,
                t_res=self.t_res
            )
        elif self.rep == 'lastfired':
            self.data = create_lastfired(
                df_devices,
                t_res=self.t_res
            )
        else:
            raise ValueError


    def fit_transform(self, df_devices):
        self.fit(df_devices)
        return self.data

    def inverse_transform(self, raw):
        """
        """
        raise NotImplementedError

    def set_params(self, t_res=None, sample_strat=None):
        if t_res is not None:
            self.t_res = t_res
        if sample_strat is not None:
            self.sample_strat = sample_strat

    def transform(self, df_devices=None):
        if self.rep == 'raw':
            return create_raw(
                df_devices,
                t_res=self.t_res,
                sample_strat=self.sample_strat
            )
        elif self.rep == 'changepoint':
            return create_changepoint(
                df_devices,
                t_res=self.t_res
            )
        elif self.rep == 'lastfired':
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
        """ labels data and creates the numeric representations 
        Parameters
        ----------
        df_activities : pd.DataFrame
            Columns are end_time, start_time, activity. 
        """
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

    def transform(self, x):
        """
        
        """
        # if the input is a dataframe of activities, than fit and
        # transform the data accordingly 
        col_names = {'start_time', 'end_time', 'activity'}
        if isinstance(x, pd.DataFrame) and set(x.columns) == col_names:
            df = label_data(self.df_devices, x, self.idle)
            encoded_labels = self._lbl_enc.transform(df['activity'].values)
            return pd.DataFrame(index=df.index, data=encoded_labels, columns=['activity'])
        
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
        self._lbl_enc.fit(df['activity'].values)

    def fit_transform(self, df_activities):
        df = label_data(self.df_index, df_activities, self.idle)
        df = df.iloc[self.window_size:,:]
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
        encoded_labels = self._lbl_enc.transform(df['activity'].values)
        return pd.DataFrame(index=df.index, data=encoded_labels, columns=['activity'])

