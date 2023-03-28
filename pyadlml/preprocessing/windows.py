from datetime import datetime, timezone
from typing_extensions import override
from pyadlml.dataset._core.acts_and_devs import label_data
from pyadlml.dataset.plot.matplotlib.util import save_fig
from pyadlml.pipeline import XOrYTransformer, XAndYTransformer, YTransformer
import numpy.ma as ma
from pyadlml.constants import ACTIVITY, DEVICE, END_TIME, OTHER, START_TIME, TIME, VALUE
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats.kde import gaussian_kde
import dask
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from pyadlml.dataset.plot.util import  \
    fmt_seconds2time, \
    fmt_seconds2time_log

class Windows(BaseEstimator, TransformerMixin):
    REP_M2M = 'many-to-many'
    REP_M2O = 'many-to-one'

    def __init__(self, rep, window_size, stride):
        self.rep = rep
        self.window_size = window_size
        self.stride = stride


    
    @override
    def nr_activities_per_win(self, y):
        raise NotImplementedError

    @save_fig    
    def plot_nr_activities_per_win(self, y, scale='linear', file_path=None):
        """ How many different activities are present in a window.
        
        Parameters
        ----------
        y: pd.Series or np.ndarray
            Including the activities and the timestamps ['time', 'activities']

        """
        import matplotlib.pyplot as plt
        win_size_str = int(self.window_size) if isinstance(self.window_size, float) else self.window_size
        title = f'Activities per window (size={win_size_str})'

        df = self.nr_activities_per_win(y)
        x, y = df['activity per window'], df['count']
        fig, ax = plt.subplots(1,1)
        ax.set_title(title)
        ax.bar(x, y)
        ax.xaxis.set_ticks(x)

        if scale == 'log':
            ax.set_yscale('log')
            ax.set_ylim(0, 10**np.ceil(np.log10(y.max())))

        return fig

    @classmethod
    def _gen_default_window_sizes(cls):
        return np.arange(4, 2000, 100) 

    @classmethod
    def plot_winsize_vs_activities(cls, df_devs, df_acts, window_sizes=None, scale='linear', n_bins=100):
        """


        """
        other = True
        window_sizes = cls._gen_default_window_sizes() if window_sizes is None else window_sizes

        title = 'Window size vs. activities'
        time_unit = 's'

        x_labeled = label_data(df_devs, df_acts, other=other)[[TIME, ACTIVITY]]
        if not other:
            x_labeled = x_labeled.dropna(axis=0)\
                                .reset_index(drop=True)

        max_acts, _ = cls(window_sizes=max(window_sizes)).activities_hist(x_labeled)

        z = np.zeros((max_acts.max(), len(window_sizes)), dtype=np.float32)
        for j, s in enumerate(window_sizes):
            x, y = cls(window_sizes=s).activities_hist(x_labeled)
            z[x-1, j] = y


        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        pcm = ax.pcolormesh(z, cmap='viridis', label='win_lengths', norm=scale)
        x_tick_pos = np.arange(0, len(window_sizes), np.ceil(len(window_sizes)*0.1), dtype=np.int32)
        ax.set_xticks(x_tick_pos, labels=window_sizes[x_tick_pos])

        from pyadlml.dataset.plot.matplotlib.util import make_axes_locatable
        ax.set_xlabel('#events per window')
        ax.set_ylabel('#activities per window')
        ax.set_title(title)


        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="4%", pad=0.1, pack_start=False)
        fig.add_axes(cax)
        fig.colorbar(pcm, cax=cax, orientation="vertical")
        return fig



class ExplicitWindow(BaseEstimator, TransformerMixin, XOrYTransformer):
    """
    https://www.mdpi.com/1424-8220/21/18/6037

    Divide the data stream into segments by 

    """
    def __init__(self, rep: str = 'many-to-many'):
        TransformerMixin.__init__(self)
        XOrYTransformer.__init__(self)
        self.rep = rep

    def fit(self, X, y):
        assert self.rep in ['many-to-many', 'many-to-one']
        return self

    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X,y)

    @XOrYTransformer.x_or_y_transform
    def transform(self, X: pd.DataFrame, y:pd.DataFrame) -> pd.DataFrame:
        """

        Parameters
        ----------
        X: pd.DataFrame

        y: pd.DataFrame


        .. Note 
        -------

        Sequences are padded with NaNs to the length of the longest sequence in order to form a tensor.


        Returns
        -------
        X: np.ndarray with shape (S, F, T)
            where S is the #sequences, F is the #features, and T is the maxmimum sequence length
        y: np.ndarray with shape (S, T)
            where S is the #sequences and T is the maximum sequence length

        """
        assert len(y.columns) == 1

        y = y.copy().reset_index(drop=True)
        X = X.copy().reset_index(drop=True)

        # Assign each sequence a unique number
        y['tmp'] = (y[ACTIVITY] != y[ACTIVITY].shift(1)).cumsum()
        S = y['tmp'].iat[-1]
        T = y['tmp'].value_counts().max()
        F = len(X.columns)

        Xt = np.full((S, F, T), np.nan, dtype='object')
        if self.rep == 'many-to-many':
            yt = np.full((S, T), np.nan, dtype='object')
        else:
            yt = np.full((S), '', dtype='object')

        for s, (_, grp) in enumerate(y.groupby('tmp')):
            X_tmp = np.swapaxes(X.loc[grp.index, :].values, 0,1)
            Xt[s, :, :X_tmp.shape[-1]] = X_tmp
            if self.rep == 'many-to-many':
                y_tmp = y.loc[grp.index, y.columns[:-1]].values.squeeze(-1)
                yt[s, :y_tmp.shape[-1]] = y_tmp
            else:
                yt[s] = y.at[grp.index[0], ACTIVITY]

        return Xt, yt

class TimeWindow(Windows):
    """ Divide data stream into time segments with a regular interval.

    Note
    ----
    Also works with regular sampled data (dt='xs').
    Van Kasteren et al. recommend 60s. 

    """
    def __init__(self, window_size : str, stride: str = None, rep: str ='many-to-many',  drop_empty_intervals=True):
        TransformerMixin.__init__(self)
        XOrYTransformer.__init__(self)
        Windows.__init__(self, rep, window_size, stride)
        self.drop_empty_intervals = drop_empty_intervals

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X=None, y=None):

        # Bring params into right format
        assert self.rep in [self.REP_M2M, self.REP_M2O]
        self.window_size = pd.Timedelta(self.window_size)
        self.stride = pd.Timedelta(self.stride) if self.stride is not None else self.window_size
        return self

    @XOrYTransformer.x_or_y_transform
    def transform(self, X, y=None) -> np.ndarray:
        """
        
        Parameters 
        ----------- (S, T, F) and (S, T)
        """
        # TODO refactor add conversion for different input types
        assert isinstance(X, pd.DataFrame) or X is None
        assert isinstance(y, pd.DataFrame) or y is None

        df = X.copy().sort_values(by=TIME)
        st = X[TIME].iloc[0] - pd.Timedelta('1s')
        et = X[TIME].iloc[-1] + pd.Timedelta('1s')
        win_st = st

        X_list = []
        y_list = []
        max_seq_length = 0
        i = 0


        st_windows = pd.date_range(st, et-self.window_size, freq=self.stride)
        if self.drop_empty_intervals:
            times = df[TIME].copy()
            while win_st + self.window_size <= et:

                win = (win_st, win_st + self.window_size)
                # Important the interval is [st,et) right closed
                event_idxs = df[(win[0] <= df[TIME]) & (df[TIME] < win[1])].index
                
                if not event_idxs.empty:
                    X_list.append(X.iloc[event_idxs].copy())
                    if y is not None:
                        y_list.append(y.iloc[event_idxs].copy())
                    max_seq_length = max(max_seq_length, len(X_list[i]))
                    win_st = win_st + self.stride
                    i += 1
                else:
                    # Get the next first window that covers the next event 
                    next_event_time = times[win_st < times].iloc[0]
                    win_min_idx_not_containing_ev = (st_windows <= next_event_time - self.window_size).cumsum().max() - 1
                    if win_min_idx_not_containing_ev == len(st_windows) - 1:
                        # When there is no event for the enxt windows 
                        break
                    win_st = st_windows[win_min_idx_not_containing_ev + 1]
        else: 
            for win_st in st_windows:
                win = (win_st, win_st + self.window_size)
                event_idxs = df[(win[0] <= df[TIME]) & (df[TIME] < win[1])].index
                
                X_list.append(X.iloc[event_idxs].copy())
                if y is not None:
                    y_list.append(y.iloc[event_idxs].copy())
                max_seq_length = max(max_seq_length, len(X_list[i]))
                i += 1


        S = len(X_list)
        F = len(X.columns)
        T = max_seq_length
        Xt = np.full((S, T, F), np.nan, dtype='object')

        if y is not None:
            Fy = 1 if len(y.shape) == 1 else y.shape[-1]
            if self.rep == 'many-to-many':
                yt = np.full((S, T, Fy), np.nan, dtype='object')
            else:
                yt = np.full((S, Fy), np.nan, dtype='object')

        for s in range(len(X_list)):
            X_tmp = X_list[s].values
            Xt[s, :X_tmp.shape[0], :] = X_tmp

            if y is not None:
                y_tmp = y_list[s].values
                if y_tmp.size > 0:
                    if self.rep == 'many-to-many':
                        y_tmp = y_tmp.squeeze(-1)
                        yt[s, :y_tmp.shape[0], :] = y_tmp
                    else:
                        yt[s, :] = y_tmp[-1]

        yt = None if y is None else yt
        return Xt, yt 

    @classmethod
    def nr_windows(cls, X, win_size, stride='10s', drop_empty_intervals=False):
        df = X.copy().sort_values(by=TIME)
        st = X[TIME].iloc[0] - pd.Timedelta('1s')
        et = X[TIME].iloc[-1] + pd.Timedelta('1s')
        st_windows = pd.date_range(st, et-win_size, freq=stride)
        return st_windows

    def nr_activities_per_win(self, y):
        """ How many different activities are present in a window

        Parameters
        ----------
        y: pd.DataFrame
            A table with at least one column named 'activity' and 'time'

        Returns
        -------
        pd.DataFrame
        """
        assert (y.columns == [TIME, ACTIVITY]).all()
        self.rep = self.REP_M2M
        contains_nans = y[ACTIVITY].isna().any()
        #if contains_nans:
        #    act2int = { v:k for k, v in enumerate(y[ACTIVITY].unique())} 
        #    from pandas._libs.tslibs.nattype import NaT
        #    #act2int[NaT] = np.inf
        #    y = self.transform(X=y)
        #    y = np.vectorize(act2int.get)(y).astype(np.int32)
        #    y = np.where(y == act2int[NaT], np.inf, y)
        #    y = ma.masked_invalid(y)
        #    counts_per_row = (y[:,1:] != y[:,:-1]).sum(axis=1)+1
        #    x, y = np.unique(counts_per_row, return_counts=True)
        #    df = pd.DataFrame(columns=['activity per window', 'count'], data=np.stack([x,y]).T)
        #    return df.sort_values(by='activity per window')
        #    
        #else:
        if True:
            act2int = { v:k for k, v in enumerate(y[ACTIVITY].unique())} 
            yt = self.transform(X=y)    # S, T, 2 (timestamps, activities) in feature_y dim
            yt = yt[:, :, 1] # remove timestamps -> (S, T)

            # Hotfix nan issue
            yt = np.where(pd.isnull(yt), -1, yt)
            act2int[-1] = -1
            yt = np.vectorize(act2int.get)(yt).astype(np.int32)
            yt = np.where(yt == -1, np.inf, yt)
            yt = ma.masked_invalid(yt)
            counts_per_row = (yt[:,1:] != yt[:,:-1]).sum(axis=1)+1
            bins, counts = np.unique(counts_per_row, return_counts=True)
            return pd.DataFrame(columns=['activity per window', 'count'], data=np.stack([bins, counts]).T)


    @save_fig
    def plot_nr_events_per_window(self, times, y_scale='linear', n_bins=20, file_path=None):
        """ Plot the number of events per window

        Parameters
        ----------
        times : pd.DataFrame
            TODO
        y_scale : str, one of ['linear', 'log'], default='linear'
            TODO
        n_bins : int, default=20
            TODO
        file_path : None or str

        Returns
        -------
        fig or None
        """

        # Get nr events per window
        yt = self.transform(X=times) 
        yt = yt[:, :, 1] # Remove timestamps -> (S, T)
        counts = np.count_nonzero(~pd.isnull(yt), axis=1)

        title = f'#Events per window (size={self.window_size})'

        import matplotlib.pyplot as plt
        from pyadlml.dataset.plot.matplotlib.util import plot_hist
        fig, ax = plt.subplots(1,1)

        bins = np.linspace(counts.min(), counts.max(), n_bins+1)

        hist, _ = np.histogram(counts, bins=bins)
        mean = counts.mean() 
        median = np.median(counts)

        if y_scale == 'log':
            y_max = ax.set_ylim(1, 10**np.ceil(np.log10(hist.max())))
        else:
            y_max = hist.max()
            y_max += y_max*0.05

        ax.hist(counts, bins, label='')
        ax.set_title(title, y=1.08)
        ax.set_xlabel('#events')
        ax.set_ylabel('count')

        if y_scale == 'log':
            ax.set_yscale('log')
            ax.set_ylim(1, top=y_max)
        else:
            ax.set_ylim(0, y_max)

        ax.plot([mean]*2, [0, y_max], label=f'mean {mean:.2f}')
        ax.plot([median]*2, [0, y_max], label=f'median {median:.2f}')

        ax.legend()

        return fig



    def activities_hist(self, Xt):
        """
        """
        # Retrieve windows and only activities
        #Xt = Xt[:, None, :] if win_size == 1 else Xt

        Xt = Xt[:,:,1]

        # Get histogram
        counts_per_row = (Xt[:,1:] != Xt[:,:-1]).sum(axis=1)+1
        x, y = np.unique(counts_per_row, return_counts=True)
        return x, y

    @classmethod
    def _gen_default_window_sizes(cls):
        return np.arange(4, 2000, 100) 

    @classmethod
    @save_fig
    def plot_winsize_vs_activities(cls, df_devs, df_acts, window_sizes=None, scale='linear', other=False, stride='10s', use_dask=False, file_path=None):
        """
        
        Parameters
        ----------

        """
        window_sizes = pd.timedelta_range('1h', '20D', freq='1h') if window_sizes is None else window_sizes

        title = 'Window size vs. activities'
        scale = 'log'

        x_labeled = label_data(df_devs, df_acts, other=other)[[TIME, ACTIVITY]]

        if not other:
            x_labeled = x_labeled.dropna(axis=0)\
                                .reset_index(drop=True)

        max_acts = cls(window_size=max(window_sizes), stride=stride).fit()\
                .nr_activities_per_win(x_labeled)['activity per window']\
                .max()
        if use_dask:
            z = np.zeros((max_acts.max(), len(window_sizes)), dtype=np.float32)
            res = []
            def fun(s, x):
                df = cls(window_size=s, stride=stride).fit().nr_activities_per_win(x)
                return df['count'], df['activity per window']
            x_labeled = dask.delayed(x_labeled)
            for s in window_sizes:
                res.append(dask.delayed(fun)(s, x_labeled))

            res = dask.compute(*res)
            for xi, (zi, y) in enumerate(res):
                z[y-1, xi] = zi

        else:
            z = np.zeros((max_acts.max(), len(window_sizes)), dtype=np.float32)
            for xi, s in enumerate(window_sizes):
                df = cls(window_size=s, stride=stride).fit().nr_activities_per_win(x_labeled)
                zi, y = df['count'], df['activity per window']
                z[y-1, xi] = zi


        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        pcm = ax.pcolormesh(z, cmap='viridis', label='win_lengths', norm=scale)
        x_tick_pos = np.arange(0, len(window_sizes), np.ceil(len(window_sizes)*0.1), dtype=np.int32)
        ax.set_xticks(x_tick_pos + 1 , labels=window_sizes[x_tick_pos])

        from pyadlml.dataset.plot.matplotlib.util import make_axes_locatable
        ax.set_xlabel('window $dt$')
        ax.set_ylabel('#activities per window')
        ax.set_title(title)


        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="4%", pad=0.1, pack_start=False)
        fig.add_axes(cax)
        fig.colorbar(pcm, cax=cax, orientation="vertical")
        ax.tick_params(axis='x', rotation=-45)
        plt.tight_layout()
        return fig

    def nr_events_per_win(self, times):
        """

        # Get nr events per window
        # Pick first value after timestamps. The value is irrelevant just that it is not None
        # -> (S, T, F)
        """
        yt = self.transform(X=times) 
        yt = yt[:, :, 1] # Remove timestamps -> (S, T)
        counts_per_row = np.count_nonzero(~pd.isnull(yt), axis=1)
        bins, counts = np.unique(counts_per_row, return_counts=True)
        return pd.DataFrame(columns=['events per window', 'count'], data=np.stack([bins, counts]).T)

    def get_event_rate(self, X):
        """
        
        """
        self.drop_empty_intervals = False
        st = X[TIME].iloc[0] - pd.Timedelta('1s')
        et = X[TIME].iloc[-1] + pd.Timedelta('1s')
        st_windows = pd.date_range(st, et-self.window_size, freq=self.stride)
        st_windows = st_windows + self.window_size
        # -> (S, T, F)
        Xt = self.transform(X=X) 
        counts_per_row = np.count_nonzero(~pd.isnull(Xt[:,:,0]), axis=1)
        df = pd.DataFrame([counts_per_row, st_windows]).T
        df.columns = [VALUE, TIME]
        df[DEVICE] = 'Time window: Event rate'

        return df[[TIME, DEVICE, VALUE]].sort_values(by=TIME).reset_index(drop=True)



    @classmethod
    @save_fig
    def plot_winsize_vs_events(cls, df_devs, window_sizes=None, scale='log', stride='10s', use_dask=False, file_path=None):
        """
        
        Parameters
        ----------

        """
        window_sizes = pd.timedelta_range('1h', '20D', freq='1h') if window_sizes is None else window_sizes

        title = 'Window size vs. events'

        x_labeled = df_devs

        max_acts = int(cls(window_size=max(window_sizes), stride=stride).fit()\
                .nr_events_per_win(df_devs)['events per window']\
                .max())
        if use_dask:
            z = np.zeros((max_acts, len(window_sizes)), dtype=np.float32)
            res = []
            def fun(s, x):
                df = cls(window_size=s, stride=stride).fit().nr_events_per_win(x)
                return df['count'], df['events per window']
            x_labeled = dask.delayed(x_labeled)
            for s in window_sizes:
                res.append(dask.delayed(fun)(s, x_labeled))

            res = dask.compute(*res)
            for xi, (zi, y) in enumerate(res):
                z[y-1, xi] = zi

        else:
            z = np.zeros((max_acts.max(), len(window_sizes)), dtype=np.float32)
            for xi, s in enumerate(window_sizes):
                df = cls(window_size=s, stride=stride).fit().nr_activities_per_win(x_labeled)
                zi, y = df['count'], df['events per window']
                z[y-1, xi] = zi


        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        pcm = ax.pcolormesh(z, cmap='viridis', label='win_lengths', norm=scale)
        x_tick_pos = np.arange(0, len(window_sizes), np.ceil(len(window_sizes)*0.1), dtype=np.int32)
        ax.set_xticks(x_tick_pos + 1 , labels=window_sizes[x_tick_pos])

        from pyadlml.dataset.plot.matplotlib.util import make_axes_locatable
        ax.set_xlabel('window $dt$')
        ax.set_ylabel('#events per window')
        ax.set_title(title)

        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="4%", pad=0.1, pack_start=False)
        fig.add_axes(cax)
        fig.colorbar(pcm, cax=cax, orientation="vertical")
        ax.tick_params(axis='x', rotation=-45)
        plt.tight_layout()
        return fig

class FuzzyTimeWindows():
    """
    """
    def __init__(self):
        raise NotImplementedError


class EventWindow(Windows, XOrYTransformer):
    """ Generate subsequences from a 




    Many-To-Many
    ^^^^^^^^^^^^

    To get *many-to-many* batches use the window size :math:`w` to split the data

    .. math::
        f(X_{N,K},y_{N}) \rightarrow (X_{W, N,K}, y_{W, N})

    In addition you can specify a stride.


    .. code:: python

        from pyadlml.dataset import fetch_kasteren
        from pyadlml.preprocessing import EventWindows
        data = fetch_kasteren()

        raw = StateVectorEncoder(encode='raw', t_res='10s')\
              .fit_transform(data.df_devices)
        labels = LabelEncoder().fit_transform(raw, data.df_activities)

        X = raw.values
        y = labels.values

        X, y = EventWindows(rep='many-to-many', window_length=10, stride=2)\
               .fit_transform(X, y)

    Many-To-One
    ^^^^^^^^^^^

    .. math::
        f(X_{N,K},y_{N}) \rightarrow (X_{W, N, K}, y_{N})


    .. code:: python

        from pyadlml.preprocessing import SequenceDicer

        raw = StateVectorEncoder(encode='raw', t_res='10s')\
              .fit_transform(data.df_devices)
        labels = LabelEncoder().fit_transform(raw, data.df_activities)

        X = raw.values
        y = labels.values

        X, y = SequenceDicer(rep='many-to-one', window_length='20s')\
               .fit_transform(X, y)

    """
    def __init__(self, rep: str ='many-to-many', window_size: int =10, stride: int=1):
        """
        Parameters
        ----------
        rep: str 
        window_size: int
        stride: int

        """

        TransformerMixin.__init__(self)
        XOrYTransformer.__init__(self)
        Windows.__init__(self, rep, window_size, stride)
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns if isinstance(X, pd.DataFrame) else None
        return self


    def fit_transform(self, X, y=None):
        """
        Parameters
        ----------
        X : np.array
            Some kind of numpy array or pandas Dataframe
        y : np.array

        Returns
        -------
        x : np.array of shape ()
        y : np.array
            todo
        """
        self.fit(X, y)
        return self.transform(X, y)

    @XOrYTransformer.x_or_y_transform
    def transform(self, X=None, y=None):
        """

        """
        assert self.rep in [self.REP_M2M, self.REP_M2O]

        if X is not None:
            assert self.window_size < len(X), f'Length X: {len(X)} >= {self.window_size} window size!'
            assert self.stride < len(X)

            X = self._transform_X(X)

        if y is not None:
            assert self.window_size < len(y)
            assert self.stride < len(y)

            y = self._transform_Y(y)

        return X, y

    def _calc_new_N(self, n_old):
        return int(np.floor((n_old-self.window_size)/self.stride)+1)

    def _transform_X(self, X):
        """

        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()

        n_prime = self._calc_new_N(X.shape[0])
        new_shape =[n_prime, self.window_size, *X.shape[1:]]

        res = np.zeros(shape=new_shape, dtype=X.dtype)

        for r, i in enumerate(range(0, n_prime*self.stride, self.stride)):
            res[r] = X[i:i+self.window_size]
            #res[r, :, :] = X[i:i+self.window_size, :]

        return res.squeeze()

    def _transform_Y(self, y):
        """


        """

        if isinstance(y, pd.DataFrame):
            y = y.to_numpy().squeeze()
        elif isinstance(y, pd.Series):
            y = y.to_numpy()

        if self.rep == self.REP_M2M:
            n_prime = self._calc_new_N(y.shape[0])
            new_shape = [n_prime, self.window_size, *y.shape[1:]]

            res = np.zeros(shape=new_shape, dtype=y.dtype)

            for r, i in enumerate(range(0, n_prime*self.stride, self.stride)):
                res[r, :] = y[i: i+self.window_size]

            return res.squeeze()

        elif self.rep == self.REP_M2O:
            res = y[np.arange(self.window_size-1, y.shape[0], self.stride)]
            return res.squeeze()

    def nr_activities_per_win(self, y):
        """ How many different activities are present in a window

        Parameters
        ----------
        y: pd.DataFrame
            A table with at least one column named 'activity' and 'time'

        Returns
        -------
        pd.DataFrame
        """
        assert (y.columns == [TIME, ACTIVITY]).all()
        self.rep = self.REP_M2M
        contains_nans = y[ACTIVITY].isna().any()
        if contains_nans:
            act2int = { v:k for k, v in enumerate(y[ACTIVITY].unique())} 
            from pandas._libs.tslibs.nattype import NaT
            #act2int[NaT] = np.inf
            y = self.transform(X=y)[:,:,1]
            y = np.vectorize(act2int.get)(y).astype(np.int32)
            y = np.where(y == act2int[NaT], np.inf, y)
            y = ma.masked_invalid(y)

            counts_per_row = (y[:,1:] != y[:,:-1]).sum(axis=1)+1
            x, y = np.unique(counts_per_row, return_counts=True)
            df = pd.DataFrame(columns=['activity per window', 'count'], data=np.stack([x,y]).T)
            return df.sort_values(by='activity per window')
            
        else:
            act2int = { v:k for k, v in enumerate(y[ACTIVITY].unique())} 
            y = self.transform(X=y[ACTIVITY])
            y = np.vectorize(act2int.get)(y)
            counts_per_row = (y[:,1:] != y[:,:-1]).sum(axis=1)+1
            x, y = np.unique(counts_per_row, return_counts=True)
            return pd.DataFrame(columns=['activity per window', 'count'], data=np.stack([x,y]).T)
            


    def get_coverage(self, times, n_samples=1000000, dt='5s'):
        """ Function of amount of intervals covering at times. A increment and decrement
            process based on interval start end end times. 

        Parameters
        ----------
        times : pd.Series or np.ndarray

        n_samples : int
            The number of monte carlo samples to draw

        Returns
        -------
        pd.DataFrame
            A device dataframe with columns ['time', 'device', 'value']
        """

        # Get intervals from windows
        x = self.fit_transform(times)
        df = pd.DataFrame(data=np.stack([x[:,0], x[:,-1]]).T, columns=[START_TIME, END_TIME])
        df = df.sort_values(by=START_TIME).reset_index(drop=True)
        steps = (df[END_TIME].iloc[-1] - df[START_TIME].iloc[0])//pd.Timedelta(dt)

        assert n_samples > len(df)
        n_samples_per_int = n_samples//len(df)


        # Transform to unix time 
        df = df.applymap(lambda x: x.replace(tzinfo=timezone.utc).timestamp())

        points = np.zeros((n_samples_per_int*len(df)), dtype=np.float32)
        # Sample point from intervals
        for i in range(df.shape[0]):
            intv = df.iloc[i]
            unx_sample = np.random.uniform(intv[START_TIME], intv[END_TIME], n_samples_per_int)
            j = i*n_samples_per_int
            points[j:j+n_samples_per_int] = unx_sample

        # Kernel density estimation 
        x = np.linspace(df[START_TIME].min(), df[END_TIME].max(), steps)
        y = gaussian_kde(points).pdf(x)

        df_res = pd.DataFrame(data=np.vectorize(datetime.utcfromtimestamp)(x), columns=[TIME])
        df_res[TIME] = pd.to_datetime(df_res[TIME])
        df_res[VALUE] = y
        df_res[DEVICE] = 'EventWin Coverage'
        return df_res


    def get_durations_per_window(self, times): 
        x = self.fit_transform(times)
        time_unit = 's'
        dur = (x[:,-1] - x[:,0])/np.timedelta64(1, time_unit)
        return dur

    def plot_duration_per_window(self, times, y_scale='linear', x_scale='linear', n_bins=20):
        dur = self.get_durations_per_window(times)

        title = f'Window lengths (size={int(self.window_size)})'

        import matplotlib.pyplot as plt
        from pyadlml.dataset.plot.matplotlib.util import plot_hist
        fig, ax = plt.subplots(1,1)
        
        
        if x_scale == 'log':
            ax.set_xscale('log')
            # Ensures bins are equally spaced
            bins = np.logspace(np.log10(dur.min()), np.log10(dur.max()), n_bins)
        else:
            bins = np.linspace(dur.min(), dur.max(), n_bins+1)

        hist, _ = np.histogram(dur, bins=bins)
        mean = dur.mean() 
        median = np.median(dur)

        if y_scale == 'log':
            y_max = ax.set_ylim(1, 10**np.ceil(np.log10(hist.max())))
        else:
            y_max = hist.max()
            y_max += y_max*0.05

        ax.hist(dur, bins, label='win_lengths')
        ax.set_title(title, y=1.08)
        ax.set_xlabel('time in seconds')
        ax.set_ylabel('count')

        if y_scale == 'log':
            ax.set_yscale('log')
            ax.set_ylim(1, top=y_max)
        else:
            ax.set_ylim(0, y_max)

        ax.plot([mean]*2, [0, y_max], label='mean')
        ax.plot([median]*2, [0, y_max], label='median')

        ax.legend()

        from pyadlml.dataset.plot.matplotlib.util import add_top_axis_time_format
        add_top_axis_time_format(ax)
        return fig


    # TODO refactor
    #def plotly_windows_into_acts_and_devs(self, df_devs, df_acts):
    #    """

    #    """

    #    # Get times 
    #    Xt = self.fit_transform(df_devs) # (S, T, 3)


    #    from pyadlml.plot import plotly_activities_and_devices
    #    fig = plotly_activities_and_devices(df_devs, df_acts)

    #    # Plot times as vertical bars into acts_and_devs plot 
    #    for i in range(Xt.shape[0]):
    #        st = Xt[i][0, 0]
    #        et = Xt[i][-1, 0]
    #        fig.add_vline(x=st, line_width=2, line_color="Red", line_dash="dash")
    #        fig.add_vline(x=et, line_width=2, line_color="Red")

    #    return fig


    @classmethod
    def plot_winsize_vs_activities(cls, df_devs, df_acts, window_sizes=None, scale='linear', n_bins=100):
        """


        """
        other = True
        window_sizes = np.arange(4, 2000, 100) if window_sizes is None else window_sizes

        title = 'Window size vs. activities'
        time_unit = 's'

        x_labeled = label_data(df_devs, df_acts, other=other)[[TIME, ACTIVITY]]
        if not other:
            x_labeled = x_labeled.dropna(axis=0)\
                                .reset_index(drop=True)
        def get_histogram(win_size, x_labeled):
            # Retrieve windows and only activities
            Xt = cls(window_size=win_size).fit_transform(x_labeled)
            Xt = Xt[:, None, :] if win_size == 1 else Xt

            Xt = Xt[:,:,1]

            # Get histogram
            counts_per_row = (Xt[:,1:] != Xt[:,:-1]).sum(axis=1)+1
            x, y = np.unique(counts_per_row, return_counts=True)
            return x, y

        max_acts, _ = get_histogram(max(window_sizes), x_labeled)
        z = np.zeros((max_acts.max(), len(window_sizes)), dtype=np.float32)
        for j, s in enumerate(window_sizes):
            x, y = get_histogram(s, x_labeled)
            z[x-1, j] = y


        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        pcm = ax.pcolormesh(z, cmap='viridis', label='win_lengths', norm=scale)
        x_tick_pos = np.arange(0, len(window_sizes), np.ceil(len(window_sizes)*0.1), dtype=np.int32)
        ax.set_xticks(x_tick_pos, labels=window_sizes[x_tick_pos])

        from pyadlml.dataset.plot.matplotlib.util import make_axes_locatable
        ax.set_xlabel('#events per window')
        ax.set_ylabel('#activities per window')
        ax.set_title(title)


        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="4%", pad=0.1, pack_start=False)
        fig.add_axes(cax)
        fig.colorbar(pcm, cax=cax, orientation="vertical")
        return fig

    @classmethod
    def plot_winsize_vs_lengths(cls, X, y, window_sizes=None, z_scale='linear', y_scale='linear'):
        assert window_sizes[0] > 1, 'Length is not defined for window containing one timestamp'

        # Reduce X to only time dimension
        # TODO, refactor 
        x_times = X[TIME].values

        title = 'Window size vs. duration'
        cbarlabel = 'length' if z_scale == 'linear' else 'log length'
        n_bins = 100
        time_unit = 's'
        _ = None


        z = np.zeros((n_bins, len(window_sizes)), dtype=np.float32)
        dur_list = []
        dur_min = float('inf') 
        dur_max = float('-inf')
        for s in window_sizes:
            Xt = cls(window_size=s).fit_transform(x_times)
            # Get normalized time lengths
            Xt_dur = (Xt[:,-1] - Xt[:,0])/np.timedelta64(1, time_unit)
            
            dur_min = min(dur_min, Xt_dur.min())
            dur_max = max(dur_max, Xt_dur.max())
            dur_list.append(Xt_dur)

        # Make equal bin size from max to min
        # If log, Bins are equally sized in log space 
        if y_scale == 'log':
            dur_min, dur_max = np.log10(dur_min), np.log10(dur_max)
            bins = np.logspace(dur_min, dur_max, n_bins+1)
        else:
            bins = np.linspace(dur_min, dur_max, n_bins+1)


        for pos, d in enumerate(dur_list):
            hist, edges = np.histogram(d, bins=bins)
            z[:, pos] = hist

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        X, Y = np.meshgrid(window_sizes, bins[:-1])
        pcm = ax.pcolormesh(X, Y, z, cmap='viridis', label='win_lengths', norm=z_scale)
        #x_tick_pos = np.arange(0, len(window_sizes), np.ceil(len(window_sizes)*0.1), dtype=np.int32)
        #ax.set_xticks(x_tick_pos, labels=window_sizes[x_tick_pos])
        #y_tick_pos = np.arange(0, len(bins), np.ceil(len(bins)*0.1), dtype=np.int32)
        #ax.set_yticks(y_tick_pos, labels=np.vectorize(lambda x: f'{x:.2f}')(bins[y_tick_pos]))
        if y_scale == 'log':
            ax.set_yscale('log')
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        from pyadlml.dataset.plot.matplotlib.util import add_axis_time_format, make_axes_locatable
        add_axis_time_format(ax, 'right', scale='linear')
        ax.set_xlabel('#events per window')
        ax.set_ylabel('duration seconds')
        ax.set_title(title)

        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="4%", pad=0.7, pack_start=False)
        fig.add_axes(cax)
        fig.colorbar(pcm, cax=cax, orientation="vertical")
        return fig
        

    @classmethod
    def plotly_winsize_vs_length(cls, X, y, window_sizes=None, z_scale='linear', y_scale='linear'):
        """
        """
        import plotly.graph_objects as go
        window_sizes = np.arange(2, 1000, 10) if window_sizes is None else window_sizes

        assert window_sizes[0] > 1, 'Length is not defined for window containing one timestamp'

        # Reduce X to only time dimension
        # TODO, refactor 
        x_times = X[TIME].values

        title = 'Window size vs. window_length'
        cbarlabel = 'length' if z_scale == 'linear' else 'log length'
        n_bins = 100
        time_unit = 's'
        _ = None


        z = np.zeros((n_bins, len(window_sizes)), dtype=np.float32)
        dur_list = []
        dur_min = float('inf') 
        dur_max = float('-inf')
        for s in window_sizes:
            Xt = cls(window_size=s).fit_transform(x_times)
            # Get normalized time lengths
            Xt_dur = (Xt[:,-1] - Xt[:,0])/np.timedelta64(1, time_unit)
            
            dur_min = min(dur_min, Xt_dur.min())
            dur_max = max(dur_max, Xt_dur.max())
            dur_list.append(Xt_dur)

        # Make equal bin size from max to min
        # If log, Bins are equally sized in log space 
        #bin_gen = np.logspace if y_scale == 'log' else np.linspace
        #if y_scale == 'log':
        #    dur_min, dur_max = np.log10(dur_min), np.log10(dur_max)
        bin_gen = np.linspace
        bins = bin_gen(dur_min, dur_max, n_bins+1)


        for pos, d in enumerate(dur_list):
            hist, edges = np.histogram(d, bins=bins)
            z[:, pos] = hist

        z = np.log(z) if z_scale == 'log' else z

        fig = go.Figure(data=go.Heatmap(
                            x=window_sizes,
                            z=z,
                            colorscale='Viridis',
        ))
        if z_scale == 'log':
            tmpltstr = '%{x} event-length<br>' \
                     + ' -> %{customdata[1]} windows durating %{customdata[0]}.<extra></extra>'
            fig.data[0].hovertemplate = tmpltstr
            y_fmt = bins[:-1].copy().astype(object)

            for i in range(len(y_fmt)):
                if y_scale == 'log':
                    y_fmt[i] = fmt_seconds2time_log(y_fmt[i])
                else:
                    y_fmt[i] = fmt_seconds2time(y_fmt[i])

            y_fmt = np.repeat(y_fmt, len(window_sizes)).reshape(len(y_fmt), len(window_sizes))

            tmp = (np.exp(z.copy()) if z_scale == 'log' else z).astype(np.int32) 

            cd = np.array([y_fmt, tmp])
            fig['data'][0]['customdata'] = np.moveaxis(cd, 0, -1)

        #fig.update_yaxes(type='log')
        fig.update_layout(yaxis=dict(tickmode='array', tickvals=np.arange(len(bins[:-1])), ticktext=edges[:-1]))

        from pyadlml.dataset.plot.plotly.activities import _set_compact_title
        _set_compact_title(fig, title=title)
        return fig


    def construct_target_times(self, times: np.ndarray) -> np.ndarray:
        times = self._transform_Y(times)
        
        if self.rep == self.REP_M2M:
            # TODO take only last y-obs per sequence??? 
            # Or average y for each position when plotting 
            raise NotImplementedError
        
        return times


    def construct_X_at_target(self, X: pd.DataFrame, dev_slice=slice(None, None, None)) -> np.ndarray:
        """ Reconstruct the datapoints that are at the points
            where the target is evaluated

        Parameters
        ----------
        X: pd.DataFrame, (Tt, F)
            Transformed data right before windowing 

        Returns
        -------
        Xtarget : (S,)
        """

        # -> (S, T) or (S, T, F)
        X = self._transform_X(X)
        if self.rep == self.REP_M2O:
            if X.ndim == 2:
                X = X[:,:, None]

            # Select last event for T
            X = X[:, -1, dev_slice]
            if X.ndim == 3:
                X = X.squeeze(1)

            assert X.ndim == 2
            return X
        else:
            raise NotImplementedError
    

    def inverse_transform(self, X, y):
        """
        
        
        """
        if self.rep == self.REP_M2M:
            raise NotImplementedError

        elif self.rep == self.REP_M2O:
            raise NotImplementedError
