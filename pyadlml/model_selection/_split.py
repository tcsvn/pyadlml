
import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

from pyadlml.constants import ACTIVITY, TIME, START_TIME, END_TIME, DEVICE, VALUE
from sklearn.model_selection import TimeSeriesSplit as SklearnTSSplit, KFold as SklearnKFold

from pyadlml.dataset.util import get_last_states, select_timespan
from pyadlml.dataset.cleaning.misc import remove_days


def train_test_split(df_devs, df_acts, split='leave_one_day_out', temporal=False,
                     return_day=False, return_init_states=False):
    """
    Splits the data into training and test set.

    Parameters
    ----------
    df_devs : pd.DataFrame
        todo
    df_acts : pd.DataFrame
        todo
    split : one of {'leave_one_day_out', (float, float), (float, float, float)}, default='leave_one_day_out'
        Determines the way the data is split into train and test set.
        leave_one_day_out
            one day is selected at random and used as test set. The emerging gap is closed by moving all
            succeeding events by one day into the past.
        tuple 
            First float is the train set ratio, second the ratio for the test set. All floats have to sum to one.
        triplet
            First is the ratio for the train set, second the ratio for the validation set, the 
            third is the ratio for the test. All floats have to sum to one.
    temporal : bool, default=False
        If true the split using the ratio is done with respect to time rather then to datapoints.
    return_day : bool, default=False
        If set, returns the day that is used for the test-set as fifth argument. Takes only
        an effect if `split=leave_one_day_out`.
    return_test_init_states : bool, default=False
        If set returns the last device values before the split. These values are useful when generating
        StateVectors for the test set.

    Examples
    --------
    .. code python:
        from pyadlml.dataset import fetch_amsterdam
        from pyadlml.model_selection import train_test_split

        data = fetch_amsterdam()
        df_devs, df_acts = data['devices'], data['activities']

        X_train, X_test, y_train, y_test = train_test_split(
            df_devs, df_acts, split='leave_one_day_out'
        )
        

    Returns
    -------
    X_train, X_test, y_train, y_test : all pd.DataFrames

    """
    SPLIT_LODA = 'leave_one_day_out'
    TD = '1D'
    epsilon = pd.Timedelta('1ns')
    is_train_val_test_split = isinstance(split, tuple) and len(split) == 3
    is_train_test_split = not is_train_val_test_split
    ratio_split = isinstance(split, tuple)

    if ratio_split and isinstance(split, tuple):
        for elem in split:
            assert isinstance(elem, float) and 0 < elem and elem < 1.0
        assert sum(split) == 1
        assert len(split) == 2 or len(split) == 3
    else:
        assert split == SPLIT_LODA

    assert isinstance(temporal, bool) & isinstance(return_day, bool) & isinstance(return_init_states, bool)
    assert not (ratio_split and return_day)

    # just to make sure
    df_devs = df_devs.sort_values(by=TIME).reset_index(drop=True)

    if split == SPLIT_LODA:
        # Split by leaving one day out
        rnd_day = _get_rnd_day(df_devs, padding=1)
        X_train, y_train = remove_days(df_devs, df_acts, days=[rnd_day])
        X_test, y_test = select_timespan(df_devs, df_acts, 
                start_time=rnd_day, end_time=rnd_day + pd.Timedelta(TD),
                clip_activities=True
        )
    elif len(split) == 3:
        # Split by fraction 
        if temporal:
            total_range = (df_devs.at[len(df_devs)-1, TIME] - df_devs.at[0, TIME]) 
            train_range = total_range * split[0]
            val_range = total_range * (split[0] + split[1])

            train_end_time = df_devs.at[0, TIME] + train_range
            val_end_time = df_devs.at[0, TIME] + val_range
        else:
            train_end_time = df_devs.at[int(len(df_devs) * split[0]), TIME]
            val_end_time = df_devs.at[int(len(df_devs) * (split[0] + split[1])), TIME]

        X_train, y_train = select_timespan(df_devs, df_acts, 
               end_time=train_end_time - epsilon, clip_activities=True
        )
        X_val, y_val = select_timespan(df_devs, df_acts, 
               start_time=train_end_time + epsilon, 
               end_time=val_end_time - epsilon, clip_activities=True
        )
        X_test, y_test = select_timespan(df_devs, df_acts, 
                start_time=val_end_time + epsilon, 
                clip_activities=True
        )


    else:
        # Split by fraction 
        if temporal:
            train_range = (df_devs.at[len(df_devs)-1, TIME] - df_devs.at[0, TIME]) * split[0]
            split_time = df_devs.at[0, TIME] + train_range
        else:

            split_idx = int(len(df_devs) * split[0])
            split_time = df_devs.at[split_idx, TIME]

        X_train, y_train = select_timespan(df_devs, df_acts, 
               end_time=split_time - epsilon, clip_activities=True
        )
        X_test, y_test = select_timespan(df_devs, df_acts, 
                start_time=split_time + epsilon, clip_activities=True
        )


    if is_train_val_test_split:
        res_lst = [X_train, X_val, X_test, y_train, y_val, y_test]
    else:
        res_lst = [X_train, X_test, y_train, y_test]

    if return_day:
        res_lst.append([rnd_day, rnd_day + pd.Timedelta(TD)])

    if return_init_states:
        last_states_train = get_last_states(X_train)    

        if is_train_val_test_split:
            last_states_val = get_last_states(X_val)
            for dev in last_states_train.keys():
                if dev not in last_states_val.keys():
                    last_states_val[dev] = last_states_train[dev]
        
        if is_train_val_test_split:
            states_dict = dict(
                init_states_val=last_states_train,
                init_states_test=last_states_val
            )
        else:
            states_dict = dict(
                init_states_test=last_states_train,
            )
        
        res_lst.append(states_dict)


    # Sanity checks
    # 1. check if activity or device appears in test or val set that is not present in the train set
    if is_train_val_test_split:
        dev_in_train_but_not_in_val =  set(X_train[DEVICE].unique()) - set(X_val[DEVICE].unique())
        dev_in_val_but_not_in_train =  set(X_val[DEVICE].unique()) - set(X_train[DEVICE].unique())
        if dev_in_train_but_not_in_val:
            print('Warning: Devices from train set %s produce no event in validation set!'%dev_in_train_but_not_in_val)
        assert not dev_in_val_but_not_in_train, 'Unseen device in validation set not contained in train set %s. Data cleaning required!!!'%dev_in_val_but_not_in_train

    dev_in_train_but_not_in_test =  set(X_train[DEVICE].unique()) - set(X_test[DEVICE].unique())
    dev_in_test_but_not_in_train =  set(X_test[DEVICE].unique()) - set(X_train[DEVICE].unique())
    if dev_in_train_but_not_in_test:
        print('Warning: Devices from train set %s produce no event in test set!'%dev_in_train_but_not_in_test)
    assert not dev_in_test_but_not_in_train, 'Unseen device in test set not contained in train set %s. Data cleaning required!!!'%dev_in_test_but_not_in_train


    if is_train_val_test_split:
        act_in_train_but_not_in_val =  set(y_train[ACTIVITY].unique()) - set(y_val[ACTIVITY].unique())
        act_in_val_but_not_in_train =  set(y_val[ACTIVITY].unique()) - set(y_val[ACTIVITY].unique())
        if act_in_train_but_not_in_val:
            print('Warning: Activities from train set %s do not appear in validation set!'%act_in_train_but_not_in_val)
        assert not act_in_val_but_not_in_train, 'Unseen activity in validation set, not contained in train set %s. Data cleaning required!!!'%act_in_val_but_not_in_train


    act_in_train_but_not_in_test =  set(y_train[ACTIVITY].unique()) - set(y_test[ACTIVITY].unique())
    act_in_test_but_not_in_train =  set(y_test[ACTIVITY].unique()) - set(y_train[ACTIVITY].unique())
    if act_in_train_but_not_in_test:
        print('Warning: Activities from train set %s do not appear in test set!'%act_in_train_but_not_in_test)
    assert not act_in_test_but_not_in_train, 'Unseen activity in test set, not contained in train set %s. Data cleaning required!!!'%act_in_test_but_not_in_train



    return res_lst


def _get_rnd_day(df_devs, retain_other_days=False, padding=0):
    """ Retrieves a random day from the dataset

    Parameters
    ----------
    X : pd.DataFrame
        with timeindex
    retain_other_days : bool, default=False
        determines whether all other days except for the random day are also returned
    padding : int, default=0
        How many days from start and from end should not be used

    Returns
    -------
    str or list

    """

    # get all days
    days = list(df_devs[TIME].dt.floor('d').value_counts().index)
    days = days[padding:len(days)-padding-1]

    # select uniformly a random day
    rnd_idx = np.random.randint(0, high=len(days)-1)
    rnd_day = days[rnd_idx]
    if retain_other_days:
        return rnd_day, days.pop(rnd_idx)
    else:
        return rnd_day


def _split_devs(df_devs, rnd_day, temporal=False):
    """ get indices of all data for that day and the others """
    if temporal:
        raise NotImplementedError
    else:
        rnd_dayp1 = rnd_day + pd.Timedelta('1D')
        mask = (rnd_day < df_devs[TIME]) & (df_devs[TIME] < rnd_dayp1)
        idxs_test = df_devs[mask].index.values
        idxs_train = df_devs[~mask].index.values
        return idxs_train, idxs_test


def _split_acts(df_acts : pd.DataFrame, rnd_day, temporal : bool =False) -> list:
    """ Get indices of all activities for that day and the others 
    
    Parameters
    ----------
    df_acts : pd.DataFrame
    temporal : 

    """
    if temporal:
        raise NotImplementedError
    else:
        rnd_dayp1 = rnd_day + pd.Timedelta('1D')
        mask_test = (rnd_day < df_acts[END_TIME]) & (df_acts[START_TIME] < rnd_dayp1)
        mask_train = (df_acts[START_TIME] < rnd_day) | (rnd_dayp1 < df_acts[END_TIME])
        idxs_test = df_acts[mask_test].index.values
        idxs_train = df_acts[mask_train].index.values
        return idxs_train, idxs_test


class TimeSeriesSplit(_BaseKFold):
    """
    Parameters
    ----------
    n_splits : int, default=5
        number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.
    test_size : int, default=None
        Used to limit the size of the test set. Defaults to n_samples // (n_splits + 1), which is the maximum allowed value with gap=0.
    gap : int, default=0
        Number of samples to exclude from the end of each train set before the test set.
    return_timestamp : bool, default=False
        When true timestamp intervals are returned rather than indicies. This is
        useful whenever data is upscaled or downscaled as the indicies in the testset c
        can not be known beforehand.
    temporal_split : bool, default=False
        If set, the splits are made based on the time rather than on the datapoints. This
        allows for rescaling of the data and applying the split afterwards.

    Examples
    --------
    >>> import os

    """
    EPS = pd.Timedelta('5ms')

    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0, return_timestamp=False,
                 temporal_split=False):
        self.return_timestamp = return_timestamp
        self.temporal_split = temporal_split
        self.max_train_size = max_train_size
        self.test_size = test_size

        if self.temporal_split:
            self.gap = pd.Timedelta('0s') if gap == 0 else gap
            assert isinstance(self.gap, pd.Timedelta)
        else:
            self.gap = gap

        super().__init__(n_splits, shuffle=False, random_state=None)


    def _temporal_split(self, X, y, groups):
        # create time_range from first device to last device
        assert isinstance(self.gap, pd.Timedelta)
        assert self.max_train_size is None or isinstance(self.max_train_size, pd.Timedelta)
        assert self.test_size is None or isinstance(self.test_size, pd.Timedelta)

        data_start = X[TIME].iloc[0]
        data_end = X[TIME].iloc[-1]
        n_folds = self.n_splits + 1 # |--|--|--|--|  k=3
        test_size = self.test_size if self.test_size is not None \
            else (data_end - data_start) // n_folds

        test_starts = pd.date_range(data_end - self.n_splits*test_size, data_end, freq=test_size)[:-1]

        res_lst = []
        for test_st in test_starts:
            train_et = test_st - self.gap - self.EPS
            test_et = test_st + test_size

            if self.max_train_size and self.max_train_size < train_et:
                train_st = train_et - self.max_train_size
            else:
                train_st = data_start - self.EPS

            if self.return_timestamp:
                res_lst.append(((train_st, train_et), (test_st, test_et)))
            else:
                train_idx = X[(train_st < X[TIME]) & (X[TIME] < train_et)].index.values
                test_idx = X[(test_st < X[TIME]) & (X[TIME] < test_et)].index.values
                res_lst.append((train_idx, test_idx))
        return res_lst


    def _index_split(self, X, y, groups):
        """ Blatantly copied from the original sklearn Timeseries split
        """
        assert isinstance(self.gap, int)
        assert self.max_train_size is None or isinstance(self.max_train_size, int)
        assert self.test_size is None or isinstance(self.test_size, int)

        n_samples = len(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = self.test_size if self.test_size is not None \
            else n_samples // n_folds

        # Make sure we have enough samples for the given split parameters
        if n_folds > n_samples:
            raise ValueError(
                (f"Cannot have number of folds={n_folds} greater"
                 f" than the number of samples={n_samples}."))
        if n_samples - gap - (test_size * n_splits) <= 0:
            raise ValueError(
                (f"Too many splits={n_splits} for number of samples"
                 f"={n_samples} with test_size={test_size} and gap={gap}."))

        indices = np.arange(n_samples)
        test_starts = range(n_samples - n_splits * test_size,
                            n_samples, test_size)
        res_lst = []
        for test_start in test_starts:
            train_end = test_start - gap
            if self.max_train_size and self.max_train_size < train_end:
                train_idxs = indices[train_end - self.max_train_size:train_end]  # sliding window
            else:
                train_idxs = indices[:train_end]  # expanding window
            test_idxs = indices[test_start:test_start + test_size]

            # own implementation addition
            if not self.return_timestamp:
                res_lst.append((train_idxs, test_idxs))
            else:
                train_st = X.iloc[train_idxs[0]][TIME] - self.EPS
                train_et = X.iloc[train_idxs[-1]][TIME] + self.EPS

                val_st = X.iloc[test_idxs[0]][TIME] - self.EPS
                val_et = X.iloc[test_idxs[-1]][TIME] + self.EPS

                res_lst.append(
                    ((train_st, train_et), (val_st, val_et))
                )

        return res_lst


    def split(self, X, y=None, groups=None):
        """ Generate indices or intervals to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        """
        if self.temporal_split:
            return self._temporal_split(X, y, groups)
        else:
            return self._index_split(X, y, groups)


class LeaveKDayOutSplit(object):
    """ LeaveKDayOut cross-validator

    Provides train/test indices to split data in train/test sets. Split
    dataset into one day out folds.

    Read more in the :ref:`User Guide <leave_one_day_out>`

    Parameters
    ----------
    k : int, default=1
        The number of days to use for the test set.
    n_splits : int, default=1
        The number of splits. All splits are exclusive, meaning there will not be more t TODO
    return_timestamps : bool, default=False
        When true timestamp intervals are returned rather than indicies. This is
        useful whenever data is upscaled or downscaled as the indicies in the testset
        can not be known beforehand.
    epsilon : str, default='5ms'
        the offset that is used to pad before the first and after the last interval for
        the timestamps. Has only an effect if *return_timestamps* is set to *true*
    offset : str, default='0s'
        The offset that is used to shift the start of a day
    shift : bool, defaul=False
        Determines whether to shift the

    Examples
    --------
    >>> import os
    """
    def __init__(self, n_splits=1, k=1, return_timestamps=False, epsilon='5ms', offset='0s', shift=False):
        self.n_splits = n_splits
        self.return_timestamp = return_timestamps
        self.k = k
        self.offset = pd.Timedelta(offset)
        self.eps = pd.Timedelta(epsilon)
        self.shift = shift

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        """ Generate indices to split data into training and test set.

        Parameters
        ----------
        X : pd.DataFrame
            device dataframe
        y : pd.Series
            activity dataframe

        Returns
        -------
        splits : list
            Returns tuples of splits of train and test sets
            example: [(train1, test1), ..., (trainn, testn)]
        """

        X = X.copy()
        first_day = X[TIME].iloc[0].floor('d')
        last_day = X[TIME].iloc[-1].ceil('d')
        days = pd.date_range(first_day, last_day, freq='1D').values
        print(days)
        days[1:-2] = days[1:-2] + self.offset

        N = len(days)
        if self.k is None:
            self.k = (N-2)//self.n_splits

        assert self.k <= (N-2)//self.n_splits, "The number of days for each split exceeds the possible"

        step_size = N//self.n_splits
        res = []
        for i in range(self.n_splits):
            test = (days[i*step_size], days[i*step_size + self.k])
            if i == 0:
                # case when | test | train |
                train = (days[i*step_size + self.k], days[-1])
            elif i == self.n_splits-1 and (i*step_size+self.k) == N-1:
                # case  when | train | test|
                train = (days[0], days[i*step_size])
            else:
                # case when | train | test | train |
                train = ((days[0], days[i*step_size]),
                         (days[i*step_size + self.k], days[-1]))

            if self.return_timestamp:
                res.append((train, test))
            else:
                def get_indices(df, l_bound, r_bound):
                    return df[(l_bound < df[TIME]) & (df[TIME] < r_bound)].index.values
                test_idxs = get_indices(X, test[0], test[1])
                if i == 0 or (i == self.n_splits-1 and (i*step_size+self.k) == N-1):
                    train_idxs = get_indices(X, train[0], train[1])
                else:
                    train_idxs_int_1 = get_indices(X, train[0][0], train[0][1])
                    train_idxs_int_2 = get_indices(X, train[1][0], train[1][1])
                    train_idxs = np.concatenate([train_idxs_int_1, train_idxs_int_2])

                    if self.shift:
                        # shift the second interval by that amount of days into the past
                        X.loc[train_idxs_int_2, TIME] = X[TIME] - pd.Timedelta(str(self.k) + 'D')
                        if y is not None:
                            y.loc[train_idxs_int_2, TIME] = y[TIME] - pd.Timedelta(str(self.k) + 'D')

                res.append((train_idxs, test_idxs))

        return res