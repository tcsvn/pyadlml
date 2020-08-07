import pandas as pd
import numpy as np
from enum import Enum


class DataRep(Enum):
    RAW = 'raw'
    CHANGEPOINT ='changed'
    LAST_FIRED = 'last_fired'

"""
    df_activities:
        start_time | end_time   | activity
        ---------------------------------
        timestamp   | timestamp | act_name

    df_devices:
        toggle_time | device_1 | ...| device_n
        ----------------------------------------
        timestamp   | state_0  | ...| state_n
"""
class Data():
    def __init__(self, activities, devices):
        self.df_activities = activities
        self.df_devices = devices


class _Dataset(object):



    def get_contingency_table(self):
        """
        Returns
        -------
        cont_tab : pd.Dataframe
            a table containing the contingencies
        """
        df_act = self._acts._act_data.copy()
        df_devs = self._devs._dev_data.copy() # type: pd.DataFrame
        act_series = self._label_data(df_devs)
        labeled_act_series = act_series.apply(self._ids2label)
        tmplist = labeled_act_series.to_list()
        cols = df_devs.columns.values
        for col in cols:
            lbl = self._devs.get_lbl_reverse_hashmap()[col]
            new_off_lbl = lbl + '-off'
            new_on_lbl = lbl + '-on'
            df_devs[new_off_lbl] = ~df_devs[col]
            df_devs.rename(columns={col:new_on_lbl}, inplace=True)

        df_devs['activities'] = tmplist

        tmp = df_devs.groupby(by='activities').aggregate(np.sum)
        return tmp
        print()
        # todo create crosstab

    def _label_data(self, dev_df_test: pd.DataFrame) -> pd.Series:
        """
        for each row in the dataframe select the corresponding activity from the
        timestamp and create a np array with the activity labels
        :param dev_df_test:
            Name                    0      1      2      3   ...     10     11     12     13
            Time                                             ...
            2008-03-20 00:34:38  False  False  False  False  ...  False  False  False   True
            2008-03-20 00:34:39  False  False  False  False  ...  False  False  False  False
            ...
        :return:
            numpy ndarray 1D
        """
        test_x = np.zeros((len(dev_df_test)), dtype=np.int64)
        #arr = np.zeros((num_cols), dtype=np.int64)
        series = pd.Series(test_x)
        i = 0
        for date in dev_df_test.index:
            try:
                act_id = self._acts._time2actid(date)
            except LookupError:
                """
                if the observation doesn't match an activity assign the idle activity
                """
                act_id = self._acts._activity_label_hashmap[ACT_IDLE_NAME]

            series[i] = act_id
            i += 1
        return series

    def load_data(self):
        """
        is called by controller
        :param test_repr
            is either None
                then the test data equals the train data
            'one_day_out'
                then a day is picked out

        :return: None
        """
        self._acts.load_basic()
        self._devs.load_basic()

        test_all_x, test_all_y = self._load_all_labeled_data()
        self._test_all_x = test_all_x
        self._test_all_y = test_all_y

        train_y = None
        test_x = None
        test_y = None

        if self._test_sel == 'all':
            self._devs.train_eq_test_dat()
            self._acts.train_eq_test_dat()
        elif self._test_sel == 'one_day_out':
            if self._test_day is None:
                test_day = self._acts.get_random_day()
            else:
                import datetime
                test_day = datetime.datetime.strptime(
                    self._test_day, '%Y-%m-%d').date()

            self._devs.split_train_test_dat(test_day)
            self._acts.split_train_test_dat(test_day)
        else:
            raise ValueError

        train_y, test_x, test_y = self._get_specific_repr()
        assert train_y is not None
        assert test_x is not None
        assert test_y is not None

        self._train_y = train_y
        self._test_x = test_x
        self._test_y = test_y

    def _get_specific_repr(self):
        """loads either raw, last fired or changepoint representation of data

        Raises
        ------
        ValueError
            no representation was chosen in the init process

        Returns
        -------
        None

        """
        train_y = None
        test_x = None
        test_y = None
        if self._data_representation == DataRep.RAW:
            test_y, train_y = self._devs.get_raw()
        elif self._data_representation == DataRep.LAST_FIRED:
            test_y, train_y = self._devs.get_last_fired()
        elif self._data_representation == DataRep.CHANGEPOINT:
            test_y, train_y = self._devs.get_changepoint()
        else:
            print('no data representation was chosen')
            raise ValueError
        df_test = self._devs.get_test_df()
        test_x, test_y = self._acts.label_data(df_test.copy(), test_y, idle=False)
        return train_y, test_x, test_y



START_TIME = 'Start time'
END_TIME = 'End time'
ACT_IDLE_NAME = 'idle'

class _ActManager():
    def __init__(self, include_idle=False):
        self._activity_label = {}
        self._activity_label_hashmap = {}
        self._activity_label_reverse_hashmap = {}
        self._include_idle = include_idle

    def load_basic(self):
        """
        Returns
        -------

        """
        self._act_data, self._activity_label_hashmap, \
            self._activity_label_reverse_hashmap = self._load_basic()

    def _load_basic(self):
        """
        has to somehow load the data and return a dataframe where in a
        single row the Start time and the end time is denoted with the activity mapped
        onto the Idea column
        Returns
        -------
            act_data (pd.Dataframe)
                           Start time            End time  Idea
                0 2008-02-25 19:40:26 2008-02-25 20:22:58     5
                1 2008-02-25 20:23:12 2008-02-25 20:23:35     6

            activity_label <class 'dict'>:

            activity_label_hashmap <class 'dict'>:

                {'leave house': 0, 'use toilet': 1, 'take shower': 2, ... }
            activity_label_reverse_hashmap

                {0: 'leave house', 1: 'use toilet', 2: 'take shower', 3: 'go to bed', ... }
        """
        raise NotImplementedError


    def get_lbl_hashmap(self):
        return self._activity_label_hashmap

    def get_lbl_reverse_hashmap(self):
        return self._activity_label_reverse_hashmap

    def get_lbl_list(self):
        """
        Returns
        -------
            activity label list
            returns the decoded activities as a list
        """
        return list(self._activity_label_hashmap.keys())

    def get_activities_count(self):
        """
        Returns
        -------
        res pd.Dataframe
                       leave house  use toilet  ...  prepare Dinner  get drink
            occurence           33         111  ...              10         19

        """
        df = self._act_data.groupby('Idea').count()     # type: pd.DataFrame
        df = df.drop(columns=[END_TIME])
        df.columns = ['occurence']
        df = df.transpose()
        print('asdf')
        lst = df.columns
        new_lst = []
        for item in lst:
            new_lst.append(
                self._activity_label_reverse_hashmap[item]
            )
        df.columns = new_lst
        return df

    def get_total_act_duration(self, freq='sec'):
        """
        counts the timedeltas of the activites and calculates the
        percentage of activities in relation to each other
        Parameters
        ----------
        freq : str
           the frequency the stuff should be get
        Returns
        -------
        res : pd.Dataframe
                       leave house  use toilet  ...  prepare Dinner  get drink
            perc               0.3         ...                 0.2        0.01

        """
        label = 'total'
        df = self._get_act_dur_helper(label, freq='sec')
        # convert to seconds
        df = df.transpose()
        df.columns = self._df_idcol2lblcol(df.columns)
        return df

    def _get_act_dur_helper(self, label, freq='sec'):
        df = self._act_data
        df['DIFF'] = df[END_TIME] - df[START_TIME]
        df = df.drop(columns=[END_TIME, START_TIME])
        df = self._act_data.groupby('Idea').sum()     # type: pd.DataFrame
        df.columns = [label]
        if freq == 'sec':
            df[label] = df[label].apply(lambda x: x.total_seconds())
        elif freq == 'min':
            df[label] = df[label].apply(lambda x: x.total_minutes())
        else:
            df[label] = df[label].apply(lambda x: x.total_hours())
        return df

    def _df_idcol2lblcol(self, cols):
        """ turn the columns of a dataframe with ids into labeled columns
        Parameters
        ----------
        cols : lst
        Returns
        -------
        new_cols: lst
        """
        new_cols = []
        for item in cols:
            new_cols.append(
                self._activity_label_reverse_hashmap[item]
            )
        return new_cols


    def label_data(self, dev_df_test: pd.DataFrame, test_y, idle=True) -> np.ndarray:
        """
        for each row in the dataframe select the corresponding activity from the
        timestamp and create a np array with the activity labels
        :param dev_df_test:
            Name                    0      1      2      3   ...     10     11     12     13
            Time                                             ...
            2008-03-20 00:34:38  False  False  False  False  ...  False  False  False   True
            2008-03-20 00:34:39  False  False  False  False  ...  False  False  False  False
            ...
        :param idle: boolean
            if true this leads to datapoints not falling into a logged activity to be
            labeled as idle
        :return:
            numpy ndarray 1D
        """
        test_x = np.zeros((len(dev_df_test)), dtype=np.int64)
        i = 0
        not_labeled_lst = []
        for date in dev_df_test.index:
            try:
                act_id = self._time2actid(date)
            except LookupError:
                """
                if the observation doesn't match an activity assign the idle activity
                """
                if idle:
                    act_id = self._activity_label_hashmap[ACT_IDLE_NAME]
                else:
                    not_labeled_lst.append(i)
                    act_id = -1
            test_x[i] = act_id
            i += 1
        # correct test_y and test_x
        corrected_test_x = np.delete(test_x, not_labeled_lst, axis=0)
        corrected_test_y = np.delete(test_y, not_labeled_lst, axis=0)
        assert len(corrected_test_y) == len(corrected_test_x)

        return corrected_test_x, corrected_test_y

    def split_train_test_dat(self, test_day):
        """
        is called after a random test_day is selected
        :param test_day:
        :return:
        """
        df = self._act_data
        mask_st_days = (df[START_TIME].dt.day == test_day.day)
        mask_st_months = (df[START_TIME].dt.month == test_day.month)
        mask_st_year = (df[START_TIME].dt.year == test_day.year)
        mask_et_days = (df[END_TIME].dt.day == test_day.day)
        mask_et_months = (df[END_TIME].dt.month == test_day.month)
        mask_et_year = (df[END_TIME].dt.year == test_day.year)
        mask1 = mask_st_days & mask_st_months & mask_st_year
        mask2 = mask_et_days & mask_et_months & mask_et_year
        mask = mask1 | mask2
        test_df = df[mask]
        train_df = df[~mask]
        self._test_df = test_df
        self._train_df = train_df

TIME_STEP_SIZE = '30min'
    def get_devices_count(self):
        """
        Returns
        -------
        res pd.Dataframe
                       binary_sensor.  use toilet  ...  prepare Dinner  get drink
            occurence               33         111  ...              10         19

        """
        df_cp = self._apply_change_point(self._dev_data.copy())
        cnt = df_cp.apply(pd.value_counts)
        cnt.drop(False, inplace=True)
        new_columns = []
        for item in cnt.columns:
            new_columns.append(
                self._sensor_label_reverse_hashmap[item]
            )
        cnt.columns = new_columns
        return cnt

    def get_changepoint(self):
        """
        Returns
        -------

        """
        df_test = self._df_test
        df_train = self._df_train

        assert self._freq is not None

        df_res = self._resample_data(df_test, self._freq)
        df_cp = self._apply_change_point(df_res.copy())
        self._df_test = df_cp

        df_train = self._resample_data(df_train, self._freq)
        df_train = self._apply_change_point(df_train)
        self._df_train = df_train

        test_y = df_cp.values  # type: np.ndarray
        train_y = df_train.values  # type: np.ndarray

        return test_y, train_y

    def _ffill_bfill_inv_first_dev(self, new_df):
        """
        :param new_df:
                    Name                    0      1      2    3   ...   10   11     12     13
            Time                                           ...
            2008-02-25 00:20:14    NaN    NaN    NaN  NaN  ...  NaN  NaN    NaN   True
            2008-02-25 00:22:57    NaN    NaN    NaN  NaN  ...  NaN  NaN    NaN  False
            2008-02-25 09:33:41    NaN    True   NaN  NaN  ...  NaN  NaN    NaN   True
            2008-02-25 09:33:42    NaN    NaN    NaN  NaN  ...  NaN  NaN    NaN  False
        :return:
                    Name                    0      1      2    3   ...   10   11     12     13
            Time                                            ...
            2008-02-25 00:20:14    True False  False   True ... False False True  True
            2008-02-25 00:22:57    True False  False   True ... False False True False
            2008-02-25 09:33:41    True True   False   True ... False False True  True
            2008-02-25 09:33:42    True True   False   True ... False False True False

        """
        new_df2 = new_df.fillna(method='ffill')  # type: pd.DataFrame
        for col in new_df2.columns:
            """
            gets index of first nan and fills up the rest with the opposite value
            """
            tmp = new_df2[col]
            idx = tmp.first_valid_index()
            val = tmp[idx]
            new_df2[col].fillna(not val, inplace=True)
        return new_df2



    def split_train_test_dat(self, test_day):
        """
        :param df:
            dataframe
                                    Name          Start time            End time  Val
                1199   Hall-Bedroom door 2008-02-25 00:20:14 2008-02-25 00:22:57    1
                1200   Hall-Bedroom door 2008-02-25 09:33:41 2008-02-25 09:33:42    1
        :param test_day:
            pandas. timestamp
            e.g 2008-03-15
        :return:
            df without test day
            df with test day
        """
        df = self._dev_data
        mask_days = (df.index.day == test_day.day)
        mask_months = (df.index.month == test_day.month)
        mask_year = (df.index.year == test_day.year)
        mask = mask_days & mask_months & mask_year
        train_df = df[~mask]
        test_df = df[str(test_day)]
        self._df_test = test_df
        self._df_train = train_df





    def _apply_change_point(self, df):
        """

        Parameters
        ----------
        df

        Returns
        -------

        """
        i = 0
        insert_stack = []
        pushed_prev_it_on_stack = False
        len_of_row = len(df.iloc[0])
        series_all_false = self._gen_row_false_idx_true(len_of_row, [])
        for j in range(1, len(df.index)+2):
            if (len(insert_stack) == 2 or (len(insert_stack) == 1 and not pushed_prev_it_on_stack)) \
                    and (i-1 >= 0):
                """
                the case when either the stack is maxed out ( == 2) or a row from before
                2 iterations is to be written to the dataframe and it is not the first two rows
                """
                item = insert_stack[0]
                df.iloc[i-1] = item
                insert_stack.remove(item)
            else:
                df.iloc[i-1] = series_all_false
            if pushed_prev_it_on_stack:
                pushed_prev_it_on_stack = False

            if j >= len(df.index):
                # this is to process the last two lines also as elements are appended
                # 2 rows in behind
                i += 1
                continue

            rowt = df.iloc[i]   # type: pd.Series
            rowtp1 = df.iloc[j] # type: pd.Series

            if not rowt.equals(rowtp1):
                idxs = self._get_cols_that_changed(rowt, rowtp1) # type: list
                row2insert = self._gen_row_false_idx_true(len(rowt), idxs)
                insert_stack.append(row2insert)
                pushed_prev_it_on_stack = True
            i += 1
        return df

    def _gen_row_false_idx_true(self, len_row, idxs):
        """
        generates a row for the dataframe with everyithing set to false
        instead of the indicies
        Returns
        -------
        """
        row = pd.Series([False for i in range(len_row)])
        for i in idxs:
            row.iloc[i] = True
        return row


    def get_last_fired(self):
        """
        gets the observations aligned
        :param test_day:
        :return:
            train_y
                2d nd.array
            test_y
                2d nd.array
        """
        df_test = self._df_test
        df_train = self._df_train

        assert self._freq is not None
        df_test = self._resample_data(df_test, self._freq)
        df_test = self._apply_last_fired(df_test)
        self._df_test = df_test

        df_train = self._resample_data(df_train, self._freq)
        df_train = self._apply_last_fired(df_train)
        self._df_train = df_train

        test_y = df_test.values  # type: np.ndarray
        train_y = df_train.values  # type: np.ndarray

        return test_y, train_y

    def _row2false_except_col(self, row, idx):
        """

        :param row:
        :param idx:
        :return:
        """
        row = row.apply(lambda x: False)
        row.iloc[idx] = True
        return row

    def get_raw(self):
        """ upsamples the raw observations to the set frequency
        :param test_day:
        :return:
            train_x
                2d nd.array
            test_x
                2d nd.array
        """
        df_test = self._df_test
        df_train = self._df_train

        assert self._freq is not None
        df_test = self._resample_data(df_test, self._freq)
        self._df_test = df_test

        df_train = self._resample_data(df_train, self._freq)
        self._df_train = df_train

        test_y = self._df_test.values  # type: np.ndarray
        train_y = self._df_train.values  # type: np.ndarray
        return test_y, train_y

    def _idx_dev_first_turn_true(self, df):
        """ returns the index of the device that first turned on
        if there are multiple devices choose the one, that has the
        shortest time to stay true, because this distorts data the least

        :param df:
            Name                    0      1      2      3   ...     10     11     12    13
            Time                                             ...
            2008-03-04 09:00:00  False   True  False  False  ...  False  False  False  True
            2008-03-04 09:30:00  False   True  False  False  ...  False  False  False  True
            ...

        :return:
            idx (int)
                the index of the row where a device first changed
            row_loc (int)
                the location of the row where a device first changed
        """
        row_first_change = None
        row0 = None
        for j in range(0, len(df.index)):
            row = df.iloc[j]
            idxs_where_row_is_true = [i for i, x in enumerate(row) if x]
            if len(idxs_where_row_is_true) > 0:
                if len(idxs_where_row_is_true) > 1:
                    row0 = row
                    break
                else:
                    # the case when only one entry is true and found
                    return idxs_where_row_is_true[0], j

        """
        get the first change in the selected indicies and return them
        """
        for j in range(1, len(df.index)):
            row = df.iloc[j] # type: pd.Series
            if not row.equals(row0):
                idxs = self._get_cols_that_changed(row0, row)
                return self._cols_select_rand(idxs), j

    def _cols_select_rand(self, idxs):
        """
        Parameters
        ----------
        idxs (list)
            list of indicies of a panda dataframe

        Returns
        -------
        idx (int)
            a random index
        """
        if len(idxs) == 1:
            return idxs[0]
        else:   # multiple indicies changed at once, then get one random
            rand_idx = np.random.random_integers(0, len(idxs)-1)
            return idxs[rand_idx]

    def _apply_last_fired(self, df):
        """

        Parameters
        ----------
        df

        Returns
        -------

        """
        idx_last_fired, row_first_change = self._idx_dev_first_turn_true(df)

        df = self._alf_set_rows_false_till_change(df, row_first_change)
        df.iloc[0][idx_last_fired] = True
        curr_row = df.iloc[0].copy()  # type: pd.Series

        for j in range(1, len(df.index)):
            row = df.iloc[j]
            if curr_row.equals(row):
                df.iloc[j] = self._row2false_except_col(row, idx_last_fired)
                continue
            else:
                # get change idx
                idxs = self._get_cols_that_changed(curr_row, row) # type: list
                idx_last_fired = self._cols_select_rand(idxs)
                idxs.remove(idx_last_fired)
                for idx in idxs:
                    row[idx] = not row[idx]


                # save how change in old df looks like
                curr_row = row.copy()

                # set row j to 0000 idx 0000
                df.iloc[j] = self._row2false_except_col(row, idx_last_fired)
                sum = df.iloc[j].sum()
                sum2 = df.iloc[j-1].sum()
                if sum > 1 or sum2 > 1:
                    print('')
                assert sum2 == 1 and sum == 1
        return df

    def _alf_set_rows_false_till_change(self, df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """ sets the upper section of a dataframe to false until the index of the first
        changed device
        Parameters
        ----------
        df (pd.Dataframe)
            Name                    0      1      2      3   ...     10     11     12    13
            Time                                             ...
            2008-03-04 09:00:00  False   True  False  False  ...  False  False  False  True
            2008-03-04 09:30:00  False   True  False  False  ...  False  False  False  True
            2008-03-04 10:00:00  False   True  True   False  ...  False  False  False  True

        idx (int)
            e.g 3

        Returns
        -------
        df (pd.Dataframe)

            Name                    0      1      2      3   ...     10     11     12    13
            Time                                             ...
            2008-03-04 09:00:00  False   False False  False  ...  False  False  False  False
            2008-03-04 09:30:00  False   False False  False  ...  False  False  False  False
            2008-03-04 10:00:00  False   True  True   False  ...  False  False  False  True
        """
        for col in df.columns:
            df[col].values[:idx] = False
        return df

    def _get_cols_that_changed(self, row1, row2):
        ""
        bool_mask = row1.eq(row2)
        idx_diff = [i for i, x in enumerate(bool_mask) if not x]
        return idx_diff

