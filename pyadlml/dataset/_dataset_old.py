

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

