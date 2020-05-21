import pandas as pd
import numpy as np
#from .kasteren import DatasetKasteren
START_TIME = 'Start time'
END_TIME = 'End time'
ID = 'Idea'
VAL = 'Val'
NAME = 'Name'
TIME_STEP_SIZE = 60


class DevManager():

    def __init__(self, kasteren):
        self._kast = kasteren
        self._sens_file_path = None
        self._dev_data = None
        self._sensor_label = {}
        self._sensor_label_hashmap = {}
        self._sensor_label_reverse_hashmap = {}

    def get_lbl_hashmap(self):
        return self._sensor_label_hashmap

    def get_lbl_reverse_hashmap(self):
        return self._sensor_label_reverse_hashmap


    def set_file_path(self, file_path):
        self._sens_file_path = file_path


    def _split_test_train_data(self, df, test_day):
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
        START_TIME = 'Start time'
        END_TIME = 'End time'
        test_df = df.loc[df[START_TIME].dt.date == test_day]# or \
                         #df[END_TIME].dt.date == test_day]

        train_df = df.loc[df[START_TIME].dt.date != test_day]# and \
                          #df[END_TIME].dt.date != test_day]

        #test_df = df[test_day_data]
        #train_df = df[~test_day_data]
        return train_df, test_df

    def _df2raw_seq(self, df: pd.DataFrame) -> np.ndarray:
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
        # todo not the whole thing is loaded and processed correctly
        # exit(-1)
        df = self._sorted_df_after_time(df)
        arr_len = len(df.index)
        res_arr = np.empty((arr_len), dtype=object)
        for i in range(0, arr_len):
            single_row_df = df.iloc[i]
            label = single_row_df['Name']
            value = single_row_df['Val']
            enc_lbl = self._sensor_label_hashmap[label][value]
            res_arr[i] = enc_lbl
            i +=1
        return res_arr

    def get_test_sorted_test_arr(self, test_day):
        """

        :return:
        """
        df_train, df_test = self._split_test_train_data(self._dev_data, test_day)
        df_test = self._sorted_df_after_time(df_test)
        return df_test

    def get_raw(self, test_day):
        """
        gets the observations aligned
        :param test_day:
        :return:
            train_x
                nd.array
            test_x
                nd.array
        """
        df = self._dev_data.copy()
        df_train, df_test = self._split_test_train_data(df, test_day)

        test_seq = self._df2raw_seq(df_test)    # type: np.ndarray
        train_seq = self._df2raw_seq(df_train)  # type: np.ndarray
        return test_seq, train_seq

    def load_last_dev_fired(self):
        pass

    def load_changed(self):
        pass

    def _debug_print_window_of_nparr(self, obs_seq, idx, end_time):
        if idx >= 10:
            print('end_time: ', end_time)
            print('obs_seq: ', obs_seq[idx-10:idx+10])
        else:
            print('end_time: ', end_time)
            print('obs_seq: ', obs_seq)

    def _mul_ts(self, c):
        """
        returns the time step multiplied by a constant c
            example:
                timestep = 30 # for 30 seconds
                then the method return 30*c seconds a s Timedelta format
        :return:
        """
        return pd.Timedelta(pd.offsets.Second(c*TIME_STEP_SIZE))

    def _df_get_time(self, df_row):
        """
        :param df: pandas dataframe of size
                                   Name                Time  Val
                1199  Hall-Bedroom door 2008-02-25 00:20:14    1
        :return: 2008-02-25 00:20:14
        """
        return df_row[1]


    def _df_get_label(self, df_row):
        """
        :param df: pandas dataframe of size
                                   Name                Time  Val
                1199  Hall-Bedroom door 2008-02-25 00:20:14    1
        :return: Hall-Bedroom door
        """
        return df_row[1][0]

    def _df_get_value(self, df_row):
        """
        :param df: pandas dataframe of size
                                   Name                Time  Val
                1199  Hall-Bedroom door 2008-02-25 00:20:14    1
        :return: 1
        """
        return df_row[1][2]

    def _get_df_last_row(self, df):
        return df.tail(1).iloc[0]

    def _get_df_first_row(self, df):
        return df.iloc[0]

    def _get_obs_from_first_row_from_slice(self, df_slice):
        first_row = df_slice.head(1).iloc[0]
        label = first_row['Name']
        val = first_row['Val']
        return self._sensor_label_hashmap[label][val]

    def _sorted_df_after_time(self, df):
        """
        transformes start time end time row to another row with observation val = 0
        :param df:
            Example:
                                    Name          Start time            End time  Val
                1199   Hall-Bedroom door 2008-02-25 00:20:14 2008-02-25 00:22:57    1
                1200   Hall-Bedroom door 2008-02-25 09:33:41 2008-02-25 09:33:42    1
                1201   Hall-Bedroom door 2008-02-25 09:33:47 2008-02-25 17:21:12    1
        :return:
            Example:
                                    Name                Time  Val
                1199   Hall-Bedroom door 2008-02-25 00:20:14    1
                1199   Hall-Bedroom door 2008-02-25 00:22:57    0
        """
        df_start = df.copy()
        df_end = df.copy()
        df_end = df_end.loc[:, df_end.columns != START_TIME]
        df_start = df_start.loc[:, df_start.columns != END_TIME]
        df_end[VAL] = 0

        # rename column 'End Time' and 'Start Time' to 'Time'
        new_columns = df_end.columns.values
        new_columns[1] = 'Time'
        df_end.columns = new_columns
        df_start.columns = new_columns
        new_df = pd.concat([df_end, df_start])
        new_df = new_df.sort_values('Time')
        return new_df


    def _sensor_file_to_df(self, path_to_file):
        sens_data = pd.read_csv(path_to_file,
                    sep="\t",
                    skiprows=23,
                    skipfooter=1,
                    parse_dates=True,
                    names=[START_TIME, END_TIME, ID, VAL],
                    engine='python' #to ignore warning for fallback to python engine because skipfooter
                    #dtype=[]
                    )
        sens_label = pd.read_csv(path_to_file,
                                 sep="    ",
                                 skiprows=6,
                                 nrows=14,
                                 skipinitialspace=4,
                                 names=[ID, NAME],
                                 engine='python'
                                 )
        sens_label[ID] = sens_label[ID].apply(lambda x: x[1:-1])
        sens_label[NAME] = sens_label[NAME].apply(lambda x: x[1:-1])
        sens_label[ID] = pd.to_numeric(sens_label[ID])

        # todo declare at initialization of dataframe
        sens_data[START_TIME] = pd.to_datetime(sens_data[START_TIME])
        sens_data[END_TIME] = pd.to_datetime(sens_data[END_TIME])

        res = pd.merge(sens_label, sens_data, on=ID, how='outer')
        res = res.sort_values('Start time')
        del res[ID]
        return sens_label, res


    def _df_to_seq(self, labels, df):

        import pandas as pd

        # create hashmap of sensor labels
        # duplicate all values
        #for row in labels.iterrows():
        #    labels.loc[-1] = [row[1][0],row[1][1]]
        #    labels.index = labels.index + 1
        #    labels = labels.sort_values(by=['Idea'])
        #print(labels)

        # create alternating zeros and 1 label dataframe
        lb_zero = labels.copy()
        lb_zero[VAL] = 0
        lb_zero = lb_zero.loc[:, lb_zero.columns != 'Idea']
        lb_one = labels.copy()
        lb_one[VAL] = 1
        lb_one = lb_one.loc[:, lb_one.columns != 'Idea']
        new_label = pd.concat([lb_one, lb_zero]).sort_values(NAME)
        new_label = new_label.reset_index(drop=True)

        #
        N=25
        #print(pd.Series(string.ascii_uppercase) for _ in range(N))

        #
        df_start = df.copy()
        df_end = df.copy()
        df_end = df_end.loc[:, df_end.columns != START_TIME]
        df_start = df_start.loc[:, df_start.columns != END_TIME]
        df_end[VAL] = 0

        # rename column 'End Time' and 'Start Time' to 'Time'
        new_columns = df_end.columns.values
        new_columns[1] = 'Time'
        df_end.columns = new_columns
        df_start.columns = new_columns


        #print(df_start.head(5))
        #print(df_end.head(5))

        new_df = pd.concat([df_end, df_start])
        new_df = new_df.sort_values('Time')
        #print(df.head(10))

        cut = int(len(new_df.index)*0.8)
        idx = 0
        for row in new_df.iterrows():
            label =row[1][0]
            time = row[1][1]
            value = row[1][2]
            # lookup id in new_label
            correct_row = new_label[(new_label[NAME] == label) \
                                    & (new_label[VAL] == value)]
            ide = correct_row.index[0]
            if idx < cut:
                self._train_seq.append(ide)
            else:
                self._test_seq.append((ide, time))
            idx += 1
        #new_label[NAME] = new_label[NAME].str.strip()
        #print(lst)
        #print(len(lst))
        #print(len(new_df.index))
        #self._train_seq = lst[:cut]
        #self._test_seq = lst[cut:]


        self._df = new_df
        self._sensor_label = new_label

        # create hashmap instead of shitty dataframe >:/
        self._sensor_label_hashmap = {}
        self._sensor_label_reverse_hashmap = {}
        idx = 0
        for row in new_label.iterrows():
            name = str(row[1][0])
            value = row[1][1]
            if idx%2 == 0:
                self._sensor_label_hashmap[name] = {}
                self._sensor_label_hashmap[name][value] = idx
                self._sensor_label_reverse_hashmap[idx] = name
            else:
                self._sensor_label_hashmap[name][value] = idx
                self._sensor_label_reverse_hashmap[idx] = name
            #if idx%5 == 0:
            #    print(self._sensor_label)
            #    for item in self._sensor_label_hashmap:
            #        print(item)

            idx+=1
            #print('--')

    def _create_activity_on_off_map(self, labels):
        """
        from the file create single single an on off dataframe
        :param labels:
            example:
                    Idea                Name
                0      1           Microwave
                1      5    Hall-Toilet door


        :return: pandas dataframe:
            example:
                    Name            Val
                0   Cups cupboard   1
                1   Cups cupboard   0
                ...
        """
        lb_zero = labels.copy()
        lb_zero[VAL] = 0
        lb_zero = lb_zero.loc[:, lb_zero.columns != 'Idea']
        lb_one = labels.copy()
        lb_one[VAL] = 1
        lb_one = lb_one.loc[:, lb_one.columns != 'Idea']
        new_label = pd.concat([lb_one, lb_zero]).sort_values(NAME)
        new_label = new_label.reset_index(drop=True)
        return new_label


    def load_basic(self):
        labels, df = self._sensor_file_to_df(self._sens_file_path)
        self._dev_data = df
        # create alternating zeros and 1 label dataframe
        new_label = self._create_activity_on_off_map(labels)
        self._create_device_hashmap(new_label)


    def _create_device_hashmap(self, new_label):
        self._sensor_label = new_label
        self._sensor_label_hashmap = {}
        self._sensor_label_reverse_hashmap = {}
        idx = 0
        for row in new_label.iterrows():
            name = str(row[1][0])
            value = row[1][1]
            if idx%2 == 0:
                self._sensor_label_hashmap[name] = {}
                self._sensor_label_hashmap[name][value] = idx
                self._sensor_label_reverse_hashmap[idx] = name
            else:
                self._sensor_label_hashmap[name][value] = idx
                self._sensor_label_reverse_hashmap[idx] = name
            #if idx%5 == 0:
            #    print(self._sensor_label)
            #    for item in self._sensor_label_hashmap:
            #        print(item)

            idx+=1
            #print('--')
