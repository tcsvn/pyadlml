import pandas as pd
import numpy as np
#from .kasteren import DatasetKasteren
START_TIME = 'Start time'
END_TIME = 'End time'
ID = 'Idea'
VAL = 'Val'
NAME = 'Name'
TIME_STEP_SIZE = 60

class ActManager():
    #def __init__(self, kasteren: DatasetKasteren):
    def __init__(self, kasteren):
        self._kast = kasteren
        self._act_data = None # type: pd.DataFrame
        self._act_file_path = None
        self._activity_label = {}
        self._activity_label_hashmap = {}
        self._activity_label_reverse_hashmap = {}

    def get_lbl_hashmap(self):
        return self._activity_label_hashmap

    def get_lbl_reverse_hashmap(self):
        return self._activity_label_reverse_hashmap

    def set_file_path(self, file_path):
        self._act_file_path = file_path

    def get_random_day(self):
        """
        returns a random date of the activity dataset
        :return:
            datetime object
        """
        assert self._act_data is not None
        max_idx = len(self._act_data.index)
        rnd_idx = np.random.random_integers(0, max_idx)
        rnd_start_time = self._act_data.iloc[rnd_idx]['Start time'] # type: pd.Timestamp
        return rnd_start_time.date()

    def _check_if_any_parallel_acts(self, df):
        """
        :return:
            True if there is an activity that ends beyond another acitivies beginning
            False if case above is not true
        """
        for row in df.iterrows():
            start_time = row[1]['Start time']
            tmp = df[df['Start time'] >= start_time]
            val = tmp[tmp['End time'] <= start_time] # type: pd.DataFrame
            if not val.empty:
                return False
        return True


    def _time2_corr_act_id(self, time):
        """

        :param time:
            timestamp
            2008-02-26 00:39:25
        :return:
            int
            key value of an encoded activity
        """
        act_row =  self._act_data[(self._act_data['Start time'] <= time) \
                & (self._act_data['End time'] >= time)]
        self._check_if_any_parallel_acts(self._act_data)
        if len(act_row.index) != 1:
            if act_row.empty:
                print('time: %s not in arr'%(time))
                raise KeyError
            elif len(act_row.index) == 2:
                # more elements than one in the array
                # overlapping activities
                if 1 in act_row['Idea'].values:
                    for row in act_row.iterrows():
                        idea = row[1]['Idea']
                        label = self._activity_label_reverse_hashmap[idea]
                        if label == 'go to bed':
                            idea_key = row['Idea'].iloc[0]
                            return int(idea_key)
                        else:
                            print('lulula'*10)
                            print(act_row)
                            print(label)
                            print('lulula'*10)
                else:
                    print('time: %s lead to more elem'%(time))
                    print(act_row)
                    raise KeyError
            else:
                print('lalu'*100)
                print('time: ', time)
                print('idx: ', act_row.index)
                print(act_row)
                print('lalu'*100)

        idea_key = act_row['Idea'].iloc[0]
        return int(idea_key)

    def _dev_row2enc(self, row):
        label = row['Name']
        value = row['Val']
        return self._kast.encode_obs_lbl(label, value)

    def get_raw(self, df: pd.DataFrame, test_x: np.ndarray, test_day):
        """
        creates the corresponding activity sequence for test_x, the true
        activities for each corresp. observation
        :param df:
            dataframe holding the test device data of the day in sorted
            and
        :param test_x:
           [12 13 24 25 16 17 ... ]
        :param test_day:
            timestamp:

        :return:
            nd array of type in32
            e.g
            [0 2 3 0 0 0 1 ... ]
        """
        test_y = np.empty((len(test_x)), dtype=np.int32)
        arr_len = len(test_x)
        print(df.head(10))
        print(test_x[:10])
        to_delete_index = []    # contains the index of elements to be deleted
        for i in range(0, arr_len):
            # get sensor reading and its timestamp
            # look up in which category of pd it falls
            # finally assign label
            single_row = df.iloc[i]     # type: pd.DataFrame
            time = single_row['Time']
            #enc_label = self._dev_row2enc(single_row)
            try:
                enc_act = self._time2_corr_act_id(time)
            except KeyError:
                """
                is reached if a device firing outside of recorded activity
                """
                test_y[i] = -1
                to_delete_index.append(i)
            except Exception as e:
                print('exec: ', e)
            assert enc_act is not None
            test_y[i] = enc_act
        print('~'*100)
        print(to_delete_index)
        print('y: ', test_y)
        print('len y: ', len(test_y))
        test_y = np.delete(test_y, to_delete_index)
        print('y: ', test_y)
        print('len y: ', len(test_y))
        exit(-1)
        return test_x, test_y

    def _remap_ids(self, id_map, act_data):
        """
        ma
        :param id_map
            id_map:  {1: 0, 4: 1, 5: 2, 10: 3, 13: 4, 15: 5, 17: 6}

        :param act_data:
            e.g
                         Start time            End time  Idea
              0 2008-02-25 19:40:26 2008-02-25 20:22:58    15
              1 2008-02-25 20:23:12 2008-02-25 20:23:35    17

        :return:
                           Start time            End time  Idea
                0 2008-02-25 19:40:26 2008-02-25 20:22:58     5
                1 2008-02-25 20:23:12 2008-02-25 20:23:35     6
        """
        act_data['Idea'] = act_data['Idea'].map(id_map)
        return act_data

    def load_basic(self):
        """
        :param path_to_file:
        :return:
        """
        id_map = self._create_data_activity_labels()
        act_data = self._ld_act_dat_from_file()
        self._act_data = self._remap_ids(id_map, act_data)
        self._act_data_time_merged = self._act_dat_merge_time_sort(self._act_data)

    def _activity_file_to_df(self, path_to_file):
        """
        :param path_to_file:
        :return:
        """
        pass

    def _create_test_seq(self, act_data):
        TIME = 'time'
        # label test sequence
        self._test_arr = np.zeros((len(self._test_seq), 2))
        """
        offset is used to correct index if an observation falls out of the 
        measuered range
        """
        offset = 0
        for idx in range(0, len(self._test_seq)):
            entry = self._test_seq.pop(0)
            timestamp = entry[1]
            #mask = (act_data[TIME] > timestamp) & (act_data[TIME] <= timestamp)
            mask_lower = (act_data[TIME] <= timestamp)
            lower_slice = act_data.loc[mask_lower][-1:]
            lidx = int(lower_slice.index[0])
            test = act_data.loc[lidx]

            # time of the next value above lower_slice
            upper_time = test[-1:].loc[lidx][TIME]
            if not upper_time >= timestamp:
                offset += 1
            else:
                idea = int(test[-1:].loc[lidx][ID])
                self._test_arr[idx-offset][0] = entry[0]
                self._test_arr[idx-offset][1] = idea

        self._test_arr = self._test_arr[:-offset]


    def _create_data_activity_labels(self):
        """
        loads the encoded activity labels from kasteren data file
        creates a mapping between
        :return:
            <class 'dict'>:
                {'leave house': 0,
                 'use toilet': 1, 'take shower': 2, ... }
            <class 'dict'>:
                {0: 'leave house', 1: 'use toilet',
                2: 'take shower', 3: 'go to bed', ... }
             <class 'dict'>:
                {1: 0, 4: 1, 5: 2, 10: 3, 13: 4, 15: 5, 17: 6}
        """
        act_label = pd.read_csv(self._act_file_path,
                                sep=":",
                                skiprows=5,
                                nrows=7,
                                skipinitialspace=5,
                                names=[ID, NAME],
                                engine='python'
                                )

        act_label[NAME] = act_label[NAME].apply(lambda x: x[1:-1])
        act_label[ID] = pd.to_numeric(act_label[ID])


        # create encoded mapping
        activity_label_hashmap = {}
        activity_label_reverse_hashmap = {}
        id_map = {}
        i = 0
        for row in act_label.iterrows():
            name = str(row[1][1])
            value = row[1][0]
            id_map[value] = i
            activity_label_hashmap[name] = i
            activity_label_reverse_hashmap[i] = name
            i += 1
        self._activity_label = act_label
        self._activity_label_hashmap = activity_label_hashmap
        self._activity_label_reverse_hashmap = activity_label_reverse_hashmap
        return  id_map

    def _act_dat_merge_time_sort(self, df):
        """
        merges start time and end time columns into time and sorts the dataframe
        :param df:
                dataframe
                              Start time            End time  Idea
                0   2008-02-25 19:40:26 2008-02-25 20:22:58    15
                1   2008-02-25 20:23:12 2008-02-25 20:23:35    17
        :return:
            dataframe:
                                    time  Idea
                0   2008-02-25 20:22:58    15
                1   2008-02-25 20:23:35    17
                2   2008-02-25 21:52:36     4

        """
        #reformating datetime
        df_start = df.copy()
        df_end = df.copy()
        df_end = df_end.loc[:, df_end.columns != START_TIME]
        df_start = df_start.loc[:, df_start.columns != END_TIME]
        TIME = 'time'
        # rename column 'End Time' and 'Start Time' to 'Time'
        new_columns = df_end.columns.values
        new_columns[0] = TIME
        df_end.columns = new_columns
        df_start.columns = new_columns

        new_df = pd.concat([df_end, df_start])
        df = new_df.sort_values(TIME)
        df[TIME] = pd.to_datetime(df[TIME])
        return df

    def _ld_act_dat_from_file(self):
        """
        creates a valid dataframe from a file
        :return:
        """
        df = pd.read_csv(self._act_file_path,
                         sep="\t",
                         skiprows=23,
                         skipfooter=1,
                         parse_dates=True,
                         names=[START_TIME, END_TIME, ID],
                         engine='python' #to ignore warning for fallback to python engine because skipfooter
                         #dtype=[]
                         )
        df[START_TIME] = pd.to_datetime(df[START_TIME])
        df[END_TIME] = pd.to_datetime(df[END_TIME])
        return df
