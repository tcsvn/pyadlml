from datetime import timedelta

import pandas as pd
import numpy as np
import yaml

from hassbrain_algorithm.benchmarks.datasets.dataset import DataInterfaceHMM
from detective.core import HassDatabase

START_TIME = 'Start time'
END_TIME = 'End time'
TIME = 'time'
ID = "ide"
STATE = "state"
ID = 'Idea'
VAL = 'Val'
NAME = 'Name'

class DatasetHomeassistant(DataInterfaceHMM):
    def __init__(self):
        self._database_path = None
        self._act_file_path = None
        # if this path is set a yaml file containing the activitys and
        # labels are presupposed
        self._labels_path = None

        # this lists can be set by the controller
        # should be used when hassbrain web activitys are used
        self._act_list = None
        self._dev_list = None

        self._train_seq = []
        self._test_seq = []

        self._is_multi_seq_train = False

        self._sensor_label_hashmap = {}
        self._sensor_label_reverse_hashmap = {}

        self._activity_label_hashmap = {}
        self._activity_label_reverse_hashmap = {}


        # Homeassistant specific stuff
        self._df = None


    """
    is used by web interface to set the activity and devices before stuff is done  
    """
    def set_state_list(self, act_list):
        self._act_list = act_list

    def set_obs_list(self, dev_list):
        self._dev_list = dev_list

    def get_state_lbl_hashmap(self):
        return self._activity_label_hashmap

    def get_state_lbl_reverse_hashmap(self):
        return self._activity_label_reverse_hashmap

    def get_obs_lbl_hashmap(self):
        return self._sensor_label_hashmap

    def get_obs_lbl_reverse_hashmap(self):
        return self._sensor_label_reverse_hashmap

    def set_file_paths(self, dict):
        self._database_path = dict['database_path']
        self._act_file_path = dict['act_file_path']
        try:
            self._labels_path = dict['labels_path']
        except:
            pass

    def get_test_labels_and_seq(self):
        """
        :return:
            list of lists of observations,
            list of lists of states
        """
        obs_seqs = []
        lbl_seqs = []
        for seq in self._test_seqs:
            s = seq.T
            obs_seqs.append(s[0])
            lbl_seqs.append(s[1])

        return lbl_seqs, obs_seqs

    def is_multi_seq_train(self):
        """
        if the dataset supports multiple sequences or consists of one sequence in the
        training
        :return:
        """
        return False


    def get_train_seq(self):
        return self._train_seq

    def get_train_seqs(self):
        raise NotImplementedError

    def get_obs_list(self):
        """
        returns the set of encoded observations that one can make
        :return:
        """
        lst = []
        for key, value in self._sensor_label_reverse_hashmap.items():
            lst.append(key)
        return lst

    def get_state_list(self):
        """
        retrievie all encoded activity
        :return:
            sorted list of all states
        """
        #print(self._activity_label_reverse_hashmap)
        lst = []
        for key, value in self._activity_label_reverse_hashmap.items():
            lst.append(key)
        return lst

    def encode_state_lbl(self, label):
        """
        returns the id of an activity given the corresp. label
        :param label:
        :return:
        """
        return self._activity_label_hashmap[label]

    def decode_state_lbl(self, id):
        """
        returns the label of an activity given the corresp. id
        :param label:
        :return:
        """
        return self._activity_label_reverse_hashmap[id]

    def encode_obs_lbl(self, label, state):
        """
        returns the id of a sensor given a label
        :param label:
        :return:
        """
        return self._sensor_label_hashmap[label][state]

    def decode_obs_lbl(self, id):
        """
        retrieves the label given a sensor id
        :param id:
        :return:
        """
        return self._sensor_label_reverse_hashmap[id]


    def decode_obs_seq(self, seq):
        """
        decodes an iterable list of encoded observations to the labels
        :param seq:
        :return:
        """
        new_seq = []
        for item in seq:
            new_seq.append(self.decode_obs_lbl(item))
        return new_seq

    def load_data(self):
        self._create_label_hashmaps()
        print('*I'*100)
        print(self._activity_label_hashmap)
        print('--')
        print(self._activity_label_reverse_hashmap)
        print('--')
        print(self._sensor_label_hashmap)
        print('--')
        print(self._sensor_label_reverse_hashmap)
        print('*I'*100)

        self._create_sequences()

    def _create_label_hashmaps(self):
        """
        this method presupposes that either a path is set
        :return:
        """
        # either load by file or this was set previously by controller
        if self._labels_path is not None:
            with open(self._labels_path, 'r') as stream:
                data_loaded = yaml.load(stream)
                if self._act_list is None:
                    self._act_list = data_loaded['activitys']
                if self._dev_list is None:
                    self._dev_list = data_loaded['devices']

        # encode strings with numbers
        for i, activity in enumerate(self._act_list):
            self._activity_label_hashmap[activity] = i
            self._activity_label_reverse_hashmap[str(i)] = activity
            i += 1

        #print('act_lbl_hm: ', self._activity_label_hashmap)
        #print('act_lbl_rev_hm: ', self._activity_label_reverse_hashmap)

        i = 0
        for device in self._dev_list:
            self._sensor_label_hashmap[device] = {}
            self._sensor_label_hashmap[device][0] = i
            self._sensor_label_reverse_hashmap[i] = device
            i += 1
            self._sensor_label_hashmap[device][1] = i
            self._sensor_label_reverse_hashmap[i] = device
            i += 1

        #print('dev_lbl_hm: ', self._sensor_label_hashmap)
        #print('dev_lbl_rev_hm: ', self._sensor_label_reverse_hashmap)


    def _format_mat_full(self, x):
        pd.set_option('display.max_rows', len(x))
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        pd.set_option('display.float_format', '{:20,.10f}'.format)
        pd.set_option('display.max_colwidth', -1)
        s = str(x)
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.float_format')
        pd.reset_option('display.max_colwidth')
        return s


    def _act_file_to_df(self):

        df_act = pd.read_csv(self._act_file_path,
                            sep=" ",
                            names=[TIME, ID, STATE],
                            engine='python'
        )
        #print(df_act.head(10))
        df_act[TIME] = pd.to_datetime(df_act[TIME])

        # remove activities that are not set in the state list
        df_act = df_act[df_act[ID].isin(self._act_list)]

        return df_act


    def _create_sequences(self):
        """
        use the database sql and the hashmaps to create these ultra long sequences
        :return:
        """

        # load activity csv to df
        act_df = self._act_file_to_df()

        # load database
        from detective.core import HassDatabase
        if self._database_path is None:
            raise ValueError

        #print('-'*100)
        #print(self._database_path)
        db = HassDatabase('sqlite:////' + self._database_path)
        db.fetch_all_data()

        self._df = db.master_df
        from detective.core import BinarySensors

        # query
        sensors_binary_df = BinarySensors(db.master_df).data

        # split df into test sequence and train sequence
        # test sequence is only annotated data
        # train sequence consists of days where no data was annotated
        DAY = '%Y-%m-%d'
        # first get series of days which are to be omitted
        days_to_omit = act_df[TIME].map(lambda x: x.strftime(DAY)).drop_duplicates()

        days_where_sth_was_logged = sensors_binary_df.index.strftime(DAY).isin(days_to_omit)

        #print(df[criteria])
        #print(days_to_omit)
        # split into test and train dataset
        test_df = sensors_binary_df[days_where_sth_was_logged]
        self._create_test_sequences(test_df, act_df)

        # the days where nothing was logged
        train_df = sensors_binary_df[~days_where_sth_was_logged]
        train_df = self._exclude_not_listed_sensors(train_df)
        self._create_train_sequence(train_df)

    def _create_test_sequences(self, test_df, act_df):
        #print(test_df)
        #print(len(test_df))
        #print('--'*10)
        #for idx in range(len(test_df)):

        """
        create a list of dataframes for each coherent logging
        sessions
        """
        df_seq = []
        tmp_row = None
        offset = 0
        for i, row in enumerate(act_df.iterrows()):
            if i == 0:
                tmp_row = row[1][0]
                #print(tmp_row)
                #print('~'*10)
                continue
            act_row = row[1][0]
            diff = act_row - tmp_row
            state = row[1][2]
            #print('-'*100)
            #print('prev: ', i-1, ' iter', i)
            #print('state: ', state)
            #print('diff: ', diff)
            #print(act_row)
            if (state != 1 and diff > timedelta(minutes=1)) or i == len(act_df)-1:
                #print('condition was true')
                """
                case if the logged activity isn't ended and the timedelta is
                greater than one is meaning that a new logging sequence is started.
                Reminder: if an activity is changed in the app, then there is nearly no
                timedelay between the switches. 
                """
                #print('offset: ', offset)
                #print('i: ', i)
                #print(act_df[offset:i+1])
                #print('--'*10)
                df_seq.append(act_df[offset:i+1])
                offset = i+1
            tmp_row = act_row
            #print('--'*20)



        """
        for each logging session get subset of the data
        and save it in a list
        """
        test_df = self._exclude_not_listed_sensors(test_df)
        #print('#'*20)
        #print(act_df)
        #print('--'*100)
        #print(df_seq)
        self._test_seqs = []
        for i, act_seq_df in enumerate(df_seq):
            #print('*'*100)
            #print(act_seq_df)
            #print('-'*100)
            #print(test_df)
            #print('-'*100)
            # get start time and endtime of curr. log seq
            start_date = act_seq_df.iloc[0][0]
            end_date = act_seq_df.iloc[len(act_seq_df)-1][0]

            # get
            #print(start_date)
            #print(end_date)
            #print('--'*10)

            #mask = (test_df[TIME] > start_date) & (test_df[TIME] <= end_date)
            df_slice = test_df.loc[start_date:end_date]
            #print('~'*100)
            #print(df_slice)
            #print(act_seq_df)
            #print('~'*100)
            """
            for each subset tag the observations with the different
            activity labels 
            """
            trainmat = np.zeros((len(df_slice), 2),dtype=object)
            # go pairwise through the act seq
            for i in range(0, len(act_seq_df), 2):
                start_act = act_seq_df
                #print('~'*10)
                act_start_time = start_act.iloc[i][0]
                act_end_time = start_act.iloc[i+1][0]
                act_slice = df_slice.loc[act_start_time:act_end_time]
                act_name = start_act.iloc[i][1]
                act_name2 = start_act.iloc[i+1][1]
                if act_name2 != act_name:
                    print('the pairwise iteration of the sequences didn\'t work')
                    raise ValueError
                #print('\tname1n2: ', act_name, act_name2)
                #print('\tact_start: \t', act_start_time)
                #print('\tact_end: \t', act_end_time)
                #print('\tact_slice: \t', act_slice)
                #print('\tact_name: ', act_name)
                #print('*'*10)
                #if act_slice
                """
                filter the act_slice to only include binary sensor that are
                specified by either the config file or the web application
                """

                if int(len(act_slice)) != 0:
                    for i, row in enumerate(act_slice.iterrows()):
                        #print('\t' + '+'*10 + ': ' + str(i))
                        label_name, state = self._row2labelnstate(row)
                        trainmat[i][0] = self.encode_obs_lbl(label_name, state)
                        #print('\t\t' + str(self._activity_label_hashmap))
                        trainmat[i][1] = self.encode_state_lbl(act_name)
                        #print('\t\t', trainmat)
                        #print('\t\t' + '--'*10)
                        #print('\t\t', label_name)
                        #print('\t\t', state)
                        #print('\t\t', self._sensor_label_hashmap, '\n')

                    self._test_seqs.append(trainmat)
            #print(self._format_mat_full(test_df))

    def _create_train_sequence(self, train_df):
        #print('\n')
        #print('\n')
        #print('\n')
        #print('#'*100)
        # 2019-04-16, 2019-04-17
        #print(self._format_mat_full(train_df))
        # make from train_def the train_seq
        for row in train_df.iterrows():
            label_name, state = self._row2labelnstate(row)
            self._train_seq.append(self.encode_obs_lbl(
                label_name,
                state
            ))


    def _exclude_not_listed_sensors(self, df):
        # exclude binary sensors not listed in sensor list
        #print('*'*100)
        #print(self._dev_list)
        #print(df)
        df = df[self._dev_list]
        # drop the rows that are all NaN
        df = df.dropna(how='all')
        #print(df)
        #print('*'*100)
        return df


    def _row2labelnstate(self, row):
        """
        gets a row from the dataframe of homeassistant or activities
        and return the label and the state of the entity
        :param row:
        :return:
        """
        entity = row[1].dropna()
        label_name = entity.index[0]

        if entity[0]:
            state = 1
        else:
            state = 0
        return label_name, state
        #print(self._train_seq)


        # make from test_df and states the test_seq
        #query = ("SELECT state, last_changed"
        #         "FROM states"
        #         "WHERE entity_id "
        #         "in ('binary_sensor.motion_hallway')"
        #         " AND NOT state='unknown'")

        #response = db.perform_query(query)
        #df = pd.DataFrame(response.fetchall())
        #print(df.head(10))
