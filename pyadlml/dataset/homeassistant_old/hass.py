from datetime import timedelta

import pandas as pd
import numpy as np
import yaml
from hassbrain_algorithm.datasets._dataset import _Dataset
from hassbrain_algorithm.datasets.homeassistant.act_manager import HassActManager
from hassbrain_algorithm.datasets.homeassistant.dev_manager import HassDevManager

START_TIME = 'Start time'
END_TIME = 'End time'
TIME = 'time'
ID = "ide"
STATE = "state"
ID = 'Idea'
VAL = 'Val'
NAME = 'Name'

class DatasetHomeassistant(_Dataset):
    def __init__(self, **params):
        self._database_path = None
        self._act_file_path = None
        # if this path is set a yaml file containing the activitys and
        # labels are presupposed
        self._labels_path = None

        # this lists can be set by the controller
        # should be used when hassbrain web activitys are used
        self._set_act_and_dev_list_manual = True
        self._act_list = None
        self._dev_list = None

        _Dataset.__init__(self, **params)
        try:
            freq = params['freq']
        except KeyError:
            freq = None
        self._devs = HassDevManager(self, freq=freq)
        self._acts = HassActManager(self, include_idle=self._include_idle)



    """
    is used by web interface to set the activity and devices before stuff is done  
    """
    def set_state_list(self, act_list):
        self._act_list = act_list

    def set_obs_list(self, dev_list):
        self._dev_list = dev_list

    def set_file_paths(self, conf_dict):
        with open(conf_dict['path_to_config']) as f:
            data = yaml.safe_load(f)
            self._devs.set_file_path(data['database_path'])
            self._acts.set_file_path(data['act_file_path'])
            if self._set_act_and_dev_list_manual:
                self._acts.set_act_labels_path(data['labels_path'])
                self._devs.set_dev_labels_path(data['labels_path'])

    def is_multi_seq_train(self):
        """
        if the dataset supports multiple sequences or consists of one sequence in the
        training
        :return:
        """
        return False

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


    def _create_sequences(self):
        """
        use the database sql and the hashmaps to create these ultra long sequences
        :return:
        """

        # load_basic activity csv to df
        act_df = self._act_file_to_df()

        # load_basic database
        from detective.core import HassDatabase
        if self._database_path is None:
            raise ValueError

        #print('-'*100)
        #print(self._database_path)
        db = HassDatabase('sqlite:////' + self._database_path)
        db.fetch_all_data()

        self._df = db.master_df
        from detective.core import HassbrainCompatibleDevices

        # load_basic multiple sensors, (lights, switches and binary_sensors as things with on of value)
        sensors_binary_df = HassbrainCompatibleDevices(db.master_df).data
        self._df = sensors_binary_df
        #self._print_full(sensors_binary_df.head(20))
        #print('~'*10)
        #self._print_full(self._df.head(20))
        #exit(-1)

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

    @classmethod
    def load_domain_knowledge(cls, file_path):
        with open(file_path, 'r') as file:
            import json
            data = json.load(file)
            act_data = data['activity_data']
            loc_data = data['loc_data']

        import datetime
        for act_data_point in act_data:
            act_data_point['start'] = datetime.datetime.strptime(
                act_data_point['start'],
                '%H:%M:%S').time()
            act_data_point['end'] = datetime.datetime.strptime(
                act_data_point['end'],
                '%H:%M:%S').time()
        return act_data, loc_data
