import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas._libs.index import timedelta

from hassbrain_algorithm.datasets._dataset import DataInterfaceHMM
from .act_manager import ActManager
from .dev_manager import DevManager

START_TIME = 'Start time'
END_TIME = 'End time'
ID = 'Idea'
VAL = 'Val'
NAME = 'Name'
TIME_STEP_SIZE = 60


class DatasetKasteren(DataInterfaceHMM):
    def __init__(self, repr='raw'):
        # different representations
        if repr == 'raw':
            self._load_raw = True
            self._load_changed, self._load_last_fired = [False, False]
        elif repr == 'changed':
            self._load_changed = True
            self._load_raw, self._load_last_fired = [False, False]
        elif repr == 'last_fired':
            self._load_last_fired = False
            self._load_raw, self._load_changed = [False, False]

        self._devs = DevManager(self)
        self._acts = ActManager(self)

        self._train_seq = None

        # true x
        self._test_acts = None
        # true y
        self._test_obs = None

    def set_file_paths(self, dict):
        self._devs.set_file_path(dict['sens_file_path'])
        self._acts.set_file_path(dict['act_file_path'])

    def get_state_lbl_hashmap(self):
        return self._acts.get_lbl_hashmap()

    def get_state_lbl_reverse_hashmap(self):
        return self._acts.get_lbl_reverse_hashmap()

    def get_obs_lbl_hashmap(self):
        return self._devs.get_lbl_hashmap()

    def get_obs_lbl_reverse_hashmap(self):
        return self._devs.get_lbl_reverse_hashmap()

    def get_obs_list(self):
        """
        returns the set of encoded observations that one can make
        :return:
        """
        lst = []
        for key, value in self._devs.get_lbl_reverse_hashmap().items():
            lst.append(key)
        return lst


    def get_state_list(self):
        """
        retrieve all encoded activity
        :return:
            sorted list of all states
        """
        lst = []
        for key, value in self._acts.get_lbl_reverse_hashmap().items():
            lst.append(key)
        return lst

    def encode_state_lbl(self, label):
        return self._acts.get_lbl_hashmap()[label]

    def decode_state_lbl(self, id):
        return self._acts.get_lbl_reverse_hashmap()[id]

    def encode_obs_lbl(self, label, state):
        """
            self._devs.get_raw()
        returns the id of a sensor given a label
        :param label:
        :return:
        """
        return self._devs.get_lbl_hashmap()[label][state]

    def decode_obs_lbl(self, id):
        """
        retrieves the label given a sensor id
        :param id:
        :return:
        """
        return self._devs.get_lbl_reverse_hashmap()[id]

    def is_multi_seq_train(self):
        return False

    def _gen_rnd_test_day(self):
        """
        generates a random day inside of the dataset
        this is used for testing after the one day of priniciple
        :return:
        """
        return self._acts.get_random_day()

    def load_data(self):
        """
        is called by controller
        :return:
        """
        self._devs.load_basic()
        self._acts.load_basic()
        test_day = self._gen_rnd_test_day()
        train_x = None
        test_x = None
        test_y = None
        if self._load_raw:
            test_x, train_x = self._devs.get_raw(
                test_day=test_day
            )
            df_test = self._devs.get_test_sorted_test_arr(test_day)
            test_x, test_y = self._acts.get_raw(
                df_test,
                test_x,
                test_day=test_day
            )
        elif self._load_last_fired:
            self._load_last_dev_fired()

        elif self._load_changed:
            pass
        else:
             raise ValueError

        assert train_x is not None
        assert test_x is not None
        assert test_y is not None

        self._train_seq = train_x
        self._test_acts = test_x
        self._test_obs = test_y

    def get_test_labels_and_seq(self):
        """
        :return:
            z: np.array of observations
            x, list of lists of states
        """
        z, x = None
        tmp = self._test_arr.T
        return [tmp[0]], [tmp[1]]

    def get_train_seq(self):
        """
        the train seq is an
        :return:
            train seq
        """
        return self._train_seq

    def _create_time_df(self, df, timestep):
        """
            turns the dataframe into a new dataframe where for each timestep a row is
            generated where the activities of all sensors are printed
        :param df:
            Example:
                                    Name                Time  Val
                1199   Hall-Bedroom door 2008-02-25 00:20:14    1
                1199   Hall-Bedroom door 2008-02-25 00:22:57    0
        :return:
            Example:
                23  1   3   4   5   1   3   2   1   9   8   7   1

        """
    def _load_raw_devs(self):
        """
        discretised in timeslices of del t = 60 sec
        example:
            sensor s4 changed state at t1 to 1 and at t4 to 0
                s1 s2 s3 s4
            t0  0  0  0  0
            t1  0  0  0  1
            t2  0  0  0  1
            t3  0  0  0  1
            t4  0  0  0  1
            t5  0  0  0  0

        :return:
        """
        # calculate start timestamp
        # calculate end timestamp
        # calculate timesteps between
        # create np array
        #
        raise NotImplementedError


    def _load_change_point_devs(self):
        """
        this representation assings 1 to timeslices where sensor changes state and 0 otherwise
        example:
            sensor s4 changed state at t1 and t4
                s1 s2 s3 s4
            t1  0  0  0  1
            t2  0  0  0  0
            t3  0  0  0  0
            t4  0  0  0  1
        :return:
        """
        # in case of hmm learning there would be an extra observation for a row to be zero at all
        raise NotImplementedError

    def _load_df_last_dev_fired(self):
        """
        last sensor rep a
            sensor changes state and rep continues to assign a 1 to the last sensor
            that changes state until a new sensor changes its state
        example:
            sensor s4 changed state at t1 and t4
                s1 s2 s3 s4
            t0  0  0  1  0
            t1  0  0  0  1
            t2  0  0  0  1
            t3  0  0  0  1
            t4  0  0  0  1
            t5  0  0  0  1
            t6  0  1  0  0
        :return: numpy array
        """

    def _load_last_dev_fired(self):
        """
        last sensor rep a
            sensor
            sensor changes state and rep continues to assign a 1 to the last sensor
            that changes state until a new sensor changes its state
        example:

        :return:
            1D numpy array, np.int32
            example:
                [1,2,0,....,]
        """

        labels, df = self._sensor_file_to_df(self._sens_file_path)
        self._df_to_seq(labels, df)

        # create alternating zeros and 1 label dataframe
        new_label = self._create_activity_on_off_map(labels)
        self._create_device_hashmap(new_label)
        # sort the data in ascending order
        df = self._sorted_df_after_time(df)




        start_time = self._df_get_time(self._get_df_first_row(df))
        end_time = self._df_get_time(self._get_df_last_row(df))
        #print('start_time: ', start_time)
        #print('end_time: ', end_time)
        #print('type(starttime): ', type(start_time))
        i = end_time - start_time # type: timedelta
        #print('timedel: ', td)
        #print('type(td): ', type(td))
        #print('secs: ',  td.seconds)
        #print('days: ',  td.days)
        # calculate timesteps
        timesteps = int((i.seconds + (i.days*24*60*60)) / TIME_STEP_SIZE)

        # make time as an index
        df.set_index('Time', inplace=True)
        print(df.head(10))
        obs_seq = np.zeros(timesteps, dtype=np.int32)
        # do first iteration outside
        tmp_obs = self._get_obs_from_first_row_from_slice(df)
        obs_seq[0] = tmp_obs
        # this counter is incremented if multiple observations are appended
        # to obs_seq and
        td_offset_multi = 0
        print_debug = False
        td = 1
        for i in range(1, timesteps):
            td += 1
            #print('~'*100)
            td_offset = self._mul_ts(td)
            end_time = start_time + td_offset
            df_slice = df.loc[start_time:end_time] # type: DataFrame
            row_count = len(df_slice.index)
            if row_count > 2:
                """
                this places a problem as multiple device changes where made within a 
                timestep. As this only concernes the last fired sensor the timeslice 
                is tagged with the last sensor that fired 
                todo mabye seek a different solution 
                """
                print('this shit has to be tacklejdkll'*100)
                print('end_time: ', end_time)
                print('df_slice: ', df_slice)
                # extracts the timestamp from datetimeindex
                last_entry_time = df_slice.tail(1).index[0]
                #df_slice = df_slice.iloc[1:] # remove first row
                df_slice = df_slice.iloc[-1:] # remove first row
                print('df_slice: ', df_slice)
                print('*'*100)
                #tmp_tmp_obs = []
                #for i in range(len(df_slice.index)):
                #    # for each row remove row and get obs_symbol
                #    tmp_tmp_obs.append(
                #        self._get_obs_from_first_row_from_slice(df_slice)
                #    )
                #    df_slice = df_slice.iloc[1:]
                # extend the array by number
                #print(tmp_tmp_obs)
                #print('len obsseq: ', len(obs_seq))
                #print('insert ...')
                #print('len obsseq: ', len(obs_seq))

                #print('*'*100)

                #print('last_entry: ', last_entry_time)
                #print('z'*10)
                start_time = last_entry_time
                td = 1
                #print('new_start_time: ', start_time)
                df_slice = df.loc[start_time:end_time] # type: DataFrame
                #print('new_df_slice: ', df_slice)
                tmp_obs = self._get_obs_from_first_row_from_slice(
                    df_slice
                )
            elif row_count == 2:
                if print_debug:
                    print('lulu'*20)
                    print('start_time: ', start_time)
                    print('diff: ', end_time - pd.offsets.Second(i*TIME_STEP_SIZE))

                #print('new start_time: ', start_time)
                print('x*'*10)
                print('start_time: ', start_time)
                print('end_time: ', end_time)
                start_time = end_time - self._mul_ts(1)
                td = 1
                print('new start_time: ', start_time)
                print('df_slice: ' ,df_slice)
                last_entry_time = df_slice.tail(1).index[0]
                print('last_entry_time: ', last_entry_time)
                #df_slice = df.loc[start_time:last_entry_time] # type: DataFrame
                df_slice = df.tail(1)
                print('new df_slice: ' ,df_slice)
                print('x*'*10)
                tmp_obs = self._get_obs_from_first_row_from_slice(
                    df_slice
                )
            if print_debug:
                pass
                #print('df_slice: ', df_slice)

            obs_seq[i] = tmp_obs
            # get dataframe
            # check if sensor has changed
            # if true than make
                # reassign new start_time
                # concat other observation
            # if false append
                # concat same observation



            # debug
            #if pd.datetime.fromisoformat("2008-02-25 09:32:00") < end_time:
            #    if not print_debug:
            #        print(df.head(10))
            #        print_debug= True
            #    print('td_offset: ', td_offset)
            #    print('start_time: ', start_time)
            #    print('end_time: ', end_time)
            #    self._debug_print_window_of_nparr(obs_seq, i, end_time)
            #    print('--'*100)
            #time_to_break = "2008-02-25 09:38:00"
            #if pd.datetime.fromisoformat(time_to_break) < end_time:
            #    print('rup_time: ', pd.datetime.fromisoformat(time_to_break))
            #    print('end_time: ', end_time)
            #    break

        #    label = self._df_get_label(row)
        #    time = row[1][1]
        #    value = row[1][2]
        #    obs_id = self._sensor_label_hashmap[label][value]
        #    print('label: ', label)
        #    print('time: ', time)
        #    print('val: \t', value)
        #    print('obs_id: ', obs_id)
        #    ide = obs_id
        #    if idx < cut:
        #        self._train_seq.append(ide)
        #    else:
        #        self._test_seq.append((ide, time))
        #    idx += 1
        #    break
        #new_label[NAME] = new_label[NAME].str.strip()
        #print(lst)
        #print(len(lst))
        #print(len(new_df.index))
        #self._train_seq = lst[:cut]
        #self._test_seq = lst[cut:]
        self._train_seq = obs_seq
        print(np.info(obs_seq))
        print(obs_seq)

