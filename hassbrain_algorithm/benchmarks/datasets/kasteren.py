import pandas as pd
import numpy as np
from hassbrain_algorithm.benchmarks.datasets.dataset import DataInterfaceHMM

START_TIME = 'Start time'
END_TIME = 'End time'
ID = 'Idea'
VAL = 'Val'
NAME = 'Name'

class DatasetKasteren(DataInterfaceHMM):
    def __init__(self):
        self._sens_file_path = None
        self._act_file_path = None
        self._train_seq = []
        self._test_seq = []

        #N x 2 numpy array
        #first value is id of observation
        #second value is id of state
        self._test_arr = None

        self._df = None

        self._sensor_label = None
        self._sensor_label_hashmap = None
        self._sensor_label_reverse_hashmap = None

        self._activity_label = None
        self._activity_label_hashmap = None
        self._activity_label_reverse_hashmap = None

    def set_file_paths(self, dict):
        self._sens_file_path = dict['sens_file_path']
        self._act_file_path = dict['act_file_path']

    def get_test_labels_and_seq(self):
        """
        :return:
            list of observations,
            list of states
        """
        tmp = self._test_arr.T
        return tmp[0], tmp[1]

    def get_train_seq(self):
        return self._train_seq

    def get_obs_list(self):
        """
        returns the set of encoded observations that one can make
        :return:
        """
        lst = []
        for key, value in self._sensor_label_reverse_hashmap.items():
            lst.append(key)
        return lst

    def load_data(self):
        self._load_sensors()
        self._load_activitys()

    def get_state_list(self):
        """
        retrievie all encoded activity
        :return:
            sorted list of all states
        """
        lst = []
        for key, value in self._activity_label_reverse_hashmap.items():
            lst.append(key)
        return lst

    def decode_state_label(self, id):
        return self._activity_label_reverse_hashmap[id]

    def encode_state_label(self, label):
        return self._activity_label_hashmap[label]

    def encode_obs_lbl(self, label, state):
        """
        returns the id of a sensor given a label
        :param label:
        :return:
        """
        return self._sensor_label_hashmap[label][state]

    def decode_obs_label(self, id):
        """
        retrieves the label given a sensor id
        :param id:
        :return:
        """
        return self._sensor_label_reverse_hashmap[id]


    # todo flag for deletion
    #def get_sensor_labels(self):
    #    lst = []
    #    modu = 0
    #    for key, value in self._sensor_label_reverse_hashmap.items():
    #        if modu%2 == 0:
    #            lst.append(value)
    #        modu +=1
    #    return lst





    def _load_sensors(self):
        labels, df = self._sensor_file_to_df(self._sens_file_path)
        self._df_to_seq(labels, df)

    def _load_activitys(self):
        """
        todo load data too
        :param path_to_file:
        :return:
        """
        self._load_activity_labels()
        self._load_activity_data()

    def _load_activity_labels(self):
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
        self._activity_label_hashmap = {}
        self._activity_label_reverse_hashmap = {}
        for row in act_label.iterrows():
            name = str(row[1][1])
            value = row[1][0]
            self._activity_label_hashmap[name] = value
            self._activity_label_reverse_hashmap[value] = name
        self._activity_label = act_label

    def _load_activity_data(self):
        act_data = pd.read_csv(self._act_file_path,
            sep="\t",
            skiprows=23,
            skipfooter=1,
            parse_dates=True,
            names=[START_TIME, END_TIME, ID],
            engine='python' #to ignore warning for fallback to python engine because skipfooter
            #dtype=[]
            )
        act_data[START_TIME] = pd.to_datetime(act_data[START_TIME])
        act_data[END_TIME] = pd.to_datetime(act_data[END_TIME])


        #reformating datetime
        df_start = act_data.copy()
        df_end = act_data.copy()
        df_end = df_end.loc[:, df_end.columns != START_TIME]
        df_start = df_start.loc[:, df_start.columns != END_TIME]
        TIME = 'time'
        # rename column 'End Time' and 'Start Time' to 'Time'
        new_columns = df_end.columns.values
        new_columns[0] = TIME
        df_end.columns = new_columns
        df_start.columns = new_columns

        new_df = pd.concat([df_end, df_start])
        act_data = new_df.sort_values(TIME)
        act_data[TIME] = pd.to_datetime(act_data[TIME])

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





    def _activity_file_to_df(self, path_to_file):
        """
        :param path_to_file:
        :return:
        """
        pass

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
        import string

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


