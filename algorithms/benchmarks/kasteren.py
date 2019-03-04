import pandas as pd

START_TIME = 'Start time'
END_TIME = 'End time'
ID = 'Idea'
VAL = 'Val'
NAME = 'Name'

class DatasetKasteren():
    def __init__(self):
        self._train_seq = []
        self._test_seq = []
        self._df = None
        self._sensor_label = None
        self._sensor_label_hashmap = None
        self._sensor_label_reverse_hashmap = None

        self._activity_label = None
        self._activity_label_hashmap = None
        self._activity_label_reverse_hashmap = None


    def get_test_seq(self):
        return self._test_seq

    def get_train_seq(self):
        return self._train_seq

    def get_sensor_labels(self):
        lst = []
        modu = 0
        for key, value in self._sensor_label_reverse_hashmap.items():
            if modu%2 == 0:
                lst.append(value)
            modu +=1
        return lst

    def get_sensor_list(self):
        """
        returns the set of encoded observations that one can make
        :return:
        """
        lst = []
        for key, value in self._sensor_label_reverse_hashmap.items():
            lst.append(key)
        return lst

    def get_activity_list(self):
        """
        retrievie all encoded activity
        :return:  list
        """
        lst = []
        for key, value in self._activity_label_reverse_hashmap.items():
            lst.append(key)
        return lst

    def get_activity_id_from_label(self, label):
        return self._activity_label_hashmap[label]

    def get_activity_label_from_id(self, id):
        return self._activity_label_reverse_hashmap[id]

    def get_sensor_id_from_label(self, label, state):
        """
        returns the id of a sensor given a label
        :param label:
        :return:
        """
        return self._sensor_label_hashmap[label][state]

    def get_sensor_label_from_id(self, id):
        """
        retrieves the label given a sensor id
        :param id:
        :return:
        """
        return self._sensor_label_reverse_hashmap[id]



    def load_sensors(self, path_to_file):
        labels, df = self._sensor_file_to_df(path_to_file)
        self._df_to_seq(labels, df)

    def load_activitys(self, path_to_file):
        """
        todo load data too
        :param path_to_file:
        :return:
        """
        act_label = pd.read_csv(path_to_file,
                                 sep=":",
                                 skiprows=5,
                                 nrows=7,
                                 skipinitialspace=5,
                                 names=[ID, NAME],
                                 engine='python'
                                 )

        act_label[NAME] = act_label[NAME].apply(lambda x: x[1:-1])
        act_label[ID] = pd.to_numeric(act_label[ID])

        self._activity_label_hashmap = {}
        self._activity_label_reverse_hashmap = {}
        for row in act_label.iterrows():
            name = str(row[1][1])
            value = row[1][0]
            self._activity_label_hashmap[name] = value
            self._activity_label_reverse_hashmap[value] = name
        self._activity_labe = act_label

        # todo add activity data


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
        #print(new_df.head(20))

        lst = []
        for row in new_df.iterrows():
            label =row[1][0]
            value = row[1][2]
            # lookup id in new_label
            correct_row = new_label[(new_label[NAME] == label) \
                                    & (new_label[VAL] == value)]
            ide = correct_row.index[0]
            lst.append(ide)
        #new_label[NAME] = new_label[NAME].str.strip()
        #print(lst)
        #print(len(lst))
        #print(len(new_df.index))
        cut = int(len(lst)*0.8)
        self._train_seq = lst[:cut]
        self._test_seq = lst[cut:]



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


