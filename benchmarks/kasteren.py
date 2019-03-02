import pandas as pd

START_TIME = 'Start time'
END_TIME = 'End time'
ID = 'Idea'
VAL = 'Val'
NAME = 'Name'

class DatasetKasteren():
    def __init__(self):
        self._obs_seq = []
        self._df = None
        self._label = None
        self._label_hashmap = None
        self._label_reverse_hashmap = None


    def get_id_from_label(self, label, state):
        """
        returns the id of a sensor given a label
        :param label:
        :return:
        """
        return self._label_hashmap[label][state]

    def get_label_from_id(self, id):
        """
        retrieves the label given a sensor id
        :param id:
        :return:
        """
        return self._label_reverse_hashmap[id]



    def load_sensors(self, path_to_file):
        labels, df = self._sensor_file_to_df(path_to_file)
        self._df_to_seq(labels, df)

    def load_activitys(self, path_to_file):
        # todo implement
        pass

    def get_obs_seq(self):
        return self._obs_seq

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
        counter = 0
        #print(new_label)
        #print(new_df.head(5))
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
        self._obs_seq = lst
        self._df = new_df
        self._label = new_label

        # create hashmap instead of shitty dataframe >:/
        self._label_hashmap = {}
        self._label_reverse_hashmap = {}
        idx = 0
        for row in new_label.iterrows():
            name = str(row[1][0])
            value = row[1][1]
            if idx%2 == 0:
                self._label_hashmap[name] = {}
                self._label_hashmap[name][value] = idx
                self._label_reverse_hashmap[idx] = name
            else:
                self._label_hashmap[name][value] = idx
                self._label_reverse_hashmap[idx] = name
            idx+=1


