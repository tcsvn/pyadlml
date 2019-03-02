import benchmarks.kasteren as kasteren
from enum import Enum

KASTEREN_SENS_PATH = '/mnt/external_D/code/hassbrain_algorithm/datasets/kasteren/kasterenSenseData.txt'
KASTEREN_ACT_PATH = '/mnt/external_D/code/hassbrain_algorithm/datasets/kasteren/kasterenSenseData.txt'
HASS_PATH = ''

class Dataset(Enum):
    HASS = 'hass'
    KASTEREN = 'kasteren'
    MAVPAD2005 = 'mavpad'
    ARAS = 'aras'
    CASAS_ARUBA = 'CASAS'

class Bench():
    def __init__(self):
        self._model = None
        self._loaded_datasets = {}

    def load_dataset(self, data_name):
        """
        loads the dataset into ram
        :param data_name:
        :return:
        """
        if data_name == Dataset.KASTEREN:
            labels, df = kasteren.sensor_file_to_df(KASTEREN_SENS_PATH)
            self.load_kasteren(labels, df)

        elif data_name == Dataset.HASS:
            return
        elif data_name == Dataset.MAVPAD2005:
            return
        elif data_name == Dataset.ARAS:
            return
        elif data_name == Dataset.CASAS_ARUBA:
            return

    def load_kasteren(self, labels, df):
        import pandas as pd
        import string
        self._loaded_datasets[Dataset.KASTEREN.name] = {}
        self._loaded_datasets[Dataset.KASTEREN.name]['sensors'] = df

        # create hashmap of sensor labels
        # duplicate all values
        #for row in labels.iterrows():
        #    labels.loc[-1] = [row[1][0],row[1][1]]
        #    labels.index = labels.index + 1
        #    labels = labels.sort_values(by=['Idea'])
        #print(labels)

        # create alternating zeros and 1 label dataframe
        lb_zero = labels.copy()
        lb_zero['Val'] = 0
        lb_zero = lb_zero.loc[:, lb_zero.columns != 'Idea']
        lb_one = labels.copy()
        lb_one['Val'] = 1
        lb_one = lb_one.loc[:, lb_one.columns != 'Idea']
        new_label = pd.concat([lb_one, lb_zero]).sort_values('Name')
        new_label = new_label.reset_index(drop=True)

        #
        N=25
        #print(pd.Series(string.ascii_uppercase) for _ in range(N))

        #
        df_start = df.copy()
        df_end = df.copy()
        df_end = df_end.loc[:, df_end.columns != 'Start time']
        df_start = df_start.loc[:, df_start.columns != 'End time']
        df_end['Val'] = 0

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
        print(new_label)
        print(new_df.head(5))
        for row in new_df.iterrows():
            label =row[1][0]
            value = row[1][2]
            # lookup id in new_label
            correct_row = new_label[(new_label['Name'] == label)\
                                    & (new_label['Val'] == value)]
            ide = correct_row.index[0]
            lst.append(ide)
        print(lst)
        print(len(lst))
        print(len(new_df.index))



        #df = kasteren.activity_file_to_df(KASTEREN_ACT_PATH)
        #self._loaded_datasets[Dataset.KASTEREN.name]['activity'] = df

    def register_model(self, model):
        """
        setter for model
        :param model:
        """
        self._model = model


    def train_model(self, data_name):
        """
        trains the model on the sequence of the data
        :param data_name:
        :return:
        """
        if data_name == Dataset.KASTEREN:
            self.train_on_kasteren()

    def train_on_kasteren(self):
        # 1. convert dataframe to sequence
        # 2. encode labels
        pass

    def report(self):
        """
        creates a report including accuracy, precision, recall, training convergence
        :return:
        """
        pass


    def plot_accuracy(self, acc):
        pass


    def plot_precision(self, prec):
        pass


    def plot_recall(self, recall):
        pass


    def plot_convergence(self, conv_seq):
        pass

