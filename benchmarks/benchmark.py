from enum import Enum
from benchmarks.kasteren import DatasetKasteren

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
            print('loading sensors...')
            kasteren = DatasetKasteren()
            self._loaded_datasets[Dataset.KASTEREN.name] = kasteren
            kasteren.load_sensors(KASTEREN_SENS_PATH)
            # todo
            # kasteren.load_activitys(KASTEREN_ACT_PATH)

        elif data_name == Dataset.HASS:
            return
        elif data_name == Dataset.MAVPAD2005:
            return
        elif data_name == Dataset.ARAS:
            return
        elif data_name == Dataset.CASAS_ARUBA:
            return

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
            kasteren = self.load_dataset[KASTEREN_SENS_PATH]
            obs_seq = kasteren.get_seq()

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

