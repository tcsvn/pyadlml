import logging
from enum import Enum

import matplotlib
from numpy import genfromtxt
import  matplotlib.pyplot as plt
from algorithms.benchmarks.kasteren import DatasetKasteren
import os

from algorithms.benchmarks.mnist.dataset import DatasetMNIST

dirname = os.path.dirname(__file__)[:-22]
#KASTEREN_SENS_PATH = '/mnt/external_D/code/hassbrain_algorithm/datasets/kasteren/kasterenSenseData.txt'
#KASTEREN_ACT_PATH = '/mnt/external_D/code/hassbrain_algorithm/datasets/kasteren/kasterenActData.txt'
KASTEREN_SENS_PATH = dirname + '/datasets/kasteren/kasterenSenseData.txt'
KASTEREN_ACT_PATH = dirname + '/datasets/kasteren/kasterenActData.txt'
HASS_PATH = ''

LOGGING_FILENAME='train_model.log'

class Dataset(Enum):
    HASS = 'hass'
    KASTEREN = 'kasteren'
    MAVPAD2005 = 'mavpad'
    ARAS = 'aras'
    CASAS_ARUBA = 'CASAS'
    MNIST = 'mnist'

class Bench():
    def __init__(self):
        self._model = None
        self._loaded_datasets = {}
        self._conv_plot = None
        self._conv_data = None

    def setup_training_logging(self):
        logging.basicConfig(
            level=logging.DEBUG,
            filename=LOGGING_FILENAME,
            filemode='w',
            format='%(message)s'
        )



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
            kasteren.load_activitys(KASTEREN_ACT_PATH)

        elif data_name == Dataset.MNIST:
            print('loading numbers...')
            mnist = DatasetMNIST()
            self._loaded_datasets[Dataset.MNIST.name] = mnist
            mnist.load_files()

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

    def init_model_on_dataset(self, data_name):
        if data_name == Dataset.KASTEREN:
            kasteren = self._loaded_datasets[Dataset.KASTEREN.name]
            self._model.model_init(
                kasteren.get_activity_list(),
                kasteren.get_sensor_list()
            )
        elif data_name == Dataset.MNIST:j


    def train_model(self, data_name):
        """
        trains the model on the sequence of the data
        :param data_name:
        :return:
        """
        # enable
        self.setup_training_logging()
        logger = logging.getLogger()
        logger.disabled = False

        # train on given dataset
        if data_name == Dataset.KASTEREN:
            kasteren = self._loaded_datasets[Dataset.KASTEREN.name]
            train_seq = kasteren.get_train_seq()[:19]
            self._model.train(train_seq)
        elif data_name == Dataset.HASS:
            pass

        logger = logging.getLogger()
        logger.disabled = True


        # read logged file and read in data for plotting
        data = genfromtxt(LOGGING_FILENAME, delimiter='\n')
        self._conv_data = data
        self.generate_conv_plot(data)

    def generate_conv_plot(self, data):
        #matplotlib.rcParams['text.usetex'] = True
        plt.plot(data)
        #plt.ylabel('$P(X|\Theta)$')
        plt.ylabel('P(X|Theta)')
        plt.xlabel('training steps')


    def create_report(self):
        """
        creates a report including accuracy, precision, recall, training convergence
        :return:
        """
        start_prob_X = self._conv_data[0]
        end_prob_X = self._conv_data[len(self._conv_data)-1]
        s = "Report"
        s += "\n"
        s += "_"*100
        s += "\n"
        s += "Start\tP( X|Theta ) = " + str(start_prob_X) + "\n"
        s += "Trained\tP( X|Theta ) = " + str(end_prob_X) +"\n"
        s += "_"*100
        s += "\n"
        s += "Metric on test dataset:" + "\n"
        s += "\tAccuracy: \t" + "todo" + "\n"
        s += "\tPrecision: \t" + "todo" + "\n"
        s += "\tRecall: \t" + "todo" + "\n"
        s += "*"*100
        return s

    def render_model(self, data_name):
        if data_name == Dataset.KASTEREN:
            kasteren = self._loaded_datasets[Dataset.KASTEREN.name]
            dot = self._model.draw(kasteren.get_activity_label_from_id)
        return dot

    def show_plot(self):
        plt.show()

    def calc_accuracy(self, acc):
        pass


    def calc_precision(self, prec):
        pass


    def calc_recall(self, recall):
        pass


    def plot_convergence(self, conv_seq):
        pass
