import os
import logging
import yaml
from enum import Enum
#from algorithms.model import load_model
from benchmarks.benchmark import Benchmark
from benchmarks.datasets.dataset import DataInterfaceHMM
from benchmarks.datasets.kasteren import DatasetKasteren
from benchmarks.datasets.pendigits import DatasetPendigits

class Dataset(Enum):
    HASS = 'hass'
    KASTEREN = 'kasteren'
    MAVPAD2005 = 'mavpad'
    ARAS = 'aras'
    CASAS_ARUBA = 'CASAS'
    PENDIGITS = 'pendigits'

class Controller():
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._model = None
        self._bench = None # type: Benchmark
        self._dataset = None # type: DataInterfaceHMM

        dirname = os.path.dirname(__file__)
        self._path_to_config = dirname + '/config.yaml'

    def load_paths(self, dataset):
        """
            loads the paths to different files for a given dataset
        :param dataset:
            enum Dataset
        :return:
            dictionary of paths
            Example:
                { 'test_file_path: '/path/to/file', 'test_file_2': 'asdf' }
        """
        with open(self._path_to_config) as f:
            data = yaml.safe_load(f)
            return data['datasets'][dataset.value]

    def load_dataset(self, data_name):
        """
        loads the dataset into ram
        :param data_name:
        :return:
        """
        self.logger.info("load dataset...")
        if data_name == Dataset.KASTEREN:
            self._dataset = DatasetKasteren()
            self._dataset.set_file_paths(self.load_paths(Dataset.KASTEREN))

        elif data_name == Dataset.PENDIGITS:
            self._dataset = DatasetPendigits()
            self._dataset.set_file_paths(self.load_paths(Dataset.PENDIGITS))

        self._dataset.load_data()
        #elif data_name == Dataset.HASS:
        #    return
        #elif data_name == Dataset.MAVPAD2005:
        #    return
        #elif data_name == Dataset.ARAS:
        #    return
        #elif data_name == Dataset.CASAS_ARUBA:
        #    return

    def register_model(self, model):
        self._model = model

    def init_model_on_dataset(self):
        self._model.model_init(self._dataset)

    def register_benchmark(self):
        self._bench = Benchmark(self._model)
        self._model.register_benchmark(self._bench)

    def disable_benchmark(self):
        self._bench = None

    def train_model(self, args=None):
        """
        trains the model on the sequence of the data
        :param args:
            list of parameters for the model
        :return:
        """

        # is used to log the convergence rate during training
        if self._bench is not None:
            self._bench.enable_logging()
            self._bench.notify_model_was_trained()

        # train model on dataset
        self._model.train(self._dataset, args)

        if self._bench is not None:
            self._bench.disable_logging()
            self._bench.read_in_conv_plot()

    def generate_observations(self):
        return self._model.gen_obs()

    def plot_observations(self):
        obs_seq = self._model.gen_obs()
        self._dataset.plot_obs_seq(obs_seq, 3)

    def save_model(self):
        self._model.save_model(1)

    def load_model(self):
        self._model = self._model.load_model(1)

    def render_model(self):
        dot = self._model.draw(self._dataset.decode_state_label)
        return dot

    def show_plot(self):
        self._bench.show_plot()

    def decode_state(self, state_number):
        return self._dataset.decode_state_label(state_number)

    def create_report(self, conf_matrix=False, accuracy=False, precision=False, recall=False, f1=False):
        if accuracy or precision or recall or conf_matrix:
            y_true, y_pred = self._model.create_pred_act_seqs(self._dataset)
        if accuracy:
            self._bench.calc_accuracy(y_true, y_pred)
        if precision:
            self._bench.calc_precision(y_true, y_pred)
        if recall:
            self._bench.calc_recall(y_true, y_pred)
        if conf_matrix:
            self._bench.calc_conf_matrix(y_true, y_pred)
        if f1:
            self._bench.calc_f1_score(y_true, y_pred)

        return self._bench.create_report()
