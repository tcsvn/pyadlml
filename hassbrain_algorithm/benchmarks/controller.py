import os
import logging
import yaml
from enum import Enum
from hassbrain_algorithm.benchmarks.datasets.dataset import DataInterfaceHMM

class Dataset(Enum):
    HASS = 'homeassistant'
    KASTEREN = 'kasteren'
    MAVPAD2005 = 'mavpad'
    ARAS = 'aras'
    CASAS_ARUBA = 'CASAS'
    PENDIGITS = 'pendigits'

class Controller():
    def __init__(self,path_to_config=None):
        self.logger = logging.getLogger(__name__)
        self._model = None
        self._bench = None
        self._dataset = None
        self._dataset_enm = None

        if path_to_config is None:
            dirname = os.path.dirname(__file__)
            self._path_to_config = dirname + '/config.yaml'
        else:
            self._path_to_config = path_to_config

        md_conf_dict = self.load_md_conf()
        self._md_folder = md_conf_dict['folder']


    def load_md_conf(self):
        with open(self._path_to_config) as f:
            data = yaml.safe_load(f)
            return data['models']

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
            from hassbrain_algorithm.benchmarks.datasets.kasteren import DatasetKasteren
            self._dataset_enm = Dataset.KASTEREN
            self._dataset = DatasetKasteren()
            self._dataset.set_file_paths(self.load_paths(Dataset.KASTEREN))

        elif data_name == Dataset.PENDIGITS:
            from hassbrain_algorithm.benchmarks.datasets.pendigits import DatasetPendigits
            self._dataset_enm = Dataset.PENDIGITS
            self._dataset = DatasetPendigits()
            self._dataset.set_file_paths(self.load_paths(Dataset.PENDIGITS))

        elif data_name == Dataset.HASS:
            from hassbrain_algorithm.benchmarks.datasets.homeassistant import DatasetHomeassistant
            self._dataset_enm = Dataset.HASS
            self._dataset = DatasetHomeassistant()
            self._dataset.set_file_paths(self.load_paths(Dataset.HASS))

        self._dataset.load_data()

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
        from hassbrain_algorithm.benchmarks.benchmark import Benchmark
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
        print(obs_seq)
        exit(-1)
        self._dataset.plot_obs_seq(obs_seq, 3)

    def _generate_file_name(self):
        modelname = str(self._model.__class__.__name__)
        datasetname = self._dataset_enm.value
        return modelname + "_" + datasetname

    def get_model_file_path(self):
        filename = self._generate_file_name()
        return self._model.generate_file_path(
            path_to_folder=self._md_folder,
            filename=filename)


    def save_model(self):
        self._model.save_model(
            path_to_folder=self._md_folder,
            filename=self._generate_file_name()
        )

    def load_model(self):
        self._model = self._model.load_model(
             path_to_folder=self._md_folder,
            filename=self._generate_file_name()
        )

    def render_model(self):
        dot = self._model.draw(self._dataset.decode_state_lbl)
        return dot

    def show_plot(self):
        self._bench.show_plot()

    def decode_state(self, state_number):
        return self._dataset.decode_state_lbl(state_number)

    def get_bench_metrics(self):
        # only call this after creating a report
        self.create_report(True, True, True, True, True)
        acc = self._bench.get_accuracy()
        prec = self._bench.get_precision()
        rec = self._bench.get_recall()
        f1 = self._bench.get_f1score()
        return acc, prec, rec, f1

    def create_report(self, conf_matrix=False, accuracy=False, precision=False, recall=False, f1=False):

        if accuracy or precision or recall or conf_matrix:
            y_true, y_pred = self._model.create_pred_act_seqs(self._dataset)
            print('went here')
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
