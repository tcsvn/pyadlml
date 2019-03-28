import os
import yaml
from enum import Enum
from benchmarks.benchmark import Benchmark
from benchmarks.kasteren import DatasetKasteren
from benchmarks.pendigits import DatasetPendigits

#dirname = os.path.dirname(__file__)[:-22]
#PD_LABEL_FILE = dirname + '/datasets/mnist_sequences/sequences/'
#PD_TEST_FILE = dirname + '/algorithms/bchmarks/mnist_data/data/pendigits-test'
#KASTEREN_SENS_PATH = '/mnt/external_D/code/hassbrain_algorithm/datasets/kasteren/kasterenSenseData.txt'
#KASTEREN_ACT_PATH = '/mnt/external_D/code/hassbrain_algorithm/datasets/kasteren/kasterenActData.txt'
#KASTEREN_SENS_PATH = dirname + '/datasets/kasteren/kasterenSenseData.txt'
#KASTEREN_ACT_PATH = dirname + '/datasets/kasteren/kasterenActData.txt'
#HASS_PATH = ''


class Dataset(Enum):
    HASS = 'hass'
    KASTEREN = 'kasteren'
    MAVPAD2005 = 'mavpad'
    ARAS = 'aras'
    CASAS_ARUBA = 'CASAS'
    PENDIGITS = 'pendigits'

class Controller():
    def __init__(self):
        self._model = None
        self._bench = None # type: Benchmark
        self._loaded_datasets = {}

        # if set true this is used to log training convergence to a file
        self._conv_logging = False

        dirname = os.path.dirname(__file__)
        self._path_to_config = dirname + '/config.yaml'

    def load_paths(self, data_name):
        with open(self._path_to_config) as f:
            data = yaml.safe_load(f)

            if data_name == Dataset.KASTEREN:
                kast_sens_path = data['datasets']['kasteren']['sens_file_path']
                kast_act_path = data['datasets']['kasteren']['act_file_path']
                return kast_sens_path, kast_act_path
            if data_name == Dataset.PENDIGITS:
                train_path = data['datasets']['pendigit']['train_file_path']
                test_path = data['datasets']['pendigit']['test_file_path']
                return train_path, test_path

    def load_dataset(self, data_name):
        """
        loads the dataset into ram
        :param data_name:
        :return:
        """
        if data_name == Dataset.KASTEREN:
            print('loading sensors...')
            kast_sens_path, kast_act_path = self.load_paths(Dataset.KASTEREN)
            kasteren = DatasetKasteren(kast_sens_path, kast_act_path)
            self._loaded_datasets[Dataset.KASTEREN.name] = kasteren
            kasteren.load_sensors()
            kasteren.load_activitys()

        elif data_name == Dataset.PENDIGITS:
            print('loading numbers...')
            pd_train_path, pd_test_path = self.load_paths(Dataset.PENDIGITS)
            pd = DatasetPendigits(pd_train_path, pd_test_path)
            self._loaded_datasets[Dataset.PENDIGITS.name] = pd
            pd.load_files()

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
            self._model.model_init(kasteren)

        elif data_name == Dataset.PENDIGITS:
            pd = self._loaded_datasets[Dataset.PENDIGITS.name]
            self._model.model_init(
                activity_list = pd.get_state_list(),
                observation_list=pd.get_observations_list()
            )

    def enable_benchmark(self):
        self._bench = Benchmark(self._model)
        self._model.register_benchmark(self._bench)

    def disable_benchmark(self):
        self._bench = None


    def train_model(self, data_name, args=None):
        """
        trains the model on the sequence of the data
        :param data_name:
        :return:
        """
        # enable
        if self._bench is not None:
            self._bench.enable_logging()
            self._bench.notify_model_was_trained()

        # train on given dataset
        if data_name == Dataset.KASTEREN:
            kasteren = self._loaded_datasets[Dataset.KASTEREN.name]
            train_dump = kasteren.get_train_seq()
            train_seq_1 = train_dump[:30]
            train_seq_2 = train_dump[30:60]
            train_seq_3 = train_dump[60:90]
            train_seq_4 = train_dump[90:120]
            train_seq_5 = train_dump[120:150]
            self._model.train(train_seq_1, args)
            self._model.train(train_seq_2, args)
            #self._model.train(train_seq_3, args)
            #self._model.train(train_seq_4, args)
            #self._model.train(train_seq_5, args)
        elif data_name == Dataset.HASS:
            pass

        if self._bench is not None:
            self._bench.disable_logging()
            self._bench.read_in_conv_plot()

    def render_model(self, data_name):
        if data_name == Dataset.KASTEREN:
            kasteren = self._loaded_datasets[Dataset.KASTEREN.name]
            dot = self._model.draw(kasteren.get_activity_label_from_id)
        return dot

    def show_plot(self):
        self._bench.show_plot()

    def decode_state(self, state_number, dk):
        dataset = self._loaded_datasets[dk.name]
        return dataset.get_activity_label_from_id(state_number)

    def create_report(self, conf_matrix=False, accuracy=False, precision=False, recall=False, f1=False):
        kasteren = self._loaded_datasets[Dataset.KASTEREN.name]
        test_seq = kasteren.get_test_arr()
        if accuracy or precision or recall or conf_matrix:
            y_true, y_pred = self._model.create_pred_act_seqs(test_seq)
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
