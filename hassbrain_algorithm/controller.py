import os
import logging
import yaml
from enum import Enum
from hassbrain_algorithm.benchmark import Benchmark

class Dataset(Enum):
    HASS_CHRIS = 'hass_chris'
    HASS = 'homeassistant'
    HASS_SIMON = 'hass_simon'
    HASS_TESTING = 'hass_testing'
    KASTEREN = 'kasteren'
    MAVPAD2005 = 'mavpad'
    ARAS = 'aras'
    CASAS_ARUBA = 'casas'
    PENDIGITS = 'pendigits'

class Controller():
    def __init__(self,path_to_config=None, config=None):
        self.logger = logging.getLogger(__name__)
        self._model = None # type: Model
        self._bench = None # type: Benchmark
        self._dataset = None
        self._dataset_enm = None

        """
        is used to induce custom config, that isn't specified in the configuration
        file
        """
        self._config = config


        if path_to_config is None:
            dirname = os.path.dirname(__file__)
            self._path_to_config = dirname + '/config.yaml'
        else:
            self._path_to_config = path_to_config

        md_conf_dict = self.load_md_conf()
        self._md_folder = md_conf_dict['folder']

    def set_custom_state_list(self, state_list):
        self._dataset.set_state_list(state_list)

    def set_custom_obs_list(self, obs_list):
        self._dataset.set_obs_list(obs_list)

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
        if self._config is None:
            with open(self._path_to_config) as f:
                data = yaml.safe_load(f)
                print(data)
                return data['datasets'][dataset.value]
        else:
            # this is the case if a config was induced
            # for example from homeassistant web for the own database
            return self._config['datasets'][dataset.value]

    def set_dataset(self, data_name):
        """
        loads the dataset into ram
        :param data_name:
        :return:
        """
        self.logger.info("load dataset...")
        if data_name == Dataset.KASTEREN:
            from hassbrain_algorithm.datasets.kasteren import DatasetKasteren
            self._dataset_enm = Dataset.KASTEREN
            self._dataset = DatasetKasteren()
            self._dataset.set_file_paths(self.load_paths(Dataset.KASTEREN))

        elif data_name == Dataset.PENDIGITS:
            from hassbrain_algorithm.datasets.pendigits import DatasetPendigits
            self._dataset_enm = Dataset.PENDIGITS
            self._dataset = DatasetPendigits()
            self._dataset.set_file_paths(self.load_paths(Dataset.PENDIGITS))

        elif data_name == Dataset.HASS_TESTING:
            from hassbrain_algorithm.datasets.homeassistant import DatasetHomeassistant
            self._dataset_enm = Dataset.HASS_TESTING
            self._dataset = DatasetHomeassistant()
            self._dataset.set_file_paths(self.load_paths(Dataset.HASS_TESTING))

        elif data_name == Dataset.HASS_CHRIS:
            from hassbrain_algorithm.datasets.homeassistant import DatasetHomeassistant
            self._dataset_enm = Dataset.HASS_CHRIS
            self._dataset = DatasetHomeassistant()
            self._dataset.set_file_paths(self.load_paths(Dataset.HASS_CHRIS))

        elif data_name == Dataset.HASS_SIMON:
            from hassbrain_algorithm.datasets.homeassistant import DatasetHomeassistant
            self._dataset_enm = Dataset.HASS_SIMON
            self._dataset = DatasetHomeassistant()
            self._dataset.set_file_paths(self.load_paths(Dataset.HASS_SIMON))

        elif data_name == Dataset.HASS:
            from hassbrain_algorithm.datasets.homeassistant import DatasetHomeassistant
            self._dataset_enm = Dataset.HASS
            self._dataset = DatasetHomeassistant()
        #elif data_name == Dataset.MAVPAD2005:
        #    return
        #elif data_name == Dataset.ARAS:
        #    return
        #elif data_name == Dataset.CASAS_ARUBA:
        #    return

    def manual_load_paths(self, data):
        """
        is used when the website loads the custom data for the home assistant
        dataset and base and
        :param data:
        :return:
        """
        self._dataset.set_file_paths(data)

    def load_dataset(self):
        self._dataset.load_data()

    def register_model(self, model):
        self._model = model

    def init_model_on_dataset(self):
        self._model.model_init(self._dataset)

    def register_benchmark(self):
        from hassbrain_algorithm.benchmark import Benchmark
        self._bench = Benchmark(self._model)
        self._model.register_benchmark(self._bench)

    def register_acc_file_path(self, path_to_file):
        """
        sets the path in benchmark to training loss
        :param path_to_file:
        :return:
        """
        self._bench.register_train_acc_file_path(path_to_file)

    def register_loss_file_path(self, path_to_file):
        """
        sets the path in benchmark to training loss
        :param path_to_file:
        :return:
        """
        self._bench.register_train_loss_file_path(path_to_file)

    def save_loss_plot_to_file(self, path_to_file):
        """
        sets the path of the file where the training loss per training step
        should be logged to
        :param path_to_file:
        :return:
        """
        self._bench.save_train_loss_plot(path_to_file)


    def save_visualization_to_file(self, path_to_file):
        """
        sets the path to the file where the img should be saved
        and enables the action that if the algorithm is trained the img
        is actually saved to this location
        :param path_to_file:
        :return:
        """
        self._model.save_visualization(path_to_file)

    def register_location_info(self, loc_data):
        if not self._model.are_hashmaps_created():
            self._model.gen_hashmaps(self._dataset)
        self._model.register_loc_info(loc_data)

    def register_activity_info(self, act_data):
        if not self._model.are_hashmaps_created():
            self._model.gen_hashmaps(self._dataset)
        self._model.register_act_info(act_data)

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
        #if self._bench is not None:
            #self._bench.enable_logging()
            #self._bench.notify_model_was_trained()

        # train model on dataset
        self._model.train(self._dataset, args)

        #if self._bench is not None:
            #self._bench.disable_logging()
            #self._bench.read_train_loss_file()

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

    def splt_path2folder_fnm(self, file_path):
        import os
        dir = os.path.dirname(file_path)
        file = os.path.basename(file_path)
        return dir, file

    def save_model(self, file_path=None):
        if file_path is None:
            self._model.save_model(
                path_to_folder=self._md_folder,
                filename=self._generate_file_name()
            )
        else:
            path_to_folder, filename = self.splt_path2folder_fnm(file_path)
            self._model.save_model(
                path_to_folder=path_to_folder,
                filename=filename
            )

    def load_model(self, file_path=None):
        if file_path is None:
            self._model = self._model.load_model(
                 path_to_folder=self._md_folder,
                filename=self._generate_file_name()
            )
        else:
            path_to_folder, filename = self.splt_path2folder_fnm(file_path)
            self._model = self._model.load_model(
                 path_to_folder=path_to_folder,
                filename=filename
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
