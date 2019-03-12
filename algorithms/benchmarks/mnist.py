from algorithms.benchmarks.dataset import DataInterface
from algorithms.benchmarks.mnist_data.parsing import parser, digit
#from algorithms.benchmarks.mnist_data.analysis import training, sampling, testing, classify
#from algorithms.benchmarks.mnist_data.config import settings



class DatasetMNIST():
    def __init__(self):
        pass


    def load_files(self, path_test_file, path_train_file):
        pars = parser.Parser()
        self._train_digits = pars.parse_file(path_train_file)
        self._test_digits = pars.parse_file(path_test_file)


    def get_train_seq(self):
        return self._train_digits

    def get_test_seq(self):
        return self._test_digits

    def get_observation_list(self):
        pass

    def get_state_list(self):
        pass
