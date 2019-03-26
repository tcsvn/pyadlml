# baseclass for dataset
from abc import ABCMeta, abstractmethod

class DataInterface():

    @abstractmethod
    def get_test_seq(self): raise NotImplementedError

    @abstractmethod
    def get_train_seq(self): raise NotImplementedError

    @abstractmethod
    def load_data(self, path): raise NotImplementedError

