# baseclass for dataset
from abc import ABCMeta, abstractmethod

class DataInterfaceHMM():

    """
    encodes labels into numbers
    """
    @abstractmethod
    def load_data(self, path):
        raise NotImplementedError



    """
    the following is functionality the dataset should provide
    to map state and observation labels that have meaning to 
    numbers/representants that don't waste memory.
    An example is activity to numbers mapping 
        and sensors to numbers mapping
    """
    def encode_state_lbl_list(self, list):
        res_list = []
        for item in list:
            res_list.append(self.encode_state_lbl(item))

    def decode_state_lbl(self, repr):
        """
        returns the correct label for the used repr
        :param label:
        :return:
        """
        raise NotImplementedError

    def encode_state_lbl(self, lbl):
        """
        returns the corresp. represenant for a given label
        :param label:
        :return:
        """
        raise NotImplementedError

    def decode_obs_lbl(self, repr):
        raise NotImplementedError

    def encode_obs_lbl(self, lbl):
        raise NotImplementedError


    """
    the following methods are definitions how the dataset should be 
    preprocessed and presented to the model
    """

    # HMM specific
    @abstractmethod
    def get_state_list(self):
        """
        :return:
            a list of encoded symbols representing all possible
            states
        """
        raise NotImplementedError

    @abstractmethod
    def get_obs_list(self):
        """
        :return:
            a list of encoded symbols representing all possible
            observations
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_seq(self):
        """
        :return:
            a list of symbols representing the observations
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_seqs(self):
        """
        :return:
            a list of lists.
            [[1,2,3],[2,1,2, ... ],... ]
        """
        raise NotImplementedError

    @abstractmethod
    def set_file_paths(self, dict):
        raise NotImplementedError

    @abstractmethod
    def get_test_labels_and_seq(self):
        """
        :return:
            y_labels
            y_observations
        """
        raise NotImplementedError

    def is_multi_seq_train(self):
        """
        :return:
            true if the dataset supports multiple sequences
            false if the dataset supports only one long sequence
        """
        raise NotImplementedError