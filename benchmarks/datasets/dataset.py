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

    def decode_state_label(self, representant):
        """
        returns the correct label for the used repr
        :param label:
        :return:
        """
        raise NotImplementedError
    def encode_state_label(self, label):
        """
        returns the corresp. represenant for a given label
        :param label:
        :return:
        """
        raise NotImplementedError

    def decode_obs_label(self, repr):
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
