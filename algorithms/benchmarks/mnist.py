from algorithms.benchmarks.dataset import DataInterface
#from algorithms.benchmarks.mnist_data.parsing import parser, digit
#from algorithms.benchmarks.mnist_data.analysis import training, sampling, testing, classify
#from algorithms.benchmarks.mnist_data.config import settings
import numpy as np

class Digit():
    def __init__(self, np_seq):
        self._np_seq = np_seq

    def get_seq(self):
        return self._np_seq

class DatasetMNIST():
    def __init__(self):
        self._digits = None

    def load_files(self, label_file, data_file, label=0):
        NUM_COLUMNS = 4
        """

        :param label:
            specify the label, which number that should be learned
        :return:
        """

        # select the filenames that mach the number label
        # for which we want to train the hmm
        with open(label_file, 'r') as f:
            ln = 0
            files_to_parse = []
            print(label)
            for line in f:
                #if label == int(line):
                if True:
                    files_to_parse.append(ln)
                ln += 1
                #if ln == 2000:
                #    break
            print(files_to_parse)

        digits = []
        # create Digits from files
        for file in files_to_parse:
            filename = data_file.replace('num', str(file))
            with open(filename, 'r') as f:
                np_arr = np.fromfile(filename, dtype=int, sep=" ")
                rows = int(len(np_arr)/NUM_COLUMNS)
                np_arr = np.reshape(np_arr, (rows ,NUM_COLUMNS))
                digits.append(Digit(np_arr))
        self._digits = digits
        arr = np.zeros((len(digits)))
        for i, d in enumerate(self._digits):
            arr[i] = (np.max(d.get_seq()))
        print(arr)
        print(arr.max())
        # encode labels
        # decode labels

        # create training sequence

    def get_train_seq(self):
        return self._train_digits

    def get_test_seq(self):
        return self._test_digits

    def get_observation_list(self):
        pass

    def get_state_list(self):
        pass
