import numpy as np

class ProbabilityMassFunction():
    def __init__(self, obs_alphabet):
        self._E = np.full(len(obs_alphabet), 1/len(obs_alphabet))
        self._alphabet = obs_alphabet

    def __str__(self):
        s = "ProbMassFct\n"
        s += str(self._E)
        return s

    def _get_idx(self, label):
        for idx, obs in enumerate(self._alphabet):
            if label == obs:
                return idx

    def get_probs(self):
        return self._E

    def set_probs(self, np_arr):
        """
        :param np_arr:
        :return:
        """
        # todo check how to manage that stuff doesn't sum to one
        if len(self._alphabet) == len(np_arr):#and np_arr.sum() == 1:
            self._E = np_arr
        else:
            return -1

    def prob(self, obs_label):
        return self._E[self._get_idx(obs_label)]
