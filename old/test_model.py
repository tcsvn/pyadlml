from hassbrain_algorithm.models.hmm.bhmm import BernoulliHMM
import numpy as np

class BHMMTestModel(BernoulliHMM):
    def __init__(self, ctrl):
        BernoulliHMM.__init__(self, ctrl)

    def predict(self, X):
        """
        computes the the probs of classes for every feature vector
        x in the column seperatly.
        todo is not optimal (how can I do this???)
            i want a sequence to be estimated
            do this with function input formatter
            do this with output formatter

        :param X:
        :return:
        nd array (1D)
            n times the predictions
        """
        if len(X.shape) == 1:
            X = np.reshape(X, (-1, len(X)))
        T, _ = X.shape
        resnp = np.zeros((T, self.K), dtype=np.float64)
        for i, row in enumerate(X):
            row = np.reshape(row, (-1, len(row)))
            tmp = super().classify_multi(row, as_dict=False)
            resnp[i, :] = tmp
        return resnp

from hassbrain_algorithm.models.hmm.bhsmm import BHSMM
class BHSMMTestModel(BHSMM):
    def __init__(self, ctrl):
        BHSMM.__init__(self, ctrl)

    def predict(self, X):
        """
        computes the the probs of classes for every feature vector
        x in the column seperatly.
        todo is not optimal (how can I do this???)
            i want a sequence to be estimated
            do this with function input formatter
            do this with output formatter

        :param X:
        :return:
        nd array (1D)
            n times the predictions
        """
        if len(X.shape) == 1:
            X = np.reshape(X, (-1, len(X)))
        T, _ = X.shape
        resnp = np.zeros((T, self.K), dtype=np.float64)
        for i, row in enumerate(X):
            row = np.reshape(row, (-1, len(row)))
            tmp = super().classify_multi(row, as_dict=False)
            resnp[i, :] = tmp
        return resnp
