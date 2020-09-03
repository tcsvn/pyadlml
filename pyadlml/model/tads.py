import math
from hassbrain_algorithm.datasets._dataset import _Dataset
from hassbrain_algorithm.models._model import Model
import numpy as np

class TADS(Model):

    def __init__(self, controller):
        self._activity_dur_dist = None
        self._cum_act_dur_dist = None
        Model.__init__(self, "test", controller)

    def __str__(self):
        s = ""
        return s

    def _model_init(self, dataset : _Dataset):
        import pandas as pd
        tmp = dataset.get_rel_act_durations() # type: pd.Dataframe
        hm = self._state_lbl_hashmap
        tmp.rename(columns=hm,
                 inplace=True)
        self._activity_dur_dist = tmp.copy()
        self._cum_act_dur_dist = tmp.cumsum(axis=1)

    def get_act_dur_dist(self):
        rev_hm = self._state_lbl_rev_hashmap
        assert self._activity_dur_dist is not None
        tmp = self._activity_dur_dist.rename(columns=rev_hm,
                 inplace=False)
        return tmp

    def save_visualization(self, path_to_file):
        """
        save a visualization of the model to the given filepath
        :param path_to_file:
        :return:
        """
        raise NotImplementedError

    def predict_state_sequence(self, test_y):
        l, _ = test_y.shape
        pred_z, _ = self._sample(l)
        return pred_z


    def predict_obs_sequence(self, test_y):
        raise NotImplemented

    def can_predict_next_obs(self):
        return False

    def _draw_sample(self):
        s = np.random.uniform(low=0.0, high=1.0)
        for i, val in enumerate(self._cum_act_dur_dist.iloc[0]):
            if s < val:
                return int(i)
        raise ValueError

    def _sample(self, n, obs_seq=None):
        res = np.zeros((n), dtype=np.int64)
        for i in range(n):
            res[i] = self._draw_sample()
        z = res
        obs = None
        return z, obs

    def can_predict_prob_devices(self):
        return False

    def _train(self, dataset):
        return None

    def _classify(self, obs_seq):
        """
        get the most probable state/activity
        :param obs_seq:
        :return:
        """
        assert len(obs_seq[0]) == self._hmm.D
        state_seq = self._hmm.most_likely_states(obs_seq)
        pred_state = state_seq[-1:][0]
        return pred_state

    def _classify_multi(self, obs_seq):
        """
        computes the last omega slice of viterbi which is
        equivalent to
        :param obs_seq:
        :return:
            numpy array with each index being the id for the label
        """
        tmp = self._hmm.filter(obs_seq)
        last_alpha = tmp[-1:][0]


        assert math.isclose(last_alpha.sum(), 1.)

        K = self._hmm.K
        return last_alpha
