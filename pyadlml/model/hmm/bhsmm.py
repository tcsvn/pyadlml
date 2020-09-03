import math

from hassbrain_algorithm.models._model import Model
import autograd.numpy as np
import autograd.numpy.random as npr
from scipy.stats import nbinom
import matplotlib.pyplot as plt
import ssm
from ssm.util import rle, find_permutation
from hassbrain_algorithm.models.util import create_xset_onebitflip

class BHSMM(Model):

    def __init__(self, controller):
        #self._cm = controller # type: Controller
        # training parameters
        self._training_steps = 50
        self._epsilon = None
        self._use_q_fct = False

        Model.__init__(self, "test", controller)

    def __str__(self):
        if self._hsmm is None:
            return "hmm has to be inits"
        else:
            s = "--"*10 + "\n"
            s += "States: " + str(self._hsmm.K) + "\n"
            s += "Obs:\t " + str(self._hsmm.D) + "\n"
            s += "pi:\t " + str(self._hsmm.init_state_distn) + "\n"
            s += "Trans:\t " + str(self._hsmm.transitions) + "\n"
            s += "M:\t " + str(self._hsmm.M) + "\n"
            return s

    def _model_init(self, dataset):
        K = len(self._state_list)       # number of discrete states
        D = len(self._observation_list) # dimension of the observation
        self._hsmm = ssm.HSMM(K, D, observations='bernoulli')

    def set_training_steps(self,  val):
        assert val > 1 and isinstance(val, int)
        self._training_steps = val


    def _sample(self, n, obs_seq=None):
        tmp1, tmp2 = self._hsmm.sample(T=n)
        return tmp1, tmp2

    def save_visualization(self, path_to_file):
        """
        save a visualization of the model to the given filepath
        :param path_to_file:
        :return:
        """
        pass

    def save_visualization_helper_decode_labels(self, label):
        """
        :return:
        """
        return self.decode_state_lbl(label)


    def predict_state_sequence(self, test_y):
        """

        :param test_y:
        :return:
        """
        pred_x = self._hsmm.most_likely_states(test_y)
        return pred_x

    def predict_obs_sequence(self, test_y):
        _, pred_y = self._hsmm.sample(test_y)
        return pred_y

    def draw(self, act_retrieval_meth):
        self._hmm.plot()
        return self._hsmm.generate_graphviz_dot_ext_lbl(act_retrieval_meth)
        #vg.render('test.gv', view=True)

    def _train_loss_callback(self, hmm, loss, *args):
        # todo in log models convert to normal Probability
        """
        :param hmm:
        :param loss:
        :param args:
        :return:
        """
        # this is due to the the loss param is actually the likelihood of the P(X|Model)
        # therefore the loss can be 1 - P(X|Model)
        loss = 1-loss
        self._bench.train_loss_callback(hmm, loss)

    def train(self, dataset):
        y_train = dataset.get_train_data()
        hsmm_em_lls = self._hsmm.fit(
            y_train,
            method="em",
            num_em_iters=self._training_steps)

        test_x, test_y = dataset.get_all_labeled_data()
        self.assign_states(test_x, test_y)
        return hsmm_em_lls

    def assign_states(self, true_z, true_y):
        """
        assigns the unordered hidden states of the trained model (on true_y)
        to the most probable state labels in alignment of true_z
        :param true_z
            the true state sequence of a labeled dataset
        :param true_y
            the true corresp. observation sequence of a labeled dataset

        assign
           z = true state seq [1,2,1,....,]
           tmp3 = pred. state seq [3,4,1,2,...,]
           match each row to different column in such a way that corresp
           sum is minimized
           select n el of C, so that there is exactly one el.  in each row
           and one in each col. with min corresp. costs


        match states [1,2,...,] of of the
        :return:
            None
        """
        # Plot the true and inferred states
        tmp1 = self._hsmm.most_likely_states(true_y)
        # todo temporary cast to int64 remove for space saving solution
        # todo as the amount of states only in range [0, 30
        true_z = true_z.astype(np.int64)
        tmp2 = find_permutation(true_z, tmp1)
        self._hsmm.permute(tmp2)

#-------------------------------------------------------------------
    # RT Node stuff

    def _classify(self, obs_seq):
        """
        get the most probable state/activity
        :param obs_seq:
        :return:
        """
        assert len(obs_seq[0]) == self._hsmm.D
        state_seq = self._hsmm.most_likely_states(obs_seq)
        pred_state = state_seq[-1:][0]
        return pred_state

    def _classify_multi(self, obs_seq):
        """
        calculates the probability of being in state z after observing obs_seq
        :param obs_seq:
            np array 2D
            e.g [ [0,1,0,0,1,...,0] , [0,0,1,0,...,1], ... ]
        :return:
            score_dict (dict)
                {0: 0.008754, 1: 0.1481, ... }
        """
        tmp = self._hsmm.filter(obs_seq)
        last_alpha = tmp[-1:][0]

        assert math.isclose(last_alpha.sum(), 1.)
        K = self._hsmm.K

        score_dict = {}
        for i in range(K):
            label = i
            score = last_alpha[i]
            score_dict[label] = score
        return score_dict

    def can_predict_next_obs(self):
        return True

    def can_predict_prob_devices(self):
        return True

    def _predict_next_obs(self, obs_seq):
        """
        returns index of most probable next observation
        """
        samples = create_xset_onebitflip(obs_seq[-1:][0])
        log_prob_states = self._hsmm.predict_xnp1(data=obs_seq,x=samples)
        # get max of pre_states
        max_idx = np.argmax(log_prob_states)
        if max_idx == len(log_prob_states) -1:
            return np.argmax(log_prob_states[:len(log_prob_states)-1])
        else:
            return max_idx

    def _predict_prob_xnp1(self, obs_seq):
        samples = create_xset_onebitflip(obs_seq[-1:][0])
        log_prob_states = self._hsmm.predict_xnp1(data=obs_seq,x=samples)
        normalizer = ssm.util.logsumexp(log_prob_states)
        norm_log_prob = log_prob_states - normalizer
        norm_log_prob = norm_log_prob[:len(norm_log_prob)-1]
        norm_log_prob = np.exp(norm_log_prob)
        return norm_log_prob

#class BHSMM_Categorical(BHSMM):
#    def __init__(self, controller):
#        BHSMM.__init__(self, controller)
#
#    def _model_init(self, dataset):
#        K = len(self._state_list)       # number of discrete states
#        D = len(self._observation_list) # dimension of the observation
#        self._hsmm = ssm.HSMM(K, D, observations='categorical')
