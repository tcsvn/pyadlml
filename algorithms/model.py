"""
    this is an abstract class of an implemented algorithm like hmm
    model is used for benchmarks
    model is also called from hmm events
"""
from algorithms.hmm._hmm_base import HiddenMarkovModel
from benchmarks.controller import Controller, Dataset


class Model():
    def __init__(self, activity_alphabet, sensor_alphabet):
        self._bench = None

    def register_benchmark(self, bench):
        self._bench = bench

    def model_init(self, dataset):
        """
        initialize model on dataset
        :param dataset:
        :return:
        """
        pass

    def train(self, dataset):
        """
         gets a dataset and trains the model on the data
        :param dataset:
        :return:
        """
        pass

    def predict_next_observation(self, args):
        """

        :param args:
        :return:
        """
        pass

    def draw(self):
        """
         somehow visualize the model
        :return: an image png or jpg
        """
        pass

from algorithms.hmm.distributions import ProbabilityMassFunction
from algorithms.hmm.hmm_scaled import HMM_Scaled
import numpy as np

# wrapper class for hmm
class Proxy_HMM(Model):

    def __init__(self, controller):
        self._cm = controller # type: Controller
        self._hmm = None # type: HiddenMarkovModel

        # training parameters
        self._training_steps = 100
        self._epsilon = None
        self._use_q_fct = False

    def __str__(self):
        return self._hmm.__str__()

    def use_q_fct(self, value):
        if value == True or value == False:
            self._use_q_fct = value
        else:
            raise ValueError

    def model_init(self, dataset):
        state_list = dataset.get_activity_list()
        observation_list = dataset.get_sensor_list()

        init_pi = HiddenMarkovModel.gen_rand_pi(len(state_list))
        # init markov model in normal way
        self._hmm = HMM_Scaled(state_list,
                               observation_list,
                               ProbabilityMassFunction,
                               init_pi)

    def get_conv_plot_y_label(self):
        if self._use_q_fct:
            return 'Q(Theta, Theta_old)'
        else:
            return 'P(X|Theta)'

    def get_conv_plot_x_label(self):
        return 'training steps'


    def draw(self, act_retrieval_meth):
        return self._hmm.generate_visualization_2(act_retrieval_meth)
        #vg.render('test.gv', view=True)

    def train(self, obs_seq, use_q_fct=False):
        if use_q_fct == True or use_q_fct == False:
            self.use_q_fct(use_q_fct)

        obs_seq = obs_seq
        self._hmm.train(obs_seq,
                        self._epsilon,
                        self._training_steps,
                        self._use_q_fct)

    def get_label_list(self):
        lst = []
        for state_nr in self._hmm._z:
            lst.append(self._cm.decode_state(state_nr, dk=Dataset.KASTEREN))
        return lst

    # todo flag deletion
    # works but use sklearn instead
    #def tmp_create_confusion_matrix(self, test_obs_arr):
    #    K = len(self._hmm._z)
    #    N = len(test_obs_arr)
    #    print(K)
    #    print(N)
    #    #print(test_obs_arr)
    #    # predicted class X actual class
    #    conf_mat = np.zeros((K, K))
    #    obs_seq = []
    #    for n in range(N):
    #        obs_seq.append(int(test_obs_arr[n][0]))
    #        state_seq = self._hmm.viterbi(obs_seq)
    #        predicted_state = state_seq[-1:][0]
    #        actual_state = int(test_obs_arr[n][1])
    #        idx_pred_state = self._hmm._idx_state(predicted_state)
    #        idx_act_state = self._hmm._idx_state(actual_state)
    #        conf_mat[idx_act_state][idx_pred_state] += 1

    #        #print(self._cm.decode_state(predicted_state, Dataset.KASTEREN))
    #    return conf_mat

    def create_pred_act_seqs(self, test_obs_arr):
        """

        :param test_obs_arr:
            N x 2 numpy array
            first value is id of observation
            second value is id of state
        """
        K = len(self._hmm._z)
        N = len(test_obs_arr)
        #print(test_obs_arr)
        obs_seq = []
        y_pred = np.zeros((N))
        y_true = np.zeros((N))

        for n in range(N):
            obs_seq.append(int(test_obs_arr[n][0]))
            state_seq = self._hmm.viterbi(obs_seq)
            predicted_state = state_seq[-1:][0]
            actual_state = int(test_obs_arr[n][1])
            idx_pred_state = self._hmm._idx_state(predicted_state)
            idx_act_state = self._hmm._idx_state(actual_state)

            y_pred[n] = idx_pred_state
            y_true[n] = idx_act_state
        return y_true, y_pred


    def predict_next_observation(self, obs_seq):
        pass
