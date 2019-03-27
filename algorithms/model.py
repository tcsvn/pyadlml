"""
    this is an abstract class of an implemented algorithm like hmm
    model is used for benchmarks
    model is also called from hmm events
"""
from algorithms.hmm._hmm_base import HiddenMarkovModel


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
        self._cm = controller
        self._hmm = None

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

        obs_seq = obs_seq[:30]
        self._hmm.train(obs_seq,
                        self._epsilon,
                        self._training_steps,
                        self._use_q_fct)

    def calc_accuracy(self, test_obs_seq):
        return 3.5


    def predict_next_observation(self, obs_seq):
        pass
