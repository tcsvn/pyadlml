"""
    this is an abstract class of an implemented algorithm like hmm
    model is used for benchmarks
    model is also called from hmm events
"""
from algorithms.hmm._hmm_base import HiddenMarkovModel
from benchmarks.controller import Controller
#from benchmarks.controller import Controller, Dataset
import copy
import numpy as np
from hmmlearn.hmm import MultinomialHMM
from algorithms.hmm._hmm_base import  HiddenMarkovModel
from algorithms.hmm.distributions import ProbabilityMassFunction
from algorithms.hmm.hmm_scaled import HMM_Scaled
from algorithms.hmm._hmm_log import HMM_log
from algorithms.hmm._hmm_log import HMM_log_scaled
import numpy as np
from benchmarks.datasets.kasteren import DatasetKasteren
from benchmarks.datasets.pendigits import DatasetPendigits
import math
import joblib

MD_FILE_NAME = "model_%s.joblib"


class Model():

    def __init__(self, name, controller):
        self._bench = None
        self._cm = controller # type: Controller
        self._model_name = MD_FILE_NAME

    def register_benchmark(self, bench):
        self._bench = bench

    def save_model(self, key):
        joblib.dump(self, self._model_name%(key))

    def load_model(self, key):
        return joblib.load(self._model_name%(key))

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




# wrapper class for hmm
class ModelHMM(Model):

    def __init__(self, controller):
        self._cm = controller # type: Controller
        self._hmm = None # type: HiddenMarkovModel

        # training parameters
        self._training_steps = 20
        self._epsilon = None
        self._use_q_fct = False
        Model.__init__(self, "Test", controller)

    def __str__(self):
        return self._hmm.__str__()

    def use_q_fct(self, value):
        if value == True or value == False:
            self._use_q_fct = value
        else:
            raise ValueError

    def model_init(self, dataset):
        state_list = dataset.get_state_list()
        observation_list = dataset.get_obs_list()
        K = len(state_list)
        D = len(observation_list)
        # init markov model in normal way
        #self._hmm = HMM_Scaled(state_list,
        #self._hmm = HMM_log(state_list,
        self._hmm = HiddenMarkovModel(state_list,
                               observation_list,
                               ProbabilityMassFunction,
                               initial_dist=None)
        init_pi = HiddenMarkovModel.gen_rand_pi(K)
        self._hmm.set_pi(init_pi)

        em_matrix = HiddenMarkovModel.gen_rand_emissions(K,D)
        self._hmm.set_emission_matrix(em_matrix)

        trans_mat = HiddenMarkovModel.gen_rand_transitions(K)
        self._hmm.set_transition_matrix(trans_mat)

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

    def train(self, dataset, use_q_fct=False):
        obs_seq = dataset.get_train_seq()
        if use_q_fct == True or use_q_fct == False:
            self.use_q_fct(use_q_fct)

        seq_lst = [obs_seq[:30], obs_seq[30:60], obs_seq[60:90], obs_seq[120:150]]
        self._hmm.train_seqs(seq_lst,
                            self._epsilon,
                            self._training_steps)

        #for i, subseq  in enumerate(lst):
        #    print(self._hmm)
        #    self._hmm.train(subseq,
        #                    self._epsilon,
        #                    self._training_steps,
        #                    self._use_q_fct)
        #self._hmm.train(obs_seq,
        #                     self._epsilon,
        #                     self._training_steps,
        #                     self._use_q_fct)


    def get_label_list(self):
        lst = []
        for state_symb in self._hmm._z:
            lst.append(self._cm.decode_state(state_symb))
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

    def create_pred_act_seqs(self, dataset : DatasetKasteren):
        """
        for a given list of states ans observation compute the predicted states given
        the list of observations
        :param state_list:
        :param obs_list:
        :return:
            y_true: list of true states
            y_pred: list of predicted states
        """

        # todo is conceptual sth like score  in hmm
        # todo therefore move into hmm_base class
        #obs_list, state_list = self._dataset.get_test_labels_and_seq()
        state_list, obs_list = dataset.get_test_labels_and_seq()
        K = len(self._hmm._z)
        N = len(state_list)

        obs_seq = []
        y_pred = np.zeros((N))
        y_true = np.zeros((N))

        for n in range(10):
        #for n in range(N):
            obs_seq.append(int(obs_list[n]))
            state_seq = self._hmm.viterbi(obs_seq)
            predicted_state = state_seq[-1:][0]
            actual_state = int(state_list[n])
            #idx_pred_state = self._hmm._idx_state(predicted_state)
            #idx_act_state = self._hmm._idx_state(actual_state)
            y_pred[n] = predicted_state
            y_true[n] = actual_state
            #y_pred[n] = idx_pred_state
            #y_true[n] = idx_act_state
        return y_true, y_pred


    def predict_next_observation(self, obs_seq):
        pass


class ModelHMM_log(ModelHMM):
    def __init__(self, controller):
        ModelHMM.__init__(self, controller)

    def model_init(self, dataset):
        state_list = dataset.get_state_list()
        observation_list = dataset.get_obs_list()
        K = len(state_list)
        D = len(observation_list)
        # init markov model in normal way
        #self._hmm = HMM_Scaled(state_list,
        self._hmm = HMM_log(state_list,
                                      observation_list,
                                      ProbabilityMassFunction,
                                      initial_dist=None)
        init_pi = HMM_log.gen_rand_pi(K)
        self._hmm.set_pi(init_pi)

        em_matrix = HMM_log.gen_rand_emissions(K,D)
        self._hmm.set_emission_matrix(em_matrix)

        trans_mat = HMM_log.gen_rand_transitions(K)
        self._hmm.set_transition_matrix(trans_mat)


class ModelHMM_log_scaled(ModelHMM):
    def __init__(self, controller):
        ModelHMM.__init__(self, controller)

    def model_init(self, dataset):
        state_list = dataset.get_state_list()
        observation_list = dataset.get_obs_list()
        K = len(state_list)
        D = len(observation_list)
        # init markov model in normal way
        #self._hmm = HMM_Scaled(state_list,
        self._hmm = HMM_log_scaled(state_list,
                            observation_list,
                            ProbabilityMassFunction,
                            initial_dist=None)
        init_pi = HMM_log.gen_rand_pi(K)
        self._hmm.set_pi(init_pi)

        em_matrix = HMM_log.gen_rand_emissions(K,D)
        self._hmm.set_emission_matrix(em_matrix)

        trans_mat = HMM_log.gen_rand_transitions(K)
        self._hmm.set_transition_matrix(trans_mat)

class ModelHMM_scaled(ModelHMM):
    def __init__(self, controller):
        ModelHMM.__init__(self, controller)

    def model_init(self, dataset):
        state_list = dataset.get_state_list()
        observation_list = dataset.get_obs_list()
        K = len(state_list)
        D = len(observation_list)
        # init markov model in normal way
        #self._hmm = HMM_Scaled(state_list,
        self._hmm = HMM_Scaled(state_list,
                            observation_list,
                            ProbabilityMassFunction,
                            initial_dist=None)
        init_pi = HMM_Scaled.gen_rand_pi(K)
        self._hmm.set_pi(init_pi)

        em_matrix = HMM_Scaled.gen_rand_emissions(K,D)
        self._hmm.set_emission_matrix(em_matrix)

        trans_mat = HMM_Scaled.gen_rand_transitions(K)
        self._hmm.set_transition_matrix(trans_mat)

class ModelPendigits(Model):
    """
    this model trains 10 hmms. Each hmm is trained on a sequence of drawn numbers

    """
    def __init__(self, controller, name):
        # the number of classes that should the directions should be
        # divided into
        # todo doesn't work for 30 training steps why!!!
        self._training_steps = 20
        self._epsilon = None
        self._use_q_fct = False

        # is used to know what to plot or generate sequence about
        self._selected_number = 0

        # dictionary to hold all 10 models each representing a number
        self._model_dict = {}
        Model.__init__(self, name, controller)

    def __str__(self):
        s = "-"*20
        s += "\n"
        s += str(id(self)) + "\n"
        s += self._model_name + "\n"
        s += str(self._model_dict)
        return s

    def select_number(self, num):
        if 0 <= num <= 9:
            self._selected_number = int(num)
        else:
            raise ValueError


    def _use_q_fct(self, value):
        if value is True or value is False:
            self._use_q_fct = value
        else:
            raise ValueError

    def model_init(self, dataset : DatasetPendigits):
        state_list = dataset.get_state_list()
        obs_list = dataset.get_obs_list()
        state_count = len(state_list)
        em_count = len(obs_list)

        for i in range(0, 10):
            #cls = HiddenMarkovModel
            cls = HMM_log
            #cls = HMM_Scaled
            #cls = HMM_log_scaled

            init_dist = cls.gen_eq_pi(state_count)
            # generate equal sized start probabilitys
            em_mat = cls.gen_rand_emissions(state_count, em_count)
            # generate equal sized transition probabilitys
            trans_mat = cls.gen_rand_transitions(state_count)
            em_dist = ProbabilityMassFunction

            model = cls(
                latent_variables=state_list,
                observations=obs_list,
                em_dist=em_dist,
                initial_dist=init_dist)

            model.set_transition_matrix(trans_mat)
            model.set_emission_matrix(em_mat)
            self._model_dict[i] = model

    def get_conv_plot_y_label(self):
        if self._use_q_fct:
            return 'Q(Theta, Theta_old)'
        else:
            return 'P(X|Theta)'

    def get_conv_plot_x_label(self):
        return 'training steps'


    def draw(self, act_retrieval_meth):
        lst = []
        for key, item in self._model_dict.items():
            lst.append(item.generate_visualization_2(act_retrieval_meth))
        return lst
        # vg.render('test.gv', view=True)

    def train(self, dataset:DatasetPendigits, use_q_fct=False):
        if use_q_fct == True or use_q_fct == False:
            self._use_q_fct(use_q_fct)
        # train 10 models for each digit one
        for i in range(0, 10):
            #print('-'*100)
            #print('training model nr %s for digit %s ' % (i, i))
            enc_data, lengths = dataset.get_train_seq_for_nr(i)
            #print("\ntraining %s digits for number %s" % (len(lengths), i))
            #print("total length of sequence %s observations" % (sum(lengths)))
            idx = 0
            curr_hmm = self._model_dict[i] # type: HiddenMarkovModel
            digits_skipped = 0
            seq_lst = []
            N = len(lengths)
            # todo flag for deletion
            N = 5
            for j in range(0, N):
                new_idx = idx + lengths[j]
                seq = enc_data[idx:new_idx]
                idx = new_idx
                seq_lst.append(seq)


            curr_hmm.train_seqs(seq_lst, steps=self._training_steps)
            print(curr_hmm)

    def get_label_list(self):
        """
        benchmark uses this to print the labels of the model
        :return:
        """
        lst = []
        for i in range(10):
            lst.append(str(i))
        return lst


    def get_models(self):
        return self._model_dict


    def load_model(self, key):
        dict =  {}
        for k in range(0, 10):
            name = self._model_name%(k)
            dict[k] = joblib.load(name)
        new_md = ModelPendigits(self._cm, self._model_name)
        new_md._model_dict = dict
        return new_md

    def save_model(self, k):
        # save hmm models
        for key, model in self._model_dict.items():
            joblib.dump(model, self._model_name%(key))

    def create_pred_act_seqs(self, dataset : DatasetPendigits):
        """
        for a given list of observations predict states and return the sequence
        :param: dataset
            get observation sequence from dataset
        :return:
            y_true: list of true states
            y_pred: list of predicted states
        """
        length = 10
        y_true = dataset.get_test_labels_and_seq()
        y_pred = []
        #N = len(y_true)
        N = 30
        y_true = y_true[:N]
        #print(y_true)
        #print(self._model_dict)
        # for every number
        for i in range(N):
            # stores prob_x of each hmm number on the number
            llks = np.zeros(length)

            # get test data for one number
            # lengths should be 0
            seq, tmp = dataset._create_test_seq(i)

            # let each model score on the number
            for j in range(length):
                try:
                    llks[j] = self._model_dict[j].forward_backward(seq)
                except ValueError:
                    break

            # get highest average probX of all different hmms for the
            # type of number i
            # save number of hmm that scored the most
            y_pred.append(np.argmax(llks))

        return y_true, y_pred

    def gen_obs(self):
        return self.generate_observations([])

    def generate_observations(self, seq):
        """
        todo maybe this should also be part of the hmm class
        :param: num
            is the number to draw/ the generative hmm to use
        :param: n
            n is the number of steps to generate in the future
        :return:
        """
        num = self._selected_number
        hmm = self._model_dict[num] # type: HMM_Scaled

        # in the pendigit dataset a digit always begins
        # with pen down, get symbols for these operations
        pen_down = hmm._o[-2:][0]
        pen_up = hmm._o[-1:]
        max_pred_count = 25
        """
        the prediction process ends when the pen_up symbol is predicted
        but for numbers as 4, 5, 7 the prediction process should end after
        two pen_up are observed
        """
        retouch = num in [4,5,7]
        if seq is []:
            seq = [pen_down]
        for i in range(max_pred_count):
            new_symbol = hmm.predict_xnp1(seq)
            seq.append(new_symbol)
            print(seq)
            if new_symbol == pen_up:
                if retouch:
                    retouch = False
                else:
                    break
        return seq

    def predict_next_observation(self, obs_seq):
        new_symbol = self._hmm.predict_xnp1(obs_seq)
