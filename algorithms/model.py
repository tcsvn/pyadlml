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
#from algorithms.hmm.hmm_scaled import HMM_Scaled as HiddenMarkovModel
from algorithms.hmm._hmm_base import  HiddenMarkovModel
from algorithms.hmm.distributions import ProbabilityMassFunction
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
        state_list = dataset.get_state_list()
        observation_list = dataset.get_obs_list()

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

    def train(self, dataset, use_q_fct=False):
        obs_seq = dataset.get_train_seq()
        if use_q_fct == True or use_q_fct == False:
            self.use_q_fct(use_q_fct)

        obs_seq = obs_seq
        train_seq_1 = obs_seq[:30]
        train_seq_2 = obs_seq[30:60]
        train_seq_3 = obs_seq[60:90]
        train_seq_4 = obs_seq[120:150]
        self._hmm.train(train_seq_1,
                        self._epsilon,
                        self._training_steps,
                        self._use_q_fct)
        #self._hmm.train(train_seq_2,
        #                self._epsilon,
        #                self._training_steps,
        #                self._use_q_fct)

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
            init_dist = HiddenMarkovModel.gen_eq_pi(state_count)
            # generate equal sized start probabilitys
            em_mat = HiddenMarkovModel.gen_rand_emissions(state_count, em_count)
            # generate equal sized transition probabilitys
            trans_mat = HiddenMarkovModel.gen_rand_transitions(state_count)
            em_dist = ProbabilityMassFunction

            model = HiddenMarkovModel(
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
            curr_hmm = self._model_dict[i]
            digits_skipped = 0
            for j in range(0, len(lengths)):
                new_idx = idx + lengths[j]
                seq = enc_data[idx:new_idx]
                tmp_hmm = copy.deepcopy(curr_hmm)
                try:
                    curr_hmm.train(seq, steps=self._training_steps)
                except:
                    digits_skipped += 1
                    print('sequence to long => model crashes')
                    # rollback changes
                    curr_hmm = tmp_hmm
                idx = new_idx
                # todo debug only learn one digit
                if j == 0:
                    break
            #print('~'*100)
            #print(len(lengths))
            #print(digits_skipped)
            #print(curr_hmm)

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
        number_of_obs_to_gen = 20
        num = 3
        return self.generate_observations(num, number_of_obs_to_gen)

    def generate_observations(self, num, n):
        """
        todo maybe this should also be part of the hmm class
        :param: num
            is the number to draw/ the generative hmm to use
        :param: n
            n is the number of steps to generate in the future
        :return:
        """
        hmm = self._model_dict[num] # type: HMM_Scaled

        # in the pendigit dataset a digit always begins
        # with pen down, get symbols for these operations
        pen_down = hmm._o[-2:][0]
        pen_up = hmm._o[-1:]
        """
        the prediction process ends when the pen_up symbol is predicted
        but for numbers as 4, 5, 7 the prediction process should end after
        two pen_up are observed
        """
        retouch = num in [4,5,7]
        seq = [pen_down]
        for i in range(n):
            new_symbol = hmm.predict_xnp1(seq)
            seq.append(new_symbol)
            if new_symbol == pen_up:
                if retouch:
                    retouch = False
                else:
                    break
        return seq

    def predict_next_observation(self, obs_seq):
        pass
