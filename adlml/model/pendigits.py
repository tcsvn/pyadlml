from hassbrain_algorithm.algorithms.hmm.distributions import ProbabilityMassFunction
from sklearn.externals import joblib
from hassbrain_algorithm.datasets.pendigit.pendigits import DatasetPendigits
from hassbrain_algorithm.models._model import Model
from hbhmm.hmm.hmm_log import HMM_log
import numpy as np


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

    def _model_init(self, dataset : DatasetPendigits):
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
            lst.append(item.generate_graphviz_dot_ext_lbl(act_retrieval_meth))
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

    def get_state_label_list(self):
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

    def _predict_next_observation(self, obs_seq):
        new_symbol = self._hmm.predict_xnp1(obs_seq)