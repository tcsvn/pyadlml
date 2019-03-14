import numpy as np
from algorithms.benchmarks.pendigit.loadUniPenData import loadUnipenData
from algorithms.benchmarks.pendigit.plotting import plotUniPenData
from algorithms.benchmarks.pendigit.normalize import normalize_example
from hmmlearn.hmm import MultinomialHMM
from algorithms.hmm.hmm import HiddenMarkovModel
from algorithms.hmm.distributions import ProbabilityMassFunction
import math
import joblib

MODEL_NAME="mnHmm_%s.joblib"

class DatasetPendigits():
    def __init__(self):
        self._train_labels = None
        self._train_data = None

        self._test_labels = None
        self._test_data = None

        self._model_dict = {}


    def load_files(self, training_file, test_file):
        train_data, train_labels = loadUnipenData(training_file)
        self._train_labels = train_labels

        #tdata is data without penUp and penDown
        train_data, tdata = normalize_example(train_data)
        self._train_data = train_data

        # load testdata
        test_data, test_labels = loadUnipenData(test_file)
        test_data, ptest_data = normalize_example(test_data)
        self._test_data = test_data


    def get_models(self):
        return self._model_dict

    def init_models(self):
        for i in range(10):
            #print('--'*10)
            # params=ste means transitions emissions and start probabiilitys are updated during training
            # initially emissions are generated at random

            # get emission labels
            enc_data, lengths = self._create_train_seq(i)
            print(enc_data)
            print(lengths)
            #print(self._train_labels)
            #print(self._train_data)
            exit(-1)
            self._model_dict[i].fit(enc_data, lengths)


            state_count = 10
            em_count = 0
            init_dist = HiddenMarkovModel.gen_eq_pi(state_count)
            # generate equal sized start probabilitys
            em_dist = HiddenMarkovModel.gen_rand_emissions(state_count, em_count)
            # generate equal sized transition probabilitys
            trans_mat = HiddenMarkovModel.gen_rand_transitions(state_count)
            em_dist = ProbabilityMassFunction
            state_list = self._train_labels
            observation_alphabet = []

            model = HiddenMarkovModel(
                latent_variables=state_list,
                observations=asdf,
                em_dist=em_dist,
                initial_dist=init_dist)

            model.set_transition_matrix(trans_mat)
            model.set_emission_matrix(em_dist)

            self._model_dict[i] = model

    def init_models_hmmlearn(self):
        # generate
        for i in range(10):
            #print('--'*10)
            # params=ste means transitions emissions and start probabiilitys are updated during training
            # initially emissions are generated at random
            model = MultinomialHMM(n_components=10,
                                   n_iter=20,
                                   tol=100,
                                   verbose=True,
                                   params='ste',
                                   init_params='e')
            init = 1. / 10

            # generate equal sized start probabilitys
            model.startprob_ = np.full(10, init)
            # generate equal sized transition probabilitys
            model.transmat_ = np.full((10, 10), init)
            self._model_dict[i] = model

    def save_models(self):
        # save models
        for key, model in self._model_dict.items():
            joblib.dump(model, MODEL_NAME%(key))

    def train_models_hmmlearn(self):
        for i in range(10):
            enc_data, lengths = self._create_train_seq(i)
            self._model_dict[i].fit(enc_data, lengths)

    def _create_train_seq(self, number):
        print('-'*100)
        ind = np.where(np.array(self._train_labels) == number)
        #print(ind)
        digit_data = np.array(self._train_data)[ind]
        #print(digit_data)
        enc_data, lengths = self._encode_direction(digit_data)
        return enc_data, lengths

    def load_models(self):
        for key in range(0,10):
            self._model_dict[key] = joblib.load(MODEL_NAME%(key))

    def plot_example(self, number):
        data, tdata = normalize_example(self._train_data)
        plotUniPenData(data[number])

    def _encode_direction(self, raw_data):
        enc_data = []
        lengths = []
        for example in raw_data:
            print(example)
            sq = []
            for point in example:
                print('-')
                x = point[0]
                y = point[1]
                print(x)
                print(y)
                if x == -1 and y == 1:
                    sq.append([8])
                    xp = float('inf')
                # the case where
                elif x == -1 and y == -1:
                    sq.append([9])
                    xp = float('inf')
                else:
                    if xp != float('inf'):
                        dx = xp - x
                        dy = yp - y
                        direction = (int(math.ceil(math.atan2(dy, dx) / (2 * math.pi / 8))) + 8) % 8
                        print(direction)
                        sq.append([direction])
                    xp = x
                    yp = y
            enc_data.extend(sq)
            lengths.append(len(sq))
            exit(-1)
        return enc_data, lengths

    def _encode_direction_ex(self, example):
        sq = []
        for point in example:
            x = point[0]
            y = point[1]
            if x == -1 and y == 1:
                sq.append([8])
                xp = float('inf')
            elif x == -1 and y == -1:
                sq.append([9])
                xp = float('inf')
            else:
                if xp != float('inf'):
                    dx = xp - x
                    dy = yp - y
                    direction = (int(math.ceil(math.atan2(dy, dx) / (2 * math.pi / 8))) + 8) % 8
                    sq.append([direction])
                xp = x
                yp = y
        return sq

    def benchmark(self):
        plabels = []
        for j in range(len(self._test_data)):
            llks = np.zeros(10)
            enc_data = np.atleast_2d(self._encode_direction_ex(self._test_data[j]))
            for i in range(10):
                llks[i] = self._model_dict[i].score(enc_data)
            plabels.append(np.argmax(llks))
        print(float(np.sum(np.array(plabels) == np.array(self._test_labels))) / len(plabels))