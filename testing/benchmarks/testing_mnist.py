import os
import unittest

from hassbrain_algorithm.benchmarks import DatasetMNIST
from hassbrain_algorithm.algorithms.hmm._hmm_base import HiddenMarkovModel
from hassbrain_algorithm.algorithms.hmm import ProbabilityMassFunction
import numpy as np


#from algorithms.benchmarks.mnist_data.analysis import training


class TestMNIST(unittest.TestCase):
    def setUp(self):
        # set of observations
        #self._bench = Bench()
        #self._bench.load_dataset(Dataset.MNIST)
        #self._mnist_obj = self._bench._loaded_datasets[Dataset.MNIST.name]
        pass




    def tearDown(self):
        pass

    def test_own_parser(self):
        dirname = os.path.dirname(__file__)[:-25]
        print(dirname)
        MNIST_LABEL_FILE = dirname + '/datasets/mnist_sequences/trainlabels.txt'
        MNIST_DATA_FILES = dirname + '/datasets/mnist_sequences/trainimg-num-inputdata.txt'
        dmnist = DatasetMNIST()
        dmnist.load_files(MNIST_LABEL_FILE, MNIST_DATA_FILES, label=5)


#    def test_own_hmm(self):
#        n_observation_classes = 256
#        n_hidden_states = 30
#        n_iter = 10
#        train_digits = self._mnist_obj.get_train_seq()
#        centroids = training.get_digit_kmeans_centroids(
#            train_digits, n_observation_classes - 3)
#        training.set_digit_observations(
#            train_digits,
#            centroids,
#            n_observation_classes)
#
#        # train digit  4
#        DIGIT_LABEL = 4
#
#        # ------
#        current_digits = []
#        digit_observations = [] # set of observations occuring in observation sequences
#        samples = [] # concatenated observation sequences
#        lengths = [] # lengths of observation sequences in samples
#        for dig in train_digits:
#            if dig.label == DIGIT_LABEL:
#                for observation in dig.observations:
#                    if not observation in digit_observations:
#                        digit_observations.append(observation)
#                current_digits.append(dig)
#                samples += dig.observations
#                lengths.append(len(dig.observations))
#
#        """
#        Sequence for a training image relative distance dx, dy
#        and a marker for end of sequence (eos) and end of string (eos)
#
#            dx | dy | eos | eod
#            18 |  4 |  0  |  0
#            -1 |  1 |  0  |  0
#            -1 |  0 |  0  |  0
#            ...
#            0  |  0 |  1  |  1
#
#        """
#
#        le = preprocessing.LabelEncoder()
#        X = np.array(samples)
#        X = le.fit_transform(X)
#        print(X)
#        print('-'*100)
#        print(X)

        #hmm = HiddenMarkovModel(
        #    latent_variables=,
        #    observations=,
        #    em_dist=,
        #    initial_dist=
        #)





    def test_dataset(self):
        n_observation_classes = 256
        n_hidden_states = 30
        n_iter = 10
        tol = 0.1
        train_digits = self._mnist_obj.get_train_seq()
        centroids = training.get_digit_kmeans_centroids(
            train_digits, n_observation_classes - 3)
        training.set_digit_observations(
            train_digits,
            centroids,
            n_observation_classes)
        print('-'*10)
        print('-'*10)
        print(train_digits[0])
        print('-'*10)
        print(type(train_digits[0]))
        print('-'*10)
        print(train_digits[0].observations)
        print('-'*10)
        hmm = training.train_hmm(
            train_digits,
            n_observation_classes,
            n_hidden_states,
            n_iter,
            tol)

        #self._bench.register_model(HMM_Model())
        #self._bench.init_model_on_dataset(Dataset.MNIST)


    def test_train(self):
        # initalize with random hmm
        trans_matrix = np.random.random_sample((7,7))
        row_sums = trans_matrix.sum(axis=1)
        trans_matrix = trans_matrix/row_sums[:, np.newaxis]

        em_matrix = np.random.random_sample((7,28))
        row_sums = em_matrix.sum(axis=1)
        em_matrix = em_matrix/row_sums[:, np.newaxis]

        init_pi = np.random.random_sample(7)
        init_pi = init_pi/sum(init_pi)

        observation_alphabet =  self._kast_obj.get_sensor_list()
        states = self._kast_obj.get_state_list()
        # init markov model
        hmm = HiddenMarkovModel(states,
                                observation_alphabet,
                                ProbabilityMassFunction,
                                init_pi)
        hmm.set_emission_matrix(em_matrix)
        hmm.set_transition_matrix(trans_matrix)


        # do stuff
        obs_seq = self._kast_obj.get_train_seq()[:250]#[:19]
        print(hmm)
        print(obs_seq)
        print(len(obs_seq))
        alpha = hmm.forward(obs_seq)
        beta = hmm.backward(obs_seq)
        hmm._prob_X(alpha, beta)
        print('!'*100)
        hmm.train(obs_seq, None, 200)
        print('!'*100)
        #hmm.train(obs_seq, 0.000001, None)
        print(hmm)
        alpha = hmm.forward(obs_seq)
        beta = hmm.backward(obs_seq)
        print(hmm._prob_X(alpha, beta))

        #for i in range(0,300):
        #    alpha_before = hmm.forward(obs_seq)
        #    prob_x_before = hmm.prob_X(alpha_before)
        #    hmm.training_step(obs_seq)
        #    alpha_after = hmm.forward(obs_seq)
        #    prob_x_after = hmm.prob_X(alpha_after)
        #    if i == 199:
        #        #self.print_full(hmm.emissions_to_df())
        #        print('~'*10)
        #        #print(alpha_before)
        #        print(prob_x_before)
        #        #print(alpha_after)
        #        print(prob_x_after)
        #        print('~'*10)
        #        #print(hmm._z)
        #        #print(self._kast_obj.get_activity_label_from_id(hmm._z[6]))
        #        #print(obs_seq)

        #        # test if emission sum to 1
        #        df = hmm.emissions_to_df()
        #        df['row_sum'] = df.sum(axis=1)
        #        print(df)

        #        # test if transition sum to one
        #        df = hmm.transitions_to_df()
        #        df['row_sum'] = df.sum(axis=1)
        #        print(df)
        #        print('#'*100)



    def test_training_step_model(self):
        # initalize with random hmm
        trans_matrix = np.random.random_sample((7,7))
        row_sums = trans_matrix.sum(axis=1)
        trans_matrix = trans_matrix/row_sums[:, np.newaxis]

        em_matrix = np.random.random_sample((7,28))
        row_sums = em_matrix.sum(axis=1)
        em_matrix = em_matrix/row_sums[:, np.newaxis]

        init_pi = np.random.random_sample(7)
        init_pi = init_pi/sum(init_pi)

        observation_alphabet =  self._kast_obj.get_sensor_list()
        states = self._kast_obj.get_state_list()
        # init markov model
        hmm = HiddenMarkovModel(states,
                                observation_alphabet,
                                ProbabilityMassFunction,
                                init_pi)
        hmm.set_emission_matrix(em_matrix)
        hmm.set_transition_matrix(trans_matrix)

        #print(hmm)

        obs_seq = self._kast_obj.get_train_seq()[:19]
        alpha_before = hmm.forward(obs_seq)
        prob_x_before = hmm._prob_X(alpha_before)
        hmm.training_step(obs_seq)
        alpha_after = hmm.forward(obs_seq)
        prob_x_after = hmm._prob_X(alpha_after)
        self.print_full(hmm.emissions_to_df())
        print('~'*10)
        #print(alpha_before)
        print(prob_x_before)
        #print(alpha_after)
        print(prob_x_after)
        print('~'*10)
        print(hmm._z)
        print(self._kast_obj.decode_state_label(hmm._z[6]))
        print(obs_seq)
        print('#'*100)

        # test if emission sum to 1
        df = hmm.emissions_to_df()
        df['e'] = df.sum(axis=1)
        print(df['e'])

        # test if transition sum to one
        df = hmm.transitions_to_df()
        print(df)
        df['e'] = df.sum(axis=1)
        print(df['e'])


    def test_id_from_label(self):
        id1 = self._kast_obj.encode_obs_lbl('Cups cupboard', 0)
        id2 = self._kast_obj.encode_obs_lbl('Cups cupboard', 1)
        id3 = self._kast_obj.encode_obs_lbl('Washingmachine', 1)
        id4 = self._kast_obj.encode_obs_lbl('Groceries Cupboard', 1)
        id5 = self._kast_obj.encode_obs_lbl('Hall-Bathroom door', 0)
        self.assertEqual(1, id1)
        self.assertEqual(0, id2)
        self.assertEqual(26, id3)
        self.assertEqual(10, id4)
        self.assertEqual(13, id5)

    def test_label_from_id(self):
        print(self._kast_obj._label)
        print(self._kast_obj._label_hashmap)
        id1 = self._kast_obj.decode_obs_label(0)
        id2 = self._kast_obj.decode_obs_label(11)
        id3 = self._kast_obj.decode_obs_label(10)
        id4 = self._kast_obj.decode_obs_label(27)
        id5 = self._kast_obj.decode_obs_label(5)
        self.assertEqual('Cups cupboard', id1)
        self.assertEqual('Groceries Cupboard', id2)
        self.assertEqual('Groceries Cupboard', id3)
        self.assertEqual('Washingmachine', id4)
        self.assertEqual('Freezer', id5)

    def test_load_df(self):
        pass
        #seq = self._kast_obj.get_obs_seq()
        #df = self._kast_obj._df
        #self.assertEqual(2638, len(seq))
        #self.assertEqual(2638, len(df.index))

