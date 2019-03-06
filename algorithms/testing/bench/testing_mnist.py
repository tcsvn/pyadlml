import unittest
from algorithms.benchmarks.benchmark import Bench
from algorithms.benchmarks.benchmark import Dataset
from algorithms.hmm.hmm import HiddenMarkovModel
from algorithms.hmm.distributions import ProbabilityMassFunction
import pandas as pd
import numpy as np


class TestMNIST(unittest.TestCase):
    def setUp(self):
        # set of observations
        self._bench = Bench()
        self._bench.load_dataset(Dataset.MNIST)
        self._mnist_obj = self._bench._loaded_datasets[Dataset.MNIST.name]


    def tearDown(self):
        pass


    def test_train_model(self):
        pass



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
        states = self._kast_obj.get_activity_list()
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
        hmm.prob_X(alpha, beta)
        print('!'*100)
        hmm.train(obs_seq, None, 200)
        print('!'*100)
        #hmm.train(obs_seq, 0.000001, None)
        print(hmm)
        alpha = hmm.forward(obs_seq)
        beta = hmm.backward(obs_seq)
        print(hmm.prob_X(alpha, beta))

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
        states = self._kast_obj.get_activity_list()
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
        prob_x_before = hmm.prob_X(alpha_before)
        hmm.training_step(obs_seq)
        alpha_after = hmm.forward(obs_seq)
        prob_x_after = hmm.prob_X(alpha_after)
        self.print_full(hmm.emissions_to_df())
        print('~'*10)
        #print(alpha_before)
        print(prob_x_before)
        #print(alpha_after)
        print(prob_x_after)
        print('~'*10)
        print(hmm._z)
        print(self._kast_obj.get_activity_label_from_id(hmm._z[6]))
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
        id1 = self._kast_obj.get_sensor_id_from_label('Cups cupboard', 0)
        id2 = self._kast_obj.get_sensor_id_from_label('Cups cupboard', 1)
        id3 = self._kast_obj.get_sensor_id_from_label('Washingmachine', 1)
        id4 = self._kast_obj.get_sensor_id_from_label('Groceries Cupboard', 1)
        id5 = self._kast_obj.get_sensor_id_from_label('Hall-Bathroom door', 0)
        self.assertEqual(1, id1)
        self.assertEqual(0, id2)
        self.assertEqual(26, id3)
        self.assertEqual(10, id4)
        self.assertEqual(13, id5)

    def test_label_from_id(self):
        print(self._kast_obj._label)
        print(self._kast_obj._label_hashmap)
        id1 = self._kast_obj.get_sensor_label_from_id(0)
        id2 = self._kast_obj.get_sensor_label_from_id(11)
        id3 = self._kast_obj.get_sensor_label_from_id(10)
        id4 = self._kast_obj.get_sensor_label_from_id(27)
        id5 = self._kast_obj.get_sensor_label_from_id(5)
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

