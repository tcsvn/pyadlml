import unittest

from benchmarks.controller import Controller
from benchmarks.controller import Dataset
from algorithms.hmm._hmm_base import HiddenMarkovModel
from algorithms.hmm.distributions import ProbabilityMassFunction
from algorithms.hmm.testing.hmm2.hmm.discrete.DiscreteHMM import DiscreteHMM

import pandas as pd
import numpy as np


class TestKasteren(unittest.TestCase):
    def setUp(self):
        # set of observations
        self._bench = Controller()
        self._bench.load_dataset(Dataset.KASTEREN)
        self._kast_obj = self._bench._loaded_datasets[Dataset.KASTEREN.name] # type: DatasetKasteren

    def tearDown(self):
        pass

    def test_train_model(self):
        pass

    def test_train(self):
        # initalize with random hmm
        state_count = 7
        obs_count = 28
        training_steps = 200
        trans_matrix = HiddenMarkovModel.gen_rand_transitions(state_count)
        em_matrix = HiddenMarkovModel.gen_rand_emissions(state_count, obs_count)
        init_pi = HiddenMarkovModel.gen_rand_pi(state_count)

        observation_alphabet = self._kast_obj.get_sensor_list()
        states = self._kast_obj.get_activity_list()

        print(states)
        print(observation_alphabet)
        # init markov model
        hmm = HiddenMarkovModel(states,
                                observation_alphabet,
                                ProbabilityMassFunction,
                                init_pi)
        hmm.set_emission_matrix(em_matrix)
        hmm.set_transition_matrix(trans_matrix)

        hmm2 = DiscreteHMM(state_count, obs_count,
                           trans_matrix,
                           em_matrix, init_pi, init_type='user')


        import math
        # do stuff
        #obs_seq = self._kast_obj.get_train_seq()[:19]  # [:250]#[:19]
        obs_seq = self._kast_obj.get_train_seq()[:250]  # [:250]#[:19]
        print(math.exp(hmm2.forwardbackward(obs_seq)))
        hmm2.mapB(obs_seq)
        hmm2.train(obs_seq, training_steps)
        print(hmm2.pi)
        print(hmm2.A)
        print(math.exp(hmm2.forwardbackward(obs_seq)))

        #print('-'*100)
        #print(hmm)
        #print(obs_seq)

        #print(hmm.forward_backward(obs_seq))
        #hmm.train(obs_seq, None, training_steps)
        ##hmm.train(obs_seq, 0.000001, None)
        #print(hmm)

        #print(hmm.forward_backward(obs_seq))

        # for i in range(0,300):
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

    #def test_training_step_model(self):
    #    # initalize with random hmm
    #    trans_matrix = np.random.random_sample((7, 7))
    #    row_sums = trans_matrix.sum(axis=1)
    #    trans_matrix = trans_matrix / row_sums[:, np.newaxis]

    #    em_matrix = np.random.random_sample((7, 28))
    #    row_sums = em_matrix.sum(axis=1)
    #    em_matrix = em_matrix / row_sums[:, np.newaxis]

    #    init_pi = np.random.random_sample(7)
    #    init_pi = init_pi / sum(init_pi)

    #    observation_alphabet = self._kast_obj.get_sensor_list()
    #    states = self._kast_obj.get_activity_list()
    #    # init markov model
    #    hmm = HiddenMarkovModel(states,
    #                            observation_alphabet,
    #                            ProbabilityMassFunction,
    #                            init_pi)
    #    hmm.set_emission_matrix(em_matrix)
    #    hmm.set_transition_matrix(trans_matrix)

    #    # print(hmm)

    #    obs_seq = self._kast_obj.get_train_seq()[:19]
    #    alpha_before = hmm.forward(obs_seq)
    #    beta_before = hmm.backward(obs_seq)
    #    prob_x_before = hmm._prob_X(alpha_before, beta_before)
    #    hmm.training_step(obs_seq)
    #    alpha_after = hmm.forward(obs_seq)
    #    beta_after = hmm.backward(obs_seq)
    #    prob_x_after = hmm._prob_X(alpha_after, beta_after)
    #    self.print_full(hmm.emissions_to_df())
    #    print('~' * 10)
    #    # print(alpha_before)
    #    print(prob_x_before)
    #    # print(alpha_after)
    #    print(prob_x_after)
    #    print('~' * 10)
    #    print(hmm._z)
    #    print(self._kast_obj.get_activity_label_from_id(hmm._z[6]))
    #    print(obs_seq)
    #    print('#' * 100)

    #    # test if emission sum to 1
    #    df = hmm.emissions_to_df()
    #    df['e'] = df.sum(axis=1)
    #    print(df['e'])

    #    # test if transition sum to one
    #    df = hmm.transitions_to_df()
    #    print(df)
    #    df['e'] = df.sum(axis=1)
    #    print(df['e'])

    def test_test_list(self):
        test_arr = self._kast_obj.get_test_arr()
        #print(test_arr)
        #self._kast_obj.load_activitys()


    def print_full(self, x):
        pd.set_option('display.max_rows', len(x))
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        pd.set_option('display.float_format', '{:20,.10f}'.format)
        pd.set_option('display.max_colwidth', -1)
        print(x)
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.float_format')
        pd.reset_option('display.max_colwidth')

    def test_get_sensor_list(self):
        res = self._kast_obj.get_sensor_list()
        result = [1, 5, 6, 8, 9, 12, 13, 14, 17, 18, 20, 23, 24]

        # print(res)
        # print(result)
        # print(len(res))
        # print(self._kast_obj._sensor_label)
        # print(self._kast_obj.get_obs_seq())
        self.assertEqual(len(res), 28)
        # res = self._kast_obj.get_sensor_labels()
        # result = ['Cups cupboard', 'Dishwasher', 'Freezer', 'Fridge',
        #          'Frontdoor', 'Groceries Cupboard', 'Hall-Bathroom door',
        #          'Hall-Bedroom door', 'Hall-Toilet door', 'Microwave',
        #          'Pans Cupboard', 'Plates cupboard', 'ToiletFlush',
        #          'Washingmachine']
        # self.assertSetEqual(set(res), set(result))

    #def test_get_activity_list(self):
    #    res = self._kast_obj.get_activity_list()
    #    result = ['leave house', 'use toilet', 'take shower',
    #              'go to bed', 'prepare Breakfast', 'prepare Dinner',
    #              'get drink']
    #    self.assertSetEqual(set(res), set(result))

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
        seq = self._kast_obj._train_seq + self._kast_obj._test_seq
        df = self._kast_obj._df
        self.assertEqual(2638, len(seq))
        self.assertEqual(2638, len(df.index))