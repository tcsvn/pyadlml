import unittest
import numpy as np
from hbhmm.hmm import HiddenMarkovModel
from hbhmm.hmm.distributions import ProbabilityMassFunction

S1 = 'State 1'
S2 = 'State 2'
E = 'eggs'
N = 'no eggs'

"""
this example is straight from Wikipedia 
    https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
    
    there is a chicken form which eggs are collected everyday
    two unknown factors determine whether the chicken lays eggs
"""
class TestHmmWiki(unittest.TestCase):
    def setUp(self):
        # set of observations
        observation_alphabet = [N, E]
        states = [S1, S2]
        init_dist = [0.2, 0.8]

        # init markov model
        self.hmm = HiddenMarkovModel(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm.set_transition_matrix(np.array([[0.5,0.5],
                                                 [0.3,0.7]]))
        self.hmm.set_emission_matrix(np.array([[0.3,0.7],[0.8,0.2]]))

        self.obs_seq = [N,N,N,N,N,E,E,N,N,N]


    def tearDown(self):
        pass

    def test_learning_transition_matrix(self):
        obs_seq = self.obs_seq




    def test_learning(self):
        new_trans_matrix = np.array([[0.0598, 0.0908],
                                    [0.2179, 0.9705]])

        new_trans_matrix_after_norm = np.array([[0.3973, 0.6027],
                                     [0.1833, 0.8167]])

        new_em_state_1 = np.array([0.876,0.8769])
        new_em_state_2 = np.array([1.0,0.7385])
        new_em_state_1_after_norm = np.array([0.0908,0.9092])
        new_em_state_2_after_norm = np.array([0.5752,0.4248])


    def test_xi_2(self):
        obs_seq = self.obs_seq
        result_s1_to_s2 = [0.024,0.024,0.024,0.024,0.006,0.014,0.056,0.024,0.024]

        forward_matrix = self.hmm.forward(obs_seq)
        backward_matrix = self.hmm.backward(obs_seq)
        prob_X = self.hmm._prob_X(forward_matrix)
        val = self.hmm.test_xi(S1, S2, 1,
            forward_matrix, backward_matrix,
            prob_X, obs_seq)
        print('*'*30)
        print(val)


    def test_xi(self):
        obs_seq = self.obs_seq
        result_s1_to_s2 = [0.024,0.024,0.024,0.024,0.006,0.014,0.056,0.024,0.024]

        forward_matrix = self.hmm.forward(obs_seq)
        backward_matrix = self.hmm.backward(obs_seq)
        prob_X = self.hmm._prob_X(forward_matrix)
        xi = self.hmm.xi(forward_matrix,backward_matrix, prob_X, obs_seq)
        #print('-'*30)
        #print(xi)
        # todo add assertion
        # znm1 x zn x t
        s1_to_S2 = xi[0][1]
        print(s1_to_S2)
        for idx_xn, xn in enumerate(obs_seq):
            if idx_xn == 0: continue
            prob = s1_to_S2[idx_xn]
            print('Prob of | ' + obs_seq[idx_xn-1] +"-"+ xn + "| = " + str(prob))
            print('vs: ' + str(result_s1_to_s2[idx_xn]))




    def test_gamma(self):
        obs_seq = self.obs_seq
        forward_matrix = self.hmm.forward(obs_seq)
        backward_matrix = self.hmm.backward(obs_seq)
        gamma = self.hmm.gamma(forward_matrix, backward_matrix)
        print(gamma)
        # todo add assertion

    def test_prob_X(self):
        obs_seq = self.obs_seq
        forward_matrix = self.hmm.forward(obs_seq)
        prob_X = self.hmm._prob_X(forward_matrix)
        self.assertEqual(round(prob_X,2), 0.16)

    def test_backward(self):
        obs_seq = [N,N,N,N,N,E,E,N,N,N]

        result = np.array([[0.15768,0.063],
                            [0.276, 0.21] ,
                            [0.4,0.7],
                            [1.0, 1.0]])


        backward_matrix = self.hmm.backward(obs_seq)
        print(backward_matrix)

        #print(result)
        #self.assertTrue(np.allclose(result, backward_matrix))

    def test_forward(self):
        obs_seq = [N,N,N,N,N,E,E,N,N,N]
        #result = np.array([[1.0, 0.48, 0.2304, 0.027648],
        #                   [0.0, 0.12, 0.0936, 0.130032]])
        forward_matrix = self.hmm.forward(obs_seq)
        print(forward_matrix)

        #self.assertTrue(np.allclose(result, forward_matrix))


    def test_state_sequence(self):
        #self.hmm.render_graph()
        seq = self.obs_seq
        prob = self.hmm.prob_state_seq(seq)
        self.assertEqual(prob, 0.0128)


    def test_getter_emission(self):
        self.assertEqual(self.hmm.prob_x_given_z(N, S1), 0.3)
        self.assertEqual(self.hmm.prob_x_given_z(E, S1), 0.7)
        self.assertEqual(self.hmm.prob_x_given_z(N, S2), 0.8)
        self.assertEqual(self.hmm.prob_x_given_z(E, S2), 0.2)

    def test_getter_transition(self):
        print(self.hmm.transitions_to_df())
        self.assertEqual(self.hmm.prob_za_given_zb(S1, S1), 0.5)
        self.assertEqual(self.hmm.prob_za_given_zb(S1, S2), 0.3)
        self.assertEqual(self.hmm.prob_za_given_zb(S2, S1), 0.5)
        self.assertEqual(self.hmm.prob_za_given_zb(S2, S2), 0.7)

    def test_viterbi(self):
        obs_seq = [A,A,B]
        best_state_seq = [S0, S0, S1]

        # test
        res = self.hmm.viterbi(seq=obs_seq)
        self.assertListEqual(best_state_seq, res)
