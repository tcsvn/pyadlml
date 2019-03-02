import unittest
import numpy as np
import pomegranate
import matplotlib.pyplot as plt
from hmm.hmm_scaled import HiddenMarkovModel
from hmm.hmm import ProbabilityMassFunction

S0 = 'S0'
S1 = 'S1'
A = 'A'
B = 'B'


class TestHmmScaledExampleL08HMM(unittest.TestCase):
    def setUp(self):
        # set of observations
        observation_alphabet = [A, B]
        states = [S0, S1]
        init_dist = [1.0, 0.0]

        # init markov model
        self.hmm = HiddenMarkovModel(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm.set_transition_matrix(np.array([[0.6,0.4],[0.0,1.0]]))
        self.hmm.set_emission_matrix(np.array([[0.8,0.2],[0.3,0.7]]))


    def tearDown(self):
        pass

    def test_xi(self):
        obs_seq = [A,A,B]
        forward_matrix = self.hmm.forward(obs_seq)
        backward_matrix = self.hmm.backward(obs_seq)
        xi = self.hmm.xi(forward_matrix,backward_matrix, obs_seq)
        #print('-'*30)
        #print(xi)
        # todo add assertion

    def test_gamma(self):
        obs_seq = [A,A,B]
        forward_matrix = self.hmm.forward(obs_seq)
        backward_matrix = self.hmm.backward(obs_seq)
        gamma = self.hmm.gamma(forward_matrix, backward_matrix)
        print(gamma)
        # todo add assertion

    def test_cn(self):
        obs_seq = [A,A,B]
        cn = self.hmm.cn(obs_seq)
        self.assertEqual(round(cn[0],2),0.67)
        self.assertEqual(round(cn[1],2),0.67)
        self.assertEqual(round(cn[2],2),0.33)


    def test_prob_X(self):
        obs_seq = [A,A,B]
        forward_matrix = self.hmm.forward(obs_seq)
        prob_X = self.hmm.prob_X(forward_matrix)
        self.assertEqual(round(prob_X,2), 0.16)


        # test normalized variants
        cn = self.hmm.cn(obs_seq)
        norm_prob_X = self.hmm.norm_prob_X(cn)
        print(round(norm_prob_X,2))
        # is nearly the same as 0.16 ...
        # todo confirm with other results

    def test_backward(self):
        obs_seq = [A,A,B]
        result = np.array([[0.15768, 0.276, 0.4, 1.0],
                           [0.063, 0.21, 0.7, 1.0]])

        backward_matrix = self.hmm.backward(obs_seq)
        print(backward_matrix)

        print(result)
        self.assertTrue(np.allclose(result, backward_matrix))

    def test_forward(self):
        obs_seq = [A,A,B]
        result = np.array([[1.0, 0.48, 0.2304, 0.027648],
                           [0.0, 0.12, 0.0936, 0.130032]])
        forward_matrix = self.hmm.forward(obs_seq)
        cn = self.hmm.cn(obs_seq)
        norm_forward_matrix = self.hmm.norm_forward(obs_seq, cn)
        print(forward_matrix)
        print('--'*30)
        print(norm_forward_matrix)

        self.assertTrue(np.allclose(result, forward_matrix))


    def test_state_sequence(self):
        #self.hmm.render_graph()
        seq = [A,A,B]
        prob = self.hmm.prob_state_seq(seq)
        self.assertEqual(prob, 0.0128)


    def test_getter_emission(self):
        #self.hmm.draw()
        #print(self.hmm.emissions_to_df())
        self.assertEqual(self.hmm.prob_x_given_z(A, S0), 0.8)
        self.assertEqual(self.hmm.prob_x_given_z(B, S0), 0.2)
        self.assertEqual(self.hmm.prob_x_given_z(A, S1), 0.3)
        self.assertEqual(self.hmm.prob_x_given_z(B, S1), 0.7)

    def test_getter_transition(self):
        #self.hmm.draw()
        #print(self.hmm.transitions_to_df())
        self.assertEqual(self.hmm.prob_za_given_zb(S0, S0), 0.6)
        self.assertEqual(self.hmm.prob_za_given_zb(S0, S1), 0.0)
        self.assertEqual(self.hmm.prob_za_given_zb(S1, S0), 0.4)
        self.assertEqual(self.hmm.prob_za_given_zb(S1, S1), 1.0)

    def test_viterbi(self):
        obs_seq = [A,A,B]
        best_state_seq = [S0, S0, S1]

        # test
        res = self.hmm.viterbi(seq=obs_seq)
        self.assertListEqual(best_state_seq, res)
