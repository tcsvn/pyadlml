import unittest
import numpy as np
from algorithms.hmm import HiddenMarkovModel
from algorithms.hmm import ProbabilityMassFunction

S0 = 'S0'
S1 = 'S1'
A = 'A'
B = 'B'


class TestHmmExampleL08HMM(unittest.TestCase):
    def setUp(self):
        # set of observations
        observation_alphabet = [A, B]
        states = [S0, S1]
        init_dist = [1.0, 0.0]
        self.obs_seq = [A,A,B]

        # init markov model
        self.hmm = HiddenMarkovModel(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm.set_transition_matrix(np.array([[0.6,0.4],[0.0,1.0]]))
        self.hmm.set_emission_matrix(np.array([[0.8,0.2],[0.3,0.7]]))


    def tearDown(self):
        pass


    def test_probX(self):
        obs_seq = [A,A,B]
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        n0 = self.hmm.prob_X_n(alpha, beta, 0)
        n1 = self.hmm.prob_X_n(alpha, beta, 1)
        n2 = self.hmm.prob_X_n(alpha, beta, 2)
        #print(n0)
        #print(n1)
        #print(n2)
        self.assertAlmostEqual(n0, n1, 10)
        self.assertAlmostEqual(n1, n2, 10)

    def test_xi(self):
        obs_seq = [A,A,B]
        forward_matrix = self.hmm.forward(obs_seq)
        backward_matrix = self.hmm.backward(obs_seq)
        xi = self.hmm.xi(forward_matrix,backward_matrix, obs_seq)
        #print('-'*30)
        #print(xi)
        # todo add assertion

    def test_predict_xnp1(self):
        obs_seq = self.obs_seq
        print(self.hmm)
        #self.hmm.train(obs_seq, 0.000001)
        #print(self.hmm)
        symbol = self.hmm.predict_xnp1(obs_seq)
        print(symbol)

    def test_predict_probs_xnp(self):
        obs_seq = self.obs_seq
        print(self.hmm)
        #self.hmm.train(obs_seq, 0.000001)
        #print(self.hmm)
        predicted_probs = self.hmm._predict_probs_xnp(obs_seq)
        self.assertAlmostEqual(1.0, predicted_probs.sum())


    def test_gamma(self):
        obs_seq = [A,A,B]
        forward_matrix = self.hmm.forward(obs_seq)
        backward_matrix = self.hmm.backward(obs_seq)
        gamma = self.hmm.gamma(forward_matrix, backward_matrix)
        print(gamma)
        # todo add assertion

    def test_prob_X(self):
        obs_seq = [A,A,B]
        forward_matrix = self.hmm.forward(obs_seq)
        prob_X = self.hmm._prob_X(forward_matrix)
        self.assertEqual(round(prob_X,2), 0.16)

    def test_backward(self):
        obs_seq = [A,A,B]
        result = np.array([[ 0.276, 0.4, 1.0],
                           [ 0.21, 0.7, 1.0]]).T

        backward_matrix = self.hmm.backward(obs_seq)
        print(backward_matrix)

        print(result)
        self.assertTrue(np.allclose(result, backward_matrix))

    def test_forward(self):
        obs_seq = [A,A,B]
        result = np.array([[0.48, 0.2304, 0.027648],
                           [0.12, 0.0936, 0.130032]]).T
        forward_matrix = self.hmm.forward(obs_seq)
        print(forward_matrix)
        print('-'*200)
        print(result)

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
