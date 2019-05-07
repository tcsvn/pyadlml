import unittest
import numpy as np
from math import exp
from hbhmm.hmm._hmm_base import HMM
from hbhmm.hmm.distributions import ProbabilityMassFunction
from hbhmm.testing.hmm2.discrete.DiscreteHMM import DiscreteHMM

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
        init_dist2 = np.array([1.0, 0.0])
        self.obs_seq = [A,A,B]
        trans_mat = np.array([[0.6,0.4],[0.0,1.0]])
        em_mat = np.array([[0.8,0.2],[0.3,0.7]])

        # init markov model
        self.hmm = HMM(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm.set_transition_matrix(trans_mat)
        self.hmm.set_emission_matrix(em_mat)

        # init hmm2
        self.hmm2 = DiscreteHMM(2, 3, trans_mat, em_mat, init_dist2, init_type='user')
        self.obs_seq = [A,A,B]
        self.obs_seq2 = [0,0,1]
        self.hmm2._mapB(self.obs_seq2)


    def tearDown(self):
        pass


    def test_xi(self):
        obs_seq = [A,A,B,A]
        obs_seq2 = [0,0,1,0]

        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        prob_X = self.hmm._prob_X(alpha, beta)
        xi = self.hmm.xi(obs_seq, alpha, beta, prob_X)

        self.hmm2._mapB(obs_seq2)
        hmm2_alpha = self.hmm2._calcalpha(obs_seq2)
        hmm2_beta = self.hmm2._calcbeta(obs_seq2)
        hmm2_xi = self.hmm2._calcxi(obs_seq2, hmm2_alpha, hmm2_beta)

        for i in range(0, len(self.hmm._z)):
            for j in range(0, len(self.hmm._z)):
                for n in range(0, len(obs_seq)-1):
                    self.assertAlmostEqual(xi[n][i][j], hmm2_xi[n][i][j])


    def test_gamma(self):
        obs_seq = self.obs_seq
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        gamma = self.hmm.gamma(alpha, beta)

        hmm2_alpha = self.hmm2._calcalpha(obs_seq)
        hmm2_beta = self.hmm2._calcbeta(obs_seq)
        hmm2_xi = self.hmm2._calcxi(obs_seq, hmm2_alpha, hmm2_beta)
        hmm2_gamma = self.hmm2._calcgamma(hmm2_xi, len(obs_seq))

        self.assertTrue(np.allclose(hmm2_gamma[:2], gamma[:2]))


    def test_prob_X(self):
        """
        slide 29, 0.03 + 0.13 = 0.16
        :return:
        """
        obs_seq = [A,A,B]
        obs_seq2 = self.obs_seq2
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        prob_X = self.hmm._prob_X(alpha, beta)
        hmm2_prob_X = exp(self.hmm2.forwardbackward(obs_seq2))
        #self.assertEqual(round(prob_X,2), 0.16)
        self.assertEqual(hmm2_prob_X, prob_X)

    def test_backward(self):
        """
        slide 33, beta values
        :return:
        """
        obs_seq = [A,A,B]
        result = np.array([[ 0.276, 0.4, 1.0],
                           [ 0.21, 0.7, 1.0]]).T

        beta = self.hmm.backward(obs_seq)

        hmm2_beta = self.hmm2._calcbeta(self.obs_seq2)


        self.assertTrue(np.allclose(hmm2_beta,beta))
        self.assertTrue(np.allclose(result,beta))


    def test_forward(self):
        """
        slide 29, alpha values
        :return:
        """
        obs_seq = [A,A,B]
        #result = np.array([[0.48, 0.2304, 0.027648],
        #                   [0.12, 0.0936, 0.130032]]).T

        alpha = self.hmm.forward(obs_seq)
        hmm2_alpha = self.hmm2._calcalpha(self.obs_seq2)

        #self.assertTrue(np.allclose(result, alpha))
        self.assertTrue(np.allclose(hmm2_alpha, alpha))

    # todo delete or verify
    #def test_state_sequence(self):
    #    #self.hmm.render_graph()
    #    seq = [S0,S0,S1]
    #    prob = self.hmm.prob_state_seq(seq)
    #    self.assertEqual(prob, 0.0128)


    def test_getter_emission(self):
        self.assertEqual(0.8, self.hmm.prob_x_given_z(A, S0))
        self.assertEqual(0.2, self.hmm.prob_x_given_z(B, S0))
        self.assertEqual(0.3, self.hmm.prob_x_given_z(A, S1))
        self.assertEqual(0.7, self.hmm.prob_x_given_z(B, S1))

    def test_getter_transition(self):
        self.assertEqual(0.6, self.hmm.prob_za_given_zb(S0, S0))
        self.assertEqual(0.0, self.hmm.prob_za_given_zb(S0, S1))
        self.assertEqual(0.4, self.hmm.prob_za_given_zb(S1, S0))
        self.assertEqual(1.0, self.hmm.prob_za_given_zb(S1, S1))

    def test_viterbi(self):
        obs_seq = [A,A,B]
        best_state_seq = [S0, S0, S1]

        # test
        res = self.hmm.viterbi(seq=obs_seq)
        self.assertListEqual(best_state_seq, res)
