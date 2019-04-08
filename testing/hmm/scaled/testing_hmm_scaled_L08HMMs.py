import unittest
import numpy as np
from math import exp
from hassbrain_algorithm.algorithms.hmm._hmm_base import HiddenMarkovModel as norm_HMM
from hassbrain_algorithm.algorithms.hmm import HMM_Scaled as sc_HMM
from hassbrain_algorithm.algorithms.hmm import ProbabilityMassFunction
from testing.testing import DiscreteHMM

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
        init_dist2 = np.array([1.0, 0.0])
        trans_matrix = np.array([[0.6,0.4],[0.0,1.0]])
        em_matrix = np.array([[0.8,0.2],[0.3,0.7]])
        # init markov model
        self.hmm = norm_HMM(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm.set_transition_matrix(trans_matrix)
        self.hmm.set_emission_matrix(em_matrix)

        self.hmm_scaled = sc_HMM(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm_scaled.set_transition_matrix(trans_matrix)
        self.hmm_scaled.set_emission_matrix(em_matrix)

        # init hmm2 markov model
        self.hmm2 = DiscreteHMM(2, 3, trans_matrix, em_matrix, init_dist2, init_type='user')
        self.obs_seq = [A,A,B]
        self.obs_seq2 = [0,0,1]
        self.hmm2.mapB(self.obs_seq2)


    def tearDown(self):
        pass


    def test_train(self):
        obs_seq = self.obs_seq
        prob_xt = []
        prob_xt.append(self.hmm_scaled.forward_backward(obs_seq))
        self.hmm_scaled.training_step(obs_seq)
        prob_xt.append(self.hmm_scaled.forward_backward(obs_seq))
        self.hmm_scaled.training_step(obs_seq)
        prob_xt.append(self.hmm_scaled.forward_backward(obs_seq))
        self.hmm_scaled.training_step(obs_seq)
        prob_xt.append(self.hmm_scaled.forward_backward(obs_seq))
        self.hmm_scaled.training_step(obs_seq)
        prob_xt.append(self.hmm_scaled.forward_backward(obs_seq))
        self.hmm_scaled.training_step(obs_seq)
        prob_xt.append(self.hmm_scaled.forward_backward(obs_seq))

        # todo why is not prob(Xt) < prob(Xt+1) forall t
        print(prob_xt)


    def test_xi(self):
        obs_seq = [A,A,B,A]
        obs_seq2 = [0,0,1,0]
        n_alpha, cn = self.hmm_scaled.forward(obs_seq)
        n_beta = self.hmm_scaled.backward(obs_seq, cn)
        n_xi = self.hmm_scaled.xi(n_alpha, n_beta, cn, obs_seq)

        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        prob_X = self.hmm._prob_X(alpha, beta)
        xi = self.hmm.xi(obs_seq, alpha, beta, prob_X)


        self.hmm2.mapB(obs_seq2)
        hmm2_alpha = self.hmm2.calcalpha(obs_seq2)
        hmm2_beta = self.hmm2.calcbeta(obs_seq2)
        hmm2_xi = self.hmm2.calcxi(obs_seq2, hmm2_alpha, hmm2_beta)


        for i in range(0, len(self.hmm._z)):
            for j in range(0, len(self.hmm._z)):
                for n in range(0, len(obs_seq)-1):
                    self.assertAlmostEqual(xi[n][i][j], n_xi[n][i][j])
                    self.assertAlmostEqual(n_xi[n][i][j], hmm2_xi[n][i][j])





    def test_gamma(self):
        obs_seq = self.obs_seq
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        gamma = self.hmm.gamma(alpha, beta)

        n_alpha, cn = self.hmm_scaled.forward(obs_seq)
        n_beta = self.hmm_scaled.backward(obs_seq, cn)
        n_gamma = self.hmm_scaled.gamma(n_alpha, n_beta)

        hmm2_alpha = self.hmm2.calcalpha(obs_seq)
        hmm2_beta = self.hmm2.calcbeta(obs_seq)
        hmm2_xi = self.hmm2.calcxi(obs_seq, hmm2_alpha, hmm2_beta)
        hmm2_gamma = self.hmm2.calcgamma(hmm2_xi, len(obs_seq))

        for n in range(0, len(obs_seq)):
            for k in range(0, len(self.hmm_scaled._z)):
                self.assertAlmostEqual(gamma[n][k], n_gamma[n][k])

        self.assertTrue(np.allclose(hmm2_gamma[:2], n_gamma[:2]))


    def test_prob_X(self):
        obs_seq = [A,A,B]
        obs_seq2 = self.obs_seq2
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        prob_X = self.hmm._prob_X(alpha, beta)

        hmm2_prob_X = exp(self.hmm2.forwardbackward(obs_seq2))

        n_prob_X = self.hmm.forward_backward(obs_seq)
        self.assertAlmostEqual(hmm2_prob_X, prob_X)
        self.assertAlmostEqual(prob_X, n_prob_X)


    def test_backward(self):
        obs_seq = [A,A,B]
        result = np.array([[ 0.276, 0.4, 1.0],
                           [ 0.21, 0.7, 1.0]]).T

        beta = self.hmm.backward(obs_seq)

        n_alpha, cn = self.hmm_scaled.forward(obs_seq)
        n_beta = self.hmm_scaled.backward(obs_seq, cn)
        re_beta = self.hmm_scaled.nbeta_to_beta(n_beta, cn)

        hmm2_beta = self.hmm2.calcbeta(self.obs_seq2)


        for n in range(0, len(obs_seq)):
            for k in range(0, len(self.hmm_scaled._z)):
                self.assertAlmostEqual(beta[n][k], re_beta[n][k])
                self.assertAlmostEqual(beta[n][k], hmm2_beta[n][k])
                self.assertAlmostEqual(beta[n][k], result[n][k])


    def test_forward(self):
        """
        slide 29, alpha values
        :return:
        """
        obs_seq = [A,A,B]
        #result = np.array([[0.48, 0.2304, 0.027648],
        #                   [0.12, 0.0936, 0.130032]])

        alpha = self.hmm.forward(obs_seq)

        n_alpha, cn = self.hmm_scaled.forward(obs_seq)
        re_alpha = self.hmm_scaled.nalpha_to_alpha(n_alpha, cn)
        hmm2_alpha = self.hmm2.calcalpha(self.obs_seq2)


        #self.assertTrue(np.allclose(result, alpha))
        for n in range(0, len(obs_seq)):
            for k in range(0, len(self.hmm_scaled._z)):
                self.assertAlmostEqual(re_alpha[n][k], alpha[n][k])
                self.assertAlmostEqual(hmm2_alpha[n][k], alpha[n][k])


    #def test_state_sequence(self):
    #    #self.hmm.render_graph()
    #    seq = [A,A,B]
    #    prob = self.hmm.prob_state_seq(seq)
    #    self.assertEqual(prob, 0.0128)


    def test_getter_emission(self):
        self.assertEqual(self.hmm.prob_x_given_z(A, S0), 0.8)
        self.assertEqual(self.hmm.prob_x_given_z(B, S0), 0.2)
        self.assertEqual(self.hmm.prob_x_given_z(A, S1), 0.3)
        self.assertEqual(self.hmm.prob_x_given_z(B, S1), 0.7)

    def test_getter_transition(self):
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
