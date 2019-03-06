import unittest
import numpy as np
from algorithms.hmm.hmm_scaled import HiddenMarkovModel
from algorithms.hmm.distributions import ProbabilityMassFunction
from algorithms.testing.hmm.hmm2.hmm.discrete.DiscreteHMM import DiscreteHMM

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
        self.hmm = HiddenMarkovModel(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm.set_transition_matrix(trans_matrix)
        self.hmm.set_emission_matrix(em_matrix)

        # init hmm2 markov model
        self.hmm2 = DiscreteHMM(2, 3, trans_matrix, em_matrix, init_dist2, init_type='user')
        self.obs_seq = [A,A,B]
        self.obs_seq2 = [0,0,1]
        self.hmm2.mapB(self.obs_seq2)


    def tearDown(self):
        pass

    def test_xi(self):
        obs_seq = self.obs_seq

        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        xi = self.hmm.xi(obs_seq, alpha, beta)

        cn = self.hmm.calc_cn(obs_seq)
        norm_alpha = self.hmm.norm_forward(obs_seq, cn)
        norm_beta = self.hmm.norm_backward(obs_seq, cn)
        norm_xi = self.hmm.norm_xi(norm_alpha, norm_beta, cn, obs_seq)

        # test with hmm2
        hmm2_alpha = self.hmm2.calcalpha(self.obs_seq2)
        hmm2_beta = self.hmm2.calcbeta(self.obs_seq2)
        hmm2_xi = self.hmm2.calcxi(self.obs_seq2, hmm2_alpha, hmm2_beta)

        print(xi)
        print('-'*100)
        print(norm_xi)
        print('-'*100)
        print(hmm2_xi)

    def test_gamma(self):
        obs_seq = self.obs_seq
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        gamma = self.hmm.gamma(alpha, beta)

        # do it for normalized thing
        cn = self.hmm.calc_cn(obs_seq)
        norm_alpha = self.hmm.norm_forward(obs_seq, cn)
        norm_beta = self.hmm.norm_backward(obs_seq, cn)
        norm_gamma = self.hmm.norm_gamma(norm_alpha, norm_beta)

        # test with hmm2
        hmm2_alpha = self.hmm2.calcalpha(obs_seq)
        hmm2_beta = self.hmm2.calcbeta(obs_seq)
        hmm2_xi = self.hmm2.calcxi(obs_seq, hmm2_alpha, hmm2_beta)
        hmm2_gamma = self.hmm2.calcgamma(hmm2_xi, len(obs_seq))


        print(gamma)
        print(hmm2_gamma)
        print(norm_gamma)

    def test_cn(self):
        obs_seq = [A,A,B]
        cn = self.hmm.calc_cn(obs_seq)
        self.assertEqual(round(cn[0],2),0.67)
        self.assertEqual(round(cn[1],2),0.67)
        self.assertEqual(round(cn[2],2),0.33)


    def test_prob_X(self):
        obs_seq = [A,A,B]
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        prob_X = self.hmm.prob_X(alpha, beta)
        #self.assertEqual(round(prob_X,2), 0.16)

        # test normalized variants
        cn = self.hmm.calc_cn(obs_seq)
        norm_prob_X = self.hmm.norm_prob_X(cn)
        print(norm_prob_X)
        print(round(norm_prob_X,2))
        # is nearly the same as 0.16 ...
        # todo confirm with other results

    def test_backward(self):
        obs_seq = [A,A,B]
        result = np.array([[0.15768, 0.276, 0.4, 1.0],
                           [0.063, 0.21, 0.7, 1.0]])

        beta = self.hmm.backward(obs_seq)

        cn = self.hmm.calc_cn(obs_seq)
        norm_beta = self.hmm.norm_backward(obs_seq, cn)

        hmm2_beta = self.hmm2.calcbeta(self.obs_seq2)

        print(beta)
        print(hmm2_beta)
        print(norm_beta)
        #test_beta = self.hmm.norm_beta_to_beta(norm_beta, cn)
        #print(test_beta)

        #self.assertTrue(np.allclose(result, backward_matrix))

    def test_forward(self):
        obs_seq = [A,A,B]
        result = np.array([[0.48, 0.2304, 0.027648],
                           [0.12, 0.0936, 0.130032]])

        alpha = self.hmm.forward(obs_seq)

        cn = self.hmm.calc_cn(obs_seq)
        norm_alpha = self.hmm.norm_forward(obs_seq, cn)

        hmm2_alpha = self.hmm2.calcalpha(self.obs_seq2)

        print(alpha)
        print(hmm2_alpha)
        print(norm_alpha)

        #test = self.hmm.norm_alpha_to_alpha(norm_alpha, cn)
        #self.assertTrue(np.allclose(result, alpha))


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
