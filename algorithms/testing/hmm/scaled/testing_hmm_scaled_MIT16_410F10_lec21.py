import math
import unittest
import numpy as np
from algorithms.hmm.hmm_scaled import HMM_Scaled as ScaledHMM
from algorithms.hmm._hmm_base import HiddenMarkovModel as hMM
from algorithms.hmm.distributions import ProbabilityMassFunction
from algorithms.testing.hmm.hmm2.hmm.discrete.DiscreteHMM import DiscreteHMM

LA = 'Los Angeles'
NY = 'New York'
NULL = 'null'

"""
this example is straight from lecture nodes of MIT Lec 21 Nov 24, 2010
    Finding Keyser Soeze    
    travels between two cities: Lost Angeles and New York
    observations are sightings 
"""
class TestMIT16_410F10_scaled(unittest.TestCase):
    def setUp(self):
        # set of observations
        observation_alphabet = [LA, NY, NULL]
        states = [LA, NY]
        init_dist = [0.2, 0.8]
        init_dist2 = np.array([0.2, 0.8])
        trans_matrix = np.array([[0.5,0.5],[0.5,0.5]])
        em_matrix = np.array([[0.4,0.1,0.5],[0.1,0.5,0.4]])

        self.obs_seq = [NULL, LA, LA, NULL, NY, NULL, NY, NY, NY, NULL,
                        NY, NY, NY, NY, NY, NULL, NULL, LA, LA, NY]
        # init markov model
        self.hmm_scaled = ScaledHMM(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm_scaled.set_transition_matrix(np.array([[0.5, 0.5],
                                                        [0.5,0.5]]))
        self.hmm_scaled.set_emission_matrix(np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]]))

        self.hmm = hMM(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm.set_transition_matrix(np.array([[0.5, 0.5],
                                                 [0.5,0.5]]))
        self.hmm.set_emission_matrix(np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]]))

        self.obs_seq2 = [2,0,0,2,1,2,1,1,1,2,1,1,1,1,1,2,2,0,0,1]
        self.hmm2 = DiscreteHMM(2, 3, trans_matrix, em_matrix, init_dist2, init_type='user')
        self.hmm2.mapB(self.obs_seq2)


    def tearDown(self):
        pass


    def test_xi(self):
        print()
        obs_seq = self.obs_seq
        n_alpha, cn = self.hmm_scaled.forward(obs_seq)
        n_beta = self.hmm_scaled.backward(obs_seq, cn)
        n_xi = self.hmm_scaled.xi(n_alpha, n_beta, cn, obs_seq)


        #print(n_xi)
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        prob_X = self.hmm._prob_X(alpha, beta)
        xi = self.hmm.xi(obs_seq, alpha, beta, prob_X)
        #print('#'*100)
        #print(xi)
        print("~"*10)
        K = len(self.hmm._z)
        N = len(obs_seq)

        for n in range(0, N-1):
            for i in range(0, K):
                for j in range(0, K):
                    self.assertAlmostEqual(xi[n][i][j], n_xi[n][i][j])

    def test_training(self):
        """
        slide 22
        test if the training converges
        :return:
        """
        # todo check if true
        obs_seq = self.obs_seq
        # transition matrix after convergence
        result_trans_matrix = np.array([[0.6909, 0.3091],
                                 [0.0934, 0.9066]])
        result_obs_matrix = np.array([[0.5807, 0.0010, 0.4183],
                               [0.000, 0.7621, 0.2379]])

        #print("Prob of seq:\t" + str(self.hmm_scaled.forward_backward(obs_seq)))
        for i in range(0,200):
            self.hmm_scaled.training_step(obs_seq)
            #print(self.hmm_scaled)
            #print("Prob of seq:\t" + str(self.hmm_scaled.forward_backward(obs_seq)))
            #print('*#'*50)
        #print(self.hmm_scaled)
        #print("Prob of seq:\t" + str(self.hmm_scaled.forward_backward(obs_seq)))
        #print('*#'*50)
        #print('~'*10)
        #print(result_trans_matrix)
        #print(result_obs_matrix)
        ##self.assertTrue((result_trans_matrix == self.hmmA).all())
        ## todo change
        ##self.assertTrue((result_obs_matrix == self._E).all())
        #print('\n~*'*10)


    def test_gamma(self):
        """
        slide 16
        test the probability of being in a state distribution after 20 observations
        gamma_20 = (0.16667, 0.8333)
        """
        obs_seq = self.obs_seq
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        gamma = self.hmm.gamma(alpha,beta)

        n_alpha, cn= self.hmm_scaled.forward(obs_seq)
        n_beta = self.hmm_scaled.backward(obs_seq, cn)
        n_gamma = self.hmm_scaled.gamma(n_alpha, n_beta)
        gamma_20 = n_gamma[20-1]

        for n in range(0, len(obs_seq)):
            for k, zn in enumerate(self.hmm._z):
                self.assertAlmostEqual(n_gamma[n][k], gamma[n][k])

        self.assertEqual(round(gamma_20[0],4),0.1667)
        self.assertEqual(round(gamma_20[1],4),0.8333)


    def test_forward(self):
        """
        compute alpha, normalized alpha then unnormalize it and compare the
        unnormalized version with the original alpha
        """
        obs_seq = [NY, NY, NY, NULL, NY, NY, NULL, LA, NULL, LA, NY, NY, NY]
        alpha = self.hmm.forward(obs_seq)
        n_alpha , cn = self.hmm_scaled.forward(obs_seq)
        re_alpha = self.hmm_scaled.nalpha_to_alpha(n_alpha, cn)
        for n in range(0, len(obs_seq)):
            for zn in range(0, len(self.hmm._z)):
                self.assertAlmostEqual(re_alpha[n][zn], alpha[n][zn])

        obs_seq2 = [1, 1, 1, 2, 1, 1, 2, 0, 2, 0, 1, 1, 1]
        hmm2_alpha = self.hmm2.calcalpha(obs_seq2)
        for n in range(0, len(obs_seq)):
            #print(alpha[n])
            #print(hmm2_alpha[n])
            #print(re_alpha[n])
            #print(hmm2_re_beta[n])
            #print(re_beta[n])
            print('-')

    def test_backward(self):
        print()
        obs_seq = [NY, NY, NY, NULL, NY, NY, NULL, LA, NULL, LA, NY, NY, NY]
        #obs_seq2 = [1, 1, 1, 2, 1, 1, 2, 0, 2, 0, 1, 1, 1]
        #obs_seq = self.obs_seq
        #obs_seq2 = self.obs_seq2
        beta = self.hmm.backward(obs_seq)
        n_alpha, cn = self.hmm_scaled.forward(obs_seq)
        n_beta = self.hmm_scaled.backward(obs_seq, cn)
        re_beta = self.hmm_scaled.nbeta_to_beta(n_beta, cn)
        #hmm2_beta = self.hmm2.calcbeta(obs_seq2)
        #hmm2_re_beta = self.hmm_scaled.nbeta_to_beta(hmm2_beta, cn)

        print(beta)
        print('-'*100)
        print(self.hmm)
        print('-'*100)
        print(n_beta)
        #print(self.hmm2.A)
        #print(self.hmm2.B)
        for n in range(0, len(obs_seq)):
            print(beta[n])
            #print(hmm2_beta[n])
            #print(hmm2_re_beta[n])
            print(re_beta[n])
            print('-')

    def test_prob_X(self):
        #obs_seq = [NY, NY, NY, NULL, NY, NY, NULL, LA, NULL, LA, NY, NY, NY]
        obs_seq = self.obs_seq
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        prob_X0 = self.hmm._prob_X(alpha, beta)

        n_alpha , cn = self.hmm_scaled.forward(obs_seq)
        prob_X1 = self.hmm_scaled._prob_X(cn)
        print(prob_X0, prob_X1)
        self.assertAlmostEqual(prob_X1, prob_X0)

    def test_cn(self):
        obs_seq = [NY, NY, NY, NULL, NY, NY, NULL, LA, NULL, LA, NY, NY, NY]

        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)

        n_alpha , cn = self.hmm_scaled.forward(obs_seq)
        n_beta = self.hmm_scaled.backward(obs_seq, cn)

        prob_X0 = self.hmm._prob_X(alpha, beta)
        prob_X1 = self.hmm_scaled._prob_X(cn)

        print(prob_X0, prob_X1)
        gamma = self.hmm.gamma(alpha, beta)
        n_gamma = self.hmm_scaled.gamma(n_alpha, n_beta)


        print(alpha)
        print(self.hmm_scaled.nalpha_to_alpha(n_alpha, cn))

        for n in range(0, len(obs_seq)):
            for k in range(0,len(self.hmm._z)):
                self.assertAlmostEqual(n_gamma[n][k], gamma[n][k])
                #print(gamma[n][k], n_gamma[n][k])


    def test_getter_emission(self):
        self.assertEqual(self.hmm_scaled.prob_x_given_z(NY, NY), 0.5)
        self.assertEqual(self.hmm_scaled.prob_x_given_z(LA, NY), 0.1)
        self.assertEqual(self.hmm_scaled.prob_x_given_z(NY, LA), 0.1)
        self.assertEqual(self.hmm_scaled.prob_x_given_z(LA, LA), 0.4)
        self.assertEqual(self.hmm_scaled.prob_x_given_z(NULL, NY), 0.4)
        self.assertEqual(self.hmm_scaled.prob_x_given_z(NULL, LA), 0.5)

    def test_getter_transition(self):
        self.assertEqual(self.hmm_scaled.prob_za_given_zb(NY, NY), 0.5)
        self.assertEqual(self.hmm_scaled.prob_za_given_zb(NY, LA), 0.5)
        self.assertEqual(self.hmm_scaled.prob_za_given_zb(LA, NY), 0.5)
        self.assertEqual(self.hmm_scaled.prob_za_given_zb(LA, LA), 0.5)

    def test_viterbi(self):
        #todo
        # bug viterbi fix
        obs_seq = self.obs_seq
        best_state_seq = [NY,LA,LA,LA,LA,NY,LA,NY,NY,NY,LA,NY,NY,NY,NY,NY,LA,LA,LA,NY]

        # test
        res = self.hmm_scaled.viterbi(seq=obs_seq)
        print(len(best_state_seq))
        print(len(obs_seq))
        print(len(res))
        print("best_seq\t    viterbi_seq")
        print('-'*30)
        i=0
        for item1, item2 in zip(best_state_seq,res):
            i+=1
            print(str(i) + " " + item1 + "\t == " + item2)
        print(res)
        #self.assertListEqual(best_state_seq, res)
