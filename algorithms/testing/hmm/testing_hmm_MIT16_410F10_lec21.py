import unittest
import numpy as np
from algorithms.hmm import HiddenMarkovModel
from algorithms.hmm import ProbabilityMassFunction

LA = 'Los Angeles'
NY = 'New York'
NULL = 'null'

"""
this example is straight from lecture nodes of MIT Lec 21 Nov 24, 2010
    Finding Keyser Soeze    
    travels between two cities: Lost Angeles and New York
    observations are sightings 
"""
class TestHmmWiki(unittest.TestCase):
    def setUp(self):
        # set of observations
        observation_alphabet = [LA, NY, NULL]
        states = [LA, NY]
        init_dist = [0.2, 0.8]

        # init markov model
        self.hmm = HiddenMarkovModel(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm.set_transition_matrix(np.array([[0.5,0.5],
                                                 [0.5,0.5]]))
        self.hmm.set_emission_matrix(np.array([[0.4,0.1,0.5],[0.1,0.5,0.4]]))
        self.obs_seq = [NULL, LA, LA, NULL, NY, NULL, NY, NY, NY, NULL,
                        NY, NY, NY, NY, NY, NULL, NULL, LA, LA, NY]

        print('='*100)
        print('Initial_probs:')
        print(self.hmm.pi_to_df())
        print('Transition_matrix:')
        print(self.hmm.transitions_to_df())
        print('-'*100)
        print('Emission:')
        print(self.hmm.emissions_to_df())
        print('='*100)
        print('*'*100)

    def tearDown(self):
        pass


    def test_xi(self):
        obs_seq = self.obs_seq
        forward_matrix = self.hmm.forward(obs_seq)
        backward_matrix = self.hmm.backward(obs_seq)
        prob_X = self.hmm.prob_X(forward_matrix)
        xi = self.hmm.xi(forward_matrix,backward_matrix, prob_X, obs_seq)
        print('~'*30)
        print('~'*30)
        print(xi)
        # todo add assertion

    def test_train_transitions(self):
        obs_seq = self.obs_seq
        alpha_matrix = self.hmm.forward(obs_seq)
        beta_matrix = self.hmm.backward(obs_seq)
        xi = self.hmm.xi(alpha_matrix, beta_matrix,obs_seq)
        pass

    def test_expected_trans_zn(self):
        obs_seq = self.obs_seq
        alpha_matrix = self.hmm.forward(obs_seq)
        beta_matrix = self.hmm.backward(obs_seq)
        res = self.hmm.gamma(alpha_matrix, beta_matrix)
        self.hmm.expected_trans_from_z(res)

    def test_transition_matrix(self):
        obs_seq = self.obs_seq
        alpha_matrix = self.hmm.forward(obs_seq)
        beta_matrix = self.hmm.backward(obs_seq)
        gamma = self.hmm.gamma(alpha_matrix, beta_matrix)
        xi = self.hmm.xi(alpha_matrix, beta_matrix, obs_seq)

        exp_trans_zn = self.hmm.expected_trans_from_z(gamma)
        exp_trans_zn_znp1 = self.hmm.expected_trans_from_za_to_zb(xi)

        res = self.hmm.new_transition_matrix(
            exp_trans_zn,
            exp_trans_zn_znp1)

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

        for i in range(0,20):
            self.hmm.training_step(obs_seq)
            print(self.hmm.pi_to_df())
            print(self.hmm.transitions_to_df())
            print(self.hmm.emissions_to_df())
            print("Prob of seq:\t" + str(self.hmm.prob_X(self.hmm.forward(obs_seq))))
            print('*#'*50)


        #self.assertTrue((result_trans_matrix == self.hmmA).all())
        # todo change
        #self.assertTrue((result_obs_matrix == self._E).all())

    def test_gamma(self):
        """
        slide 16
        test the probability of being in a state distribution after 20 observations
        gamma_20 = (0.16667, 0.8333)
        :return:
        """
        obs_seq = self.obs_seq
        alpha_matrix = self.hmm.forward(obs_seq)
        beta_matrix = self.hmm.backward(obs_seq)
        res = self.hmm.gamma(alpha_matrix, beta_matrix)
        print('*'*100)
        print(res)
        gamma_20 = res[20-1]
        print('*'*100)
        print(gamma_20)
        self.assertEqual(round(gamma_20[0],4),0.1667)
        self.assertEqual(round(gamma_20[1],4),0.8333)

    def test_pred_step(self):
        """
        slide 16
        the probability distribution at the next period
        gamma_21 = T'gamma_20(0.5,0.5)
        todo implement
        :return:
        """
        pass

    def test_train_transition(self):
        obs_seq = self.obs_seq
        alpha_matrix = self.hmm.forward(obs_seq)
        beta_matrix = self.hmm.backward(obs_seq)
        gamma = self.hmm.gamma(alpha_matrix, beta_matrix)
        xi = self.hmm.xi(alpha_matrix, beta_matrix, obs_seq)
        print(gamma)
        print('-'*10)
        print(xi)

    def test_train_emission(self):
        obs_seq = self.obs_seq
        alpha_matrix = self.hmm.forward(obs_seq)
        beta_matrix = self.hmm.backward(obs_seq)
        gamma = self.hmm.gamma(alpha_matrix, beta_matrix)

        new_em_matrix = self.hmm.new_emissions(gamma, obs_seq)
        print(self.hmm.emissions_to_df())
        self.hmm.set_emission_matrix(new_em_matrix)
        print(self.hmm.emissions_to_df())


    def test_prob_X(self):
        """
        slide 16
        the probablity of getting that particular observation sequence given the model
        :return:
        """
        obs_seq = self.obs_seq
        alpha_matrix = self.hmm.forward(obs_seq)
        beta_matrix = self.hmm.backward(obs_seq)
        res = self.hmm.prob_X(alpha_matrix)
        self.assertEqual(round(res,11), 0.00000000019)


    def test_forward(self):
        obs_seq = [NY, NY, NY, NY, NY, LA, LA, NY, NY, NY]
        #result = np.array([[1.0, 0.48, 0.2304, 0.027648],
        #                   [0.0, 0.12, 0.0936, 0.130032]])
        forward_matrix = self.hmm.forward(obs_seq)
        print(forward_matrix)
        #self.assertTrue(np.allclose(result, forward_matrix))


    def test_getter_emission(self):
        self.assertEqual(self.hmm.prob_x_given_z(NY, NY), 0.5)
        self.assertEqual(self.hmm.prob_x_given_z(LA, NY), 0.1)
        self.assertEqual(self.hmm.prob_x_given_z(NY, LA), 0.1)
        self.assertEqual(self.hmm.prob_x_given_z(LA, LA), 0.4)
        self.assertEqual(self.hmm.prob_x_given_z(NULL, NY), 0.4)
        self.assertEqual(self.hmm.prob_x_given_z(NULL,LA ), 0.5)

    def test_getter_transition(self):
        self.assertEqual(self.hmm.prob_za_given_zb(NY, NY), 0.5)
        self.assertEqual(self.hmm.prob_za_given_zb(NY, LA), 0.5)
        self.assertEqual(self.hmm.prob_za_given_zb(LA, NY), 0.5)
        self.assertEqual(self.hmm.prob_za_given_zb(LA, LA), 0.5)

    def test_viterbi(self):
        #todo
        # bug viterbi fix
        obs_seq = self.obs_seq
        best_state_seq = [NY,LA,LA,LA,LA,NY,LA,NY,NY,NY,LA,NY,NY,NY,NY,NY,LA,LA,LA,NY]

        # test
        res = self.hmm.viterbi(seq=obs_seq)
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
