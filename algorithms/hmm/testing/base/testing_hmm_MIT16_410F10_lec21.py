import unittest
import numpy as np
from algorithms.hmm._hmm_base import HiddenMarkovModel
from algorithms.hmm.distributions import ProbabilityMassFunction
from algorithms.hmm.testing.hmm2.hmm.discrete.DiscreteHMM import DiscreteHMM

LA = 'Los Angeles'
NY = 'New York'
NULL = 'null'

"""
this example is straight from lecture nodes of MIT Lec 21 Nov 24, 2010
    Finding Keyser Soeze    
    travels between two cities: Lost Angeles and New York
    observations are sightings 
"""
class TestMIT16_410F10(unittest.TestCase):
    def setUp(self):
        precision = np.double

        # set of observations
        observation_alphabet = [LA, NY, NULL]
        states = [LA, NY]
        init_dist = [0.2, 0.8]
        init_dist2 = np.array([0.2, 0.8])
        trans_matrix = np.array([[0.5,0.5],[0.5,0.5]], dtype=precision)
        em_matrix = np.array([[0.4,0.1,0.5],[0.1,0.5,0.4]], dtype=precision)

        self.obs_seq = [NULL, LA, LA, NULL, NY, NULL, NY, NY, NY, NULL,
                NY, NY, NY, NY, NY, NULL, NULL, LA, LA, NY]

        # init markov model
        self.hmm = HiddenMarkovModel(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm.set_transition_matrix(trans_matrix)
        self.hmm.set_emission_matrix(em_matrix)


        self.obs_seq2 = [2,0,0,2,1,2,1,1,1,2,1,1,1,1,1,2,2,0,0,1]
        self.hmm2 = DiscreteHMM(2, 3, trans_matrix, em_matrix, init_dist2, init_type='user')
        self.hmm2.mapB(self.obs_seq2)

    def tearDown(self):
        pass


    def test_verify_A(self):
        """
        check if after each training step the transition matrix
        is valid
        :return: None
        """
        bol_verify = self.hmm.verify_transition_matrix()
        self.assertTrue(bol_verify)
        obs_seq = self.obs_seq

        for i in range(0,100):
            self.hmm.training_step(obs_seq)
            bol_verify = self.hmm.verify_transition_matrix()
            #if not bol_verify:
                #bol_verify = self.hmm.verify_transistion_matrix()

            self.assertTrue(bol_verify)

    def test_verify_emissions(self):
        """
        check if after each training step the transition matrix
        is valid
        :return: None
        """
        bol_verify = self.hmm.verify_emission_matrix()
        self.assertTrue(bol_verify)
        obs_seq = self.obs_seq

        for i in range(0,100):
            self.hmm.training_step(obs_seq)
            bol_verify = self.hmm.verify_emission_matrix()
            self.assertTrue(bol_verify)

    def test_new_pi(self):
        alpha = self.hmm.forward(self.obs_seq)
        beta = self.hmm.backward(self.obs_seq)
        gamma = self.hmm.gamma(alpha, beta)
        hmm_new_pi = self.hmm.new_pi(gamma)


        hmm2_gamma = self.hmm2.calcgamma(self.hmm2.calcxi(
            self.obs_seq2,
            self.hmm2.calcalpha(self.obs_seq2),
            self.hmm2.calcbeta(self.obs_seq2)),
            len(self.obs_seq2))
        # from code line 307
        hmm2_new_pi = hmm2_gamma[0]


        for zn in range(0,len(self.hmm._z)):
            self.assertAlmostEqual(hmm2_new_pi[zn], hmm_new_pi[zn])


    def test_new_A(self):
        obs_seq = self.obs_seq
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        xi = self.hmm.xi(obs_seq, alpha, beta)
        hmm_new_A = self.hmm.new_A(obs_seq, xi)


        hmm2_xi = self.hmm2.calcxi(
            self.obs_seq2,
            self.hmm2.calcalpha(self.obs_seq2),
            self.hmm2.calcbeta(self.obs_seq2))
        hmm2_gamma = self.hmm2.calcgamma(hmm2_xi, len(self.obs_seq2))
        hmm2_new_A = self.hmm2.reestimateA(self.obs_seq2, hmm2_xi, hmm2_gamma)


        for znm1 in range(0,len(self.hmm._z)):
            for zn in range(0,len(self.hmm._z)):
                self.assertAlmostEqual(hmm2_new_A[znm1][zn], hmm_new_A[znm1][zn])

    def test_gamma_xi_relation(self):
        # todo make use to speed up calculation
        obs_seq = self.obs_seq
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        xi = self.hmm.xi(obs_seq, alpha, beta)
        gamma = self.hmm.gamma(alpha, beta)

        K = len(self.hmm._z)
        N = len(obs_seq)
        for n in range(N-1):
            for i in range(K):
                sum_xi = xi[n][i].sum()
                self.assertAlmostEqual(gamma[n][i], sum_xi)



    def test_new_emission(self):
        # todo make that exactly the same
        obs_seq = self.obs_seq
        alpha = self.hmm.forward(self.obs_seq)
        beta = self.hmm.backward(self.obs_seq)
        gamma = self.hmm.gamma(alpha, beta)
        hmm_new_em = self.hmm.new_emissions(gamma, obs_seq)

        obs_seq2 = self.obs_seq2
        hmm2_xi = self.hmm2.calcxi(
            obs_seq2,
            self.hmm2.calcalpha(obs_seq2),
            self.hmm2.calcbeta(obs_seq2))
        hmm2_gamma = self.hmm2.calcgamma(hmm2_xi, len(obs_seq2))
        hmm2_new_ems = self.hmm2._reestimateB(obs_seq2, hmm2_gamma)

        print()
        print('#'*100)
        print(hmm_new_em)
        print('~'*100)
        print(hmm2_new_ems)
        print('~'*100)

        for znm1 in range(0,len(self.hmm._z)):
            for zn in range(0,len(self.hmm._z)):
                self.assertAlmostEqual(hmm2_new_ems[znm1][zn], hmm_new_em[znm1][zn])




    def test_training_step(self):

        obs_seq = self.obs_seq
        # transition matrix after convergence

        for i in range(0,20):
            self.hmm.training_step(obs_seq)


        self.hmm2.train(self.obs_seq2, 20, 0.001)

        #print(math.exp(self.hmm2.forwardbackward(self.obs_seq2)))


        #self.assertTrue((result_trans_matrix == self.hmmA).all())
        # todo change
        #self.assertTrue((result_obs_matrix == self._E).all())


    def test_training_q(self):
        """
        slide 22
        test if the training converges
        :return:
        """
        res_pi = np.array([1,0])
        res_A = np.array([[0.6909, 0.3091],
                          [0.0934, 0.9066]])
        res_E = np.array([[0.5807, 0.0010, 0.4183],
                          [0.000, 0.7621, 0.2379]])
        steps = 500
        obs_seq = self.obs_seq
        self.hmm.train(obs_seq, steps=steps, q_fct=True)
        self.hmm2.train(self.obs_seq2, iterations=steps)

        #print(self.hmm)
        #print('~'*100)
        #print(self.hmm2.pi)
        #print('-'*2)
        #print(self.hmm2.A)
        #print('-'*2)
        #print(self.hmm2.B)
        #print('~'*100)
        #print(res_pi)
        #print('-'*2)
        #print(res_A)
        #print('-'*2)
        #print(res_E)
        self.hmm.draw()
        vg.render('test.gv', view=True)

        # assert pi
        for zn in range(0, len(res_pi)):
            hmm_pi = round(self.hmm._pi[zn],4)
            self.assertAlmostEqual(res_pi[zn], hmm_pi, 2)

        # assert A
        for znm1 in range(0, len(res_A)):
            for zn in range(0,len(res_A[0])):
                hmm_val = round(self.hmm._A[znm1][zn],4)
                self.assertAlmostEqual(res_A[znm1][zn], hmm_val, 2)

        # assert Emissions
        pd_em = self.hmm.emissions_to_df()
        for k, zn in enumerate(self.hmm._z):
            for i, em in enumerate(self.hmm._o):
                hmm_val = round(pd_em[em][zn], 4)
                self.assertAlmostEqual(res_E[k][i], hmm_val, 2)


    def test_training_seqs(self):
        """
        slide 22
        test if the training converges
        :return:
        """
        res_pi = np.array([1,0])
        res_A = np.array([[0.6909, 0.3091],
                          [0.0934, 0.9066]])
        res_E = np.array([[0.5807, 0.0010, 0.4183],
                          [0.000, 0.7621, 0.2379]])
        steps = 1000
        obs_seq = self.obs_seq
        self.hmm.train(obs_seq, steps=steps)
        self.hmm2.train(self.obs_seq2, iterations=steps)

        self.hmm.train(obs_seq, steps=steps)
        self.hmm2.train(self.obs_seq2, iterations=steps)

        self.hmm.train(obs_seq, steps=steps)
        self.hmm2.train(self.obs_seq2, iterations=steps)

        print(self.hmm)
        print('~'*100)
        print(self.hmm2.pi)
        print('-'*2)
        print(self.hmm2.A)
        print('-'*2)
        print(self.hmm2.B)
        print('~'*100)
        print(res_pi)
        print('-'*2)
        print(res_A)
        print('-'*2)
        print(res_E)

        # assert pi
        for zn in range(0, len(res_pi)):
            hmm_pi = round(self.hmm._pi[zn],4)
            self.assertAlmostEqual(res_pi[zn], hmm_pi, 2)

        # assert A
        for znm1 in range(0, len(res_A)):
            for zn in range(0,len(res_A[0])):
                hmm_val = round(self.hmm._A[znm1][zn],4)
                self.assertAlmostEqual(res_A[znm1][zn], hmm_val, 2)

        # assert Emissions
        pd_em = self.hmm.emissions_to_df()
        for k, zn in enumerate(self.hmm._z):
            for i, em in enumerate(self.hmm._o):
                hmm_val = round(pd_em[em][zn], 4)
                self.assertAlmostEqual(res_E[k][i], hmm_val, 2)
    def test_training(self):
        """
        slide 22
        test if the training converges
        :return:
        """
        res_pi = np.array([1,0])
        res_A = np.array([[0.6909, 0.3091],
                          [0.0934, 0.9066]])
        res_E = np.array([[0.5807, 0.0010, 0.4183],
                          [0.000, 0.7621, 0.2379]])
        steps = 1000
        obs_seq = self.obs_seq
        self.hmm.train(obs_seq, steps=steps)
        self.hmm2.train(self.obs_seq2, iterations=steps)

        self.hmm.train(obs_seq, steps=steps)
        self.hmm2.train(self.obs_seq2, iterations=steps)

        self.hmm.train(obs_seq, steps=steps)
        self.hmm2.train(self.obs_seq2, iterations=steps)

        print(self.hmm)
        print('~'*100)
        print(self.hmm2.pi)
        print('-'*2)
        print(self.hmm2.A)
        print('-'*2)
        print(self.hmm2.B)
        print('~'*100)
        print(res_pi)
        print('-'*2)
        print(res_A)
        print('-'*2)
        print(res_E)

        # assert pi
        for zn in range(0, len(res_pi)):
            hmm_pi = round(self.hmm._pi[zn],4)
            self.assertAlmostEqual(res_pi[zn], hmm_pi, 2)

        # assert A
        for znm1 in range(0, len(res_A)):
            for zn in range(0,len(res_A[0])):
                hmm_val = round(self.hmm._A[znm1][zn],4)
                self.assertAlmostEqual(res_A[znm1][zn], hmm_val, 2)

        # assert Emissions
        pd_em = self.hmm.emissions_to_df()
        for k, zn in enumerate(self.hmm._z):
            for i, em in enumerate(self.hmm._o):
                hmm_val = round(pd_em[em][zn], 4)
                self.assertAlmostEqual(res_E[k][i], hmm_val, 2)


    def test_xi(self):
        obs_seq = self.obs_seq
        obs_seq2 = self.obs_seq2

        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        prob_X = self.hmm._prob_X(alpha, beta)
        xi = self.hmm.xi(obs_seq, alpha, beta, prob_X)

        hmm2_xi = self.hmm2.calcxi(
            obs_seq2,
            self.hmm2.calcalpha(obs_seq2),
            self.hmm2.calcbeta(obs_seq2))

        K = len(self.hmm._z)
        N = len(obs_seq)

        for n in range(0,N-1):
            for znm1k in range(0, K):
                for znk in range(0, K):
                    hmm2_val = hmm2_xi[n][znm1k][znk]
                    hmm_val = xi[n][znm1k][znk]
                    self.assertAlmostEqual(hmm2_val, hmm_val)


    def test_gamma(self):
        """
        slide 16
        test the probability of being in a state distribution after 20 observations
        gamma_20 = (0.16667, 0.8333)
        :return:
        """
        obs_seq = self.obs_seq
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        gamma = self.hmm.gamma(alpha, beta)

        hmm2_gamma = self.hmm2.calcgamma(self.hmm2.calcxi(
                self.obs_seq2,
                self.hmm2.calcalpha(self.obs_seq2),
                self.hmm2.calcbeta(self.obs_seq2)
            ),
            len(self.obs_seq2)
        )


        gamma_20 = gamma[20-1]
        self.assertEqual(round(gamma_20[0],4),0.1667)
        self.assertEqual(round(gamma_20[1],4),0.8333)

        # unfortunately the other rep doesn't calculate the last value
        # and is therefore wrong
        test_gamma = gamma[:len(gamma)-1]
        test_hmm2_gamma = hmm2_gamma[:len(gamma)-1]
        self.assertTrue(np.allclose(test_gamma,test_hmm2_gamma ))





    def print_hmm2(self):
        print(self.hmm2.A)
        print(self.hmm2.B)
        print(self.hmm2.pi)

    def test_forward(self):
        obs_seq = self.obs_seq
        obs_seq2 = self.obs_seq2

        alpha = self.hmm.forward(obs_seq)

        hmm2_alpha = self.hmm2.calcalpha(obs_seq2)

        N = len(obs_seq)
        K = len(self.hmm._z)
        for n in range(0, N):
            for k in range(0,K):
                self.assertAlmostEqual(hmm2_alpha[n][k], alpha[n][k])

        self.assertTrue(np.allclose(alpha, hmm2_alpha))

    def test_backward(self):
        obs_seq = self.obs_seq
        obs_seq2 = self.obs_seq2
        beta = self.hmm.backward(obs_seq)
        hmm2_beta = self.hmm2.calcbeta(obs_seq2)

        for n in range(0, len(obs_seq)):
            for k in range(0,len(self.hmm._z)):
                self.assertAlmostEqual(hmm2_beta[n][k], beta[n][k])

        self.assertTrue(np.allclose(beta, hmm2_beta))

    def test_prob_X(self):
        """
        slide 16
        the probablity of getting that particular observation sequence given the model
        :return:
        """
        obs_seq = self.obs_seq
        alpha = self.hmm.forward(obs_seq)
        beta = self.hmm.backward(obs_seq)
        hmm_prob_x = self.hmm._prob_X(alpha, beta)

        hmm2_prob_x = np.exp(self.hmm2.forwardbackward(self.obs_seq2))

        self.assertAlmostEqual(hmm2_prob_x, hmm_prob_x)
        #self.assertEqual(0.00000000019, round(hmm2_prob_x,11))

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
        obs_seq = self.obs_seq
        #best_state_seq = [NY,LA,LA,LA,LA,NY,LA,NY,NY,NY,LA,NY,NY,NY,NY,NY,LA,LA,LA,NY]
        best_state_seq = [NY,LA,LA,LA,NY,LA,NY,NY,NY,LA,NY,NY,NY,NY,NY,LA,LA,LA,LA,NY]

        # test
        res = self.hmm.viterbi(seq=obs_seq)
        print(len(best_state_seq))
        print(len(obs_seq))
        print(len(res))
        print("best_seq\t    viterbi_seq")
        print('-'*30)
        i=0
        for item1, item2 in zip(best_state_seq,res):
            print(str(i) + " " + item1 + "\t == " + item2)
            i+=1
        print(best_state_seq)
        print(res)
        self.assertListEqual(best_state_seq, res)
