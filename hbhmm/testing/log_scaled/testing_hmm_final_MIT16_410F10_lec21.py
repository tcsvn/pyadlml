import unittest
import numpy as np
from hbhmm.hmm._hmm_base import HMM
from hbhmm.hmm.probs import Probs
from hbhmm.hmm.hmm_log import HMM_log
from hbhmm.hmm.hmm_log_scaled import HMM_log_scaled
from hbhmm.hmm.distributions import ProbabilityMassFunction
from hbhmm.testing.hmm2.discrete.DiscreteHMM import DiscreteHMM

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
        # set of observations
        observation_alphabet = [LA, NY, NULL]
        states = [LA, NY]
        init_dist = [0.2, 0.8]
        init_dist2 = np.array([0.2, 0.8])
        trans_matrix = np.array([[0.5,0.5],[0.5,0.5]])
        em_matrix = np.array([[0.4,0.1,0.5],[0.1,0.5,0.4]])

        self.obs_seq = [NULL, LA, LA, NULL, NY, NULL, NY, NY, NY, NULL,
                NY, NY, NY, NY, NY, NULL, NULL, LA, LA, NY]

        # init markov model log
        self.hmm_log = HMM_log(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm_log.set_transition_matrix(trans_matrix)
        self.hmm_log.set_emission_matrix(em_matrix)

        self.hmm = HMM(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm.set_transition_matrix(trans_matrix)
        self.hmm.set_emission_matrix(em_matrix)

        self.hmm_log_scaled = HMM_log_scaled(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm_log_scaled.set_transition_matrix(trans_matrix)
        self.hmm_log_scaled.set_emission_matrix(em_matrix)

        self.obs_seq2 = [2,0,0,2,1,2,1,1,1,2,1,1,1,1,1,2,2,0,0,1]
        self.hmm2 = DiscreteHMM(2, 3, trans_matrix, em_matrix, init_dist2, init_type='user')
        self.hmm2._mapB(self.obs_seq2)

    def tearDown(self):
        pass

    def test_sampling(self):
        steps = 100
        obs_seq = self.obs_seq

        self.hmm.train(obs_seq, steps=steps, q_fct=True)
        self.hmm_log.train(obs_seq, steps=steps, q_fct=True)
        self.hmm_log_scaled.train(obs_seq, steps=steps, q_fct=True)
        self.hmm2.train(self.obs_seq2, iterations=steps)


    def test_viterbi_latt(self):
        steps = 100
        obs_seq = self.obs_seq

        self.hmm.train(obs_seq, steps=steps, q_fct=True)
        self.hmm_log.train(obs_seq, steps=steps, q_fct=True)
        self.hmm_log_scaled.train(obs_seq, steps=steps, q_fct=True)
        self.hmm2.train(self.obs_seq2, iterations=steps)


        obs_seq = [NULL, LA, NY]
        hmm_vit_latt = self.hmm.viterbi_latt(obs_seq)
        hmm_log_vit_latt = self.hmm_log.viterbi_latt(obs_seq)
        hmm_log_scaled_vit_latt = self.hmm_log_scaled.viterbi_latt(obs_seq)
        print('#'*100)
        print('hmm_vit_latt: \n', hmm_vit_latt)
        print('~'*10)
        print('hmm_lg_sc_latt: \n', hmm_log_scaled_vit_latt)
        Probs.print_np_arr(hmm_log_scaled_vit_latt)
        Probs.print_np_arr(np.array([1]))
        #print('hmm_lg_sc_latt: \n', Probs.np_prob_arr2exp(hmm_loc_scaled_vit_latt))
        #print(hmm_loc_scaled_vit_state)
        print('*'*10)
        self.print_arr_exp(hmm_vit_latt)
        print('~'*10)
        self.print_arr_exp(hmm_log_vit_latt)
        print('~'*10)
        self.print_arr_exp(hmm_log_scaled_vit_latt)
        print('*'*10)
        for i in range(len(hmm_vit_latt)):
            for j in range(len(hmm_vit_latt[0])):
                rnd = 7
                print('-')
                #print(hmm_vit_latt[i][j], hmm_log_scaled_vit_latt[i][j])
                #print(round(hmm_vit_latt[i][j], rnd), round(hmm_log_vit_latt[i][j], rnd))
                #print('-')
                print(hmm_log_vit_latt[i][j], hmm_log_scaled_vit_latt[i][j])
                print(round(hmm_log_vit_latt[i][j], rnd), round(hmm_log_scaled_vit_latt[i][j], rnd))
                print('-')
                #self.assertAlmostEqual(hmm_vit_latt[i][j], float(hmm_log_vit_latt[i][j]))
                self.assertAlmostEqual(hmm_log_vit_latt[i][j], hmm_log_scaled_vit_latt[i][j], 2)

    def print_arr_exp(self, arr):
        for item in arr:
            print(np.exp(item))

    def test_viterbi2(self):
        steps = 200
        obs_seq = self.obs_seq

        self.hmm.train(obs_seq, steps=steps, q_fct=True)
        self.hmm_log.train(obs_seq, steps=steps, q_fct=True)
        self.hmm_log_scaled.train(obs_seq, steps=steps, q_fct=True)
        self.hmm2.train(self.obs_seq2, iterations=steps)

        hmm_vit_state = self.hmm.viterbi(obs_seq)
        hmm_log_vit_state = self.hmm_log.viterbi(obs_seq)
        hmm_log_scaled_vit_state = self.hmm_log_scaled.viterbi(obs_seq)

        #print(hmm_vit_state)
        #print(hmm_log_vit_state)
        #print(hmm_log_scaled_vit_state)
        self.assertTrue(np.array_equal(hmm_vit_state, hmm_log_vit_state))
        self.assertTrue(np.array_equal(hmm_vit_state, hmm_log_scaled_vit_state))


    def test_verify_A(self):
        """
        check if after each training step the transition matrix
        is valid
        :return: None
        """
        bol_verify = self.hmm_log.verify_transition_matrix()
        self.assertTrue(bol_verify)
        obs_seq = self.obs_seq

        for i in range(0,100):
            self.hmm_log.training_step(obs_seq)
            bol_verify = self.hmm_log.verify_transition_matrix()
            #if not bol_verify:
                #bol_verify = self.hmm.verify_transistion_matrix()

            self.assertTrue(bol_verify)

    def test_verify_emissions(self):
        """
        check if after each training step the transition matrix
        is valid
        :return: None
        """
        bol_verify = self.hmm_log.verify_emission_matrix()
        self.assertTrue(bol_verify)
        obs_seq = self.obs_seq

        for i in range(0,100):
            self.hmm_log.training_step(obs_seq)
            bol_verify = self.hmm_log.verify_emission_matrix()
            self.assertTrue(bol_verify)

    #def test_gamma_xi_relation(self):
    #    # todo make use to speed up calculation
    #    obs_seq = self.obs_seq
    #    alpha = self.hmm_log.forward(obs_seq)
    #    beta = self.hmm_log.backward(obs_seq)
    #    xi = self.hmm_log.xi(obs_seq, alpha, beta)
    #    gamma = self.hmm_log.gamma(alpha, beta)

    #    K = len(self.hmm_log._z)
    #    N = len(obs_seq)
    #    for n in range(N-1):
    #        for i in range(K):
    #            sum_xi = xi[n][i].sum()
    #            self.assertAlmostEqual(gamma[n][i], sum_xi)

    def test_new_pi(self):
        obs_seq = self.obs_seq
        hmm_alpha = self.hmm.forward(obs_seq)
        hmm_beta = self.hmm.backward(obs_seq)
        hmm_gamma = self.hmm.gamma(hmm_alpha, hmm_beta)
        hmm_pi = self.hmm.new_pi(hmm_gamma)

        hmm_log_alpha = self.hmm_log.forward(obs_seq)
        hmm_log_beta = self.hmm_log.backward(obs_seq)
        hmm_log_gamma = self.hmm_log.gamma(hmm_log_alpha, hmm_log_beta)
        hmm_log_pi = self.hmm_log.new_pi(hmm_log_gamma)
        hmm_log_pi = Probs.np_prob_arr2exp(hmm_log_pi)

        hmm_sc_alpha, cn = self.hmm_log_scaled.forward(obs_seq)
        hmm_sc_beta = self.hmm_log_scaled.backward(obs_seq, cn)
        hmm_sc_gamma = self.hmm_log_scaled.gamma(hmm_sc_alpha, hmm_sc_beta)
        hmm_sc_pi = self.hmm_log_scaled.new_pi(hmm_sc_gamma)
        hmm_sc_pi = Probs.np_prob_arr2exp(hmm_sc_pi)

        hmm2_gamma = self.hmm2._calcgamma(self.hmm2._calcxi(
            self.obs_seq2,
            self.hmm2._calcalpha(self.obs_seq2),
            self.hmm2._calcbeta(self.obs_seq2)),
            len(self.obs_seq2))
        # from code line 307
        hmm2_new_pi = hmm2_gamma[0]

        print()
        print('~'*100)
        print(hmm_pi)
        print('~'*100)
        print(hmm_log_pi)
        print('~'*100)
        print(hmm_sc_pi)
        print('~'*100)
        print(hmm2_new_pi)
        print('~'*100)

        for zn in range(0, len(self.hmm_log._z)):
            self.assertAlmostEqual(hmm2_new_pi[zn], hmm_pi[zn])


    def test_new_A(self):
        obs_seq = self.obs_seq
        hmm_alpha = self.hmm.forward(obs_seq)
        hmm_beta = self.hmm.backward(obs_seq)
        hmm_xi = self.hmm.xi(obs_seq, hmm_alpha, hmm_beta)
        hmm_new_A = self.hmm.new_A(obs_seq, hmm_xi)



        hmm_log_alpha = self.hmm_log.forward(obs_seq)
        hmm_log_beta = self.hmm_log.backward(obs_seq)
        hmm_log_xi = self.hmm_log.xi(obs_seq, hmm_log_alpha, hmm_log_beta)
        hmm_log_new_A = self.hmm_log.new_A(obs_seq, hmm_log_xi)
        hmm_log_new_A = Probs.np_prob_arr2exp(hmm_log_new_A)


        hmm_sc_alpha, cn = self.hmm_log_scaled.forward(obs_seq)
        hmm_sc_beta = self.hmm_log_scaled.backward(obs_seq, cn)
        hmm_sc_xi = self.hmm_log_scaled.xi(hmm_sc_alpha, hmm_sc_beta, cn, obs_seq)
        hmm_sc_new_A = self.hmm_log_scaled.new_A(obs_seq, hmm_sc_xi)
        hmm_sc_new_A = Probs.np_prob_arr2exp(hmm_sc_new_A)


        hmm2_xi = self.hmm2._calcxi(
            self.obs_seq2,
            self.hmm2._calcalpha(self.obs_seq2),
            self.hmm2._calcbeta(self.obs_seq2))
        hmm2_gamma = self.hmm2._calcgamma(hmm2_xi, len(self.obs_seq2))
        hmm2_new_A = self.hmm2._reestimateA(self.obs_seq2, hmm2_xi, hmm2_gamma)

        print()
        print('~'*100)
        print(hmm_new_A)
        print('~'*100)
        print(hmm_log_new_A)
        print('~'*100)
        print(hmm_sc_new_A)
        print('~'*100)
        print(hmm2_new_A)
        print('~'*100)

        for znm1 in range(0, len(self.hmm_log._z)):
            for zn in range(0, len(self.hmm_log._z)):
                self.assertAlmostEqual(hmm_new_A[znm1][zn], hmm_log_new_A[znm1][zn])
                self.assertAlmostEqual(hmm_log_new_A[znm1][zn], hmm_sc_new_A[znm1][zn])
                self.assertAlmostEqual(hmm_sc_new_A[znm1][zn], hmm2_new_A[znm1][zn])



    def test_new_emission(self):
        obs_seq = self.obs_seq
        alpha = self.hmm_log.forward(self.obs_seq)
        beta = self.hmm_log.backward(self.obs_seq)
        gamma = self.hmm_log.gamma(alpha, beta)
        new_em = self.hmm_log.new_emissions(gamma, obs_seq)
        new_em = Probs.np_prob_arr2exp(new_em)

        hmm_sc_alpha, cn = self.hmm_log_scaled.forward(obs_seq)
        hmm_sc_beta = self.hmm_log_scaled.backward(obs_seq, cn)
        hmm_sc_gamma = self.hmm_log_scaled.gamma(hmm_sc_alpha, hmm_sc_beta)
        hmm_sc_new_em = self.hmm_log_scaled.new_emissions(hmm_sc_gamma, obs_seq)
        hmm_sc_new_em = Probs.np_prob_arr2exp(hmm_sc_new_em)


        hmm_alpha = self.hmm.forward(obs_seq)
        hmm_beta = self.hmm.backward(obs_seq)
        hmm_gamma = self.hmm.gamma(hmm_alpha, hmm_beta)
        hmm_new_em = self.hmm.new_emissions(hmm_gamma, obs_seq)

        obs_seq2 = self.obs_seq2
        hmm2_xi = self.hmm2._calcxi(
            obs_seq2,
            self.hmm2._calcalpha(obs_seq2),
            self.hmm2._calcbeta(obs_seq2))
        hmm2_gamma = self.hmm2._calcgamma(hmm2_xi, len(obs_seq2))
        hmm2_new_ems = self.hmm2._reestimateB(obs_seq2, hmm2_gamma)

        print()
        print('~'*100)
        print(new_em)
        print('~'*100)
        print(hmm_new_em)
        print('~'*100)
        print(hmm_sc_new_em)
        print('~'*100)
        print(hmm2_new_ems)
        print('~'*100)

        for znm1 in range(0, len(self.hmm_log._z)):
            for zn in range(0, len(self.hmm_log._z)):
                #self.assertAlmostEqual(hmm2_new_ems[znm1][zn], hmm_new_em[znm1][zn])
                self.assertAlmostEqual(new_em[znm1][zn], hmm_new_em[znm1][zn])




    def test_training_with_qfct(self):
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
        self.hmm_log.train(obs_seq, steps=steps, q_fct=True)
        self.hmm_log_scaled.train(obs_seq, steps=steps, q_fct=True)
        self.hmm2.train(self.obs_seq2, iterations=steps)


        print('~'*100)
        print(self.hmm)
        print('~'*100)
        print(self.hmm_log.set_str_exp(True))
        print(self.hmm_log)
        print('~'*100)
        print(self.hmm_log_scaled.set_str_exp(True))
        print(self.hmm_log_scaled)
        print('~'*100)
        print(res_pi)
        print('-'*2)
        print(res_A)
        print('-'*2)
        print(res_E)
        #vg.render('test.gv', view=True)

        # assert pi
        #for zn in range(0, len(res_pi)):
        #    hmm_pi = round(self.hmm_log._pi[zn], 4)
        #    self.assertAlmostEqual(res_pi[zn], hmm_pi, 2)

        ## assert A
        #for znm1 in range(0, len(res_A)):
        #    for zn in range(0,len(res_A[0])):
        #        hmm_val = round(self.hmm_log._A[znm1][zn], 4)
        #        self.assertAlmostEqual(res_A[znm1][zn], hmm_val, 2)

        ## assert Emissions
        #pd_em = self.hmm_log.emissions_to_df()
        #for k, zn in enumerate(self.hmm_log._z):
        #    for i, em in enumerate(self.hmm_log._o):
        #        hmm_val = round(pd_em[em][zn], 4)
        #        self.assertAlmostEqual(res_E[k][i], hmm_val, 2)


    def test_training_seqs(self):
        """
        slide 22
        test if the training converges
        test 3 times the observant sequence and if this works
        :return:
        """
        res_pi = np.array([1,0])
        res_A = np.array([[0.6909, 0.3091],
                          [0.0934, 0.9066]])
        res_E = np.array([[0.5807, 0.0010, 0.4183],
                          [0.000, 0.7621, 0.2379]])
        steps = 50
        obs_seq = self.obs_seq
        obs_seq1 = obs_seq
        obs_seq2 = obs_seq
        #obs_seq_1 = [LA, NY, NY, LA, NULL, NULL, LA, NY, NULL, NY, LA, LA,
        #                NY, NY, NULL, NULL, LA, LA, NY]
        #obs_seq_2 = [NY, NY, LA, NULL, LA, NY, LA, LA, LA, NULL, NULL, NY, LA,
        #                NY, NY, NULL, NULL, LA, LA, NY]
        seqs = [obs_seq, obs_seq1, obs_seq2]
        #self.hmm3.train(obs_seq, steps=steps)
        print('#'*100)
        self.hmm.train_seqs(seqs, steps=steps)
        self.hmm_log.train_seqs(seqs, steps=steps)
        self.hmm_log_scaled.train_seqs(seqs, steps=steps)

        print('~'*100)
        print('ready with training')
        print('~'*100)
        print('~'*100)
        print(self.hmm)
        print('~'*100)
        self.hmm_log.set_str_exp(True)
        print(self.hmm_log)
        print('~'*100)
        self.hmm_log_scaled.set_str_exp(True)
        print(self.hmm_log_scaled)
        print('~'*100)

        #print(np.exp(self.hmm_log.forward_backward(self.obs_seq)))
        #print(np.exp(self.hmm_log.forward_backward(obs_seq1)))
        #print(np.exp(self.hmm_log.forward_backward(obs_seq2)))
        #print('~'*100)
        print(res_pi)
        print('-'*2)
        print(res_A)
        print('-'*2)
        print(res_E)

        # assert pi
        #for zn in range(0, len(res_pi)):
        #    hmm_pi = round(self.hmm._pi[zn],4)
        #    self.assertAlmostEqual(self.hmm_log_scaled._pi[zn], self.hmm._pi[zn])
        #    #self.assertAlmostEqual(res_pi[zn], hmm_pi, 2)

        ## assert A
        #for znm1 in range(0, len(res_A)):
        #    for zn in range(0,len(res_A[0])):
        #        hmm_val = round(self.hmm._A[znm1][zn],4)
        #        self.assertAlmostEqual(self.hmm_log_scaled._A[znm1][zn], self.hmm._A[znm1][zn])
        #        #self.assertAlmostEqual(res_A[znm1][zn], hmm_val, 2)

        ## assert Emissions
        #pd_em = self.hmm.emissions_to_df()
        #pd_em3 = self.hmm_log_scaled.emissions_to_df()
        #for k, zn in enumerate(self.hmm._z):
        #    for i, em in enumerate(self.hmm._o):
        #        hmm_val = round(pd_em[em][zn], 4)
        #        self.assertAlmostEqual(pd_em[em][zn], pd_em3[em][zn])
        #        #self.assertAlmostEqual(res_E[k][i], hmm_val, 2)


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
        steps = 100
        obs_seq = self.obs_seq
        self.hmm.train(obs_seq, steps=steps)
        self.hmm_log_scaled.train(obs_seq, steps=steps)
        self.hmm2.train(self.obs_seq2, iterations=steps)

        self.hmm_log_scaled.set_str_exp(True)
        print('~'*100)
        print(res_pi)
        print('-'*2)
        print(res_A)
        print('-'*2)
        print(res_E)
        print('~'*100)
        print(self.hmm_log_scaled)
        print('~'*100)
        print(self.hmm)
        print('~'*100)
        print(self.hmm2)
        print('~'*100)
        print(res_pi)
        print('-'*2)
        print(res_A)
        print('-'*2)
        print(res_E)
        # assert pi
        #for zn in range(0, len(res_pi)):
        #    hmm_pi = round(self.hmm_log._pi[zn], 4)
        #    self.assertAlmostEqual(res_pi[zn], hmm_pi, 2)

        ## assert A
        #for znm1 in range(0, len(res_A)):
        #    for zn in range(0,len(res_A[0])):
        #        hmm_val = round(self.hmm_log._A[znm1][zn], 4)
        #        self.assertAlmostEqual(res_A[znm1][zn], hmm_val, 2)

        ## assert Emissions
        #pd_em = self.hmm_log.emissions_to_df()
        #for k, zn in enumerate(self.hmm_log._z):
        #    for i, em in enumerate(self.hmm_log._o):
        #        hmm_val = round(pd_em[em][zn], 4)
        #        self.assertAlmostEqual(res_E[k][i], hmm_val, 2)


    def test_xi(self):
        obs_seq = self.obs_seq
        obs_seq2 = self.obs_seq2

        alpha, cn = self.hmm_log_scaled.forward(obs_seq)
        beta = self.hmm_log_scaled.backward(obs_seq, cn)
        prob_X = self.hmm_log_scaled._prob_X(cn)
        xi = self.hmm_log_scaled.xi(alpha, beta, cn, obs_seq)
        xi = Probs.np_prob_arr2exp(xi)

        hmm2_xi = self.hmm2._calcxi(
            obs_seq2,
            self.hmm2._calcalpha(obs_seq2),
            self.hmm2._calcbeta(obs_seq2))

        K = len(self.hmm_log._z)
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
        alpha, cn = self.hmm_log_scaled.forward(obs_seq)
        beta = self.hmm_log_scaled.backward(obs_seq, cn)
        gamma = self.hmm_log_scaled.gamma(alpha, beta)
        gamma = Probs.np_prob_arr2exp(gamma)

        hmm2_gamma = self.hmm2._calcgamma(self.hmm2._calcxi(
                self.obs_seq2,
                self.hmm2._calcalpha(self.obs_seq2),
                self.hmm2._calcbeta(self.obs_seq2)
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



    def test_forward(self):
        obs_seq = self.obs_seq
        obs_seq2 = self.obs_seq2

        alpha, cn = self.hmm_log_scaled.forward(obs_seq)
        alpha = self.hmm_log_scaled.nalpha_to_alpha(alpha, cn)
        alpha = Probs.np_prob_arr2exp(alpha)

        hmm2_alpha = self.hmm2._calcalpha(obs_seq2)

        N = len(obs_seq)
        K = len(self.hmm_log._z)
        for n in range(0, N):
            for k in range(0,K):
                self.assertAlmostEqual(hmm2_alpha[n][k], alpha[n][k])

        self.assertTrue(np.allclose(alpha, hmm2_alpha))

    def test_backward(self):
        obs_seq = self.obs_seq
        obs_seq2 = self.obs_seq2
        alpha, cn = self.hmm_log_scaled.forward(obs_seq)
        beta = self.hmm_log_scaled.backward(obs_seq, cn)
        beta = self.hmm_log_scaled.nbeta_to_beta(beta, cn)
        beta = Probs.np_prob_arr2exp(beta)
        hmm2_beta = self.hmm2._calcbeta(obs_seq2)

        for n in range(0, len(obs_seq)):
            for k in range(0, len(self.hmm_log._z)):
                self.assertAlmostEqual(hmm2_beta[n][k], beta[n][k])

        self.assertTrue(np.allclose(beta, hmm2_beta))

    def test_prob_X(self):
        """
        slide 16
        the probablity of getting that particular observation sequence given the model
        :return:
        """
        obs_seq = self.obs_seq
        alpha, cn = self.hmm_log_scaled.forward(obs_seq)
        beta = self.hmm_log_scaled.backward(obs_seq, cn)
        hmm_prob_x = float(self.hmm_log_scaled._prob_X(cn))

        hmm2_prob_x = self.hmm2.forwardbackward(self.obs_seq2)

        self.assertAlmostEqual(hmm2_prob_x, hmm_prob_x)
        #print(np.exp(hmm_prob_x))
        #self.assertEqual(0.00000000019, round(hmm2_prob_x,11))

    def test_getter_emission(self):
        self.assertEqual(np.log(0.5), float(self.hmm_log_scaled.prob_x_given_z(NY, NY)))
        self.assertEqual(np.log(0.1), float(self.hmm_log_scaled.prob_x_given_z(LA, NY)))
        self.assertEqual(np.log(0.1), float(self.hmm_log_scaled.prob_x_given_z(NY, LA)))
        self.assertEqual(np.log(0.4), float(self.hmm_log_scaled.prob_x_given_z(LA, LA)))
        self.assertEqual(np.log(0.4), float(self.hmm_log_scaled.prob_x_given_z(NULL, NY)))
        self.assertEqual(np.log(0.5), float(self.hmm_log_scaled.prob_x_given_z(NULL, LA)))

    def test_getter_transition(self):
        self.assertEqual(np.log(0.5), float(self.hmm_log_scaled.prob_za_given_zb(NY, NY)))
        self.assertEqual(np.log(0.5), float(self.hmm_log_scaled.prob_za_given_zb(NY, LA)))
        self.assertEqual(np.log(0.5), float(self.hmm_log_scaled.prob_za_given_zb(LA, NY)))
        self.assertEqual(np.log(0.5), float(self.hmm_log_scaled.prob_za_given_zb(LA, LA)))

    def test_viterbi(self):
        obs_seq = self.obs_seq
        #best_state_seq = [NY,LA,LA,LA,LA,NY,LA,NY,NY,NY,LA,NY,NY,NY,NY,NY,LA,LA,LA,NY]
        best_state_seq = [NY,LA,LA,LA,NY,LA,NY,NY,NY,LA,NY,NY,NY,NY,NY,LA,LA,LA,LA,NY]

        # test
        res = self.hmm_log.viterbi(seq=obs_seq)
        self.assertListEqual(best_state_seq, res)
