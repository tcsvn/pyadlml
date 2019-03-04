import unittest
import numpy as np
from algorithms.hmm import HiddenMarkovModel

H = 'Head'
T = 'Tail'
# Board Mode: Angabe der Pin-Nummer
class TestHmmCoin(unittest.TestCase):
    def setUp(self):
        # set of observations
        observation_alphabet = [1,2,3,4,5,6]
        states = [, SN]
        init_dist = [1/3, 2/3]

        # init markov model
        self.hmm = HiddenMarkovModel(states, observation_alphabet, init_dist)
        self.hmm.set_transition_matrix(np.array([[0.6,0.4],[0.2,0.8]]))
        self.hmm.set_emission_matrix(np.array([[0.4,0.6],[0.8,0.2]]))

    def tearDown(self):
        pass

    def test_state_sequence(self):
        #self.hmm.render_graph()
        seq = [RN, RN, SN, SN, RN]
        prob = self.hmm.prob_state_seq(seq)
        self.assertEqual(prob, 0.0128)

    def test_getter_emission(self):
        #self.hmm.draw()
        #print(self.hmm.emissions_to_df())
        self.assertEqual(self.hmm.prob_x_given_z(HP, SN), 0.8)
        self.assertEqual(self.hmm.prob_x_given_z(HP, RN), 0.4)
        self.assertEqual(self.hmm.prob_x_given_z(GR, SN), 0.2)
        self.assertEqual(self.hmm.prob_x_given_z(GR, RN), 0.6)

    def test_getter_transition(self):
        #self.hmm.draw()
        #print(self.hmm.transitions_to_df())
        self.assertEqual(self.hmm.prob_za_given_zb(SN, SN), 0.8)
        self.assertEqual(self.hmm.prob_za_given_zb(SN, RN), 0.4)
        self.assertEqual(self.hmm.prob_za_given_zb(RN, SN), 0.2)
        self.assertEqual(self.hmm.prob_za_given_zb(RN, RN), 0.6)

    def test_sequence(self):
        # observation sequence
        Y = np.array([0,0,1])
        pass

    def test_probabilities(self):
        self.hmm.prob_init_x(HP)

    def test_viterbi(self):
        # calcultes the
        obs_seq = [HP, HP, GR, GR, GR, HP]
        best_state_seq = [SN, SN, SN, RN, RN, SN]
        res = self.hmm.viterbi(seq=obs_seq)
        self.assertListEqual(best_state_seq, res)