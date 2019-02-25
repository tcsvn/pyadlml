import unittest
import numpy as np
import pomegranate
import matplotlib.pyplot as plt
from hmm.hmm import HiddenMarkovModel
from hmm.hmm import ProbabilityMassFunction

S0 = 'S0'
S1 = 'S1'
A = 'A'
B = 'B'
# Board Mode: Angabe der Pin-Nummer
class TestHmmExampleL08HMM(unittest.TestCase):
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

    def test_forward_backward(self):
        obs_seq = [A,A,B]
        #self.hmm.forward_backward(obs_seq)

    def test_backward(self):
        obs_seq = [A,A,B]
        self.hmm.backward(obs_seq)

        # pomegranate
        backward_matrix = self.pom.backward(obs_seq)
        print(backward_matrix)

    def test_forward(self):
        obs_seq = [A,A,B]
        forward_matrix = self.hmm.forward(obs_seq)
        print(forward_matrix)

    def test_state_sequence(self):
        #self.hmm.render_graph()
        seq = [A,A,B]
        prob = self.hmm.prob_state_seq(seq)
        self.assertEqual(prob, 0.0128)
        #self.pom.

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
        # calcultes the
        obs_seq = [A,A,B]
        #best_state_seq = [SN, SN, SN, RN, RN, SN]
        res = self.hmm.viterbi(seq=obs_seq)
        #self.assertListEqual(best_state_seq, res)
