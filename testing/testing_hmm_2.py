import unittest
import numpy as np
import pomegranate
import matplotlib.pyplot as plt
from hmm.hmm import HiddenMarkovModel
from hmm.hmm import ProbabilityMassFunction

RN = 'Rainy'
SN = 'Sunny'
HP = 'Happy'
GR = 'Grumpy'
# Board Mode: Angabe der Pin-Nummer
class TestHmmSunnyRainyExample(unittest.TestCase):
    def setUp(self):
        # set of observations
        observation_alphabet = [HP, GR]
        states = [RN, SN]
        init_dist = [1/3, 2/3]

        # init markov model
        self.hmm = HiddenMarkovModel(states, observation_alphabet, ProbabilityMassFunction, init_dist)
        self.hmm.set_transition_matrix(np.array([[0.6,0.4],[0.2,0.8]]))
        self.hmm.set_emission_matrix(np.array([[0.4,0.6],[0.8,0.2]]))

        # init pomegranate model for comparision
        #self.pom = pomegranate.HiddenMarkovModel('testing')
        #dist_rn = pomegranate.DiscreteDistribution({HP : 0.4, GR : 0.6})
        #dist_sn = pomegranate.DiscreteDistribution({HP : 0.8, GR : 0.2})
        #state_rn = pomegranate.State(dist_rn, name=RN)
        #state_sn = pomegranate.State(dist_sn, name=SN)
        #self.pom.add_states(state_rn, state_sn)
        #self.pom.add_transition(self.pom.start, state_rn, 1.0)
        #self.pom.add_transition(state_rn, state_rn, 0.6)
        #self.pom.add_transition(state_rn, state_sn, 0.4)
        #self.pom.add_transition(state_sn, state_sn, 0.8)
        #self.pom.add_transition(state_sn, state_rn, 0.2)
        #self.pom.add_transition(state_sn, self.pom.end, 0.1)
        #self.pom.bake()

    def test_pom(self):
        #plt.figure(figsize=(10,6))
        #self.pom.plot()
        self.hmm.draw()

    def tearDown(self):
        pass

    def test_forward_backward(self):
        obs_seq = [HP, HP, GR, GR, GR, HP]
        forward_matrix = self.hmm.forward_backward(obs_seq)

    def test_train(self):
        obs_seq = [HP, HP, GR, GR, GR, HP]
        self.hmm.train(obs_seq)


    def test_backward(self):
        obs_seq = [HP, HP, GR, GR, GR, HP]
        backward_matrix = self.hmm.backward(obs_seq)
        print(backward_matrix)

        # pomegranate
        #backward_matrix = self.pom.backward(obs_seq)
        #backward_matrix = np.exp(backward_matrix)
        #print(backward_matrix)

    def test_forward(self):
        obs_seq = [HP, HP, GR, GR, GR, HP]
        self.hmm.forward(obs_seq)

    def test_state_sequence(self):
        #self.hmm.render_graph()
        seq = [RN, RN, SN, SN, RN]
        prob = self.hmm.prob_state_seq(seq)
        self.assertEqual(prob, 0.0128)
        #self.pom.

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

    def test_viterbi(self):
        # calcultes the
        obs_seq = [HP, HP, GR, GR, GR, HP]
        best_state_seq = [SN, SN, SN, RN, RN, SN]
        res = self.hmm.viterbi(seq=obs_seq)
        self.assertListEqual(best_state_seq, res)


        # validify with different framework
        res = self.pom.viterbi(obs_seq)
        pom_best_state_seq = []
        for item in res[1]:
            pom_best_state_seq.append(item[1].name)
        pom_best_state_seq.pop(0)
        pom_best_state_seq.pop(len(pom_best_state_seq)-1)
        print(pom_best_state_seq)
