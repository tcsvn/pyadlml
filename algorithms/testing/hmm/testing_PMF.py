import unittest
import numpy as np
from algorithms.hmm import ProbabilityMassFunction

RN = 'Rainy'
SN = 'Sunny'
HP = 'Happy'
GR = 'Grumpy'
NE = 'Neutral'

class TestPMF(unittest.TestCase):
    def setUp(self):
        # set of observations
        self._observation_alphabet = [HP, GR, NE]
        self._pmf = ProbabilityMassFunction(self._observation_alphabet)


    def tearDown(self):
        pass

    def test_setter(self):
        # test
        new_probs = np.array([0.3, 0.2, 0.5])
        self._pmf.set_probs(new_probs)
        self.assertEqual(self._pmf.prob(HP), 0.3)
        self.assertEqual(self._pmf.prob(GR), 0.2)
        self.assertEqual(self._pmf.prob(NE), 0.5)

        failure = self._pmf.set_probs(np.array([0.4, 0.2, 0.5]))
        self.assertEqual(failure, -1)
        self.assertEqual(self._pmf.prob(HP), 0.3)
        self.assertEqual(self._pmf.prob(GR), 0.2)
        self.assertEqual(self._pmf.prob(NE), 0.5)

        failure = self._pmf.set_probs(np.array([0.8, 0.2]))
        self.assertEqual(failure, -1)
        self.assertEqual(self._pmf.prob(HP), 0.3)
        self.assertEqual(self._pmf.prob(GR), 0.2)
        self.assertEqual(self._pmf.prob(NE), 0.5)
