import unittest
import numpy as np
from algorithms.hmm.distributions import ProbabilityMassFunction

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
        new_probs = np.array([0.3, 0.2, 0.5])
        self._pmf.set_probs(new_probs)
        self.assertEqual(0.3, self._pmf.prob(HP))
        self.assertEqual(0.2, self._pmf.prob(GR))
        self.assertEqual(0.5, self._pmf.prob(NE))

        # try setting wrong stuff
        failure = False
        try:
            self._pmf.set_probs(np.array([0.4, 0.3, 0.5]))
        except:
            failure = True
        self.assertFalse(failure)

        test_array = np.array([0.4, 0.1, 0.5])
        self._pmf.set_probs(test_array)
        self.assertEqual(0.4, self._pmf.prob(HP))
        self.assertEqual(0.1, self._pmf.prob(GR))
        self.assertEqual(0.5, self._pmf.prob(NE))

        self.assertTrue(np.array_equal(test_array, self._pmf.get_probs()))
