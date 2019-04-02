import unittest
import numpy as np
from algorithms.hmm.Probs import Probs

class TestPMF(unittest.TestCase):
    def setUp(self):
        self.smallest_num = np.nextafter(0,1)
        self.x0 = 0.0
        self.x1 = 0.1
        self.x2 = 0.2
        self.x3 = 1.0

        self.p0 = Probs(self.x0)
        self.p1 = Probs(self.x1)
        self.p2 = Probs(self.x2)
        self.p3 = Probs(self.x3)

    def tearDown(self):
        pass

    def test_initialisation(self):
        x4 = 0.000
        # invalid
        x5 = 2.0
        x6 = -1.0

        p4 = Probs(x4)
        self.assertEqual(self.smallest_num, p4.prob_to_norm())

        p5_exc = False
        try:
            p5 = Probs(x5)
        except ValueError:
            p5_exc = True
        self.assertTrue(p5_exc)

        p6_exc = False
        try:
            p6 = Probs(x6)
        except ValueError:
            p6_exc = True
        self.assertTrue(p6_exc)


    def test_prob_to_norm(self):
        self.assertEqual(self.smallest_num, self.p0.prob_to_norm())
        self.assertEqual(self.x1, self.p1.prob_to_norm())
        self.assertEqual(self.x2, self.p2.prob_to_norm())
        self.assertEqual(self.x3, self.p3.prob_to_norm())


    """
    implement arithmetic operations
    
    """

    def test_add(self):
        p_add_0 = self.p1 + self.p2 # type: Probs
        self.assertAlmostEqual(0.3, p_add_0.prob_to_norm(), 10)

        p_add_1 = Probs(0.4) + Probs(0.15)
        self.assertEqual(0.55, p_add_1.prob_to_norm())

        p_add_2 = Probs(0.6) + Probs(0.3)
        self.assertEqual(0.9, p_add_2.prob_to_norm())

        # test cases not possible
        # todo should throw exception
        p_add_3 = Probs(0.6) + Probs(0.5)
        self.assertEqual(1.1, p_add_3.prob_to_norm())

    def test_sub(self):
        p_sub_0 = self.p3 - self.p1 # type: Probs
        self.assertAlmostEqual(0.9, p_sub_0.prob_to_norm(), 10)

        p_sub_1 = Probs(0.6) - Probs(0.3)
        self.assertAlmostEqual(0.3, p_sub_1.prob_to_norm())

    def test_mul(self):
        p_mul_0 = self.p1 * self.p2
        self.assertEqual(0.02, round(p_mul_0.prob_to_norm(), 10))

    def test_div(self):
        p_div_0 = self.p1 / self.p2
        self.assertEqual(0.5, round(p_div_0.prob_to_norm(), 10))

    def test_pow(self):
        p_pow_0 = self.p1 ** 3
        self.assertEqual(0.001, round(p_pow_0.prob_to_norm(), 10))

    """
    test inplace arithmetic operations
    
    """


    def test_iadd(self):
        self.p1 += self.p2 # type: Probs
        self.assertAlmostEqual(0.3, self.p1.prob_to_norm(), 10)

        p_add_1 = Probs(0.4)
        p_add_1 += Probs(0.15)
        self.assertEqual(0.55, p_add_1.prob_to_norm())

        p_add_2 = Probs(0.6)
        p_add_2 += Probs(0.3)
        self.assertEqual(0.9, p_add_2.prob_to_norm())

        # test cases not possible
        # todo should throw exception
        p_add_3 = Probs(0.6)
        p_add_3 += Probs(0.5)
        self.assertEqual(1.1, p_add_3.prob_to_norm())

    def test_isub(self):
        p_sub_0 = self.p3
        p_sub_0 -= self.p1 # type: Probs
        self.assertAlmostEqual(0.9, p_sub_0.prob_to_norm(), 10)

        p_sub_1 = Probs(0.6)
        p_sub_1 -= Probs(0.3)
        self.assertAlmostEqual(0.3, p_sub_1.prob_to_norm())

    def test_imul(self):
        p_mul_0 = self.p1
        p_mul_0 *= self.p2
        self.assertEqual(0.02, round(p_mul_0.prob_to_norm(), 10))

    def test_idiv(self):
        p_div_0 = self.p1
        p_div_0 /= self.p2
        self.assertEqual(0.5, round(p_div_0.prob_to_norm(), 10))

    def test_ipow(self):
        p_pow_0 = self.p1
        p_pow_0 **= 3
        self.assertEqual(0.001, round(p_pow_0.prob_to_norm(), 10))

    """
    implement unary artihmetic operations
    """

    def test_neq(self):
        p3 = not self.p1

    def test_abs(self):
        p3 = abs(self.p1)
        print(p3)

    """
    implement built-in functions
    """


    """
    comparision operators below
    """

    def test_lt(self):
        lt_bool_0 = self.p1 < self.p2
        self.assertTrue(lt_bool_0)

    def test_le(self):
        le_bool_0 = self.p1 <= self.p2
        self.assertTrue(le_bool_0)

        le_bool_1 = Probs(0.1) <= Probs(0.1)
        self.assertTrue(le_bool_1)

    def test_eq(self):
        eq_bool_0 = self.p1 == self.p2
        self.assertFalse(eq_bool_0)

        eq_bool_1 = Probs(0.75) == Probs(0.76)
        self.assertFalse(eq_bool_1)

        eq_bool_2 = self.p1 == self.p1
        self.assertTrue(eq_bool_2)

    def test_ne(self):
        ne_bool_0 = self.p1 != self.p2
        self.assertTrue(ne_bool_0)

        ne_bool_1 = Probs(0.1) != Probs(0.1)
        self.assertFalse(ne_bool_1)

    def test_gt(self):
        gt_bool_0 = self.p1 > self.p2
        self.assertFalse(gt_bool_0)

        gt_bool_1 = self.p2 > self.p1
        self.assertTrue(gt_bool_1)


    def test_ge(self):
        ge_bool_0 = self.p1 >= self.p2
        self.assertFalse(ge_bool_0)

        ge_bool_1 = self.p2 >= self.p1
        self.assertTrue(ge_bool_1)

        ge_bool_2 = Probs(0.1) >= Probs(0.1)
        self.assertTrue(ge_bool_2)
