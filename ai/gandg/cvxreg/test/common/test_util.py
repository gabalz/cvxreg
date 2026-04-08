import random
import unittest
import numpy as np

from ai.gandg.cvxreg.common.util import set_random_seed, rand_direction


class TestSetRandomSeed(unittest.TestCase):
    def test_python_random(self):
        set_random_seed(19)
        self.assertEqual(random.randint(0, 10000), 708)

    def test_numpy_random(self):
        set_random_seed(19)
        np.testing.assert_array_almost_equal(
            np.random.rand(3, 2),
            np.array([[0.6356515, 0.15946741],
                      [0.42432349, 0.93350408],
                      [0.20335322, 0.5258474]]),
        )


class TestRandDirection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.x = rand_direction(10, 3)

    def test_values(self):
        np.testing.assert_array_almost_equal(
            self.x,
            np.array([[-0.78538771,  0.31285999,  0.53412056],
                      [-0.08505102,  0.08648851, -0.99261577],
                      [-0.03835023,  0.14032477, -0.98936253],
                      [0.44378856,  0.20424592, -0.87254531],
                      [-0.13436984, -0.8220843,  -0.55328307],
                      [0.41833227,  0.30613646,  0.85514828],
                      [-0.77293566, -0.57197324,  0.2746217],
                      [-0.7169452,   0.44005241,  0.54068794],
                      [0.99017269,  0.1185976,   0.07411245],
                      [-0.33350422,  0.42052367, -0.84376227]]),
        )

    def test_unit_norms(self):
        np.testing.assert_array_almost_equal(
            np.linalg.norm(self.x, axis=1),
            np.ones(10),
        )
