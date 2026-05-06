import unittest
import numpy as np

from cvxreg.common.distance import euclidean_distance, squared_distance


class TestEuclideanDistance(unittest.TestCase):
    def setUp(self):
        self.mat = np.array([[1, 2, 3], [2, 3, 4], [0, 1, 2]])

    def test_distances_from_origin(self):
        np.testing.assert_array_almost_equal(
            np.round(euclidean_distance(self.mat, 0), decimals=4),
            np.array([3.7417, 5.3852, 2.2361]),
        )


class TestSquaredDistance(unittest.TestCase):
    def setUp(self):
        self.mat = np.array([[1, 2, 3], [2, 3, 4], [0, 1, 2]])

    def test_squared_distances_from_origin(self):
        np.testing.assert_array_equal(
            squared_distance(self.mat, 0),
            np.array([14, 29, 5]),
        )
