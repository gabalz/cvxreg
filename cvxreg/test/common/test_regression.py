import unittest
import numpy as np

from cvxreg.common.distance import squared_distance
from cvxreg.common.partition import Partition
from cvxreg.common.regression import (
    partition_predict, max_affine_predict, max_affine_fit_partition, cv_indices,
)

_X = np.array([[1.1, 1.1], [-1.2, 1.2], [-1.3, -1.3], [0.4, 0.4], [1.5, -1.5]])


class TestPartitionPredict(unittest.TestCase):
    def test_basic(self):
        p = Partition(npoints=5, ncells=2, cells=(np.array([0, 3, 4]), np.array([1, 2])))
        W = np.array([[1.0, 2.0, 3.0], [-1., -2., -3.]])
        np.testing.assert_array_almost_equal(
            partition_predict(p, W, _X),
            np.array([6.5, -2.2, 5.5, 3., -0.5]),
        )


class TestMaxAffinePredict(unittest.TestCase):
    def test_single_hyperplane(self):
        W = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_almost_equal(
            max_affine_predict(W, _X),
            np.array([6.5, 2.2, -5.5, 3., -0.5]),
        )

    def test_two_hyperplanes(self):
        W = np.array([[1.0, 2.0, 3.0], [-1., -2., -3.]])
        np.testing.assert_array_almost_equal(
            max_affine_predict(W, _X),
            np.array([6.5, 2.2, 5.5, 3., 0.5]),
        )


class TestMaxAffineFitPartition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X = np.array([[1., 1.], [-1., 1.], [0., 1.], [-1., -1.], [-2., -1], [0., 1.]])
        cls.y = 0.5 * np.sum(np.square(X), axis=1)
        cls.p = Partition(npoints=6, ncells=2, cells=(np.array([0, 3, 4]), np.array([1, 2, 5])))
        cls.X = X
        cls.weights = max_affine_fit_partition(cls.p, X, cls.y)

    def test_weights_shape(self):
        self.assertEqual(self.weights.shape, (2, 3))

    def test_weights_values(self):
        np.testing.assert_array_almost_equal(
            np.round(self.weights, decimals=4),
            np.array([[1., -1.5, 1.5], [0.25, -0.5, 0.25]]),
        )

    def test_prediction_risk(self):
        yhat = max_affine_predict(self.weights, self.X)
        self.assertAlmostEqual(
            np.round(squared_distance(yhat, self.y, axis=0) / len(self.y), decimals=4), 2.8333
        )


class TestCvIndices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test, cls.train = cv_indices(10, 4)

    def test_test_folds(self):
        np.testing.assert_array_equal(self.test[0], np.array([0, 1]))
        np.testing.assert_array_equal(self.test[1], np.array([2, 3, 4]))
        np.testing.assert_array_equal(self.test[2], np.array([5, 6]))
        np.testing.assert_array_equal(self.test[3], np.array([7, 8, 9]))

    def test_train_folds(self):
        np.testing.assert_array_equal(self.train[0], np.array([2, 3, 4, 5, 6, 7, 8, 9]))
        np.testing.assert_array_equal(self.train[1], np.array([0, 1, 5, 6, 7, 8, 9]))
        np.testing.assert_array_equal(self.train[2], np.array([0, 1, 2, 3, 4, 7, 8, 9]))
        np.testing.assert_array_equal(self.train[3], np.array([0, 1, 2, 3, 4, 5, 6]))
