import unittest
import numpy as np

from ai.gandg.cvxreg.common.util import set_random_seed
from ai.gandg.cvxreg.algorithm.cap.cap import CAPEstimator


def regression_func(X):
    return 1.0 - 2.0 * X[:, 0] + X[:, 1] ** 2


class TestCAPEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(200, 2)
        cls.y = regression_func(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 2)
        cls.y_test = regression_func(cls.X_test)
        cls.cap = CAPEstimator()
        cls.model = cls.cap.train(cls.X, cls.y)

    def test_weights_shape(self):
        self.assertEqual(self.model.weights.shape, (7, 3))

    def test_in_sample_risk(self):
        yhat = self.cap.predict(self.model, self.X)
        self.assertAlmostEqual(
            np.round(np.sum(np.square(yhat - self.y)) / len(self.y), decimals=4), 0.0102)

    def test_out_of_sample_error(self):
        yhat_test = self.cap.predict(self.model, self.X_test)
        self.assertAlmostEqual(
            np.round(np.sum(np.square(yhat_test - self.y_test)) / len(self.y_test), decimals=4),
            0.0046)


class TestFastCAPEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(200, 2)
        cls.y = regression_func(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 2)
        cls.y_test = regression_func(cls.X_test)
        cls.fastcap = CAPEstimator(train_args={'nranddirs': 3})
        cls.model = cls.fastcap.train(cls.X, cls.y)

    def test_weights_shape(self):
        self.assertEqual(self.model.weights.shape, (7, 3))

    def test_in_sample_risk(self):
        yhat = self.fastcap.predict(self.model, self.X)
        self.assertAlmostEqual(
            np.round(np.sum(np.square(yhat - self.y)) / len(self.y), decimals=4), 0.0107)

    def test_out_of_sample_error(self):
        yhat_test = self.fastcap.predict(self.model, self.X_test)
        self.assertAlmostEqual(
            np.round(np.sum(np.square(yhat_test - self.y_test)) / len(self.y_test), decimals=4),
            0.0063)
