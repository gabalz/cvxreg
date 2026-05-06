import unittest
import numpy as np

from cvxreg.common.util import set_random_seed
from cvxreg.algorithm.cnls.cnls import CNLSEstimator


def regression_func(X):
    return 1.0 - 2.0 * X[:, 0] + X[:, 1] ** 2


class TestCNLSEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(200, 2)
        cls.y = regression_func(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 2)
        cls.y_test = regression_func(cls.X_test)
        cls.cnls = CNLSEstimator()
        cls.model = cls.cnls.train(cls.X, cls.y)

    def test_weights_shape(self):
        self.assertEqual(self.model.weights.shape, (200, 3))

    def test_in_sample_risk(self):
        yhat = self.cnls.predict(self.model, self.X)
        self.assertAlmostEqual(
            np.round(np.mean(np.square(yhat - self.y)), decimals=4), 0.0057)

    def test_out_of_sample_error(self):
        yhat_test = self.cnls.predict(self.model, self.X_test)
        self.assertAlmostEqual(
            np.round(np.mean(np.square(yhat_test - self.y_test)), decimals=4), 0.0141)
