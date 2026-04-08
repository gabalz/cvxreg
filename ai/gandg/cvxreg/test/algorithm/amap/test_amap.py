import unittest
import numpy as np

from ai.gandg.cvxreg.common.util import set_random_seed
from ai.gandg.cvxreg.algorithm.amap.amap import AMAPEstimator


def regression_func(X):
    return 1.0 - 2.0 * X[:, 0] + X[:, 1] ** 2


class TestAMAPEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(200, 2)
        cls.y = regression_func(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 2)
        cls.y_test = regression_func(cls.X_test)
        cls.amap = AMAPEstimator(train_args={'ncvfolds': 3})
        cls.model = cls.amap.train(cls.X, cls.y)

    def test_weights_shape(self):
        self.assertEqual(self.model.weights.shape, (5, 3))

    def test_in_sample_risk(self):
        yhat = self.amap.predict(self.model, self.X)
        self.assertAlmostEqual(
            np.round(np.sum(np.square(yhat - self.y)) / len(self.y), decimals=4), 0.0141)

    def test_out_of_sample_error(self):
        yhat_test = self.amap.predict(self.model, self.X_test)
        self.assertAlmostEqual(
            np.round(np.sum(np.square(yhat_test - self.y_test)) / len(self.y_test), decimals=4),
            0.0103)
