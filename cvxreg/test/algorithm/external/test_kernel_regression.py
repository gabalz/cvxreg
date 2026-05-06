import unittest
import numpy as np

from cvxreg.common.util import set_random_seed
from cvxreg.algorithm.external.kernel_regression import KernelRegEstimator


def regression_func(X):
    return 1.0 - 2.0 * X[:, 0] + X[:, 1] ** 2


class TestKernelRegEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(200, 2)
        cls.y = regression_func(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 2)
        cls.y_test = regression_func(cls.X_test)
        cls.kreg1 = KernelRegEstimator('normal')
        cls.model1 = cls.kreg1.train(cls.X, cls.y)

    def test_ols_out_of_sample_error(self):
        ols_model = np.linalg.lstsq(self.X.T.dot(self.X), self.X.T.dot(self.y), rcond=-1)[0]
        ols_yhat_test = np.sum(self.X_test * ols_model, axis=1)
        self.assertAlmostEqual(
            np.round(np.sum(np.square(ols_yhat_test - self.y_test)) / len(self.y_test), decimals=4),
            6.2752)

    def test_in_sample_risk(self):
        yhat1 = self.kreg1.predict(self.model1, self.X)
        self.assertAlmostEqual(
            np.round(np.mean(np.square(yhat1 - self.y)), decimals=4), 0.033)

    def test_out_of_sample_error(self):
        yhat1_test = self.kreg1.predict(self.model1, self.X_test)
        self.assertAlmostEqual(
            np.round(np.mean(np.square(yhat1_test - self.y_test)), decimals=4), 0.1667)
