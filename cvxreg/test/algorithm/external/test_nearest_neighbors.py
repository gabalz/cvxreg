import io
import unittest
import unittest.mock
import numpy as np

from cvxreg.common.util import set_random_seed
from cvxreg.algorithm.external.nearest_neighbors import NearestNeighborsEstimator


def regression_func(X):
    return 1.0 - 2.0 * X[:, 0] + X[:, 1] ** 2


class TestNearestNeighborsEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(200, 2)
        cls.y = regression_func(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 2)
        cls.y_test = regression_func(cls.X_test)

    def test_knn1_in_sample_risk(self):
        knn1 = NearestNeighborsEstimator(n_neighbors=1)
        model1 = knn1.train(self.X, self.y)
        yhat1 = knn1.predict(model1, self.X)
        self.assertAlmostEqual(np.round(np.mean(np.square(yhat1 - self.y)), decimals=4), 0.0)

    def test_knn1_out_of_sample_error(self):
        knn1 = NearestNeighborsEstimator(n_neighbors=1)
        model1 = knn1.train(self.X, self.y)
        yhat1_test = knn1.predict(model1, self.X_test)
        self.assertAlmostEqual(
            np.round(np.mean(np.square(yhat1_test - self.y_test)), decimals=4), 0.2574)

    def test_knn_formula_verbose(self):
        buf = io.StringIO()
        knn2 = NearestNeighborsEstimator(n_neighbors='n**(d/(2+d))', verbose=True)
        with unittest.mock.patch('sys.stdout', buf):
            model2 = knn2.train(self.X, self.y)
        self.assertIn('kNN, k: n**(d/(2+d)) -> 15', buf.getvalue())
        yhat2 = knn2.predict(model2, self.X)
        self.assertAlmostEqual(np.round(np.mean(np.square(yhat2 - self.y)), decimals=4), 0.8824)
        yhat2_test = knn2.predict(model2, self.X_test)
        self.assertAlmostEqual(
            np.round(np.mean(np.square(yhat2_test - self.y_test)), decimals=4), 0.6588)

    def test_knn_afpc_verbose(self):
        buf = io.StringIO()
        knn3 = NearestNeighborsEstimator(n_neighbors='AFPC', verbose=True)
        with unittest.mock.patch('sys.stdout', buf):
            model3 = knn3.train(self.X, self.y)
        self.assertIn('kNN, k: AFPC -> 15', buf.getvalue())
        yhat3 = knn3.predict(model3, self.X)
        self.assertAlmostEqual(np.round(np.mean(np.square(yhat3 - self.y)), decimals=4), 0.8824)
        yhat3_test = knn3.predict(model3, self.X_test)
        self.assertAlmostEqual(
            np.round(np.mean(np.square(yhat3_test - self.y_test)), decimals=4), 0.6588)

    def test_knn_afpc_cv_verbose(self):
        buf = io.StringIO()
        knn4 = NearestNeighborsEstimator(n_neighbors='AFPC', cv=3, afpc_ntrials=10, verbose=True)
        with unittest.mock.patch('sys.stdout', buf):
            model4 = knn4.train(self.X, self.y)
        output = buf.getvalue()
        self.assertIn('kNN, k: AFPC -> 16', output)
        self.assertIn('kCV: 7', output)
        yhat4 = knn4.predict(model4, self.X)
        self.assertAlmostEqual(np.round(np.mean(np.square(yhat4 - self.y)), decimals=4), 0.3546)
        yhat4_test = knn4.predict(model4, self.X_test)
        self.assertAlmostEqual(
            np.round(np.mean(np.square(yhat4_test - self.y_test)), decimals=4), 0.3222)
