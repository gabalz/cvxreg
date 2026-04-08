import unittest
import numpy as np

from ai.gandg.cvxreg.common.util import set_random_seed
from ai.gandg.cvxreg.common.partition import rand_voronoi_partition, find_min_dist_centers
from ai.gandg.cvxreg.algorithm.dcf.dcf import _dcf_calc_phi
from ai.gandg.cvxreg.algorithm.dcf.optim_task import (
    SmoothDCFLocalOptimTask, NonSmoothDCFLocalOptimTask, SmoothMaxMinAffineOptimTask,
)


def _regression_func(X):
    return 1.0 - 2.0 * X[:, 0] + X[:, 1] ** 2


class TestSmoothDCFLocalOptimTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        X = np.random.randn(200, 2)
        y = _regression_func(X) + 0.1 * np.random.randn(X.shape[0])
        n, d = X.shape
        K = 5

        cls.weights1 = np.random.randn(K, d+2).ravel()
        P = rand_voronoi_partition(K, X)
        cidx = find_min_dist_centers(X, P)
        centers = X[cidx, :]

        variant = '2'
        cls.sot1 = SmoothDCFLocalOptimTask(
            centers=centers,
            phi=_dcf_calc_phi(X, centers, variant, True),
            y=y,
            variant=variant,
            is_convex=False,
            is_symmetrized=False,
            L_regularizer=0.01,
            L_regularizer_offset=0.1,
            L_sum_regularizer=0.02,
        )

        variant = '+'
        cls.weights2 = np.random.randn(K, 1+2*d).ravel()
        cls.sot2 = SmoothDCFLocalOptimTask(
            centers=centers,
            phi=_dcf_calc_phi(X, centers, variant, True),
            y=y,
            variant=variant,
            is_convex=False,
            is_symmetrized=False,
            L_regularizer=0.01,
            L_regularizer_offset=0.1,
            L_sum_regularizer=0.02,
        )

        variant = '+'
        cls.weights3 = np.random.randn(K, 2*(1+2*d)).ravel()
        cls.sot3 = SmoothDCFLocalOptimTask(
            centers=centers,
            phi=_dcf_calc_phi(X, centers, variant, True),
            y=y,
            variant=variant,
            is_convex=False,
            is_symmetrized=True,
            L_regularizer=0.01,
            L_regularizer_offset=0.1,
            L_sum_regularizer=0.02,
        )

    def test_sot1_fun(self):
        self.assertAlmostEqual(
            np.round(self.sot1.fun(self.weights1), decimals=4), 8.5123)

    def test_sot1_jac(self):
        g1 = self.sot1.jac(self.weights1)
        expected = np.array([
             0.000000e+00,  3.679000e-03,  4.700000e-05, -2.132400e-02,
             1.277010e+00,  3.355553e+00,  8.872500e-01,  3.517649e+00,
             2.554640e-01,  3.180020e-01, -6.921400e-01,  7.323230e-01,
             6.194940e-01, -1.770960e+00, -3.175200e-01,  1.807908e+00,
             1.037826e+00,  1.260986e+00,  1.559265e+00,  2.098880e+00,
        ])
        np.testing.assert_array_almost_equal(np.round(g1, decimals=6), expected)

    def test_sot1_jac_finite_difference(self):
        g1 = self.sot1.jac(self.weights1)
        diff = self.sot1.jac_finite_difference(self.weights1) - g1 + 1e-7
        np.testing.assert_array_almost_equal(np.round(diff, decimals=6),
                                             np.zeros(len(g1)))

    def test_sot2_fun(self):
        self.assertAlmostEqual(
            np.round(self.sot2.fun(self.weights2), decimals=4), 3.0544)

    def test_sot2_jac(self):
        g2 = self.sot2.jac(self.weights2)
        expected = np.array([
            0.487715, 0.71809, 1.213847, -0.228823, -0.202124, 0.,
            0.007226, -0.017663, 0.021096, -0.009145, 0., 0.015854,
            0.003508, 0.009679, -0.023577, -0.212086, 0.04581, -0.062759,
            -0.371522, -0.628068, 0.146561, 0.384821, 0.259903, 0.03094,
            0.003662,
        ])
        np.testing.assert_array_almost_equal(np.round(g2, decimals=6), expected)

    def test_sot2_jac_finite_difference(self):
        g2 = self.sot2.jac(self.weights2)
        diff = self.sot2.jac_finite_difference(self.weights2) - g2 + 1e-7
        np.testing.assert_array_almost_equal(np.round(diff, decimals=6),
                                             np.zeros(len(g2)))

    def test_sot3_fun(self):
        self.assertAlmostEqual(
            np.round(self.sot3.fun(self.weights3), decimals=4), 14.0246)

    def test_sot3_jac(self):
        g3 = self.sot3.jac(self.weights3)
        expected = np.array([
            0.000000e+00, -1.845080e-01, -2.484500e-02, -2.871490e-01,
            9.729000e-03, -3.072710e-01, 3.198000e-03, -1.096700e-02,
            -1.939700e-02, 8.579000e-03, -7.886100e-01, 5.703080e-01,
            -9.369540e-01, 8.656040e-01, -4.966850e-01, -7.552300e-02,
            -8.533000e-02, -4.761700e-02, -7.759300e-02, 1.775448e+00,
            5.268320e-01, 2.460910e+00, 9.123490e-01, -4.812800e-02,
            1.789000e-03, 4.245700e-02, -9.100000e-03, 2.589002e+00,
            3.814840e-01, 3.918733e+00, 0.000000e+00, -3.842800e-02,
            -8.235000e-03, 9.336000e-03, 6.905000e-03, -9.103000e-03,
            -1.909200e-02, -1.127200e-02, 1.699000e-02, -3.015300e-02,
            -2.546503e+00, 0.000000e+00, -3.583830e-01, -2.278300e-02,
            -1.893650e+00, 1.823000e-03, -2.042164e+00, 1.832900e-02,
            -5.232470e-01, -1.181000e-02,
        ])
        np.testing.assert_array_almost_equal(np.round(g3, decimals=6), expected)

    def test_sot3_jac_finite_difference(self):
        g3 = self.sot3.jac(self.weights3)
        diff = self.sot3.jac_finite_difference(self.weights3) - g3 + 1e-7
        np.testing.assert_array_almost_equal(np.round(diff, decimals=6),
                                             np.zeros(len(g3)))


class TestNonSmoothDCFLocalOptimTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        X = np.random.randn(200, 2)
        y = _regression_func(X) + 0.1 * np.random.randn(X.shape[0])
        n, d = X.shape
        K = 5

        cls.weights1 = np.random.randn(K, X.shape[1]).ravel()
        P = rand_voronoi_partition(K, X)
        cidx = find_min_dist_centers(X, P)
        centers = X[cidx, :]

        variant = '2'
        cls.got1 = NonSmoothDCFLocalOptimTask(
            centers=centers,
            phi=_dcf_calc_phi(X, centers, variant, False),
            y=y,
            variant=variant,
            is_convex=False,
            is_symmetrized=False,
            L_regularizer=0.01,
            L_regularizer_offset=0.1,
            L_sum_regularizer=0.02,
        )

        variant = '+'
        cls.weights2 = np.random.randn(K, 1+2*d).ravel()
        cls.got2 = NonSmoothDCFLocalOptimTask(
            centers=centers,
            phi=_dcf_calc_phi(X, centers, variant, True),
            y=y,
            variant=variant,
            is_convex=False,
            is_symmetrized=False,
            L_regularizer=0.01,
            L_regularizer_offset=0.1,
            L_sum_regularizer=0.02,
        )

        variant = '+'
        cls.weights3 = np.random.randn(K, 2*(1+2*d)).ravel()
        cls.got3 = NonSmoothDCFLocalOptimTask(
            centers=centers,
            phi=_dcf_calc_phi(X, centers, variant, True),
            y=y,
            variant=variant,
            is_convex=False,
            is_symmetrized=True,
            L_regularizer=0.01,
            L_regularizer_offset=0.1,
            L_sum_regularizer=0.02,
        )

    def test_got1_fun(self):
        self.assertAlmostEqual(
            np.round(self.got1.fun(self.weights1), decimals=4), 6.7449)

    def test_got1_jac(self):
        g1 = self.got1.jac(self.weights1)
        expected = np.array([
             0.000000e+00,  3.679000e-03,  0.000000e+00, -2.132400e-02,
             9.895940e-01,  3.454384e+00,  0.000000e+00,  1.020900e-02,
             2.072485e+00,  5.364934e+00,
        ])
        np.testing.assert_array_almost_equal(np.round(g1, decimals=6), expected)

    def test_got1_jac_finite_difference(self):
        g1 = self.got1.jac(self.weights1)
        diff = self.got1.jac_finite_difference(self.weights1) - g1 + 1e-7
        np.testing.assert_array_almost_equal(np.round(diff, decimals=6),
                                             np.zeros(len(g1)))

    def test_got2_fun(self):
        self.assertAlmostEqual(
            np.round(self.got2.fun(self.weights2), decimals=4), 3.9703)

    def test_got2_jac(self):
        g2 = self.got2.jac(self.weights2)
        expected = np.array([
            0., 0.009917, -0.003147, 0.009234, -0.003819, 0.,
            -0.024148, -0.01457, -0.020874, 0.009697, -0.422496, -0.459981,
            -0.007819, -0.018364, -0.523877, 0.221276, 0.638287, -0.010905,
            -0.037867, 0.356857, 0.795751, 0.004662, 1.173281, 0.012949,
            0.003784,
        ])
        np.testing.assert_array_almost_equal(np.round(g2, decimals=6), expected)

    def test_got2_jac_finite_difference(self):
        g2 = self.got2.jac(self.weights2)
        diff = self.got2.jac_finite_difference(self.weights2) - g2 + 1e-7
        np.testing.assert_array_almost_equal(np.round(diff, decimals=6),
                                             np.zeros(len(g2)))

    def test_got3_fun(self):
        self.assertAlmostEqual(
            np.round(self.got3.fun(self.weights3), decimals=4), 12.5344)

    def test_got3_jac(self):
        g3 = self.got3.jac(self.weights3)
        expected = np.array([
            -9.502200e-02,  0.000000e+00, -4.851200e-02, -6.228000e-03,
            -3.911100e-02,  1.626300e-02, -2.386400e-02, -7.358000e-03,
            -3.264800e-02, -6.086000e-03, -3.013548e+00,  0.000000e+00,
            -4.858500e-02, -1.797400e-02, -1.862617e+00,  2.501800e-02,
            -2.576146e+00, -2.454600e-02, -1.147347e+00, -9.064000e-03,
            -3.343150e-01,  3.186278e+00, -3.581100e-02,  5.581676e+00,
            -7.425300e-02,  3.245275e+00, -1.277070e-01, -3.823600e-02,
            -1.784560e-01,  1.652640e-01,  0.000000e+00,  0.000000e+00,
            8.320000e-04,  3.041100e-02, -5.294000e-03,  4.148000e-03,
            -6.763000e-03, -6.169000e-03, -3.163800e-02, -8.935000e-03,
            -8.379650e-01,  1.094570e+00,  2.408200e-02, -7.460000e-03,
            -1.632206e+00,  2.475710e-01, -8.162810e-01,  2.848757e+00,
            -9.956800e-02,  7.361740e-01,
        ])
        np.testing.assert_array_almost_equal(np.round(g3, decimals=6), expected)

    def test_got3_jac_finite_difference(self):
        g3 = self.got3.jac(self.weights3)
        diff = self.got3.jac_finite_difference(self.weights3) - g3 + 1e-7
        np.testing.assert_array_almost_equal(np.round(diff, decimals=6),
                                             np.zeros(len(g3)))


class TestSmoothMaxMinAffineOptimTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        X = np.random.randn(200, 2)
        y = _regression_func(X) + 0.1 * np.random.randn(X.shape[0])
        n, d = X.shape
        K = 5
        d = X.shape[1]

        W = np.random.randn(K, d*2, 1+d)
        yhat = (W[:, :, 1:].dot(X.T) + W[:, :, 0][:, :, None]).min(axis=1).max(axis=0)
        resid = y - yhat
        cls.resid_half_mse = np.round(0.5 * resid.dot(resid) / len(resid), decimals=4)
        cls.weights1 = W.ravel()
        cls.mot1 = SmoothMaxMinAffineOptimTask(
            is_symmetrized=False, X=X, y=y, K=K,
            L_regularizer=0.01,
            L_regularizer_offset=0.1,
            L_sum_regularizer=0.02,
        )
        cls.mot1_adj = SmoothMaxMinAffineOptimTask(
            is_symmetrized=False, X=X, y=y, K=K,
            L_regularizer=0.01,
            L_regularizer_offset=0.1,
            L_sum_regularizer=0.02,
            adj_f=True,
        )

        W2 = np.random.randn(K, d*2, 1+d)
        cls.weights2 = np.concatenate([W.ravel(), W2.ravel()])
        cls.mot2 = SmoothMaxMinAffineOptimTask(
            is_symmetrized=True, X=X, y=y, K=K,
            L_regularizer=0.01,
            L_regularizer_offset=0.1,
            L_sum_regularizer=0.02,
        )

    def test_resid_mse(self):
        self.assertAlmostEqual(self.resid_half_mse, 6.4306)

    def test_mot1_fun(self):
        self.assertAlmostEqual(
            np.round(self.mot1.fun(self.weights1), decimals=4), 6.9105)

    def test_mot1_adj_fun(self):
        self.assertAlmostEqual(
            np.round(self.mot1_adj.fun(self.weights1), decimals=4), 6.9105)

    def test_mot1_jac(self):
        g1 = self.mot1.jac(self.weights1)
        expected = np.array([
            -2.030e-01,  7.400e-02, -4.097e-01, -2.295e-01, -1.681e-01,
            3.438e-01,  0.000e+00,  1.020e-02,  4.900e-03,  1.073e-01,
            2.638e-01, -1.170e-02, -2.024e-01, -8.760e-02,  4.072e-01,
            -1.637e-01,  8.810e-02,  4.333e-01,  0.000e+00,  3.090e-02,
            -3.200e-02,  0.000e+00,  1.100e-02, -2.380e-02,  0.000e+00,
            5.900e-03, -6.000e-04, -7.500e-03,  7.300e-03, -2.700e-02,
            -2.422e-01,  2.590e-02,  2.457e-01, -1.801e-01,  8.090e-02,
            8.410e-02, -1.332e-01,  1.018e-01,  2.004e-01, -1.313e-01,
            1.850e-01, -1.437e-01, -3.250e-02, -3.420e-02,  4.830e-02,
            0.000e+00,  7.000e-04,  3.290e-02,  0.000e+00, -2.360e-02,
            3.190e-02,  0.000e+00, -3.000e-02,  8.300e-03, -4.602e-01,
            6.483e-01, -1.368e-01, -3.865e-01,  4.504e-01,  4.863e-01,
        ])
        np.testing.assert_array_almost_equal(np.round(g1, decimals=4), expected)

    def test_mot1_jac_finite_difference(self):
        g1 = self.mot1.jac(self.weights1)
        diff = self.mot1.jac_finite_difference(self.weights1) - g1 + 1e-7
        np.testing.assert_array_almost_equal(np.round(diff, decimals=6),
                                             np.zeros(len(g1)))

    def test_mot2_fun(self):
        self.assertAlmostEqual(
            np.round(self.mot2.fun(self.weights2), decimals=4), 5.5646)

    def test_mot2_jac(self):
        g2 = self.mot2.jac(self.weights2)
        expected = np.array([
            -7.160e-02,  1.289e-01, -2.229e-01, -1.180e-01, -8.840e-02,
            2.582e-01,  0.000e+00,  1.020e-02,  4.900e-03,  8.450e-02,
            2.019e-01, -3.900e-03, -1.522e-01, -7.540e-02,  3.138e-01,
            -1.433e-01,  8.600e-02,  3.831e-01,  0.000e+00,  3.090e-02,
            -3.200e-02,  0.000e+00,  1.100e-02, -2.380e-02,  0.000e+00,
            5.900e-03, -6.000e-04,  9.900e-03,  1.230e-02, -2.190e-02,
            -1.974e-01,  1.750e-02,  2.065e-01, -1.351e-01,  7.140e-02,
            7.340e-02, -1.223e-01,  9.270e-02,  1.866e-01, -1.147e-01,
            1.606e-01, -1.260e-01, -2.050e-02, -3.250e-02,  3.600e-02,
            0.000e+00,  7.000e-04,  3.290e-02,  0.000e+00, -2.360e-02,
            3.190e-02,  0.000e+00, -3.000e-02,  8.300e-03, -3.805e-01,
            6.001e-01, -8.490e-02, -3.460e-01,  4.379e-01,  4.871e-01,
            4.400e-02, -8.730e-02,  4.510e-02,  0.000e+00, -2.220e-02,
            -1.350e-02,  2.420e-02, -6.170e-02,  3.600e-03,  0.000e+00,
            1.060e-02,  2.850e-02, -1.800e-02, -2.850e-02, -2.800e-02,
            2.940e-02,  3.040e-02, -9.080e-02,  0.000e+00, -7.100e-03,
            -3.120e-02,  0.000e+00,  2.160e-02, -1.340e-02,  3.045e-01,
            -3.308e-01, -1.298e-01,  2.669e-01, -4.619e-01,  6.370e-02,
            3.798e-01, -8.440e-02, -4.873e-01,  4.112e-01, -3.936e-01,
            -8.456e-01,  0.000e+00,  1.000e-03,  7.600e-03,  0.000e+00,
            -1.200e-03,  2.690e-02,  1.340e-02, -3.700e-03,  5.700e-03,
            2.031e-01, -8.560e-02,  3.534e-01, -9.430e-02, -1.264e-01,
            -1.098e-01,  0.000e+00,  1.370e-02,  3.150e-02,  1.959e-01,
            8.590e-02, -1.451e-01, -5.300e-02, -1.281e-01, -2.520e-02,
        ])
        np.testing.assert_array_almost_equal(np.round(g2, decimals=4), expected)

    def test_mot2_jac_finite_difference(self):
        g2 = self.mot2.jac(self.weights2)
        diff = self.mot2.jac_finite_difference(self.weights2) - g2 + 1e-7
        np.testing.assert_array_almost_equal(np.round(diff, decimals=6),
                                             np.zeros(len(g2)))
