import unittest
import numpy as np

from ai.gandg.cvxreg.common.util import set_random_seed
from ai.gandg.cvxreg.common.partition import singleton_partition
from ai.gandg.cvxreg.algorithm.apcnls.apcnls import APCNLSEstimator, apcnls_qp_data


_X = np.array([[1.1, 1.1], [-1.2, 1.2], [-1.3, -1.3], [0.4, 0.4], [1.5, -1.5]])
_y = np.array([1.1, 1.2, 1.3, 0.4, 0.5])
_p = singleton_partition(len(_y))

_A_ROWS20_COLS9 = np.array([
    [-1., -1.1, -1.1,  1.,  1.1,  1.1,  0.,  0.,  0.],
    [-1., -1.1, -1.1,  0.,  0.,  0.,  1.,  1.1,  1.1],
    [-1., -1.1, -1.1,  0.,  0.,  0.,  0.,  0.,  0.],
    [-1., -1.1, -1.1,  0.,  0.,  0.,  0.,  0.,  0.],
    [1., -1.2,  1.2, -1.,  1.2, -1.2,  0.,  0.,  0.],
    [0.,  0.,  0., -1.,  1.2, -1.2,  1., -1.2,  1.2],
    [0.,  0.,  0., -1.,  1.2, -1.2,  0.,  0.,  0.],
    [0.,  0.,  0., -1.,  1.2, -1.2,  0.,  0.,  0.],
    [1., -1.3, -1.3,  0.,  0.,  0., -1.,  1.3,  1.3],
    [0.,  0.,  0.,  1., -1.3, -1.3, -1.,  1.3,  1.3],
    [0.,  0.,  0.,  0.,  0.,  0., -1.,  1.3,  1.3],
    [0.,  0.,  0.,  0.,  0.,  0., -1.,  1.3,  1.3],
    [1.,  0.4,  0.4,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  1.,  0.4,  0.4,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.4,  0.4],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [1.,  1.5, -1.5,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  1.,  1.5, -1.5,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.5, -1.5],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
])

_A_ROWS33_COLS9 = np.array([
    [-1., -1.1, -1.1,  1.,  1.1,  1.1,  0.,  0.,  0.],
    [-1., -1.1, -1.1,  0.,  0.,  0.,  1.,  1.1,  1.1],
    [-1., -1.1, -1.1,  0.,  0.,  0.,  0.,  0.,  0.],
    [-1., -1.1, -1.1,  0.,  0.,  0.,  0.,  0.,  0.],
    [1., -1.2,  1.2, -1.,  1.2, -1.2,  0.,  0.,  0.],
    [0.,  0.,  0., -1.,  1.2, -1.2,  1., -1.2,  1.2],
    [0.,  0.,  0., -1.,  1.2, -1.2,  0.,  0.,  0.],
    [0.,  0.,  0., -1.,  1.2, -1.2,  0.,  0.,  0.],
    [1., -1.3, -1.3,  0.,  0.,  0., -1.,  1.3,  1.3],
    [0.,  0.,  0.,  1., -1.3, -1.3, -1.,  1.3,  1.3],
    [0.,  0.,  0.,  0.,  0.,  0., -1.,  1.3,  1.3],
    [0.,  0.,  0.,  0.,  0.,  0., -1.,  1.3,  1.3],
    [1.,  0.4,  0.4,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  1.,  0.4,  0.4,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.4,  0.4],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [1.,  1.5, -1.5,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  1.,  1.5, -1.5,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.5, -1.5],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
])


class TestApcnlsQpDataLSumRegularizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.H, cls.g, cls.A, cls.b, cls.cell_idx = apcnls_qp_data(
            _X, _y, _p, L_sum_regularizer=0.1)

    def test_cell_idx(self):
        np.testing.assert_array_equal(self.cell_idx, [0, 4, 8, 12, 16])

    def test_H_shape(self):
        self.assertEqual(self.H.shape, (16, 16))

    def test_H_rank(self):
        self.assertEqual(np.linalg.matrix_rank(self.H.toarray()), 16)

    def test_H_nnz(self):
        self.assertEqual(self.H.nnz, 31)

    def test_H_first9cols(self):
        expected = np.array([
            [0.2,  0.22,  0.22,  0.,  0.,  0.,  0.,  0.,  0.],
            [0.,  0.26,  0.24,  0.,  0.,  0.,  0.,  0.,  0.],
            [0.,  0.,  0.26,  0.,  0.,  0.,  0.,  0.,  0.],
            [0.,  0.,  0.,  0.2, -0.24,  0.24,  0.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.31, -0.29,  0.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.31,  0.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.2, -0.26, -0.26],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.36,  0.34],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.36],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        ])
        np.testing.assert_array_almost_equal(
            np.round(self.H.toarray()[:, :9], decimals=2), expected)

    def test_H_last3x3(self):
        expected = np.array([
            [0.47, -0.45,  0.],
            [0.,  0.47,  0.],
            [0.,  0.,  1.],
        ])
        np.testing.assert_array_almost_equal(self.H.toarray()[-3:, -3:], expected)

    def test_g(self):
        expected = np.array([
            -0.22, -0.242, -0.242, -0.24,  0.288, -0.288, -0.26,  0.338,
            0.338, -0.08, -0.032, -0.032, -0.1, -0.15,  0.15,  0.,
        ])
        np.testing.assert_array_almost_equal(self.g, expected)

    def test_A_shape(self):
        self.assertEqual(self.A.shape, (20, 16))

    def test_A_rank(self):
        self.assertEqual(np.linalg.matrix_rank(self.A.toarray()), 13)

    def test_A_first9cols(self):
        np.testing.assert_array_almost_equal(self.A.toarray()[:, :9], _A_ROWS20_COLS9)

    def test_A_nnz(self):
        self.assertEqual(self.A.nnz, 140)

    def test_b_shape(self):
        self.assertEqual(self.b.shape, (20,))

    def test_b_zero(self):
        self.assertAlmostEqual(np.sum(np.abs(self.b)), 0.0)


class TestApcnlsQpDataL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.H, cls.g, cls.A, cls.b, cls.cell_idx = apcnls_qp_data(_X, _y, _p, L=5.0)

    def test_A_shape(self):
        self.assertEqual(self.A.shape, (40, 16))

    def test_A_first33rows_first9cols(self):
        np.testing.assert_array_almost_equal(self.A.toarray()[:33, :9], _A_ROWS33_COLS9)

    def test_A_first10rows_last5cols(self):
        expected = np.array([
            [0.,  0.,  0.,  0., -1.],
            [0.,  0.,  0.,  0., -1.],
            [1.1,  0.,  0.,  0., -1.],
            [0.,  1.,  1.1,  1.1, -1.],
            [0.,  0.,  0.,  0., -1.],
            [0.,  0.,  0.,  0., -1.],
            [1.2,  0.,  0.,  0., -1.],
            [0.,  1., -1.2,  1.2, -1.],
            [0.,  0.,  0.,  0., -1.],
            [0.,  0.,  0.,  0., -1.],
        ])
        np.testing.assert_array_almost_equal(self.A.toarray()[:10, -5:], expected)

    def test_A_last5rows(self):
        expected = np.array([
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,
             0.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             1.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             -1.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  1.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0., -1.,  0.],
        ])
        np.testing.assert_array_almost_equal(self.A.toarray()[-5:, :], expected)

    def test_b(self):
        expected = np.array([
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
            5., 5., 5., 5., 5., 5.,
        ])
        np.testing.assert_array_almost_equal(self.b, expected)


class TestApcnlsQpDataLWithV0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.H, cls.g, cls.A, cls.b, cls.cell_idx = apcnls_qp_data(
            _X, _y, _p, L=5.0, V0=0.1)

    def test_A_shape(self):
        self.assertEqual(self.A.shape, (41, 16))

    def test_A_first33rows_first9cols(self):
        np.testing.assert_array_almost_equal(self.A.toarray()[:33, :9], _A_ROWS33_COLS9)

    def test_A_first10rows_last5cols(self):
        expected = np.array([
            [0.,  0.,  0.,  0., -1.],
            [0.,  0.,  0.,  0., -1.],
            [1.1,  0.,  0.,  0., -1.],
            [0.,  1.,  1.1,  1.1, -1.],
            [0.,  0.,  0.,  0., -1.],
            [0.,  0.,  0.,  0., -1.],
            [1.2,  0.,  0.,  0., -1.],
            [0.,  1., -1.2,  1.2, -1.],
            [0.,  0.,  0.,  0., -1.],
            [0.,  0.,  0.,  0., -1.],
        ])
        np.testing.assert_array_almost_equal(self.A.toarray()[:10, -5:], expected)

    def test_A_last5rows(self):
        expected = np.array([
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             1.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             -1.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  1.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0., -1.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  1.],
        ])
        np.testing.assert_array_almost_equal(self.A.toarray()[-5:, :], expected)

    def test_b(self):
        expected = np.array([
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 5., 5., 5., 5., 5., 5.,
            5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
            5., 0.1,
        ])
        np.testing.assert_array_almost_equal(self.b, expected)


class TestApcnlsQpDataLRegularizerWithV0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.H, cls.g, cls.A, cls.b, cls.cell_idx = apcnls_qp_data(
            _X, _y, _p, L_regularizer=7.0, V0=0.1)

    def test_A_shape(self):
        self.assertEqual(self.A.shape, (41, 17))

    def test_A_first33rows_first9cols(self):
        np.testing.assert_array_almost_equal(self.A.toarray()[:33, :9], _A_ROWS33_COLS9)

    def test_A_first10rows_last5cols(self):
        expected = np.array([
            [0.,  0.,  0., -1.,  0.],
            [0.,  0.,  0., -1.,  0.],
            [0.,  0.,  0., -1.,  0.],
            [1.,  1.1,  1.1, -1.,  0.],
            [0.,  0.,  0., -1.,  0.],
            [0.,  0.,  0., -1.,  0.],
            [0.,  0.,  0., -1.,  0.],
            [1., -1.2,  1.2, -1.,  0.],
            [0.,  0.,  0., -1.,  0.],
            [0.,  0.,  0., -1.,  0.],
        ])
        np.testing.assert_array_almost_equal(self.A.toarray()[:10, -5:], expected)

    def test_A_last5rows(self):
        expected = np.array([
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             1.,  0.,  0., -1.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             -1.,  0.,  0., -1.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  1.,  0., -1.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0., -1.,  0., -1.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  1.,  0.],
        ])
        np.testing.assert_array_almost_equal(self.A.toarray()[-5:, :], expected)


def _regression_func_2d(X):
    return 1.0 - 2.0 * X[:, 0] + X[:, 1] ** 2


def _regression_func_5d(X):
    return 0.5 * np.sum(np.square(X), axis=1)


class TestAPCNLSEstimatorSeed19(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(200, 2)
        cls.y = _regression_func_2d(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 2)
        cls.y_test = _regression_func_2d(cls.X_test)

        est = APCNLSEstimator()
        cls.model1 = est.train(cls.X, cls.y)
        cls.yhat1 = est.predict(cls.model1, cls.X)
        cls.yhat1_test = est.predict(cls.model1, cls.X_test)

        est2 = APCNLSEstimator()
        cls.model2 = est2.train(
            cls.X, cls.y, afpc_q=2, use_L=True, L=4.5, use_V0=True,
            L_regularizer=None, v_regularizer=1.0/cls.X.shape[0])
        cls.yhat2 = est2.predict(cls.model2, cls.X)
        cls.yhat2_test = est2.predict(cls.model2, cls.X_test)

        est3 = APCNLSEstimator()
        cls.model3 = est3.train(
            cls.X, cls.y, use_L=True, L=4.5, use_V0=False,
            L_regularizer=None, v_regularizer=1.0)
        cls.yhat3 = est3.predict(cls.model3, cls.X)
        cls.yhat3_test = est3.predict(cls.model3, cls.X_test)

    def test_ols_out_of_sample_error(self):
        ols_model = np.linalg.lstsq(
            self.X.T.dot(self.X), self.X.T.dot(self.y), rcond=-1)[0]
        ols_yhat_test = np.sum(self.X_test * ols_model, axis=1)
        self.assertAlmostEqual(
            np.round(np.sum(np.square(ols_yhat_test - self.y_test)) / len(self.y_test),
                     decimals=4), 6.2752)

    # model1: no Lipschitz constant
    def test_model1_weights_shape(self):
        self.assertEqual(self.model1.weights.shape, (15, 3))

    def test_model1_V(self):
        self.assertAlmostEqual(np.round(self.model1.V, decimals=4), 0.0688)

    def test_model1_V0_is_none(self):
        self.assertIsNone(self.model1.V0)

    def test_model1_obj_val(self):
        self.assertAlmostEqual(np.round(self.model1.obj_val, decimals=4), -4.695)

    def test_model1_proj_obj_val(self):
        self.assertAlmostEqual(np.round(self.model1.proj_obj_val, decimals=4), -4.7208)

    def test_model1_train_diff(self):
        self.assertAlmostEqual(np.round(self.model1.train_diff, decimals=4), 0.0121)

    def test_model1_cell_diff_max(self):
        self.assertAlmostEqual(np.round(self.model1.cell_diff_max, decimals=4), 2.4772)

    def test_model1_partition_radius(self):
        self.assertAlmostEqual(np.round(self.model1.partition_radius, decimals=4), 0.8883)

    def test_model1_proj_diff_corr(self):
        self.assertAlmostEqual(np.round(self.model1.proj_diff_corr, decimals=4), -0.001)

    def test_model1_in_sample_risk(self):
        self.assertAlmostEqual(
            np.round(np.sum(np.square(self.yhat1 - self.y)) / len(self.y), decimals=4),
            0.1968)

    def test_model1_out_of_sample_error(self):
        self.assertAlmostEqual(
            np.round(np.sum(np.square(self.yhat1_test - self.y_test)) / len(self.y_test),
                     decimals=4), 0.1805)

    # model2: afpc_q=2, L=4.5, use_V0=True
    def test_model2_weights_shape(self):
        self.assertEqual(self.model2.weights.shape, (7, 3))

    def test_model2_V(self):
        self.assertAlmostEqual(np.round(self.model2.V, decimals=4), 1.1916)

    def test_model2_V0(self):
        self.assertAlmostEqual(np.round(self.model2.V0, decimals=4), 13.5358)

    def test_model2_partition_radius(self):
        self.assertAlmostEqual(np.round(self.model2.partition_radius, decimals=4), 1.504)

    def test_model2_proj_diff_corr(self):
        self.assertAlmostEqual(np.round(self.model2.proj_diff_corr, decimals=4), -0.0891)

    def test_model2_in_sample_risk(self):
        self.assertAlmostEqual(
            np.round(np.sum(np.square(self.yhat2 - self.y)) / len(self.y), decimals=4),
            0.1328)

    def test_model2_out_of_sample_error(self):
        self.assertAlmostEqual(
            np.round(np.sum(np.square(self.yhat2_test - self.y_test)) / len(self.y_test),
                     decimals=4), 0.1253)

    # model3: L=4.5, use_V0=False, v_regularizer=1.0
    def test_model3_weights_shape(self):
        self.assertEqual(self.model3.weights.shape, (15, 3))

    def test_model3_V(self):
        self.assertAlmostEqual(np.round(self.model3.V, decimals=4), 0.2336)

    def test_model3_V0_is_none(self):
        self.assertIsNone(self.model3.V0)

    def test_model3_partition_radius(self):
        self.assertAlmostEqual(np.round(self.model3.partition_radius, decimals=4), 0.8883)

    def test_model3_proj_diff_corr(self):
        self.assertAlmostEqual(np.round(self.model3.proj_diff_corr, decimals=4), -0.0056)

    def test_model3_in_sample_risk(self):
        self.assertAlmostEqual(
            np.round(np.sum(np.square(self.yhat3 - self.y)) / len(self.y), decimals=4),
            0.0564)

    def test_model3_out_of_sample_error(self):
        self.assertAlmostEqual(
            np.round(np.sum(np.square(self.yhat3_test - self.y_test)) / len(self.y_test),
                     decimals=4), 0.0434)


class TestAPCNLSEstimatorSeed17(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(17)
        cls.X = np.random.randn(300, 5)
        cls.y = _regression_func_5d(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 5)
        cls.y_test = _regression_func_5d(cls.X_test)

        est4 = APCNLSEstimator()
        cls.model4 = est4.train(cls.X, cls.y)
        cls.yhat4 = est4.predict(cls.model4, cls.X)
        cls.yhat4_test = est4.predict(cls.model4, cls.X_test)

    def test_ols_out_of_sample_error(self):
        ols_model = np.linalg.lstsq(
            self.X.T.dot(self.X), self.X.T.dot(self.y), rcond=-1)[0]
        ols_yhat_test = np.sum(self.X_test * ols_model, axis=1)
        self.assertAlmostEqual(
            np.round(np.sum(np.square(ols_yhat_test - self.y_test)) / len(self.y_test),
                     decimals=4), 9.2304)

    def test_model4_weights_shape(self):
        self.assertEqual(self.model4.weights.shape, (48, 6))

    def test_model4_obj_val(self):
        self.assertAlmostEqual(np.round(self.model4.obj_val, decimals=1), -4.3)

    def test_model4_proj_obj_val(self):
        self.assertAlmostEqual(np.round(self.model4.proj_obj_val, decimals=1), -4.3)

    def test_model4_V(self):
        self.assertAlmostEqual(np.round(self.model4.V, decimals=4), 0.0009)

    def test_model4_partition_radius(self):
        self.assertAlmostEqual(np.round(self.model4.partition_radius, decimals=4), 1.8476)

    def test_model4_train_diff(self):
        self.assertAlmostEqual(np.round(self.model4.train_diff, decimals=4), 0.0002)

    def test_model4_in_sample_risk(self):
        self.assertAlmostEqual(
            np.round(np.sum(np.square(self.yhat4 - self.y)) / len(self.y), decimals=4),
            0.0637)

    def test_model4_out_of_sample_error(self):
        self.assertAlmostEqual(
            np.round(np.sum(np.square(self.yhat4_test - self.y_test)) / len(self.y_test),
                     decimals=2), 193.13)
