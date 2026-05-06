import copy
import unittest
import numpy as np

from cvxreg.common.util import set_random_seed
from cvxreg.common.partition import (
    max_affine_partition, find_min_dist_centers,
)
from cvxreg.algorithm.dcf.dcf import (
    DCFEstimatorModel, dcf_predict, dcf_fix_bias_offset, _check_cvx_mma_weights,
)


class TestDcfFixBiasOffset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(10, 4)
        weights0 = np.random.randn(3, 5)
        partition = max_affine_partition(np.insert(cls.X, 0, 1.0, axis=1), weights0)
        cls.centers = cls.X[find_min_dist_centers(cls.X, partition), :]
        cls.noise = np.random.randn(cls.X.shape[0])
        weights = np.hstack([weights0, np.random.randn(weights0.shape[0], 1)])

        # model1: convex DCF_2
        cls.model1 = DCFEstimatorModel(
            weights, cls.centers, variant=2, use_linear=True,
            is_convex=True, is_symmetrized=False)
        cls.yhat1 = dcf_predict(cls.model1, cls.X)
        cls.y1 = cls.yhat1 + cls.noise
        cls.model1b = dcf_fix_bias_offset(cls.model1, cls.X, cls.y1)
        cls.yhat1b = dcf_predict(cls.model1b, cls.X)

        # model2: symmetrized DCF_inf
        weights2 = np.hstack([weights, weights + 0.3 * np.random.randn(*weights.shape)])
        cls.model2 = DCFEstimatorModel(
            weights2, cls.centers, variant=np.inf, use_linear=True,
            is_convex=False, is_symmetrized=True)
        cls.yhat2 = dcf_predict(cls.model2, cls.X)
        cls.y2 = cls.yhat2 + cls.noise
        cls.model2b = dcf_fix_bias_offset(cls.model2, cls.X, cls.y2)
        cls.yhat2b = dcf_predict(cls.model2b, cls.X)

        # model3: from dcf_predict return_used_weights
        cls.yhat3, cls.model3 = dcf_predict(cls.model2, cls.X, return_used_weights=True)
        cls.y3 = cls.yhat3 + cls.noise
        cls.model3b = dcf_fix_bias_offset(cls.model3, cls.X, cls.y3)
        cls.yhat3b = dcf_predict(cls.model3b, cls.X)

        # model4: DCF_+ variant
        weights_p = np.hstack([weights0, np.random.randn(weights0.shape[0], weights0.shape[1]-1)])
        cls.model4 = DCFEstimatorModel(
            weights_p, cls.centers, variant='+', use_linear=True,
            is_convex=False, is_symmetrized=False)
        cls.yhat4 = dcf_predict(cls.model4, cls.X)
        cls.y4 = cls.yhat4 + cls.noise
        cls.model4b = dcf_fix_bias_offset(cls.model4, cls.X, cls.y4)
        cls.yhat4b = dcf_predict(cls.model4b, cls.X)

        # model5: with ymean/yscale
        cls.model5 = DCFEstimatorModel(
            weights, cls.centers, variant=2, use_linear=True,
            is_convex=False, is_symmetrized=False, ymean=1.5, yscale=-2.0)
        cls.yhat5 = dcf_predict(cls.model5, cls.X)
        cls.y5 = 1.5 + cls.yhat5 + cls.noise - np.mean(cls.yhat5 + cls.noise)
        cls.model5b = dcf_fix_bias_offset(cls.model5, cls.X, cls.y5)
        cls.yhat5b = dcf_predict(cls.model5b, cls.X)

        # model6: DCF_inf to_mma (non-symmetric)
        weights6 = weights.copy() - 0.3 * np.random.rand(*weights.shape)
        weights6[:, -1] = np.minimum(0.0, weights6[:, -1])
        cls.weights6_last = np.round(weights6[:, -1], decimals=4)
        cls.model6 = DCFEstimatorModel(
            weights6, cls.centers, variant=np.inf, use_linear=True,
            is_convex=False, is_symmetrized=False).to_mma()
        cls.yhat6 = dcf_predict(cls.model6, cls.X)
        cls.y6 = cls.yhat6 + cls.noise
        cls.model6b = dcf_fix_bias_offset(cls.model6, cls.X, cls.y6)
        cls.yhat6b = dcf_predict(cls.model6b, cls.X)

        # model7: symmetrized DCF_inf to_mma
        weights7 = np.hstack([weights, weights - 0.3 * np.random.rand(*weights.shape)])
        weights7[:, -2:] = np.minimum(0.0, weights7[:, -2:])
        cls.weights7_last2 = np.round(weights7[:, -2:], decimals=4)
        cls.model7 = DCFEstimatorModel(
            weights7, cls.centers, variant=np.inf, use_linear=True,
            is_convex=False, is_symmetrized=True).to_mma()
        cls.yhat7 = dcf_predict(cls.model7, cls.X)
        cls.y7 = cls.yhat7 + cls.noise
        cls.model7b = dcf_fix_bias_offset(cls.model7, cls.X, cls.y7)
        cls.yhat7b = dcf_predict(cls.model7b, cls.X)

    def test_centers_shape(self):
        self.assertEqual(self.centers.shape, (3, 4))

    # model1
    def test_model1_bias(self):
        self.assertAlmostEqual(
            np.round(np.mean(self.yhat1 - self.y1), decimals=4), 0.6435)

    def test_model1_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat1 - self.y1)), decimals=4), 1.1418)

    def test_model1b_mean_zero(self):
        self.assertLess(abs(np.mean(self.yhat1b - self.y1)), 1e-8)

    def test_model1b_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat1b - self.y1)), decimals=4), 0.7277)

    # model2
    def test_model2_bias(self):
        self.assertAlmostEqual(
            np.round(np.mean(self.yhat2 - self.y2), decimals=4), 0.6435)

    def test_model2_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat2 - self.y2)), decimals=4), 1.1418)

    def test_model2b_mean_zero(self):
        self.assertLess(abs(np.mean(self.yhat2b - self.y2)), 1e-8)

    def test_model2b_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat2b - self.y2)), decimals=4), 0.7277)

    def test_model2b_symmetry(self):
        self.assertLess(abs(2.0 * np.mean(self.model2b.weights[:, :2])), 1e-8)

    # model3
    def test_model3_predictions_equal(self):
        self.assertAlmostEqual(
            np.round(abs(np.mean(self.yhat3 - dcf_predict(self.model3, self.X))),
                     decimals=8), 0.0)

    def test_model3_bias(self):
        self.assertAlmostEqual(
            np.round(np.mean(self.yhat3 - self.y3), decimals=4), 0.6435)

    def test_model3_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat3 - self.y3)), decimals=4), 1.1418)

    def test_model3_is_symmetrized(self):
        self.assertEqual(self.model3.is_symmetrized, ([0, 2], [1, 2]))

    def test_model3b_mean_zero(self):
        self.assertLess(abs(np.mean(self.yhat3b - self.y3)), 1e-8)

    def test_model3b_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat3b - self.y3)), decimals=4), 0.7277)

    def test_model3b_mean_zero_biases(self):
        self.assertLess(
            abs(np.mean(self.model3b.weights[0][:, 0]) +
                np.mean(self.model3b.weights[1][:, 0])), 1e-8)

    # model4
    def test_model4_bias(self):
        self.assertAlmostEqual(
            np.round(np.mean(self.yhat4 - self.y4), decimals=4), 0.6435)

    def test_model4_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat4 - self.y4)), decimals=4), 1.1418)

    def test_model4b_mean_zero(self):
        self.assertLess(abs(np.mean(self.yhat4b - self.y4)), 1e-8)

    def test_model4b_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat4b - self.y4)), decimals=4), 0.7277)

    # model5
    def test_model5_bias(self):
        self.assertAlmostEqual(
            np.round(np.mean(self.yhat5 - self.y5), decimals=4), -0.6995)

    def test_model5_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat5 - self.y5)), decimals=4), 1.217)

    def test_model5b_mean_zero(self):
        self.assertLess(abs(np.mean(self.yhat5b - self.y5)), 1e-8)

    def test_model5b_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat5b - self.y5)), decimals=4), 0.7277)

    # model6
    def test_model6_weights6_last(self):
        np.testing.assert_array_equal(self.weights6_last, [0., -0.6621, 0.])

    def test_model6_bias(self):
        self.assertAlmostEqual(
            np.round(np.mean(self.yhat6 - self.y6), decimals=4), 0.6435)

    def test_model6_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat6 - self.y6)), decimals=4), 1.1418)

    def test_model6b_mean_zero(self):
        self.assertLess(abs(np.mean(self.yhat6b - self.y6)), 1e-8)

    def test_model6b_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat6b - self.y6)), decimals=4), 0.7277)

    # model7
    def test_model7_weights7_last2(self):
        expected = np.array([[-0.513, 0.], [-0.5147, -0.8415], [-0.0195, 0.]])
        np.testing.assert_array_almost_equal(self.weights7_last2, expected)

    def test_model7_bias(self):
        self.assertAlmostEqual(
            np.round(np.mean(self.yhat7 - self.y7), decimals=4), 0.6435)

    def test_model7_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat7 - self.y7)), decimals=4), 1.1418)

    def test_model7b_mean_zero(self):
        self.assertLess(abs(np.mean(self.yhat7b - self.y7)), 1e-8)

    def test_model7b_mse(self):
        self.assertAlmostEqual(
            np.round(np.mean(np.square(self.yhat7b - self.y7)), decimals=4), 0.7277)

    def test_model7b_mean_zero_biases(self):
        self.assertLess(
            abs(np.mean(self.model7b.weights[0][:, :, 0] +
                        self.model7b.weights[1][:, :, 0])), 1e-8)


class TestDcfPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(10, 4)
        weights0 = np.random.randn(3, 5)
        partition = max_affine_partition(np.insert(cls.X, 0, 1.0, axis=1), weights0)
        cls.centers = cls.X[find_min_dist_centers(cls.X, partition), :]
        weights = np.hstack([weights0, np.random.randn(weights0.shape[0], 1)])

        cls.model1 = DCFEstimatorModel(
            weights, cls.centers, variant=2, use_linear=True,
            is_convex=True, is_symmetrized=False)
        cls.model2 = DCFEstimatorModel(
            weights, cls.centers, variant=2, use_linear=True,
            is_convex=False, is_symmetrized=False)
        cls.model3 = DCFEstimatorModel(
            weights, cls.centers, variant=1, use_linear=True,
            is_convex=False, is_symmetrized=False)

        weights3i = weights.copy()
        weights3i[:, -1] = np.minimum(weights3i[:, -1], 0.0)
        cls.weights3i_last = np.round(weights3i[:, -1], decimals=6)
        cls.model3i = DCFEstimatorModel(
            weights3i, cls.centers, variant=np.inf, use_linear=True,
            is_convex=False, is_symmetrized=False)
        cls.model3mma = cls.model3i.to_mma()

        cls.model4 = DCFEstimatorModel(
            weights, cls.centers, variant=np.inf, use_linear=True,
            is_convex=False, is_symmetrized=False)

        weights2 = np.hstack([weights, weights])
        cls.model5 = DCFEstimatorModel(
            weights2, cls.centers, variant=np.inf, use_linear=True,
            is_convex=False, is_symmetrized=True)
        cls.model5i = copy.deepcopy(cls.model5)
        cls.model5i.weights[:, -2:] = np.minimum(0.0, cls.model5.weights[:, -2:])
        cls.model5immas = cls.model5i.to_mma()

        uweights = np.vstack([weights, np.array([[-10., 0., 0., 0., 0., 0.]])])
        ucenters = np.vstack([cls.centers, cls.centers[-1:, :]])
        cls.model6 = DCFEstimatorModel(
            uweights, ucenters, variant=2, use_linear=True,
            is_convex=False, is_symmetrized=False)

        pweights = np.hstack([weights0, np.random.randn(weights0.shape[0], weights0.shape[1]-1)])
        cls.model7 = DCFEstimatorModel(
            pweights, cls.centers, variant='+', use_linear=True,
            is_convex=False, is_symmetrized=False)

        upweights = np.vstack([pweights, pweights[2:3, :]])
        upcenters = np.vstack([cls.centers, cls.centers[2:3, :]])
        cls.model8 = DCFEstimatorModel(
            upweights, upcenters, variant='+', use_linear=True,
            is_convex=False, is_symmetrized=False)

    def test_centers_shape(self):
        self.assertEqual(self.centers.shape, (3, 4))

    # model1: non-symmetric, convex DCF_2 (not dropping weights)
    def test_model1_predictions(self):
        expected = np.array([-2.3494, -0.6013, -0.4297, -1.0141, -1.5567,
                             0.2401, -0.4761,  4.035, -1.0556,  0.4878])
        np.testing.assert_array_almost_equal(
            np.round(dcf_predict(self.model1, self.X), decimals=4), expected)

    def test_model1_return_used_weights_none(self):
        _, model1r = dcf_predict(self.model1, self.X, return_used_weights=True)
        self.assertIsNone(model1r)

    # model2: non-symmetric, non-convex DCF_2 (not dropping weights)
    def test_model2_predictions(self):
        expected = np.array([-2.3494, -0.6013, -0.4297, -1.0141, -1.5567,
                             0.2401, -0.4761,  4.035, -1.0556,  0.4878])
        np.testing.assert_array_almost_equal(
            np.round(dcf_predict(self.model2, self.X), decimals=4), expected)

    def test_model2_return_used_weights_none(self):
        _, model2r = dcf_predict(self.model2, self.X, return_used_weights=True)
        self.assertIsNone(model2r)

    # model3: non-symmetric DCF_1 (not dropping weights)
    def test_model3_predictions(self):
        expected = np.array([-2.2187, -0.6013, -0.4297, -0.6724, -1.3059,
                             0.8648, -0.4761,  5.3908, -0.1792,  1.0596])
        np.testing.assert_array_almost_equal(
            np.round(dcf_predict(self.model3, self.X), decimals=4), expected)

    def test_model3_return_used_weights_none(self):
        _, model3r = dcf_predict(self.model3, self.X, return_used_weights=True)
        self.assertIsNone(model3r)

    # model3i: non-symmetric MMA (dropping weights)
    def test_model3i_weights_last(self):
        np.testing.assert_array_equal(
            self.weights3i_last, [-0.608107, -0.45914, 0.])

    def test_model3i_weights_shape(self):
        self.assertEqual(self.model3i.weights.shape, (3, 6))

    def test_model3mma_centers_none(self):
        self.assertIsNone(self.model3mma.centers)

    def test_model3mma_weights_shape(self):
        self.assertEqual(self.model3mma.weights.shape, (3, 8, 5))

    def test_model3i_vs_model3mma(self):
        yhat3i = dcf_predict(self.model3i, self.X)
        expected = np.array([-2.2208, -0.6013, -0.4297, -1.5529, -1.1635,
                             -0.5079, -0.4761,  2.6081, -2.0862,  0.6363])
        np.testing.assert_array_almost_equal(np.round(yhat3i, decimals=4), expected)
        self.assertAlmostEqual(
            np.round(np.sum(np.abs(yhat3i - dcf_predict(self.model3mma, self.X))),
                     decimals=10), 0.0)

    def test_model3mma_return_used_weights(self):
        yhat3i = dcf_predict(self.model3i, self.X)
        yhat, model3mmar = dcf_predict(self.model3mma, self.X, return_used_weights=True)
        self.assertAlmostEqual(
            np.round(np.sum(np.abs(yhat3i - yhat)), decimals=10), 0.0)
        self.assertEqual(model3mmar.weights.shape, (3, 6, 5))
        yhatr, model_none = dcf_predict(model3mmar, self.X, return_used_weights=True)
        self.assertAlmostEqual(
            np.round(np.sum(np.abs(yhat3i - yhatr)), decimals=10), 0.0)
        self.assertIsNone(model_none)

    # model4: non-symmetric DCF_inf (not dropping weights)
    def test_model4_predictions(self):
        expected = np.array([-2.2208, -0.6013, -0.4297, -1.1126, -1.1635,
                             0.0113, -0.4761,  3.4604, -1.3121,  0.6363])
        np.testing.assert_array_almost_equal(
            np.round(dcf_predict(self.model4, self.X), decimals=4), expected)

    def test_model4_return_used_weights_none(self):
        _, model4r = dcf_predict(self.model4, self.X, return_used_weights=True)
        self.assertIsNone(model4r)

    # model5: symmetric DCF_inf (dropping weights)
    def test_model5_predictions(self):
        expected = np.array([-1.5616, -0.0836,  1.386,   0.8727,  0.9478,
                             -1.6409, -1.6886, -1.149,   2.1964, -2.5135])
        np.testing.assert_array_almost_equal(
            np.round(dcf_predict(self.model5, self.X), decimals=4), expected)

    def test_model5_return_used_weights(self):
        yhat5 = dcf_predict(self.model5, self.X)
        yhat5r, model5r = dcf_predict(self.model5, self.X, return_used_weights=True)
        self.assertEqual(self.model5.weights.shape, (3, 12))
        self.assertEqual(len(model5r.weights), 2)
        self.assertEqual(model5r.is_symmetrized, ([0, 2], [1, 2]))
        self.assertAlmostEqual(
            np.round(np.sum(np.abs(yhat5r - yhat5)), decimals=10), 0.0)
        yhat5rr, model5rr = dcf_predict(model5r, self.X, return_used_weights=True)
        self.assertIsNone(model5rr)
        self.assertAlmostEqual(
            np.round(np.sum(np.abs(yhat5rr - yhat5)), decimals=10), 0.0)

    # model5i / model5immas: symmetric MMA (dropping weights)
    def test_model5i_predictions(self):
        yhat5i = dcf_predict(self.model5i, self.X)
        expected = np.array([-0.7494, -0.0836,  2.234,   1.0732,  1.767,
                             -1.6409, -1.0217, -1.149,   2.7114, -1.7974])
        np.testing.assert_array_almost_equal(np.round(yhat5i, decimals=4), expected)
        self.assertAlmostEqual(
            np.round(np.sum(np.abs(yhat5i - dcf_predict(self.model5immas, self.X))),
                     decimals=10), 0.0)

    def test_model5immas_return_used_weights(self):
        yhat5i = dcf_predict(self.model5i, self.X)
        yhat, model5immasr = dcf_predict(
            self.model5immas, self.X, return_used_weights=True)
        self.assertAlmostEqual(
            np.round(np.sum(np.abs(yhat5i - yhat)), decimals=10), 0.0)
        self.assertEqual(model5immasr.weights[0].shape, (2, 3, 5))
        self.assertEqual(model5immasr.weights[1].shape, (2, 5, 5))
        yhatr, model5immasrr = dcf_predict(
            model5immasr, self.X, return_used_weights=True)
        self.assertIsNone(model5immasrr)
        self.assertAlmostEqual(
            np.round(np.sum(np.abs(yhat5i - yhatr)), decimals=10), 0.0)

    # model6: non-symmetric DCF_2 (dropping weights)
    def test_model6_predictions(self):
        yhat6 = dcf_predict(self.model6, self.X)
        expected = np.array([-2.3494, -0.6013, -0.4297, -1.0141, -1.5567,
                             0.2401, -0.4761,  4.035, -1.0556,  0.4878])
        np.testing.assert_array_almost_equal(np.round(yhat6, decimals=4), expected)

    def test_model6_return_used_weights(self):
        yhat6 = dcf_predict(self.model6, self.X)
        yhat6r, model6r = dcf_predict(self.model6, self.X, return_used_weights=True)
        self.assertEqual(self.model6.weights.shape, (4, 6))
        self.assertEqual(model6r.weights.shape, (3, 6))
        self.assertAlmostEqual(
            np.round(np.sum(np.abs(yhat6r - yhat6)), decimals=10), 0.0)

    # model7: non-symmetric DCF_+ (not dropping weights)
    def test_model7_predictions(self):
        expected = np.array([-1.7012, -0.6013, -0.4297, -0.993,  -1.8977,
                             0.0994, -0.4761,  2.7581,  0.0151,  1.207])
        np.testing.assert_array_almost_equal(
            np.round(dcf_predict(self.model7, self.X), decimals=4), expected)

    def test_model7_return_used_weights_none(self):
        _, model7r = dcf_predict(self.model7, self.X, return_used_weights=True)
        self.assertIsNone(model7r)

    # model8: non-symmetric DCF_+ (dropping weights)
    def test_model8_predictions(self):
        yhat8 = dcf_predict(self.model8, self.X)
        expected = np.array([-1.7012, -0.6013, -0.4297, -0.993,  -1.8977,
                             0.0994, -0.4761,  2.7581,  0.0151,  1.207])
        np.testing.assert_array_almost_equal(np.round(yhat8, decimals=4), expected)

    def test_model8_return_used_weights(self):
        yhat8 = dcf_predict(self.model8, self.X)
        yhat8r, model8r = dcf_predict(self.model8, self.X, return_used_weights=True)
        self.assertEqual(self.model8.weights.shape, (4, 9))
        self.assertEqual(model8r.weights.shape, (3, 9))
        self.assertAlmostEqual(
            np.round(np.sum(np.abs(yhat8r - yhat8)), decimals=10), 0.0)
        yhat8rr, model8rr = dcf_predict(model8r, self.X, return_used_weights=True)
        self.assertIsNone(model8rr)
        self.assertAlmostEqual(
            np.round(np.sum(np.abs(yhat8rr - yhat8)), decimals=10), 0.0)


class TestCheckCvxMmaWeights(unittest.TestCase):
    def test_mma_nonsymmetrized_clamps_positive(self):
        d = 3
        w1 = -np.ones((5, 2 * d, d + 1))
        w1[1, 1, 3] = 0.001
        self.assertAlmostEqual(np.max(w1.ravel()), 0.001)
        self.assertAlmostEqual(np.max(w1[:, :, -1].ravel()), 0.001)
        w2 = _check_cvx_mma_weights(w1, None, d, False, True, False, 'mma', False)[0]
        self.assertAlmostEqual(np.max(w2.ravel()), 0.0)

    def test_mma_symmetrized_clamps_positive(self):
        d = 3
        w3 = -np.ones((5, 2 * d, 2 * (d + 1)))
        w3[1, 1, 2 * d] = 0.001
        w3[2, 2, 2 * d + 1] = 0.002
        self.assertAlmostEqual(np.max(w3.ravel()), 0.002)
        w4 = _check_cvx_mma_weights(w3, None, d, False, True, True, 'mma', False)[0]
        self.assertAlmostEqual(np.max(w4.ravel()), 0.0)


class TestDcfSocpData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cvxreg.optim.socprog import SOCP_BACKEND__CLARABEL
        from cvxreg.common.partition import (
            Partition, singleton_partition, find_min_dist_centers)
        from cvxreg.algorithm.dcf.dcf import dcf_socp_data

        # --- case1: 1D, 3 cells, no regularizer ---
        X1 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]]).T.astype(float)
        y1 = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=float)
        partition1 = Partition(npoints=len(X1), ncells=3,
                               cells=[[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        center_idxs1 = [1, 4, 7]
        cls.H1, cls.g1, cls.A1, cls.b1, cls.cones1, cls.nparams_pwf1, _ = dcf_socp_data(
            X1, y1, partition1, X1[center_idxs1, :], variant=2, is_convex=False,
            is_mma=False, is_symmetrized=False, use_linear=True,
            L_regularizer=None, L_regularizer_offset=0.0,
            bias_regularizer=0.0, L_sum_regularizer=0.0,
            backend=SOCP_BACKEND__CLARABEL,
        )

        # --- case2: 2D, singleton partition, L_regularizer=3.0 ---
        X2 = np.array([[1.1, 1.1], [-1.2, 1.2], [-1.3, -1.3], [0.4, 0.4], [1.5, -1.5]])
        y2 = np.array([1.1, 1.2, 1.3, 0.4, 0.5])
        partition2 = singleton_partition(len(y2))
        center_idxs2 = find_min_dist_centers(X2, partition2)
        cls.H2, cls.g2, cls.A2, cls.b2, cls.cones2, cls.nparams_pwf2, _ = dcf_socp_data(
            X2, y2, partition2, X2[center_idxs2, :], variant=2, is_convex=True,
            is_mma=False, is_symmetrized=False, use_linear=True,
            L_regularizer=3.0, L_regularizer_offset=5.0,
            bias_regularizer=0.0, L_sum_regularizer=0.0,
            backend=SOCP_BACKEND__CLARABEL,
        )

    # --- case1 checks ---
    def test_case1_H(self):
        expected = np.array([
            [0.333, 0.,    0.222, 0.,    0.,    0.,    0.,    0.,    0.],
            [0.,    0.222, 0.,    0.,    0.,    0.,    0.,    0.,    0.],
            [0.,    0.,    0.222, 0.,    0.,    0.,    0.,    0.,    0.],
            [0.,    0.,    0.,    0.333, 0.,    0.222, 0.,    0.,    0.],
            [0.,    0.,    0.,    0.,    0.222, 0.,    0.,    0.,    0.],
            [0.,    0.,    0.,    0.,    0.,    0.222, 0.,    0.,    0.],
            [0.,    0.,    0.,    0.,    0.,    0.,    0.333, 0.,    0.222],
            [0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.222, 0.],
            [0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.222],
        ])
        np.testing.assert_array_almost_equal(
            np.round(self.H1.toarray(), decimals=3), expected, decimal=3)

    def test_case1_g(self):
        expected = np.array([-4., -0.222, -2.667, -5., -0.222, -3.333, -6., -0.222, -4.])
        np.testing.assert_array_almost_equal(
            np.round(self.g1, decimals=3), expected, decimal=3)

    def test_case1_A(self):
        expected = np.array([
            [-1.,  0.,  0.,  1., -3.,  3.,  0.,  0.,  0.],
            [-1.,  0.,  0.,  0.,  0.,  0.,  1., -6.,  6.],
            [1.,  3.,  3., -1.,  0.,  0.,  0.,  0.,  0.],
            [0.,  0.,  0., -1.,  0.,  0.,  1., -3.,  3.],
            [1.,  6.,  6.,  0.,  0.,  0., -1.,  0.,  0.],
            [0.,  0.,  0.,  1.,  3.,  3., -1.,  0.,  0.],
        ])
        np.testing.assert_array_almost_equal(
            np.round(self.A1.toarray(), decimals=3), expected, decimal=3)

    # --- case2 checks ---
    def test_case2_nparams_pwf(self):
        self.assertEqual(self.nparams_pwf2, 4)

    def test_case2_H_nnz(self):
        self.assertEqual(self.H2.nnz, 6)

    def test_case2_H_first_cols(self):
        expected = np.array([
            [0.2, 0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0.2, 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.2],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
            [0.,  0., 0., 0., 0., 0., 0., 0., 0.],
        ])
        np.testing.assert_array_almost_equal(self.H2.toarray()[:, :9], expected)

    def test_case2_H_last_rows(self):
        expected = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 3.]])
        np.testing.assert_array_almost_equal(self.H2.toarray()[-3:, -3:], expected)

    def test_case2_g(self):
        expected = np.array([-0.22, -0.,   -0.,   -0.,   -0.24, -0.,   -0.,   -0.,   -0.26,
                             -0.,   -0.,   -0.,   -0.08, -0.,   -0.,   -0.,   -0.1,  -0.,
                             -0.,   -0.,    0.])
        np.testing.assert_array_almost_equal(self.g2, expected, decimal=6)

    def test_case2_A_shape(self):
        self.assertEqual(self.A2.shape, (45, 21))

    def test_case2_A_rank(self):
        self.assertEqual(np.linalg.matrix_rank(self.A2.toarray()), 20)

    def test_case2_A_nnz(self):
        self.assertEqual(self.A2.nnz, 125)

    def test_case2_A_first_rows(self):
        expected = np.array([
            [-1.,  0.,  0.,  0.,  1.,  2.3,       -0.1,       2.30217289,  0.],
            [-1.,  0.,  0.,  0.,  0.,  0.,          0.,          0.,         1.],
            [-1.,  0.,  0.,  0.,  0.,  0.,          0.,          0.,         0.],
            [-1.,  0.,  0.,  0.,  0.,  0.,          0.,          0.,         0.],
            [1., -2.3,  0.1,  2.30217289, -1.,     0.,          0.,          0.,        0.],
            [0.,  0.,  0.,  0., -1.,  0.,          0.,          0.,         1.],
            [0.,  0.,  0.,  0., -1.,  0.,          0.,          0.,         0.],
            [0.,  0.,  0.,  0., -1.,  0.,          0.,          0.,         0.],
            [1., -2.4, -2.4,  3.39411255,  0.,     0.,          0.,          0.,        -1.],
            [0.,  0.,  0.,  0.,  1., -0.1,        -2.5,         2.5019992,  -1.],
            [0.,  0.,  0.,  0.,  0.,  0.,          0.,          0.,         -1.],
            [0.,  0.,  0.,  0.,  0.,  0.,          0.,          0.,         -1.],
            [1., -0.7, -0.7,  0.98994949,  0.,     0.,          0.,          0.,         0.],
            [0.,  0.,  0.,  0.,  1.,  1.6,        -0.8,         1.78885438,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,          0.,          0.,         1.],
            [0.,  0.,  0.,  0.,  0.,  0.,          0.,          0.,         0.],
            [1.,  0.4, -2.6,  2.63058929,  0.,     0.,          0.,          0.,         0.],
            [0.,  0.,  0.,  0.,  1.,  2.7,        -2.7,         3.81837662,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,          0.,          0.,         1.],
            [0.,  0.,  0.,  0.,  0.,  0.,          0.,          0.,         0.],
            [0.,  0.,  0., -1.,  0.,  0.,          0.,          0.,         0.],
            [0.,  0.,  0.,  0.,  0.,  0.,          0., -1.,                  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,          0.,          0.,         0.],
            [0.,  0.,  0.,  0.,  0.,  0.,          0.,          0.,         0.],
            [0.,  0.,  0.,  0.,  0.,  0.,          0.,          0.,         0.],
        ])
        np.testing.assert_array_almost_equal(self.A2.toarray()[:25, :9], expected, decimal=6)

    def test_case2_Ab_last_rows(self):
        expected = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.],
            [0., -1., 0., 0., 0.],
            [0.,  5., 0., 0., 0.],
        ])
        result = np.vstack([self.A2[-5:, :].T.toarray(), self.b2[-5:]])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_case2_b_shape(self):
        self.assertEqual(self.b2.shape, (45,))

    def test_case2_b_sum_abs(self):
        self.assertAlmostEqual(np.sum(np.abs(self.b2)), 25.0)

    def test_case2_ncones(self):
        self.assertEqual(len(self.cones2), 7)
