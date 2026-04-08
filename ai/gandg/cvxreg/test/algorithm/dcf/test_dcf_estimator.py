import io
import sys
import unittest
import numpy as np

from ai.gandg.cvxreg.common.util import set_random_seed
from ai.gandg.cvxreg.optim.socprog import SOCP_BACKEND__CLARABEL, SOCP_BACKEND__LBFGS
from ai.gandg.cvxreg.algorithm.dcf.dcf import DCFEstimator


def _regression_func(X):
    return 1.0 - 2.0 * X[:, 0] + X[:, 1] ** 2


class TestDCFEstimatorNoLocalClarabel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(200, 2)
        cls.y = _regression_func(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 2)
        cls.y_test = _regression_func(cls.X_test)
        ols_m = np.linalg.lstsq(cls.X.T.dot(cls.X), cls.X.T.dot(cls.y), rcond=-1)[0]
        cls.ols_yhat_test = np.sum(cls.X_test * ols_m, axis=1)

        ta = {
            'normalize': False,
            'L_sum_regularizer': 0.0,
            'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
            'L_regularizer_offset': 'np.log(n)',
            'local_opt_maxiter': 0,
            'backend': SOCP_BACKEND__CLARABEL,
        }

        cls.est1 = DCFEstimator(variant=2, is_convex=True, train_args=ta)
        cls.model1 = cls.est1.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat1 = cls.est1.predict(cls.model1, cls.X)
        cls.yhat1_test = cls.est1.predict(cls.model1, cls.X_test)

        cls.est2 = DCFEstimator(variant=2, is_convex=True, train_args=ta)
        cls.model2 = cls.est2.train(cls.X, cls.y)
        cls.yhat2 = cls.est2.predict(cls.model2, cls.X)
        cls.yhat2_test = cls.est2.predict(cls.model2, cls.X_test)

        cls.est3 = DCFEstimator(variant=1, is_convex=True, train_args=ta)
        cls.model3 = cls.est3.train(cls.X, cls.y, L_regularizer=None, L_regularizer_offset=0.0)
        cls.yhat3 = cls.est3.predict(cls.model3, cls.X)
        cls.yhat3_test = cls.est3.predict(cls.model3, cls.X_test)

        cls.est4 = DCFEstimator(variant=np.inf, is_convex=True, train_args=ta)
        cls.model4 = cls.est4.train(cls.X, cls.y, afpc_q=2)
        cls.yhat4 = cls.est4.predict(cls.model4, cls.X)
        cls.yhat4_test = cls.est4.predict(cls.model4, cls.X_test)

        cls.est5 = DCFEstimator(variant=np.inf, is_convex=False, train_args=ta)
        cls.model5 = cls.est5.train(cls.X, cls.y)
        cls.yhat5 = cls.est5.predict(cls.model5, cls.X)
        cls.yhat5_test = cls.est5.predict(cls.model5, cls.X_test)

        cls.est6 = DCFEstimator(variant='+', is_convex=True, train_args=ta)
        cls.model6 = cls.est6.train(cls.X, cls.y, afpc_q=2)
        cls.yhat6 = cls.est6.predict(cls.model6, cls.X)
        cls.yhat6_test = cls.est6.predict(cls.model6, cls.X_test)

        ta7 = dict(ta, negate_y=None, L_sum_regularizer=0.01)
        cls.est7 = DCFEstimator(variant='+', negate_y=True, train_args={
            'normalize': False,
            'L_sum_regularizer': 0.01,
            'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
            'L_regularizer_offset': 'np.log(n)',
            'local_opt_maxiter': 0,
            'backend': SOCP_BACKEND__CLARABEL,
        })
        cls.model7 = cls.est7.train(cls.X, cls.y)
        cls.yhat7 = cls.est7.predict(cls.model7, cls.X)
        cls.yhat7_test = cls.est7.predict(cls.model7, cls.X_test)

        cls.est8 = DCFEstimator(variant=np.inf, is_convex=False, is_symmetrized=True, train_args=ta)
        cls.model8 = cls.est8.train(cls.X, cls.y)
        cls.yhat8 = cls.est8.predict(cls.model8, cls.X)
        cls.yhat8_test = cls.est8.predict(cls.model8, cls.X_test)

        cls.est9 = DCFEstimator(variant='2q', is_convex=False, train_args=ta)
        cls.model9 = cls.est9.train(cls.X, cls.y)
        cls.yhat9 = cls.est9.predict(cls.model9, cls.X)
        cls.yhat9_test = cls.est9.predict(cls.model9, cls.X_test)

        cls.est10 = DCFEstimator(variant='2q', is_convex=True, train_args=ta)
        cls.model10 = cls.est10.train(cls.X, cls.y, afpc_q=2)
        cls.yhat10 = cls.est10.predict(cls.model10, cls.X)
        cls.yhat10_test = cls.est6.predict(cls.model10, cls.X_test)  # note: uses est6

        cls.est11 = DCFEstimator(variant='2w', is_convex=False, train_args=ta)
        cls.model11 = cls.est11.train(cls.X, cls.y)
        cls.yhat11 = cls.est11.predict(cls.model11, cls.X)
        cls.yhat11_test = cls.est11.predict(cls.model11, cls.X_test)

        cls.est12 = DCFEstimator(variant=2, is_convex=False, train_args=dict(ta, use_linear=False))
        cls.model12 = cls.est12.train(cls.X, cls.y)
        cls.yhat12 = cls.est12.predict(cls.model12, cls.X)
        cls.yhat12_test = cls.est12.predict(cls.model12, cls.X_test)

        cls.est13 = DCFEstimator(variant=2, is_convex=False, train_args=ta)
        cls.model13 = cls.est13.train(cls.X, cls.y)
        cls.yhat13 = cls.est13.predict(cls.model13, cls.X)
        cls.yhat13_test = cls.est13.predict(cls.model13, cls.X_test)

        cls.est14 = DCFEstimator(variant=2, is_convex=False, train_args={
            'verbose': 0,
            'normalize': False,
            'L_sum_regularizer': 0.0,
            'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
            'L_regularizer_offset': 'np.log(n)',
            'local_opt_maxiter': 0,
            'afpc_ntrials': 3,
            'kmeans_objval': True,
            'kmeans_kwargs': {},
            'backend': SOCP_BACKEND__CLARABEL,
        })
        cls.model14 = cls.est14.train(cls.X, cls.y)
        cls.yhat14 = cls.est14.predict(cls.model14, cls.X)
        cls.yhat14_test = cls.est14.predict(cls.model14, cls.X_test)

        cls.est15 = DCFEstimator(variant=2, is_convex=True, train_args=dict(ta, L_sum_regularizer=0.01))
        cls.model15 = cls.est15.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat15 = cls.est15.predict(cls.model15, cls.X)
        cls.yhat15_test = cls.est15.predict(cls.model15, cls.X_test)

        cls.est16 = DCFEstimator(variant=2, is_convex=True,
                                  train_args=dict(ta, L_sum_regularizer='0.01/I_k'))
        cls.model16 = cls.est16.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat16 = cls.est16.predict(cls.model16, cls.X)
        cls.yhat16_test = cls.est16.predict(cls.model16, cls.X_test)

        cls.est17 = DCFEstimator(variant=2, is_convex=True, train_args=dict(
            ta, L_sum_regularizer=0.01, afpc_min_cell_size=3))
        cls.model17 = cls.est17.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat17 = cls.est17.predict(cls.model17, cls.X)
        cls.yhat17_test = cls.est17.predict(cls.model17, cls.X_test)

        cls.est18 = DCFEstimator(variant=2, is_convex=True,
                                  train_args=dict(ta, L_sum_regularizer='(x_radius**2)/n'))
        cls.model18 = cls.est18.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat18 = cls.est18.predict(cls.model18, cls.X)
        cls.yhat18_test = cls.est18.predict(cls.model18, cls.X_test)

        cls.est19 = DCFEstimator(variant=2, is_convex=True, train_args=dict(
            ta, normalize=True, L_sum_regularizer='(x_radius**2)/n'))
        cls.model19 = cls.est19.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat19 = cls.est19.predict(cls.model19, cls.X)
        cls.yhat19_test = cls.est19.predict(cls.model19, cls.X_test)

    # --- OLS baseline ---
    def test_ols_out_of_sample_error(self):
        v = np.round(np.sum(np.square(self.ols_yhat_test - self.y_test)) / len(self.y_test), decimals=4)
        self.assertAlmostEqual(v, 6.2752)

    # --- model1: variant=2, convex, no regularizer ---
    def test_model1_weights_shape(self):
        self.assertEqual(self.model1.weights.shape, (15, 4))

    def test_model1_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model1._socp_stats.obj_val, decimals=4), 0.0104)
        self.assertAlmostEqual(np.round(self.model1._socp_stats.proj_obj_val, decimals=4), 0.0124)

    def test_model1_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model1.weights[:, 1:], axis=1)) ** 2, decimals=4),
            65.0408)

    def test_model1_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat1 - self.y)), decimals=4), 0.0248)

    def test_model1_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat1_test - self.y_test)), decimals=4), 0.041)

    # --- model2: variant=2, convex, default regularizer ---
    def test_model2_weights_shape(self):
        self.assertEqual(self.model2.weights.shape, (15, 4))

    def test_model2_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model2._socp_stats.reg_var_value, decimals=4), 0.0015)

    def test_model2_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat2 - self.y)), decimals=4), 0.0287)

    def test_model2_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat2_test - self.y_test)), decimals=4), 0.0433)

    # --- model3: variant=1, convex ---
    def test_model3_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat3 - self.y)), decimals=4), 0.0231)

    def test_model3_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat3_test - self.y_test)), decimals=4), 0.0393)

    def test_model3_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model3._socp_stats.obj_val, decimals=4), 0.0103)
        self.assertAlmostEqual(np.round(self.model3._socp_stats.proj_obj_val, decimals=4), 0.0116)

    def test_model3_weights_shape(self):
        self.assertEqual(self.model3.weights.shape, (15, 4))

    def test_model3_reg_var_value_none(self):
        self.assertIsNone(self.model3._socp_stats.reg_var_value)

    # --- model4: variant=inf, convex, afpc_q=2 ---
    def test_model4_variant(self):
        self.assertEqual(self.model4.variant, 'inf')

    def test_model4_weights_shape(self):
        self.assertEqual(self.model4.weights.shape, (7, 4))

    def test_model4_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model4._socp_stats.reg_var_value, decimals=4), 0.0014)

    def test_model4_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat4 - self.y)), decimals=4), 0.133)

    def test_model4_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat4_test - self.y_test)), decimals=4), 0.1321)

    # --- model5: variant=inf, not convex ---
    def test_model5_weights_shape(self):
        self.assertEqual(self.model5.weights.shape, (15, 4))

    def test_model5_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model5._socp_stats.reg_var_value, decimals=4), 0.0003)

    def test_model5_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat5 - self.y)), decimals=4), 0.0266)

    def test_model5_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat5_test - self.y_test)), decimals=4), 0.047)

    # --- model6: variant='+', convex, afpc_q=2 ---
    def test_model6_weights_shape(self):
        self.assertEqual(self.model6.weights.shape, (7, 5))

    def test_model6_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat6 - self.y)), decimals=4), 0.1255)

    def test_model6_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat6_test - self.y_test)), decimals=4), 0.0992)

    # --- model7: variant='+', negate_y=True, L_sum=0.01 ---
    def test_model7_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model7._socp_stats.obj_val, decimals=4), 0.6989)
        self.assertAlmostEqual(np.round(self.model7._socp_stats.proj_obj_val, decimals=4), 0.7577)

    def test_model7_weights_shape(self):
        self.assertEqual(self.model7.weights.shape, (15, 5))

    def test_model7_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model7._socp_stats.reg_var_value, decimals=4), 0.0)

    def test_model7_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat7 - self.y)), decimals=4), 0.8455)

    def test_model7_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat7_test - self.y_test)), decimals=4), 0.8954)

    # --- model8: variant=inf, not convex, symmetrized ---
    def test_model8_weights_shape(self):
        self.assertEqual(self.model8.weights.shape, (15, 8))

    def test_model8_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model8._socp_stats.reg_var_value, decimals=4), 0.0)

    def test_model8_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model8._socp_stats.obj_val, decimals=4), 0.0079)
        self.assertAlmostEqual(np.round(self.model8._socp_stats.proj_obj_val, decimals=4), 0.008)

    def test_model8_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat8 - self.y)), decimals=4), 0.016)

    def test_model8_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat8_test - self.y_test)), decimals=4), 0.0554)

    # --- model9: variant='2q', not convex ---
    def test_model9_weights_shape(self):
        self.assertEqual(self.model9.weights.shape, (15, 5))

    def test_model9_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model9._socp_stats.reg_var_value, decimals=4), 0.0002)

    def test_model9_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model9._socp_stats.obj_val, decimals=4), 0.0078)
        self.assertAlmostEqual(np.round(self.model9._socp_stats.proj_obj_val, decimals=4), 0.0121)

    def test_model9_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat9 - self.y)), decimals=4), 0.0242)

    def test_model9_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat9_test - self.y_test)), decimals=4), 0.0755)

    # --- model10: variant='2q', convex, afpc_q=2 ---
    def test_model10_weights_shape(self):
        self.assertEqual(self.model10.weights.shape, (7, 5))

    def test_model10_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat10 - self.y)), decimals=4), 0.1912)

    def test_model10_out_of_sample_error(self):
        # predicted with est6 (as in original doctest)
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat10_test - self.y_test)), decimals=4), 0.1824)

    # --- model11: variant='2w', not convex ---
    def test_model11_weights_len(self):
        self.assertEqual(len(self.model11.weights), 2)

    def test_model11_weights_shape(self):
        self.assertEqual(self.model11.weights[0].shape, (15, 4))

    def test_model11_weights_scalar(self):
        self.assertAlmostEqual(np.round(self.model11.weights[1], decimals=6), -0.011034)

    def test_model11_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model11._socp_stats.reg_var_value, decimals=4), 0.0)

    def test_model11_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat11 - self.y)), decimals=4), 0.0228)

    def test_model11_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat11_test - self.y_test)), decimals=4), 0.0376)

    # --- model12: variant=2, not convex, use_linear=False ---
    def test_model12_weights_shape(self):
        self.assertEqual(self.model12.weights.shape, (15, 2))

    def test_model12_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model12._socp_stats.reg_var_value, decimals=4), 0.0)

    def test_model12_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model12._socp_stats.obj_val, decimals=4), 0.4732)
        self.assertAlmostEqual(np.round(self.model12._socp_stats.proj_obj_val, decimals=4), 0.5728)

    def test_model12_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat12 - self.y)), decimals=4), 1.1456)

    def test_model12_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat12_test - self.y_test)), decimals=4), 1.3786)

    # --- model13: variant=2, not convex ---
    def test_model13_weights_shape(self):
        self.assertEqual(self.model13.weights.shape, (15, 4))

    def test_model13_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model13._socp_stats.reg_var_value, decimals=4), 0.0002)

    def test_model13_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat13 - self.y)), decimals=4), 0.0249)

    def test_model13_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat13_test - self.y_test)), decimals=4), 0.0418)

    # --- model14: variant=2, not convex, afpc_ntrials=3, kmeans_objval=True ---
    def test_model14_weights_shape(self):
        self.assertEqual(self.model14.weights.shape, (16, 4))

    def test_model14_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model14._socp_stats.obj_val, decimals=4), 0.008)
        self.assertAlmostEqual(np.round(self.model14._socp_stats.proj_obj_val, decimals=4), 0.0107)

    def test_model14_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model14._socp_stats.reg_var_value, decimals=4), 0.0008)

    def test_model14_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat14 - self.y)), decimals=4), 0.0213)

    def test_model14_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat14_test - self.y_test)), decimals=4), 0.0171)

    # --- model15: variant=2, convex, L_sum=0.01, no L/offset ---
    def test_model15_weights_shape(self):
        self.assertEqual(self.model15.weights.shape, (15, 4))

    def test_model15_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model15._socp_stats.obj_val, decimals=4), 0.4792)
        self.assertAlmostEqual(np.round(self.model15._socp_stats.proj_obj_val, decimals=4), 0.5389)

    def test_model15_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model15.weights[:, 1:], axis=1)) ** 2, decimals=4),
            12.9079)

    def test_model15_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat15 - self.y)), decimals=4), 0.3804)

    def test_model15_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat15_test - self.y_test)), decimals=4), 0.3254)

    # --- model16: variant=2, convex, L_sum='0.01/I_k' ---
    def test_model16_weights_shape(self):
        self.assertEqual(self.model16.weights.shape, (15, 4))

    def test_model16_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model16._socp_stats.obj_val, decimals=4), 0.1657)
        self.assertAlmostEqual(np.round(self.model16._socp_stats.proj_obj_val, decimals=4), 0.6008)

    def test_model16_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model16.weights[:, 1:], axis=1)) ** 2, decimals=4),
            19.3132)

    def test_model16_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat16 - self.y)), decimals=4), 0.1022)

    def test_model16_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat16_test - self.y_test)), decimals=4), 0.0859)

    # --- model17: variant=2, convex, L_sum=0.01, afpc_min_cell_size=3 ---
    def test_model17_weights_shape(self):
        self.assertEqual(self.model17.weights.shape, (13, 4))

    def test_model17_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model17._socp_stats.obj_val, decimals=4), 0.4392)
        self.assertAlmostEqual(np.round(self.model17._socp_stats.proj_obj_val, decimals=4), 0.4881)

    def test_model17_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model17.weights[:, 1:], axis=1)) ** 2, decimals=4),
            12.9377)

    def test_model17_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat17 - self.y)), decimals=4), 0.3274)

    def test_model17_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat17_test - self.y_test)), decimals=4), 0.2615)

    # --- model18: variant=2, convex, L_sum='(x_radius**2)/n', normalize=False ---
    def test_model18_weights_shape(self):
        self.assertEqual(self.model18.weights.shape, (15, 4))

    def test_model18_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model18._socp_stats.obj_val, decimals=4), 1.5026)
        self.assertAlmostEqual(np.round(self.model18._socp_stats.proj_obj_val, decimals=4), 1.6325)

    def test_model18_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model18.weights[:, 1:], axis=1)) ** 2, decimals=4),
            2.8956)

    def test_model18_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat18 - self.y)), decimals=4), 1.8392)

    def test_model18_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat18_test - self.y_test)), decimals=4), 1.5557)

    # --- model19: variant=2, convex, normalize=True, L_sum='(x_radius**2)/n' ---
    def test_model19_weights_shape(self):
        self.assertEqual(self.model19.weights.shape, (15, 4))

    def test_model19_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model19._socp_stats.obj_val, decimals=4), 0.2393)
        self.assertAlmostEqual(np.round(self.model19._socp_stats.proj_obj_val, decimals=4), 0.26)

    def test_model19_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model19.weights[:, 1:], axis=1)) ** 2, decimals=4),
            0.9064)

    def test_model19_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat19 - self.y)), decimals=4), 1.8392)

    def test_model19_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat19_test - self.y_test)), decimals=4), 1.5557)


class TestDCFEstimatorNoLocalLbfgs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(200, 2)
        cls.y = _regression_func(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 2)
        cls.y_test = _regression_func(cls.X_test)
        ols_m = np.linalg.lstsq(cls.X.T.dot(cls.X), cls.X.T.dot(cls.y), rcond=-1)[0]
        cls.ols_yhat_test = np.sum(cls.X_test * ols_m, axis=1)

        cls.est1 = DCFEstimator(variant=2, is_convex=True, train_args={
            'normalize': False,
            'L_sum_regularizer': 0.0,
            'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
            'L_regularizer_offset': 'np.log(n)',
            'local_opt_maxiter': 0,
            'backend': SOCP_BACKEND__LBFGS,
        })
        cls.model1 = cls.est1.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat1 = cls.est1.predict(cls.model1, cls.X)
        cls.yhat1_test = cls.est1.predict(cls.model1, cls.X_test)

    def test_ols_out_of_sample_error(self):
        v = np.round(np.sum(np.square(self.ols_yhat_test - self.y_test)) / len(self.y_test), decimals=4)
        self.assertAlmostEqual(v, 6.2752)

    def test_model1_weights_shape(self):
        self.assertEqual(self.model1.weights.shape, (15, 4))

    def test_model1_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model1._socp_stats.obj_val, decimals=3), 0.01)
        self.assertAlmostEqual(np.round(self.model1._socp_stats.proj_obj_val, decimals=3), 0.012)

    def test_model1_max_slope_norm_sq_near_45(self):
        # Instable across architectures; check within ±1 of 45
        v = np.max(np.linalg.norm(self.model1.weights[:, 1:], axis=1)) ** 2
        self.assertTrue(abs(v - 45) < 1, msg=f'Expected |v-45|<1, got v={v}')

    def test_model1_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat1 - self.y)), decimals=2), 0.02)

    def test_model1_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat1_test - self.y_test)), decimals=2), 0.04)


class TestDCFEstimatorLocalNonsmooth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(200, 2)
        cls.y = _regression_func(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 2)
        cls.y_test = _regression_func(cls.X_test)
        ols_m = np.linalg.lstsq(cls.X.T.dot(cls.X), cls.X.T.dot(cls.y), rcond=-1)[0]
        cls.ols_yhat_test = np.sum(cls.X_test * ols_m, axis=1)

        ta_base = {
            'normalize': False,
            'L_sum_regularizer': 0.0,
            'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
            'L_regularizer_offset': 'np.log(n)',
            'local_opt_type': 'nonsmooth',
            'backend': SOCP_BACKEND__CLARABEL,
        }

        cls.est1 = DCFEstimator(variant=2, is_convex=False, train_args=dict(
            ta_base, local_opt_maxiter=10, local_opt_noise_level=0.0))
        cls.model1 = cls.est1.train(cls.X, cls.y)
        cls.yhat1 = cls.est1.predict(cls.model1, cls.X)
        cls.yhat1_test = cls.est1.predict(cls.model1, cls.X_test)

        cls.est2 = DCFEstimator(variant=2, is_convex=False, is_symmetrized=True, train_args=dict(
            ta_base, local_opt_maxiter=20))
        cls.model2 = cls.est2.train(cls.X, cls.y)
        cls.yhat2 = cls.est2.predict(cls.model2, cls.X)
        cls.yhat2_test = cls.est2.predict(cls.model2, cls.X_test)

        cls.est3 = DCFEstimator(variant='+', is_convex=False, train_args=dict(
            ta_base, local_opt_maxiter=10, local_opt_noise_level=0.0))
        cls.model3 = cls.est3.train(cls.X, cls.y)
        cls.yhat3 = cls.est3.predict(cls.model3, cls.X)
        cls.yhat3_test = cls.est3.predict(cls.model3, cls.X_test)

        cls.est4 = DCFEstimator(variant='+', is_convex=False, is_symmetrized=True, train_args=dict(
            ta_base, verbose=1, L_sum_regularizer=0.01, local_opt_maxiter=20))
        buf = io.StringIO()
        sys.stdout = buf
        try:
            cls.model4 = cls.est4.train(cls.X, cls.y)
        finally:
            sys.stdout = sys.__stdout__
        cls.yhat4 = cls.est4.predict(cls.model4, cls.X)
        cls.yhat4_test = cls.est4.predict(cls.model4, cls.X_test)

        cls.est5 = DCFEstimator(variant=np.inf, is_convex=True, train_args=dict(
            ta_base, local_opt_maxiter=10))
        cls.model5 = cls.est5.train(cls.X, cls.y, afpc_q=2)
        cls.yhat5 = cls.est5.predict(cls.model5, cls.X)
        cls.yhat5_test = cls.est5.predict(cls.model5, cls.X_test)

        cls.est6 = DCFEstimator(variant='+', is_convex=True, train_args=dict(
            ta_base, local_opt_maxiter=10, local_opt_noise_level=1e-8))
        cls.model6 = cls.est6.train(cls.X, cls.y, afpc_q=2)
        cls.yhat6 = cls.est6.predict(cls.model6, cls.X)
        cls.yhat6_test = cls.est6.predict(cls.model6, cls.X_test)

        cls.est7 = DCFEstimator(variant=2, is_convex=False, train_args=dict(
            ta_base, L_sum_regularizer=0.01, local_opt_maxiter=10))
        cls.model7 = cls.est7.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat7 = cls.est7.predict(cls.model7, cls.X)
        cls.yhat7_test = cls.est7.predict(cls.model7, cls.X_test)

        cls.est8 = DCFEstimator(variant=2, is_convex=False, train_args=dict(
            ta_base, L_sum_regularizer=0.01, local_opt_maxiter=10,
            local_opt_L_regularizer_offset=1.0))
        cls.model8 = cls.est8.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat8 = cls.est8.predict(cls.model8, cls.X)
        cls.yhat8_test = cls.est8.predict(cls.model8, cls.X_test)

        cls.est10 = DCFEstimator(variant=2, is_convex=False, train_args=dict(
            ta_base, L_sum_regularizer='(x_radius**2)/n', local_opt_maxiter=10,
            local_opt_L_regularizer_offset='np.sqrt(np.log(n))'))
        cls.model10 = cls.est10.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat10 = cls.est10.predict(cls.model10, cls.X)
        cls.yhat10_test = cls.est10.predict(cls.model10, cls.X_test)

        cls.est11 = DCFEstimator(variant=2, is_convex=False, train_args={
            'normalize': True,
            'L_sum_regularizer': '(x_radius**2)/n',
            'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
            'L_regularizer_offset': 'np.log(n)',
            'local_opt_type': 'nonsmooth',
            'local_opt_maxiter': 10,
            'local_opt_L_regularizer_offset': 'np.sqrt(np.log(n))',
            'backend': SOCP_BACKEND__CLARABEL,
        })
        cls.model11 = cls.est11.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat11 = cls.est11.predict(cls.model11, cls.X)
        cls.yhat11_test = cls.est11.predict(cls.model11, cls.X_test)

    def test_ols_out_of_sample_error(self):
        v = np.round(np.sum(np.square(self.ols_yhat_test - self.y_test)) / len(self.y_test), decimals=4)
        self.assertAlmostEqual(v, 6.2752)

    # --- model1 ---
    def test_model1_weights_shape(self):
        self.assertEqual(self.model1.weights.shape, (11, 4))

    def test_model1_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model1._socp_stats.reg_var_value, decimals=4), 0.0002)

    def test_model1_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat1 - self.y)), decimals=4), 0.0093)

    def test_model1_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat1_test - self.y_test)), decimals=4), 0.0271)

    def test_model1_opt_stats(self):
        s = self.model1._socp_stats
        ls = self.model1._local_opt_stats
        vals = np.round([s.obj_val, s.proj_obj_val, ls.init_obj_val, ls.soln_obj_val], decimals=4)
        np.testing.assert_array_almost_equal(vals, [0.0105, 0.0124, 0.0124, 0.0047], decimal=4)

    # --- model2 ---
    def test_model2_weights_shape(self):
        self.assertEqual(self.model2.weights.shape, (15, 8))

    def test_model2_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model2._socp_stats.reg_var_value, decimals=4), 0.0)

    def test_model2_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat2 - self.y)), decimals=4), 0.0143)

    def test_model2_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat2_test - self.y_test)), decimals=4), 0.0613)

    def test_model2_opt_stats(self):
        s = self.model2._socp_stats
        ls = self.model2._local_opt_stats
        vals = np.round([s.obj_val, s.proj_obj_val, ls.init_obj_val, ls.soln_obj_val], decimals=4)
        np.testing.assert_array_almost_equal(vals, [0.0073, 0.0076, 0.0076, 0.0071], decimal=4)

    # --- model3 ---
    def test_model3_weights_shape(self):
        self.assertEqual(self.model3.weights.shape, (13, 5))

    def test_model3_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model3._socp_stats.reg_var_value, decimals=4), 0.0039)

    def test_model3_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat3 - self.y)), decimals=4), 0.0098)

    def test_model3_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat3_test - self.y_test)), decimals=4), 0.0067)

    def test_model3_opt_stats(self):
        s = self.model3._socp_stats
        ls = self.model3._local_opt_stats
        vals = np.round([s.obj_val, s.proj_obj_val, ls.init_obj_val, ls.soln_obj_val], decimals=4)
        np.testing.assert_array_almost_equal(vals, [0.0059, 0.0101, 0.01, 0.0049], decimal=4)

    # --- model4: symmetrized '+', verbose ---
    def test_model4_weights_structure(self):
        self.assertEqual(len(self.model4.weights), 2)
        self.assertEqual(self.model4.weights[0].shape, (5, 5))
        self.assertEqual(self.model4.weights[1].shape, (2, 5))

    def test_model4_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model4._socp_stats.reg_var_value, decimals=4), 0.0)

    def test_model4_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat4 - self.y)), decimals=4), 0.101)

    def test_model4_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat4_test - self.y_test)), decimals=4), 0.0862)

    def test_model4_opt_stats(self):
        s = self.model4._socp_stats
        ls = self.model4._local_opt_stats
        vals = np.round([s.obj_val, s.proj_obj_val, ls.init_obj_val, ls.soln_obj_val], decimals=4)
        np.testing.assert_array_almost_equal(vals, [0.3773, 0.3986, 0.3986, 0.2257], decimal=4)

    # --- model5: inf, convex, afpc_q=2 ---
    def test_model5_weights_shape(self):
        self.assertEqual(self.model5.weights.shape, (7, 4))

    def test_model5_last_col_nonneg(self):
        self.assertTrue(np.min(self.model5.weights[:, -1]) >= -1e-5)

    def test_model5_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model5._socp_stats.reg_var_value, decimals=4), 0.0014)

    def test_model5_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat5 - self.y)), decimals=4), 0.0759)

    def test_model5_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat5_test - self.y_test)), decimals=4), 0.0711)

    # --- model6: '+', convex, afpc_q=2, noise ---
    def test_model6_weights_shape(self):
        self.assertEqual(self.model6.weights.shape, (7, 5))

    def test_model6_local_opt_stats_maxiter(self):
        ls = self.model6._local_opt_stats
        self.assertEqual(ls.opt_status, 'MaxIterReached')
        self.assertEqual(ls.niterations, 10)

    def test_model6_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat6 - self.y)), decimals=4), 0.0948)

    def test_model6_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat6_test - self.y_test)), decimals=4), 0.0602)

    # --- model7 ---
    def test_model7_weights_shape(self):
        self.assertEqual(self.model7.weights.shape, (14, 4))

    def test_model7_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model7._socp_stats.obj_val, decimals=4), 0.3612)
        self.assertAlmostEqual(np.round(self.model7._socp_stats.proj_obj_val, decimals=4), 0.4195)

    def test_model7_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model7.weights[:, 1:], axis=1)) ** 2, decimals=4),
            8.1722)

    def test_model7_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat7 - self.y)), decimals=4), 0.152)

    def test_model7_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat7_test - self.y_test)), decimals=4), 0.2191)

    # --- model8 ---
    def test_model8_weights_shape(self):
        self.assertEqual(self.model8.weights.shape, (14, 4))

    def test_model8_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model8._socp_stats.obj_val, decimals=4), 0.3612)
        self.assertAlmostEqual(np.round(self.model8._socp_stats.proj_obj_val, decimals=4), 0.4195)

    def test_model8_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model8.weights[:, 1:], axis=1)) ** 2, decimals=4),
            8.0068)

    def test_model8_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat8 - self.y)), decimals=4), 0.1532)

    def test_model8_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat8_test - self.y_test)), decimals=4), 0.2209)

    # --- model10 ---
    def test_model10_weights_shape(self):
        self.assertEqual(self.model10.weights.shape, (13, 4))

    def test_model10_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model10._socp_stats.obj_val, decimals=4), 1.1057)
        self.assertAlmostEqual(np.round(self.model10._socp_stats.proj_obj_val, decimals=4), 1.301)

    def test_model10_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model10.weights[:, 1:], axis=1)) ** 2, decimals=4),
            2.9837)

    def test_model10_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat10 - self.y)), decimals=4), 1.123)

    def test_model10_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat10_test - self.y_test)), decimals=4), 1.115)

    # --- model11 ---
    def test_model11_weights_shape(self):
        self.assertEqual(self.model11.weights.shape, (14, 4))

    def test_model11_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model11._socp_stats.obj_val, decimals=4), 0.1761)
        self.assertAlmostEqual(np.round(self.model11._socp_stats.proj_obj_val, decimals=4), 0.2072)

    def test_model11_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model11.weights[:, 1:], axis=1)) ** 2, decimals=4),
            0.9658)

    def test_model11_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat11 - self.y)), decimals=4), 1.075)

    def test_model11_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat11_test - self.y_test)), decimals=4), 1.0707)


class TestDCFEstimatorLocalSmooth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(200, 2)
        cls.y = _regression_func(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 2)
        cls.y_test = _regression_func(cls.X_test)
        ols_m = np.linalg.lstsq(cls.X.T.dot(cls.X), cls.X.T.dot(cls.y), rcond=-1)[0]
        cls.ols_yhat_test = np.sum(cls.X_test * ols_m, axis=1)

        ta_base = {
            'L_sum_regularizer': 0.0,
            'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
            'L_regularizer_offset': 'np.log(n)',
            'local_opt_type': 'smooth',
            'backend': SOCP_BACKEND__CLARABEL,
        }

        cls.est1 = DCFEstimator(variant=2, is_convex=False, train_args=dict(ta_base, local_opt_maxiter=10))
        cls.model1 = cls.est1.train(cls.X, cls.y)
        cls.yhat1 = cls.est1.predict(cls.model1, cls.X)
        cls.yhat1_test = cls.est1.predict(cls.model1, cls.X_test)

        cls.est2 = DCFEstimator(variant=2, is_convex=False, is_symmetrized=True,
                                 train_args=dict(ta_base, local_opt_maxiter=20))
        cls.model2 = cls.est2.train(cls.X, cls.y)
        cls.yhat2 = cls.est2.predict(cls.model2, cls.X)
        cls.yhat2_test = cls.est2.predict(cls.model2, cls.X_test)

        cls.est3 = DCFEstimator(variant='+', is_convex=False, train_args=dict(ta_base, local_opt_maxiter=10))
        cls.model3 = cls.est3.train(cls.X, cls.y)
        cls.yhat3 = cls.est3.predict(cls.model3, cls.X)
        cls.yhat3_test = cls.est3.predict(cls.model3, cls.X_test)

        cls.est4 = DCFEstimator(variant='+', is_convex=False, is_symmetrized=True, train_args=dict(
            ta_base, verbose=1, local_opt_maxiter=20))
        buf = io.StringIO()
        sys.stdout = buf
        try:
            cls.model4 = cls.est4.train(cls.X, cls.y)
        finally:
            sys.stdout = sys.__stdout__
        cls.yhat4 = cls.est4.predict(cls.model4, cls.X)
        cls.yhat4_test = cls.est4.predict(cls.model4, cls.X_test)

        cls.est5 = DCFEstimator(variant=np.inf, is_convex=True, train_args=dict(ta_base, local_opt_maxiter=10))
        cls.model5 = cls.est5.train(cls.X, cls.y, afpc_q=2)
        cls.yhat5 = cls.est5.predict(cls.model5, cls.X)
        cls.yhat5_test = cls.est5.predict(cls.model5, cls.X_test)

        cls.est6 = DCFEstimator(variant='+', is_convex=True, train_args=dict(
            ta_base, local_opt_maxiter=10, local_opt_noise_level=1e-8))
        cls.model6 = cls.est6.train(cls.X, cls.y, afpc_q=2)
        cls.yhat6 = cls.est6.predict(cls.model6, cls.X)
        cls.yhat6_test = cls.est6.predict(cls.model6, cls.X_test)

        # est7: note duplicate L_sum_regularizer key; last one (0.01) wins
        cls.est7 = DCFEstimator(variant=2, is_convex=False, train_args=dict(
            ta_base, local_opt_maxiter=10, L_sum_regularizer=0.01))
        cls.model7 = cls.est7.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat7 = cls.est7.predict(cls.model7, cls.X)
        cls.yhat7_test = cls.est7.predict(cls.model7, cls.X_test)

        cls.est8 = DCFEstimator(variant=2, is_convex=False, train_args=dict(
            ta_base, L_sum_regularizer=0.01, local_opt_maxiter=10,
            local_opt_L_regularizer_offset=2.0))
        cls.model8 = cls.est8.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat8 = cls.est8.predict(cls.model8, cls.X)
        cls.yhat8_test = cls.est8.predict(cls.model8, cls.X_test)

        cls.est9 = DCFEstimator(variant=2, is_convex=False, train_args={
            'normalize': False,
            'L_sum_regularizer': '(x_radius**2)/n',
            'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
            'L_regularizer_offset': 'np.log(n)',
            'local_opt_type': 'smooth',
            'local_opt_maxiter': 10,
            'local_opt_L_regularizer_offset': 'np.sqrt(np.log(n))',
            'backend': SOCP_BACKEND__CLARABEL,
        })
        cls.model9 = cls.est9.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat9 = cls.est9.predict(cls.model9, cls.X)
        cls.yhat9_test = cls.est9.predict(cls.model9, cls.X_test)

        cls.est10 = DCFEstimator(variant=2, is_convex=False, train_args={
            'normalize': True,
            'L_sum_regularizer': '(x_radius**2)/n',
            'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
            'L_regularizer_offset': 'np.log(n)',
            'local_opt_type': 'smooth',
            'local_opt_maxiter': 10,
            'local_opt_L_regularizer_offset': 'np.sqrt(np.log(n))',
            'backend': SOCP_BACKEND__CLARABEL,
        })
        cls.model10 = cls.est10.train(cls.X, cls.y, L_regularizer=0.0, L_regularizer_offset=0.0)
        cls.yhat10 = cls.est10.predict(cls.model10, cls.X)
        cls.yhat10_test = cls.est10.predict(cls.model10, cls.X_test)

    def test_ols_out_of_sample_error(self):
        v = np.round(np.sum(np.square(self.ols_yhat_test - self.y_test)) / len(self.y_test), decimals=4)
        self.assertAlmostEqual(v, 6.2752)

    # --- model1 ---
    def test_model1_weights_shape(self):
        self.assertEqual(self.model1.weights.shape, (12, 4))

    def test_model1_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model1._socp_stats.reg_var_value, decimals=4), 0.0001)

    def test_model1_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat1 - self.y)), decimals=4), 0.0091)

    def test_model1_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat1_test - self.y_test)), decimals=4), 0.0273)

    def test_model1_opt_stats(self):
        s = self.model1._socp_stats
        ls = self.model1._local_opt_stats
        vals = np.round([s.obj_val, s.proj_obj_val, ls.init_obj_val, ls.soln_obj_val], decimals=4)
        np.testing.assert_array_almost_equal(vals, [0.0017, 0.002, 0.002, 0.0007], decimal=4)

    # --- model2 ---
    def test_model2_weights_shape(self):
        self.assertEqual(self.model2.weights.shape, (15, 8))

    def test_model2_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model2._socp_stats.reg_var_value, decimals=4), 0.0)

    def test_model2_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat2 - self.y)), decimals=4), 0.0143)

    def test_model2_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat2_test - self.y_test)), decimals=4), 0.0624)

    def test_model2_opt_stats(self):
        s = self.model2._socp_stats
        ls = self.model2._local_opt_stats
        vals = np.round([s.obj_val, s.proj_obj_val, ls.init_obj_val, ls.soln_obj_val], decimals=4)
        np.testing.assert_array_almost_equal(vals, [0.0012, 0.0012, 0.0012, 0.0011], decimal=4)

    # --- model3 ---
    def test_model3_weights_shape(self):
        self.assertEqual(self.model3.weights.shape, (13, 5))

    def test_model3_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model3._socp_stats.reg_var_value, decimals=4), 0.0022)

    def test_model3_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat3 - self.y)), decimals=4), 0.0099)

    def test_model3_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat3_test - self.y_test)), decimals=4), 0.0065)

    def test_model3_opt_stats(self):
        s = self.model3._socp_stats
        ls = self.model3._local_opt_stats
        vals = np.round([s.obj_val, s.proj_obj_val, ls.init_obj_val, ls.soln_obj_val], decimals=4)
        np.testing.assert_array_almost_equal(vals, [0.0009, 0.0016, 0.0016, 0.0008], decimal=4)

    # --- model4: symmetrized '+', verbose ---
    def test_model4_weights_shape(self):
        self.assertEqual(self.model4.weights.shape, (15, 10))

    def test_model4_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model4._socp_stats.reg_var_value, decimals=4), 0.0)

    def test_model4_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat4 - self.y)), decimals=4), 0.0086)

    def test_model4_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat4_test - self.y_test)), decimals=4), 0.0477)

    def test_model4_opt_stats(self):
        s = self.model4._socp_stats
        ls = self.model4._local_opt_stats
        vals = np.round([s.obj_val, s.proj_obj_val, ls.init_obj_val, ls.soln_obj_val], decimals=4)
        np.testing.assert_array_almost_equal(vals, [0.0006, 0.0036, 0.0036, 0.0007], decimal=4)

    # --- model5 ---
    def test_model5_weights_shape(self):
        self.assertEqual(self.model5.weights.shape, (7, 4))

    def test_model5_last_col_nonneg(self):
        self.assertTrue(np.min(self.model5.weights[:, -1]) >= -1e-5)

    def test_model5_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model5._socp_stats.reg_var_value, decimals=4), 0.0008)

    def test_model5_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat5 - self.y)), decimals=4), 0.0761)

    def test_model5_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat5_test - self.y_test)), decimals=4), 0.0712)

    # --- model6 ---
    def test_model6_weights_shape(self):
        self.assertEqual(self.model6.weights.shape, (7, 5))

    def test_model6_local_opt_stats_maxiter(self):
        ls = self.model6._local_opt_stats
        self.assertEqual(ls.opt_status, 'MaxIterReached')
        self.assertEqual(ls.niterations, 10)

    def test_model6_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat6 - self.y)), decimals=4), 0.0948)

    def test_model6_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat6_test - self.y_test)), decimals=4), 0.0602)

    # --- model7 ---
    def test_model7_weights_shape(self):
        self.assertEqual(self.model7.weights.shape, (13, 4))

    def test_model7_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model7._socp_stats.obj_val, decimals=4), 0.0575)
        self.assertAlmostEqual(np.round(self.model7._socp_stats.proj_obj_val, decimals=4), 0.0668)

    def test_model7_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model7.weights[:, 1:], axis=1)) ** 2, decimals=4),
            2.647)

    def test_model7_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat7 - self.y)), decimals=4), 0.13)

    def test_model7_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat7_test - self.y_test)), decimals=4), 0.186)

    # --- model8 ---
    def test_model8_weights_shape(self):
        self.assertEqual(self.model8.weights.shape, (13, 4))

    def test_model8_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model8._socp_stats.obj_val, decimals=4), 0.0575)
        self.assertAlmostEqual(np.round(self.model8._socp_stats.proj_obj_val, decimals=4), 0.0668)

    def test_model8_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model8.weights[:, 1:], axis=1)) ** 2, decimals=4),
            2.647)

    def test_model8_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat8 - self.y)), decimals=4), 0.13)

    def test_model8_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat8_test - self.y_test)), decimals=4), 0.186)

    # --- model9 ---
    def test_model9_weights_shape(self):
        self.assertEqual(self.model9.weights.shape, (15, 4))

    def test_model9_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model9._socp_stats.obj_val, decimals=4), 1.1057)
        self.assertAlmostEqual(np.round(self.model9._socp_stats.proj_obj_val, decimals=4), 1.301)

    def test_model9_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model9.weights[:, 1:], axis=1)) ** 2, decimals=4),
            2.9284)

    def test_model9_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat9 - self.y)), decimals=4), 1.1377)

    def test_model9_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat9_test - self.y_test)), decimals=4), 1.1269)

    # --- model10 ---
    def test_model10_weights_shape(self):
        self.assertEqual(self.model10.weights.shape, (15, 4))

    def test_model10_socp_stats(self):
        self.assertAlmostEqual(np.round(self.model10._socp_stats.obj_val, decimals=4), 0.1761)
        self.assertAlmostEqual(np.round(self.model10._socp_stats.proj_obj_val, decimals=4), 0.2072)

    def test_model10_max_slope_norm_sq(self):
        self.assertAlmostEqual(
            np.round(np.max(np.linalg.norm(self.model10.weights[:, 1:], axis=1)) ** 2, decimals=4),
            1.0035)

    def test_model10_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat10 - self.y)), decimals=4), 1.0352)

    def test_model10_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat10_test - self.y_test)), decimals=4), 1.0365)


class TestDCFEstimatorLocalSmoothMma(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_random_seed(19)
        cls.X = np.random.randn(200, 2)
        cls.y = _regression_func(cls.X) + 0.1 * np.random.randn(cls.X.shape[0])
        cls.X_test = np.random.randn(500, 2)
        cls.y_test = _regression_func(cls.X_test)
        ols_m = np.linalg.lstsq(cls.X.T.dot(cls.X), cls.X.T.dot(cls.y), rcond=-1)[0]
        cls.ols_yhat_test = np.sum(cls.X_test * ols_m, axis=1)

        ta_base = {
            'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
            'L_regularizer_offset': 'np.log(n)',
            'local_opt_type': 'smooth',
            'local_opt_maxiter': 10,
            'backend': SOCP_BACKEND__CLARABEL,
        }

        cls.est1 = DCFEstimator(variant='mma', is_convex=False, train_args=dict(
            ta_base, L_sum_regularizer=0.0, warn_on_nok_weights=True))
        cls.model1 = cls.est1.train(cls.X, cls.y)
        cls.yhat1 = cls.est1.predict(cls.model1, cls.X)
        cls.yhat1_test = cls.est1.predict(cls.model1, cls.X_test)

        cls.est2 = DCFEstimator(variant='mma', is_convex=False, train_args=dict(
            ta_base, L_sum_regularizer='(x_radius**2)/n'))
        cls.model2 = cls.est2.train(cls.X, cls.y)
        cls.yhat2 = cls.est2.predict(cls.model2, cls.X)
        cls.yhat2_test = cls.est2.predict(cls.model2, cls.X_test)

        cls.est3 = DCFEstimator(variant='mma', is_convex=False, is_symmetrized=True, train_args=dict(
            ta_base, L_sum_regularizer='(x_radius**2)/n'))
        cls.model3 = cls.est3.train(cls.X, cls.y)
        cls.yhat3 = cls.est3.predict(cls.model3, cls.X)
        cls.yhat3_test = cls.est3.predict(cls.model3, cls.X_test)

    def test_ols_out_of_sample_error(self):
        v = np.round(np.sum(np.square(self.ols_yhat_test - self.y_test)) / len(self.y_test), decimals=4)
        self.assertAlmostEqual(v, 6.2752)

    # --- model1 ---
    def test_model1_is_mma(self):
        self.assertTrue(self.model1.is_mma())

    def test_model1_maxL(self):
        self.assertAlmostEqual(np.round(self.model1.get_maxL(), decimals=2), 4.01)

    def test_model1_weights_shape(self):
        self.assertEqual(self.model1.weights.shape, (14, 4, 3))

    def test_model1_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model1._socp_stats.reg_var_value, decimals=4), 0.0002)

    def test_model1_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat1 - self.y)), decimals=3), 0.009)

    def test_model1_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat1_test - self.y_test)), decimals=4), 0.0093)

    def test_model1_opt_stats(self):
        s = self.model1._socp_stats
        ls = self.model1._local_opt_stats
        vals = np.round([s.obj_val, s.proj_obj_val, ls.init_obj_val, ls.soln_obj_val], decimals=4)
        np.testing.assert_array_almost_equal(vals, [0.002, 0.0021, 0.0021, 0.0007], decimal=4)

    # --- model2 ---
    def test_model2_is_mma(self):
        self.assertTrue(self.model2.is_mma())

    def test_model2_maxL(self):
        self.assertAlmostEqual(np.round(self.model2.get_maxL(), decimals=4), 1.3728)

    def test_model2_weights_shape(self):
        self.assertEqual(self.model2.weights.shape, (6, 4, 3))

    def test_model2_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model2._socp_stats.reg_var_value, decimals=4), 0.0)

    def test_model2_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat2 - self.y)), decimals=4), 1.022)

    def test_model2_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat2_test - self.y_test)), decimals=4), 1.0078)

    def test_model2_opt_stats(self):
        s = self.model2._socp_stats
        ls = self.model2._local_opt_stats
        vals = np.round([s.obj_val, s.proj_obj_val, ls.init_obj_val, ls.soln_obj_val], decimals=4)
        np.testing.assert_array_almost_equal(vals, [0.186, 0.2178, 0.515, 0.1628], decimal=4)

    # --- model3 ---
    def test_model3_is_mma(self):
        self.assertTrue(self.model3.is_mma())

    def test_model3_is_symmetrized(self):
        self.assertTrue(self.model3.is_symmetrized)

    def test_model3_maxL(self):
        self.assertAlmostEqual(np.round(self.model3.get_maxL(), decimals=4), 1.0964)

    def test_model3_weights_len(self):
        self.assertEqual(len(self.model3.weights), 2)

    def test_model3_weights_shape(self):
        self.assertEqual(self.model3.weights[0].shape, (5, 4, 3))
        self.assertEqual(self.model3.weights[1].shape, (7, 4, 3))

    def test_model3_reg_var_value(self):
        self.assertAlmostEqual(np.round(self.model3._socp_stats.reg_var_value, decimals=4), 0.0)

    def test_model3_in_sample_risk(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat3 - self.y)), decimals=4), 0.6714)

    def test_model3_out_of_sample_error(self):
        self.assertAlmostEqual(np.round(np.mean(np.square(self.yhat3_test - self.y_test)), decimals=4), 0.6812)

    def test_model3_opt_stats(self):
        s = self.model3._socp_stats
        ls = self.model3._local_opt_stats
        vals = np.round([s.obj_val, s.proj_obj_val, ls.init_obj_val, ls.soln_obj_val], decimals=4)
        np.testing.assert_array_almost_equal(vals, [0.1276, 0.1382, 0.3631, 0.1376], decimal=4)
