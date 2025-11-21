import gc
import copy
import warnings
import numpy as np
from functools import partial
from collections import namedtuple
from timeit import default_timer as timer
from scipy.sparse import csc_matrix, block_diag, coo_matrix, vstack, triu

from ai.gandg.common.estimator import EstimatorModel, Estimator
from ai.gandg.common.regression import prepare_prediction, postprocess_prediction
from ai.gandg.optim.socprog import (
    socp_solve, SOCP_BACKEND__LBFGS, convert_matrix_to_socp_solver_format,
    socp_nonnegative_cone, socp_second_order_cone,
)
from ai.gandg.common.distance import euclidean_distance
from ai.gandg.common.partition import (
    Partition, voronoi_partition, cell_radii, find_closest_centers,
)
from ai.gandg.algorithm.apcnls.fpc import (
    adaptive_farthest_point_clustering, get_data_radius,
    FPC_FIRST_IDX__DEFAULT, FPC_FIRST_IDX__RANDOM,
)


class DCFEstimatorModel(EstimatorModel):
    """The model of the DCF estimators."""
    def __init__(
        self, weights, centers, variant, use_linear, is_convex, is_symmetrized,
        clustering_stats=None, socp_stats=None, local_opt_stats=None,
        xmean=None, xscale=None, yscale=None, ymean=None,
    ):
        EstimatorModel.__init__(self, weights, xmean, xscale, yscale, ymean)
        self.centers = centers
        self.variant = variant
        self.use_linear = use_linear
        self.is_convex = is_convex
        self.is_symmetrized = is_symmetrized

        self._clustering_stats = clustering_stats
        self._socp_stats = socp_stats
        self._local_opt_stats = local_opt_stats

    def get_nparams(self, include_centers=False):
        weights = self.weights
        if not isinstance(weights, tuple):
            weights = (weights,)
        nparams = np.sum([np.prod(w.shape) for w in weights])
        if include_centers and self.centers is not None:
            nparams += np.prod(self.centers.shape)
        return nparams

    def _get_slope_weights(self):
        weights = self.weights
        if self.is_mma():
            if self.is_symmetrized:
                weights = np.vstack([weights[0].reshape((-1, weights[0].shape[-1])),
                                     weights[1].reshape((-1, weights[1].shape[-1]))])
            else:
                weights = weights.reshape((-1, weights.shape[-1]))
            weights = weights[:, 1:]
        else:
            if isinstance(weights, tuple):
                weights = weights[0]
            if self.is_symmetrized and isinstance(self.is_symmetrized, bool):
                weights = np.vstack([weights[:, 2::2], weights[:, 3::2]])
            else:
                weights = weights[:, 1:]
        return weights

    def get_maxL(self):
        """Returns the maximum of the slope weight norms."""
        weights = self._get_slope_weights()
        return np.max(np.linalg.norm(weights, axis=1))

    def get_sumsqL(self):
        """Returns the sum of the squared slope weight norms."""
        weights = self._get_slope_weights()
        return np.sum(np.linalg.norm(weights, axis=1)**2)

    def _get_mma_weights(self, weights, centers):
        wmax = np.max(weights[:, -1])
        assert wmax <= 0.0, f'Incompatible positive wmax: {wmax}!'
        K, d2 = weights.shape
        d1 = d2 - 1
        d = d1 - 1
        W = np.empty((K, 2*d, d1))
        Id = np.eye(d)
        W[:, :, 1:] = (
            weights[:, 1:-1][:, None, :]
            + weights[:, -1][:, None, None] * np.vstack([Id, -Id])[None, :, :]
        )
        W[:, :, 0] = (
            weights[:, 0][:, None]
            - (W[:, :, 1:] * centers[:, None, :]).sum(axis=2)
        )
        return W

    def is_mma(self):
        return self.variant == 'mma'

    def to_mma(self):
        assert self.use_linear
        assert not self.is_convex
        assert str(self.variant) == 'inf'
        weights = self.weights
        centers = self.centers
        symm = self.is_symmetrized
        if symm:
            if isinstance(symm, bool):
                W = (self._get_mma_weights(weights[:, 0::2], centers),
                     self._get_mma_weights(weights[:, 1::2], centers))
            else:
                W = (self._get_mma_weights(weights[:symm, :], centers[:symm, :]),
                     self._get_mma_weights(weights[symm:, :], centers[symm:, :]))
        else:
            W = self._get_mma_weights(weights, centers)

        new_model = copy.deepcopy(self)
        new_model.is_symmetrized = True if symm else False
        new_model.weights = W
        new_model.centers = None
        new_model.variant = 'mma'
        return new_model


DCFClusteringStats = namedtuple('DCFClusteringStats', [
    'ncells',              # number of cells (K)
    'max_epsilon',         # maximum cell radius (eps_n(\hat{X}_K))
    'avg_epsilon',         # average cell radius
    'runtime',             # clustering runtime in seconds
])
DCFSOCPStats = namedtuple('DCFSOCPStats', [
    'reg_var_value',       # value of the regularization variable (z)
    'obj_val',             # SOCP objective value (of g_n)
    'proj_obj_val',        # SOCP objective value after projecting the solution (g_n -> f_n)
    'niterations',         # number of iterations performed by the SOCP solver
    'runtime_data',        # SOCP data preparation runtime in seconds
    'runtime_solve',       # SOCP solver runtime in seconds
    'nparams_wo_centers',  # number of model parameters (not counting the centers)
    'nparams_w_centers',   # number of model parameters (including the centers)
])
DCFLocalOptStats = namedtuple('DCFLocalOptStats', [
    'init_obj_val',        # objective value at initial point
    'soln_obj_val',        # objective value at solution point
    'opt_status',          # optimization status (i.e., reason of termination)
    'niterations',         # number of iteratiosn performed by the solver
    'nfun_evals',          # number of function evaluations performed by the solver
    'ngrad_evals',         # number of gradient evaluations performed by the solver
    'runtime',             # solver runtime in seconds
])


class DCFEstimator(Estimator):
    """Delta-convex fitting (DCF) estimator."""
    def __init__(self, variant,
                 is_convex=False, negate_y=False, is_symmetrized=False,
                 train_args=None, predict_args=None, logger=print):
        """
        :param variant: integer defining the norm, np.inf for rhe max-norm, or '+'
        """
        train_args = {} if train_args is None else dict(train_args)
        predict_args = {} if predict_args is None else dict(predict_args)
        assert 'is_convex' not in train_args or train_args['is_convex'] == is_convex
        train_args['is_convex'] = is_convex
        assert 'negate_y' not in train_args or train_args['negate_y'] == negate_y
        train_args['negate_y'] = negate_y
        assert not is_convex or not negate_y, 'Negating y makes no sense for convex regression!'
        assert not is_symmetrized or not negate_y, (
            'Negating y makes no sense for a symmetric representation!')
        assert 'is_symmetrized' not in train_args or train_args['is_symmetrized'] == is_symmetrized
        train_args['is_symmetrized'] = is_symmetrized
        train_args['logger'] = logger
        assert not is_convex or not is_symmetrized, 'Convex representations cannot be symmetrized!'
        Estimator.__init__(
            self,
            train=partial(dcf_train, variant=variant, **train_args),
            predict=partial(dcf_predict, **predict_args),
        )


def _get_norm_p(variant):
    if variant.startswith('inf'):
        return np.inf
    if variant[-1] in ('q', 'w'):
        return int(variant[:-1])
    return int(variant)


def _get_nparams_pwf(d, variant, use_linear):
    """
    _get_nparams_pwf(2, '2', True)
    4
    _get_nparams_pwf(2, '+', True)
    5
    """
    return (1+2*d if variant == '+'
            else d*int(use_linear)+2+int(variant[-1] == 'q'))


def _dcf_calc_diffs(X, centers, variant):
    diffs = (X[:, :, None] - centers[:, :, None].T).transpose(0, 2, 1)
    norms = (None if variant == '+' else
             np.linalg.norm(diffs, ord=_get_norm_p(variant), axis=2))
    return diffs, norms


def _dcf_calc_phi(X, centers, variant, use_linear):
    """Calculates the feature tensor of shape n x K x nparams_pwf."""
    n = X.shape[0]
    K, d = centers.shape
    diffs, norms = _dcf_calc_diffs(X, centers, variant)
    nparams_pwf = _get_nparams_pwf(d, variant, use_linear)
    phi = np.empty((n, K, nparams_pwf))
    phi[:, :, 0] = 1.0
    d1 = d+1
    if variant == '+':
        phi[:, :, 1:d1] = np.maximum(0.0, diffs)
        phi[:, :, d1:] = np.maximum(0.0, -diffs)
    else:
        wlen = 2 if variant[-1] == 'q' else 1
        if use_linear:
            phi[:, :, 1:d1] = diffs
        for p, i in enumerate(range(-wlen, 0), 1):
            phi[:, :, i] = norms if p == 1 else norms**p
    if variant[-1] == 'w':
        phi = (phi, np.sum(X**2, axis=1))
    return phi


def _dcf_calc_Xw(phi, weights):
    if isinstance(phi, tuple):
        Xw = _dcf_calc_Xw(phi[0], weights[0])
        for p, w in zip(phi[1:], weights[1:]):
            Xw += w * p[:, None]
    else:
        # Calculating: Xw = (phi * weights[None, :, :]).sum(axis=2)
        Xw = np.einsum('ikj,kj->ik', phi, weights)
    return Xw


def _dcf_reg(weights, is_symmetrized,
             L_regularizer, L_regularizer_offset, L_sum_regularizer):
    if (L_regularizer is None or L_regularizer == 0.0) and L_sum_regularizer == 0.0:
        return 0.0
    if isinstance(weights, tuple):
        weights = weights[0]
    if is_symmetrized:
        weights = np.vstack([weights[:, 0::2], weights[:, 1::2]])
    wnorms = np.linalg.norm(weights[:, 1:], axis=1)
    wreg = 0.0
    if L_regularizer is not None and L_regularizer > 0.0:
        wreg += L_regularizer * max(0.0, np.max(wnorms) - L_regularizer_offset)**2
    if L_sum_regularizer > 0.0:
        wreg += L_sum_regularizer * np.sum(np.square(wnorms))
    return wreg


def dcf_fix_bias_offset(model, X, y, nsplit=10000):
    """Normalizing the bias terms without changing the training risk.

    >>> np.set_printoptions(legacy='1.25')
    >>> from ai.gandg.common.util import set_random_seed
    >>> from ai.gandg.common.partition import max_affine_partition, find_min_dist_centers
    >>> set_random_seed(19)

    >>> X = np.random.randn(10, 4)
    >>> weights0 = np.random.randn(3, 5)
    >>> partition = max_affine_partition(np.insert(X, 0, 1.0, axis=1), weights0)
    >>> centers = X[find_min_dist_centers(X, partition), :]
    >>> centers.shape
    (3, 4)
    >>> noise = np.random.randn(X.shape[0])

    >>> weights = np.hstack([weights0, np.random.randn(weights0.shape[0], 1)])

    >>> model1 = DCFEstimatorModel(weights, centers, variant=2, use_linear=True,
    ...                            is_convex=True, is_symmetrized=False)
    >>> yhat1 = dcf_predict(model1, X)
    >>> y1 = yhat1 + noise
    >>> np.round(np.mean(yhat1 - y1), decimals=4)
    0.6435
    >>> np.round(np.mean(np.square(yhat1 - y1)), decimals=4)
    1.1418
    >>> model1b = dcf_fix_bias_offset(model1, X, y1)
    >>> yhat1b = dcf_predict(model1b, X)
    >>> abs(np.mean(yhat1b - y1)) < 1e-8
    True
    >>> np.round(np.mean(np.square(yhat1b - y1)), decimals=4)
    0.7277

    >>> weights2 = np.hstack([weights, weights + 0.3 * np.random.randn(*weights.shape)])
    >>> model2 = DCFEstimatorModel(weights2, centers, variant=np.inf, use_linear=True,
    ...                            is_convex=False, is_symmetrized=True)
    >>> yhat2 = dcf_predict(model2, X)
    >>> y2 = yhat2 + noise
    >>> np.round(np.mean(yhat2 - y2), decimals=4)
    0.6435
    >>> np.round(np.mean(np.square(yhat2 - y2)), decimals=4)
    1.1418
    >>> model2b = dcf_fix_bias_offset(model2, X, y2)
    >>> yhat2b = dcf_predict(model2b, X)
    >>> abs(np.mean(yhat2b - y2)) < 1e-8
    True
    >>> np.round(np.mean(np.square(yhat2b - y2)), decimals=4)
    0.7277
    >>> abs(2.0 * np.mean(model2b.weights[:, :2])) < 1e-8
    True

    >>> yhat3, model3 = dcf_predict(model2, X, return_used_weights=True)
    >>> np.round(abs(np.mean(yhat3 - dcf_predict(model3, X))), decimals=8)
    0.0
    >>> y3 = yhat3 + noise
    >>> np.round(np.mean(yhat3 - y3), decimals=4)
    0.6435
    >>> np.round(np.mean(np.square(yhat3 - y3)), decimals=4)
    1.1418
    >>> model3.is_symmetrized
    ([0, 2], [1, 2])
    >>> model3b = dcf_fix_bias_offset(model3, X, y3)
    >>> yhat3b = dcf_predict(model3b, X)
    >>> abs(np.mean(yhat3b - y3)) < 1e-8
    True
    >>> np.round(np.mean(np.square(yhat3b - y3)), decimals=4)
    0.7277
    >>> abs(np.mean(model3b.weights[0][:, 0]) + np.mean(model3b.weights[1][:, 0])) < 1e-8
    True

    >>> weights_p = np.hstack([weights0, np.random.randn(weights0.shape[0], weights0.shape[1]-1)])
    >>> model4 = DCFEstimatorModel(weights_p, centers, variant='+', use_linear=True,
    ...                            is_convex=False, is_symmetrized=False)
    >>> yhat4 = dcf_predict(model4, X)
    >>> y4 = yhat4 + noise
    >>> np.round(np.mean(yhat4 - y4), decimals=4)
    0.6435
    >>> np.round(np.mean(np.square(yhat4 - y4)), decimals=4)
    1.1418
    >>> model4b = dcf_fix_bias_offset(model4, X, y4)
    >>> yhat4b = dcf_predict(model4b, X)
    >>> abs(np.mean(yhat4b - y4)) < 1e-8
    True
    >>> np.round(np.mean(np.square(yhat4b - y4)), decimals=4)
    0.7277

    >>> model5 = DCFEstimatorModel(weights, centers, variant=2, use_linear=True,
    ...                            is_convex=False, is_symmetrized=False,
    ...                            ymean=1.5, yscale=-2.0)
    >>> yhat5 = dcf_predict(model5, X)
    >>> y5 = 1.5 + yhat5 + noise - np.mean(yhat5 + noise)
    >>> np.round(np.mean(yhat5 - y5), decimals=4)
    -0.6995
    >>> np.round(np.mean(np.square(yhat5 - y5)), decimals=4)
    1.217
    >>> model5b = dcf_fix_bias_offset(model5, X, y5)
    >>> yhat5b = dcf_predict(model5b, X)
    >>> abs(np.mean(yhat5b - y5)) < 1e-8
    True
    >>> np.round(np.mean(np.square(yhat5b - y5)), decimals=4)
    0.7277

    >>> weights6 = weights.copy() - 0.3 * np.random.rand(*weights.shape)
    >>> weights6[:, -1] = np.minimum(0.0, weights6[:, -1])
    >>> np.round(weights6[:, -1], decimals=4)
    array([ 0.    , -0.6621,  0.    ])
    >>> model6 = DCFEstimatorModel(weights6, centers, variant=np.inf, use_linear=True,
    ...                            is_convex=False, is_symmetrized=False).to_mma()
    >>> yhat6 = dcf_predict(model6, X)
    >>> y6 = yhat6 + noise
    >>> np.round(np.mean(yhat6 - y6), decimals=4)
    0.6435
    >>> np.round(np.mean(np.square(yhat6 - y6)), decimals=4)
    1.1418
    >>> model6b = dcf_fix_bias_offset(model6, X, y6)
    >>> yhat6b = dcf_predict(model6b, X)
    >>> abs(np.mean(yhat6b - y6)) < 1e-8
    True
    >>> np.round(np.mean(np.square(yhat6b - y6)), decimals=4)
    0.7277

    >>> weights7 = np.hstack([weights, weights - 0.3 * np.random.rand(*weights.shape)])
    >>> weights7[:, -2:] = np.minimum(0.0, weights7[:, -2:])
    >>> np.round(weights7[:, -2:], decimals=4)
    array([[-0.513 ,  0.    ],
           [-0.5147, -0.8415],
           [-0.0195,  0.    ]])
    >>> model7 = DCFEstimatorModel(weights7, centers, variant=np.inf, use_linear=True,
    ...                            is_convex=False, is_symmetrized=True).to_mma()
    >>> yhat7 = dcf_predict(model7, X)
    >>> y7 = yhat7 + noise
    >>> np.round(np.mean(yhat7 - y7), decimals=4)
    0.6435
    >>> np.round(np.mean(np.square(yhat7 - y7)), decimals=4)
    1.1418
    >>> model7b = dcf_fix_bias_offset(model7, X, y7)
    >>> yhat7b = dcf_predict(model7b, X)
    >>> abs(np.mean(yhat7b - y7)) < 1e-8
    True
    >>> np.round(np.mean(np.square(yhat7b - y7)), decimals=4)
    0.7277
    >>> abs(np.mean(model7b.weights[0][:, :, 0] + model7b.weights[1][:, :, 0])) < 1e-8
    True
    """
    model = copy.deepcopy(model)
    yhat = dcf_predict(model, X, nsplit)
    bias_offset = np.mean(y) - np.mean(yhat)
    if model.yscale is not None:
        bias_offset /= model.yscale
    w = model.weights
    if model.is_symmetrized:
        if model.is_mma():
            w[0][:, :, 0] += bias_offset
            symm_offset = 0.5 * (np.mean(w[0][:, :, 0]) + np.mean(w[1][:, :, 0]))
            w[0][:, :, 0] -= symm_offset
            w[1][:, :, 0] -= symm_offset
        else:
            if isinstance(model.is_symmetrized, bool):
                w[:, 0] += bias_offset
                symm_offset = 2.0 * np.mean(w[:, :2])
                w[:, :2] -= 0.5 * symm_offset
            else:
                w[0][:, 0] += bias_offset
                symm_offset = 0.5 * (np.mean(w[0][:, 0]) + np.mean(w[1][:, 0]))
                w[0][:, 0] -= symm_offset
                w[1][:, 0] -= symm_offset
    else:
        if model.is_mma():
            w[:, :, 0] += bias_offset
        else:
            w0 = w[0] if isinstance(w, tuple) else w
            w0[:, 0] += bias_offset
    return model


def dcf_predict(model, X, nsplit=10000, return_used_weights=False):
    """
    >>> np.set_printoptions(legacy='1.25')
    >>> from ai.gandg.common.util import set_random_seed
    >>> from ai.gandg.common.partition import max_affine_partition, find_min_dist_centers
    >>> set_random_seed(19)

    >>> X = np.random.randn(10, 4)
    >>> weights0 = np.random.randn(3, 5)
    >>> partition = max_affine_partition(np.insert(X, 0, 1.0, axis=1), weights0)
    >>> centers = X[find_min_dist_centers(X, partition), :]
    >>> centers.shape
    (3, 4)

    >>> weights = np.hstack([weights0, np.random.randn(weights0.shape[0], 1)])

    ### Non-symmetric, convex DCF_2 model (not dropping weights):
    >>> model1 = DCFEstimatorModel(weights, centers, variant=2, use_linear=True,
    ...                            is_convex=True, is_symmetrized=False)
    >>> np.round(dcf_predict(model1, X), decimals=4)
    array([-2.3494, -0.6013, -0.4297, -1.0141, -1.5567,  0.2401, -0.4761,
            4.035 , -1.0556,  0.4878])
    >>> yhat, model1r = dcf_predict(model1, X, return_used_weights=True)
    >>> model1r is None
    True

    ### Non-symmetric DCF_2 model (not dropping weights):
    >>> model2 = DCFEstimatorModel(weights, centers, variant=2, use_linear=True,
    ...                            is_convex=False, is_symmetrized=False)
    >>> np.round(dcf_predict(model2, X), decimals=4)
    array([-2.3494, -0.6013, -0.4297, -1.0141, -1.5567,  0.2401, -0.4761,
            4.035 , -1.0556,  0.4878])
    >>> yhat, model2r = dcf_predict(model2, X, return_used_weights=True)
    >>> model2r is None
    True

    ### Non-symmetric DCF_1 model (not dropping weights):
    >>> model3 = DCFEstimatorModel(weights, centers, variant=1, use_linear=True,
    ...                            is_convex=False, is_symmetrized=False)
    >>> np.round(dcf_predict(model3, X), decimals=4)
    array([-2.2187, -0.6013, -0.4297, -0.6724, -1.3059,  0.8648, -0.4761,
            5.3908, -0.1792,  1.0596])
    >>> yhat, model3r = dcf_predict(model3, X, return_used_weights=True)
    >>> model3r is None
    True

    ### Non-symmetric MMA model (dropping weights):
    >>> weights.shape
    (3, 6)
    >>> weights3i = weights.copy()
    >>> weights3i[:, -1] = np.minimum(weights3i[:, -1], 0.0)
    >>> np.round(weights3i[:, -1], decimals=6)
    array([-0.608107, -0.45914 ,  0.      ])
    >>> model3i = DCFEstimatorModel(weights3i, centers, variant=np.inf, use_linear=True,
    ...                             is_convex=False, is_symmetrized=False)
    >>> yhat3i = dcf_predict(model3i, X)
    >>> np.round(yhat3i, decimals=4)
    array([-2.2208, -0.6013, -0.4297, -1.5529, -1.1635, -0.5079, -0.4761,
            2.6081, -2.0862,  0.6363])
    >>> model3mma = model3i.to_mma()
    >>> model3mma.centers is None
    True
    >>> model3mma.weights.shape
    (3, 8, 5)
    >>> np.round(np.sum(np.abs(yhat3i - dcf_predict(model3mma, X))), decimals=10)
    0.0
    >>> yhat, model3mmar = dcf_predict(model3mma, X, return_used_weights=True)
    >>> np.round(np.sum(np.abs(yhat3i - yhat)), decimals=10)
    0.0
    >>> model3mmar.weights.shape
    (3, 6, 5)
    >>> yhatr, model_none = dcf_predict(model3mmar, X, return_used_weights=True)
    >>> np.round(np.sum(np.abs(yhat3i - yhatr)), decimals=10)
    0.0
    >>> model_none is None
    True

    ### Non-symmetric DCF_inf model (not dropping weights):
    >>> model4 = DCFEstimatorModel(weights, centers, variant=np.inf, use_linear=True,
    ...                            is_convex=False, is_symmetrized=False)
    >>> np.round(dcf_predict(model4, X), decimals=4)
    array([-2.2208, -0.6013, -0.4297, -1.1126, -1.1635,  0.0113, -0.4761,
            3.4604, -1.3121,  0.6363])
    >>> yhat, model4r = dcf_predict(model4, X, return_used_weights=True)
    >>> model4r is None
    True

    ### Symmetric DCF_inf model (dropping weights):
    >>> weights2 = np.hstack([weights, weights])
    >>> model5 = DCFEstimatorModel(weights2, centers, variant=np.inf, use_linear=True,
    ...                            is_convex=False, is_symmetrized=True)
    >>> yhat5 = dcf_predict(model5, X)
    >>> np.round(yhat5, decimals=4)
    array([-1.5616, -0.0836,  1.386 ,  0.8727,  0.9478, -1.6409, -1.6886,
           -1.149 ,  2.1964, -2.5135])
    >>> yhat5r, model5r = dcf_predict(model5, X, return_used_weights=True)
    >>> model5.weights.shape, len(model5r.weights), model5r.is_symmetrized
    ((3, 12), 2, ([0, 2], [1, 2]))
    >>> np.round(np.sum(np.abs(yhat5r - yhat5)), decimals=10)
    0.0
    >>> yhat5rr, model5rr = dcf_predict(model5r, X, return_used_weights=True)
    >>> model5rr is None
    True
    >>> np.round(np.sum(np.abs(yhat5rr - yhat5)), decimals=10)
    0.0

    ### Symmetric MMA model (dropping weights):
    >>> model5i = copy.deepcopy(model5)
    >>> model5i.weights[:, -2:] = np.minimum(0.0, model5.weights[:, -2:])
    >>> yhat5i = dcf_predict(model5i, X)
    >>> np.round(yhat5i, decimals=4)
    array([-0.7494, -0.0836,  2.234 ,  1.0732,  1.767 , -1.6409, -1.0217,
           -1.149 ,  2.7114, -1.7974])
    >>> model5immas = model5i.to_mma()
    >>> np.round(np.sum(np.abs(yhat5i - dcf_predict(model5immas, X))), decimals=10)
    0.0
    >>> yhat, model5immasr = dcf_predict(model5immas, X, return_used_weights=True)
    >>> np.round(np.sum(np.abs(yhat5i - yhat)), decimals=10)
    0.0
    >>> model5immasr.weights[0].shape, model5immasr.weights[1].shape
    ((2, 3, 5), (2, 5, 5))
    >>> yhatr, model5immasrr = dcf_predict(model5immasr, X, return_used_weights=True)
    >>> model5immasrr is None
    True
    >>> np.round(np.sum(np.abs(yhat5i - yhatr)), decimals=10)
    0.0

    ### Non-symmetric DCF_2 model (dropping weights):
    >>> uweights = np.vstack([weights, np.array([[-10., 0., 0., 0., 0., 0.]])])
    >>> ucenters = np.vstack([centers, centers[-1:, :]])
    >>> model6 = DCFEstimatorModel(uweights, ucenters, variant=2, use_linear=True,
    ...                            is_convex=False, is_symmetrized=False)
    >>> yhat6 = dcf_predict(model6, X)
    >>> np.round(yhat6, decimals=4)
    array([-2.3494, -0.6013, -0.4297, -1.0141, -1.5567,  0.2401, -0.4761,
            4.035 , -1.0556,  0.4878])
    >>> yhat6r, model6r = dcf_predict(model6, X, return_used_weights=True)
    >>> model6.weights.shape, model6r.weights.shape
    ((4, 6), (3, 6))
    >>> np.round(np.sum(np.abs(yhat6r - yhat6)), decimals=10)
    0.0

    ### Non-symmetric DCF_+ model (not dropping weights)
    >>> pweights = np.hstack([weights0, np.random.randn(weights0.shape[0], weights0.shape[1]-1)])
    >>> model7 = DCFEstimatorModel(pweights, centers, variant='+', use_linear=True,
    ...                            is_convex=False, is_symmetrized=False)
    >>> yhat7 = dcf_predict(model7, X)
    >>> np.round(yhat7, decimals=4)
    array([-1.7012, -0.6013, -0.4297, -0.993 , -1.8977,  0.0994, -0.4761,
            2.7581,  0.0151,  1.207 ])
    >>> yhat7r, model7r = dcf_predict(model7, X, return_used_weights=True)
    >>> model7r is None
    True

    ### Non-symmetric DCF_+ model (dropping weights)
    >>> upweights = np.vstack([pweights, pweights[2:3, :]])
    >>> upcenters = np.vstack([centers, centers[2:3, :]])
    >>> model8 = DCFEstimatorModel(upweights, upcenters, variant='+', use_linear=True,
    ...                            is_convex=False, is_symmetrized=False)
    >>> yhat8 = dcf_predict(model8, X)
    >>> np.round(yhat8, decimals=4)
    array([-1.7012, -0.6013, -0.4297, -0.993 , -1.8977,  0.0994, -0.4761,
            2.7581,  0.0151,  1.207 ])
    >>> yhat8r, model8r = dcf_predict(model8, X, return_used_weights=True)
    >>> model8.weights.shape, model8r.weights.shape
    ((4, 9), (3, 9))
    >>> np.round(np.sum(np.abs(yhat8r - yhat8)), decimals=10)
    0.0
    >>> yhat8rr, model8rr = dcf_predict(model8r, X, return_used_weights=True)
    >>> model8rr is None
    True
    >>> np.round(np.sum(np.abs(yhat8rr - yhat8)), decimals=10)
    0.0
    """
    weights, X = prepare_prediction(model, X, extend_X1=False)
    n, d = X.shape
    centers = model.centers
    variant = str(model.variant)
    splits = list(range(0, n, nsplit))
    splits = zip(splits, splits[1:] + [n])
    symm = model.is_symmetrized
    is_mma = model.is_mma()
    if return_used_weights:
        ind = _init_used_ind(symm, is_mma)
    if symm:
        if isinstance(weights, tuple):
            w1, w2 = weights
        else:
            w1 = weights[:, 0::2]
            w2 = weights[:, 1::2]
    else:
        w1 = w2 = None
    yhat = np.zeros(n)
    for (split_start, split_end) in splits:
        Xsplit = X[split_start:split_end, :]
        if is_mma:
            if symm:
                yh1s, id1s = _mma_predict(Xsplit, w1, return_used_weights)
                yh2s, id2s = _mma_predict(Xsplit, w2, return_used_weights)
                yhat[split_start:split_end] = yh1s - yh2s
                if return_used_weights:
                    ind = (_add_used_inds(ind[0], id1s),
                           _add_used_inds(ind[1], id2s))
            else:
                yhat[split_start:split_end], ids = (
                    _mma_predict(Xsplit, weights, return_used_weights))
                if return_used_weights:
                    ind = _add_used_inds(ind, ids)
        else:
            if symm:
                if isinstance(symm, bool):
                    phi = _dcf_calc_phi(Xsplit, centers, variant, model.use_linear)
                    Xw1 = _dcf_calc_Xw(phi, w1)
                    Xw2 = _dcf_calc_Xw(phi, w2)
                else:
                    phi = _dcf_calc_phi(Xsplit, centers[symm[0]], variant, model.use_linear)
                    Xw1 = _dcf_calc_Xw(phi, w1)
                    phi = _dcf_calc_phi(Xsplit, centers[symm[1]], variant, model.use_linear)
                    Xw2 = _dcf_calc_Xw(phi, w2)
                yhat[split_start:split_end] = np.max(Xw1, axis=1) - np.max(Xw2, axis=1)
                if return_used_weights:
                    ind = _add_used_inds(ind, (np.argmax(Xw1, axis=1), np.argmax(Xw2, axis=1)))
            else:
                phi = _dcf_calc_phi(Xsplit, centers, variant, model.use_linear)
                Xw = _dcf_calc_Xw(phi, weights)
                yhat[split_start:split_end] = np.max(Xw, axis=1)
                if return_used_weights:
                    ind = _add_used_inds(ind, np.argmax(Xw, axis=1))
    res = postprocess_prediction(model, yhat)
    if return_used_weights:
        res_model = _reduce_model(model, weights, w1, w2, ind, is_mma, symm)
        res = (res, res_model)

    return res


def _reduce_model(model, weights, w1, w2, ind, is_mma, symm):
    nparams = model.get_nparams(include_centers=True)
    res_model = None
    if is_mma:
        W = copy.deepcopy(weights)
        if symm:
            if (W[0].shape[0] > len(ind[0][0]) or W[0].shape[1] > len(ind[0][1])
                    or W[1].shape[0] > len(ind[1][0]) or W[1].shape[1] > len(ind[1][1])):
                ind = ((sorted(ind[0][0]), sorted(ind[0][1])),
                       (sorted(ind[1][0]), sorted(ind[1][1])))
                res_model = copy.deepcopy(model)
                res_model.weights = (W[0][ind[0][0], :, :][:, ind[0][1], :],
                                     W[1][ind[1][0], :, :][:, ind[1][1], :])
        else:
            if W.shape[0] > len(ind[0]) or W.shape[1] > len(ind[1]):
                ind = (sorted(ind[0]), sorted(ind[1]))
                res_model = copy.deepcopy(model)
                res_model.weights = W[ind[0], :, :][:, ind[1], :]
    else:
        if symm:
            if w1.shape[0] > len(ind[0]) or w2.shape[0] > len(ind[1]):
                ind = (sorted(ind[0]), sorted(ind[1]))
                res_model = copy.deepcopy(model)
                res_model.is_symmetrized = tuple(ind)
                res_model.weights = (w1[ind[0], :], w2[ind[1], :])
        else:
            if weights.shape[0] > len(ind):
                ind = sorted(ind)
                res_model = copy.deepcopy(model)
                weights = res_model.weights[ind, :]
                res_model.weights = weights
                res_model.centers = model.centers[ind, :]
    if res_model is not None:
        res_nparams = res_model.get_nparams(include_centers=True)
        assert res_nparams < nparams, f"res_nparams:{res_nparams}, nparams:{nparams}"
    return res_model


def _init_used_ind(symm, is_mma):
    if is_mma:
        return ((set(), set()), (set(), set())) if symm else (set(), set())
    return (set(), set()) if symm else set()


def _add_used_inds(ind, ids):
    if isinstance(ind, set):
        return ind.union(set(ids))
    return (ind[0].union(set(ids[0])),
            ind[1].union(set(ids[1])))


def _mma_predict(X, W, return_used_weights):
    XW = W[:, :, 1:].dot(X.T) + W[:, :, 0][:, :, None]
    inds = None
    XWmin = XW.min(axis=1)
    if return_used_weights:
        ids = XWmin.argmax(axis=0)
        inds = (ids, XW[ids, :, :].argmin(axis=1).ravel())
    return XWmin.max(axis=0), inds


def get_kmeans_objval(X, partition, centers):
    return np.sqrt(np.sum([
        np.sum(np.sum(np.square(X[cell_idx, :] - centers[center_idx, :]), axis=1))
        for center_idx, cell_idx in enumerate(partition.cells)
    ]) / X.shape[0])


def get_dcf_partition(X, ntrials=1, q=1, dist=euclidean_distance,
                      min_cell_size=0, minimizeK=False,
                      kmeans_objval=False, kmeans_kwargs=None):
    partition = None
    objval = np.inf
    for trial in range(ntrials):
        _partition, _center_idxs = adaptive_farthest_point_clustering(
            data=X, q=q, return_center_idxs=True, dist=dist,
            first_idx=(FPC_FIRST_IDX__DEFAULT if trial == 0
                       else FPC_FIRST_IDX__RANDOM),
        )
        _centers = X[_center_idxs, :]
        if kmeans_kwargs is not None:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=_centers.shape[0],
                            init=_centers,
                            **dict(kmeans_kwargs)).fit(X)
            _centers = kmeans.cluster_centers_
            _partition = voronoi_partition(_centers, X)
        if min_cell_size > 0:
            cells = [list(c) for c in _partition.cells]
            cntrs = np.copy(_centers)
            while True:
                assert len(cells) == cntrs.shape[0]
                idx = sorted(range(len(cells)), key=lambda k: len(cells[k]))
                smallest = idx[0]
                if len(cells) <= 1 or len(cells[smallest]) >= min_cell_size:
                    break
                dropped_cell = cells[smallest]
                cells = cells[:smallest] + cells[smallest+1:]
                cntrs = np.delete(cntrs, smallest, 0)
                assert len(cells) == cntrs.shape[0]
                cidx = find_closest_centers(X[dropped_cell, :], cntrs, dist=dist)
                for i, k in zip(dropped_cell, cidx):
                    cells[k].append(i)
            cells = [np.array(sorted(c), dtype=int) for c in cells]
            _partition = Partition(X.shape[0], len(cells), cells)
            _centers = cntrs
        if kmeans_objval:
            _objval = get_kmeans_objval(X, _partition, _centers)
        else:
            _objval = max(cell_radii(X, _partition, _centers))
        if minimizeK:
            if (partition is None or _partition.ncells < partition.ncells
                    or (_partition.ncells == partition.ncells and _objval < objval)):
                partition = _partition
                centers = _centers
                objval = _objval
        elif _objval < objval:
            partition = _partition
            centers = _centers
            objval = _objval
    partition.assert_consistency()
    assert centers.shape[1] == X.shape[1]
    assert centers.shape[0] == partition.ncells
    return partition, centers, objval


def _check_input(logger, verbose, X, y, variant,
                 is_symmetrized, is_convex, use_linear):
    n, d = X.shape
    assert len(y) == n
    assert n > d
    if len(y.shape) > 1:
        assert len(y.shape) == 2 and y.shape[1] == 1
        y = y.ravel()
    if verbose > 0:
        logger(f'DCF({variant}), TRAIN, n: {n}, d: {d}')
    variant = str(variant)
    assert variant[-1] != 'w' or not is_symmetrized, \
        'Symmetrization is not supported for *w models!'
    assert variant[0] != '+' or len(variant) == 1, \
        'Model modifiers (q, w, etc...) are not supported for + models!'
    assert use_linear or not is_convex, \
        'Option use_linear=False is only supported for is_convex=False!'
    assert use_linear or (variant[-1] not in ('q', 'w')), \
        'Option use_linear=False is not supported for variant: '+variant+'!'
    assert use_linear or variant != '+', \
        'Option use_linear=False is not possible for variant="+"!'
    is_mma = (variant == 'mma')
    if is_mma:
        variant = 'inf'
    return y, variant, is_mma


def _parse_params(logger, variant, X, y, K, verbose, max_epsilon, avg_epsilon,
                  L_regularizer, L_regularizer_offset, L_sum_regularizer,
                  bias_regularizer, local_opt_maxiter, local_opt_L_regularizer_offset):
    n, d = X.shape
    K  # might be used in the evals
    x_radius = get_data_radius(X)
    y_radius = np.max(np.abs(y - np.mean(y)))
    if verbose > 0:
        logger(f'DCF({variant}), DATA, n: {n}, d: {d}'
               f', x_radius: {x_radius:.4f}, y_radius: {y_radius:.4f}')
    scale_L_sum_regularizer_by_rec_cell_size = False
    if isinstance(L_regularizer, str):
        L_regularizer = eval(L_regularizer)
    if isinstance(L_regularizer_offset, str):
        L_regularizer_offset = eval(L_regularizer_offset)
    if isinstance(bias_regularizer, str):
        bias_regularizer = eval(bias_regularizer)
    if isinstance(L_sum_regularizer, str):
        if L_sum_regularizer.endswith('/I_k'):
            scale_L_sum_regularizer_by_rec_cell_size = True
            L_sum_regularizer = L_sum_regularizer[:-4]
        L_sum_regularizer = eval(L_sum_regularizer)
    if isinstance(local_opt_maxiter, str):
        local_opt_maxiter = eval(local_opt_maxiter)
    if isinstance(local_opt_L_regularizer_offset, str):
        local_opt_L_regularizer_offset = eval(local_opt_L_regularizer_offset)

    return (
        L_regularizer, L_regularizer_offset, L_sum_regularizer,
        bias_regularizer, local_opt_maxiter, local_opt_L_regularizer_offset,
        scale_L_sum_regularizer_by_rec_cell_size,
    )


def _scaling(logger, verbose, variant, normalize, X, y, centers,
             L_regularizer, L_regularizer_offset,
             bias_regularizer, L_sum_regularizer):
    xmean = None
    xscale = None
    ymean = None
    yscale = None
    if normalize:
        xmean = np.mean(X, axis=0)
        X = X - xmean
        centers = centers - xmean
        xscale = np.linalg.norm(X, ord='fro') / np.sqrt(X.shape[0])
        X /= xscale
        centers /= xscale
        ymean = np.mean(y)
        y = y - ymean
        yscale = np.std(y)
        y /= yscale
        if L_regularizer_offset is not None:
            L_regularizer_offset *= (xscale / yscale)
        if L_regularizer is not None:
            L_regularizer /= (xscale ** 2)
        if L_sum_regularizer is not None:
            L_sum_regularizer /= (xscale ** 2)
        if bias_regularizer is not None:
            bias_regularizer /= yscale
    if verbose > 0:
        logger(f'DCF({variant}), PARAMS, xscale: {xscale or 1.0:.4f}, yscale: {yscale or 1.0:.4f}'
               f', L_regularizer_offset: {L_regularizer_offset or 0.0:.4f}'
               f', L_regularizer: {L_regularizer or 0.0:.4f}'
               f', L_sum_regularizer: {L_sum_regularizer or 0.0:.4f}'
               f', bias_regularizer: {bias_regularizer or 0.0}')
    return (
        xmean, xscale, ymean, yscale, X, y, centers,
        L_regularizer, L_regularizer_offset,
        bias_regularizer, L_sum_regularizer,
    )


def _dcf_socp_solve(X, y, partition, centers, variant, is_convex, is_mma, is_symmetrized,
                    use_linear, L_regularizer, L_regularizer_offset,
                    bias_regularizer, L_sum_regularizer, backend,
                    logger, verbose, warn_on_nok_weights,
                    socp_params, scale_L_sum_regularizer_by_rec_cell_size):
    n, d = X.shape
    K = partition.ncells
    t_start = timer()
    H, g, A, b, cones, scaled_nparams_pwf, nparams_reg = dcf_socp_data(
        X, y, partition, centers, variant, is_convex, is_mma, is_symmetrized,
        use_linear, L_regularizer, L_regularizer_offset,
        bias_regularizer, L_sum_regularizer, backend, verbose=verbose,
        scale_L_sum_regularizer_by_rec_cell_size=scale_L_sum_regularizer_by_rec_cell_size,
    )
    socp_data_seconds = timer() - t_start
    if verbose > 0:
        logger(f'DCF({variant}), SOLVE, nvars: {len(g)}'
               f', nineqcons: {cones[0][-1] if isinstance(cones[0], tuple) else cones[0].dim}'
               f', nsoccons: {len(cones)-1}, etime: {socp_data_seconds:.1f}s')
    socp_params = dict(socp_params)
    if 'verbose' not in socp_params:
        socp_params['verbose'] = (verbose > 2)
    socp_result = socp_solve(
        H, g, A, b, cones,
        backend=backend, **socp_params
    )
    socp_solve_second = timer() - t_start - socp_data_seconds
    if verbose > 0:
        logger(f'DCF({variant}), SOLVED, etime: {timer()-t_start:.1f}s')
    weights = socp_result.primal_soln
    obj_val = (
        0.5 * (weights.dot(H.dot(weights)) + weights.dot(triu(H, 1).dot(weights)))
        + g.dot(weights)
        + (0.5/n) * y.dot(y)
    )

    reg_var_value = None
    if L_regularizer is not None:
        reg_var_value = weights[-1]
    if nparams_reg > 0:
        weights = weights[:-nparams_reg]
    extra_weights = None
    if variant[-1] in ('w',):
        extra_weights = weights[-1]
        weights = weights[:-1]
    weights = np.reshape(weights, (K, -1))
    assert centers.shape[0] == weights.shape[0]
    weights, extra_weights = _check_cvx_mma_weights(
        weights, extra_weights, d,
        is_convex, is_mma, is_symmetrized,
        variant, warn_on_nok_weights,
    )
    if extra_weights is not None:
        weights = (weights, extra_weights)
    model = DCFEstimatorModel(
        weights=weights, centers=centers,
        variant=variant, use_linear=use_linear,
        is_convex=is_convex, is_symmetrized=is_symmetrized,
    )
    if is_mma:
        model = model.to_mma()
    yhat = dcf_predict(model, X)
    proj_loss = 0.5 * np.mean(np.square(y - yhat))
    proj_obj_val = proj_loss + 0.5 * _dcf_reg(
        weights,
        is_symmetrized=is_symmetrized,
        L_regularizer=L_regularizer,
        L_regularizer_offset=L_regularizer_offset,
        L_sum_regularizer=L_sum_regularizer,
    )
    socp_stats = DCFSOCPStats(
        reg_var_value=reg_var_value,
        obj_val=obj_val,
        proj_obj_val=proj_obj_val,
        niterations=socp_result.niterations,
        runtime_data=socp_data_seconds,
        runtime_solve=socp_solve_second,
        nparams_wo_centers=model.get_nparams(include_centers=False),
        nparams_w_centers=model.get_nparams(include_centers=True),
    )
    if verbose >= 2:
        train_l2 = np.mean(np.square(yhat - y))
        logger(f'obj_val: {obj_val:.4f}, '
               f'socp_stats: {socp_stats}, '
               f'train_l2: {train_l2:.4f}, '
               f'y.dot.y: {y.dot(y)/n:.4f}')
    return model, proj_loss, socp_stats


def dcf_train(
    X, y, variant, is_convex, is_symmetrized,
    use_L=False, L=None,
    L_regularizer='max(1.0, x_radius)**2 * (d*K/n)',  # theta_1
    L_regularizer_offset='(y_radius/x_radius)*np.log(n)',  # theta_3
    bias_regularizer=0.0,
    L_sum_regularizer='(x_radius/n)**2',  # theta_2
    normalize=True,
    negate_y=False,
    centers=None,
    use_linear=True,
    local_opt_lbfgs_memsize=10,
    local_opt_maxiter=10000,
    local_opt_type='smooth',
    local_opt_noise_level=0.0,
    local_opt_L_regularizer_offset='np.log(n)',  # theta_4
    afpc_ntrials=1,
    afpc_q=1,
    afpc_min_cell_size=0,
    afpc_minimizeK=False,
    kmeans_objval=False,
    kmeans_kwargs=None,
    verbose=0,
    logger=print,
    warn_on_nok_weights=False,
    backend=SOCP_BACKEND__LBFGS, socp_params={},
):
    """Training a DCF estimator.

    :param X: data matrix (each row is a sample)
    :param y: target vector
    :param variant: TPDC algorithm variant
    :param is_convex: True/False for convex/Lipschitz regression, respectively
    :param is_symmetrized: True/False for symmetrized or non-symmetrized representation
    :param use_L: not used (kept for compatibility with evaluation framework)
    :param L: not used (kept for compatibility with evaluation framework)
    :param L_regularizer: soft constraint scaler on Lipschitz constant (max-grad)
    :param L_regularizer_offset: until this value the regularization should be zero
    :param bias_regularizer: regularizer on the sum of the bias weights
    :param L_sum_regularizer: regularizer on the sum of the Lipschitz constants
    :param negate_y: whether or not the sign on the response variable y is flipped
    :param centers: providing the Voronoi partition centers or their indices (skips AFPC)
    :param use_linear: whether or not to use a linear term (needed if is_convex=True)
    :param afpc_ntrials: number of trials to get the best clustering by AFPC
    :param afpc_q: scaling parameter for AFPC stopping rule
    :param kmeans_kwargs: turns on k-means postprocessing the AFPC partition;
           can be any object which is convertible to a dict;
           passing () turns the feature on with default Scikit-learn K-Means parameters.
    :param verbose: whether to print verbose output
    :param warn_on_nok_weights: warns if weights are not ok (only applie is is_mma or is_convex is set to True)
    :param backend: SOCP solver backend
    :param socp_params: extra SOCP solver parameters
    """
    n, d = X.shape
    y, variant, is_mma = _check_input(logger, verbose, X, y, variant,
                                      is_symmetrized, is_convex, use_linear)

    t_start = timer()
    if centers is None:
        if isinstance(afpc_min_cell_size, str):
            afpc_min_cell_size = eval(afpc_min_cell_size)
        partition, centers, max_epsilon = get_dcf_partition(
            X=X, ntrials=afpc_ntrials, q=afpc_q,
            min_cell_size=afpc_min_cell_size, minimizeK=afpc_minimizeK,
            kmeans_objval=kmeans_objval, kmeans_kwargs=kmeans_kwargs)
    else:
        if np.issubdtype(centers.dtype, np.integer) and len(centers.shape) == 1:
            centers = X[centers, :]
        partition = voronoi_partition(centers, X)
        max_epsilon = max(cell_radii(X, partition, centers))
    avg_epsilon = get_kmeans_objval(X, partition, centers)
    K = partition.ncells
    if verbose > 0:
        logger(f'DCF({variant}), CLUSTER, K: {K}'
               f', max_epsilon: {max_epsilon:.2f}, avg_epsilon: {avg_epsilon:.2f}')
    clustering_stats = DCFClusteringStats(
        ncells=K,
        max_epsilon=max_epsilon,
        avg_epsilon=avg_epsilon,
        runtime=timer() - t_start,
    )

    (L_regularizer, L_regularizer_offset, L_sum_regularizer,
     bias_regularizer, local_opt_maxiter, local_opt_L_regularizer_offset,
     scale_L_sum_regularizer_by_rec_cell_size) = _parse_params(
         logger, variant, X, y, K, verbose, max_epsilon, avg_epsilon,
         L_regularizer, L_regularizer_offset, L_sum_regularizer,
         bias_regularizer, local_opt_maxiter, local_opt_L_regularizer_offset,
    )
    del max_epsilon
    del avg_epsilon

    (xmean, xscale, ymean, yscale, X, y, centers,
     L_regularizer, L_regularizer_offset,
     bias_regularizer, L_sum_regularizer) = _scaling(
         logger, verbose, variant, normalize, X, y, centers,
         L_regularizer, L_regularizer_offset,
         bias_regularizer, L_sum_regularizer)

    if negate_y:
        yscale = -1.0 if yscale is None else -yscale
        y = -y

    model, proj_loss, socp_stats = _dcf_socp_solve(
        X, y, partition, centers, variant, is_convex, is_mma, is_symmetrized,
        use_linear, L_regularizer, L_regularizer_offset,
        bias_regularizer, L_sum_regularizer, backend,
        logger, verbose, warn_on_nok_weights,
        socp_params, scale_L_sum_regularizer_by_rec_cell_size)

    local_opt_stats = None
    maxL = model.get_maxL()
    if local_opt_maxiter > 0 and maxL > 1e-8:
        weights = model.weights
        if verbose >= 2:
            logger(f"DCF({variant}), OPTIM, maxL:{maxL:.4f}")

        L_regularizer = (
            (proj_loss + L_sum_regularizer * model.get_sumsqL()) / (maxL ** 2)
        )
        L_regularizer_offset = local_opt_L_regularizer_offset * maxL
        local_opt_reg_func = partial(
            _dcf_reg,
            is_symmetrized=is_symmetrized,
            L_regularizer=L_regularizer,
            L_regularizer_offset=L_regularizer_offset,
            L_sum_regularizer=L_sum_regularizer,
        )
        init_obj_val = proj_loss + local_opt_reg_func(weights)

        symm = model.is_symmetrized
        model, local_opt_stats = _local_optim(
            local_opt_type, local_opt_maxiter,
            local_opt_lbfgs_memsize, local_opt_noise_level,
            proj_loss, model, X, y, K,
            L_sum_regularizer, L_regularizer, L_regularizer_offset,
            verbose, logger, warn_on_nok_weights,
        )
        assert model.is_symmetrized == symm
        model = dcf_fix_bias_offset(model, X, y)
        yhat, _model = dcf_predict(model, X, return_used_weights=True)
        if _model is not None:
            model = _model
        opt_l2_loss = np.mean(np.square(yhat - y))
        opt_obj_val = 0.5 * (opt_l2_loss + local_opt_reg_func(weights))
        if verbose > 0:
            logger(f'DCF({variant}), OPTIM: {local_opt_stats}')
        assert opt_obj_val < init_obj_val + 1e-8, \
            f'opt_obj_val: {opt_obj_val:.6f}, init_obj_val: {init_obj_val:.6f}'

    return DCFEstimatorModel(
        weights=model.weights, centers=model.centers,
        is_convex=model.is_convex, is_symmetrized=model.is_symmetrized,
        variant=model.variant, use_linear=model.use_linear,
        clustering_stats=clustering_stats,
        socp_stats=socp_stats,
        local_opt_stats=local_opt_stats,
        xmean=xmean, xscale=xscale, ymean=ymean, yscale=yscale,
    )


def _check_cvx_mma_weights(weights, extra_weights, d,
                           is_convex, is_mma, is_symmetrized,
                           variant, warn_on_nok_weights):
    """
    >>> np.set_printoptions(legacy='1.25')

    >>> d = 3
    >>> w1 = -np.ones((5, 2*d, d+1))
    >>> w1[1, 1, 3] = 0.001
    >>> np.max(w1.ravel())
    0.001
    >>> np.max(w1[:, :, -1].ravel())
    0.001
    >>> w2 = _check_cvx_mma_weights(w1, None, d, False, True, False, 'mma', False)[0]
    >>> np.max(w2.ravel())
    0.0

    >>> w3 = -np.ones((5, 2*d, 2*(d+1)))
    >>> w3[1, 1, 2*d] = 0.001
    >>> w3[2, 2, 2*d+1] = 0.002
    >>> np.max(w3.ravel())
    0.002
    >>> w4 = _check_cvx_mma_weights(w3, None, d, False, True, True, 'mma', False)[0]
    >>> np.max(w4.ravel())
    0.0
    """
    assert (not is_convex) or (not is_mma)
    assert not (is_convex and is_symmetrized)
    if is_convex:
        if variant == '+':
            d1 = d+1
            wsum = weights[:, 1:d1] + weights[:, d1:]
            if np.min(wsum) < 0.0:
                if warn_on_nok_weights:
                    warnings.warn(f'Non-convex weights for convex regression: {np.min(wsum)}')
                mask = wsum < 0.0
                offset = 0.5 * (weights[:, 1:d1][mask] + weights[:, d1:][mask])
                weights[:, 1:d1][mask] -= offset
                weights[:, d1:][mask] -= offset
        else:
            shifts = range(1, 2+int(variant.endswith('q')))
            for shift in shifts:
                if np.min(weights[:, -shift]) < 0.0:
                    if warn_on_nok_weights:
                        warnings.warn(f'Negative weights for convex regression:'
                                      f' {np.min(weights[:, -shift])}')
                    weights[:, -shift] = np.maximum(0.0, weights[:, -shift])
            if extra_weights is not None and min(extra_weights) < 0:
                warnings.warn(f'Negative extra_weights for convex regression:'
                              f' {np.min(extra_weights)}')
                extra_weights = np.maximum(0.0, extra_weights)
    elif is_mma:
        shifts = range(1, 2+int(is_symmetrized))
        for shift in shifts:
            slicer = (slice(None),) * (weights.ndim - 1) + (-shift,)
            if np.max(weights[slicer].ravel()) > 0.0:
                if warn_on_nok_weights:
                    warnings.warn(f'Positive weights for MMA regression:'
                                  f' {np.max(weights[slicer].ravel())}')
                weights[slicer] = np.minimum(0.0, weights[slicer])
    return weights, extra_weights


def _local_optim(opt_type, maxiter, lbfgs_memsize, noise_level,
                 init_loss, model, X, y, K,
                 L_sum_regularizer, L_regularizer, L_regularizer_offset,
                 verbose, logger, warn_on_nok_weights):
    opt_start = timer()
    d = X.shape[1]

    if model.is_mma():
        from ai.gandg.algorithm.dcf.optim_task import SmoothMaxMinAffineOptimTask
        if model.is_symmetrized:
            init_weights = np.concatenate([
                model.weights[0].ravel(),
                model.weights[1].ravel(),
            ])
        else:
            init_weights = model.weights.flatten()
        task = SmoothMaxMinAffineOptimTask(
            is_symmetrized=model.is_symmetrized,
            X=X, y=y, K=K,
            L_regularizer=L_regularizer,
            L_regularizer_offset=L_regularizer_offset,
            L_sum_regularizer=L_sum_regularizer,
            verbose=(verbose > 1),
        )
    else:
        phi = _dcf_calc_phi(X, model.centers, model.variant, model.use_linear)
        init_weights = model.weights.flatten()
        if opt_type == 'nonsmooth':
            from ai.gandg.algorithm.dcf.optim_task import NonSmoothDCFLocalOptimTask
            task = NonSmoothDCFLocalOptimTask(
                centers=model.centers, variant=model.variant,
                is_convex=model.is_convex, is_symmetrized=model.is_symmetrized,
                y=y, phi=phi,
                L_regularizer=L_regularizer,
                L_regularizer_offset=L_regularizer_offset,
                L_sum_regularizer=L_sum_regularizer,
                verbose=(verbose > 1), noise_level=noise_level,
            )
        elif opt_type == 'smooth':
            from ai.gandg.algorithm.dcf.optim_task import SmoothDCFLocalOptimTask
            task = SmoothDCFLocalOptimTask(
                centers=model.centers, variant=model.variant,
                is_convex=model.is_convex, is_symmetrized=model.is_symmetrized,
                y=y, phi=phi,
                L_regularizer=L_regularizer,
                L_regularizer_offset=L_regularizer_offset,
                L_sum_regularizer=L_sum_regularizer,
                verbose=(verbose > 1), smooth_tol=1e-6,
            )
        else:
            raise NotImplementedError('Not supported opt_type: {opt_type}!')

    from ai.gandg.optim.lbfgs import LBFGS
    opt_res = LBFGS(max_memory=lbfgs_memsize,
                    ftol=1e-6,
                    ftol_patience=20,
                    curve_tol=1e-15,
                    max_iter=maxiter).minimize(task, init_weights)

    weights = opt_res.x
    opt_time = timer() - opt_start

    init_fval = task.fun(init_weights)
    final_fval = task.fun(weights)
    assert init_fval >= final_fval, \
        (f'Increased optimization value, init_fval:{init_fval:.6f} < final_fval:{final_fval:.6f}'
         f', opt_res:\n{opt_res}')

    if verbose > 1:
        logger(f'DCF({model.variant}), OPTIM, status:{opt_res.status}'
               f', niter:{opt_res.niter}, nrestarts:{opt_res.nrestarts}')

    if model.is_mma():
        dims = (K, -1, 1+d)
        if model.is_symmetrized:
            wlen = int(0.5 * len(weights))
            weights = (np.reshape(weights[:wlen], dims),
                       np.reshape(weights[wlen:], dims))
        else:
            weights = np.reshape(weights, dims)
    else:
        weights = _check_cvx_mma_weights(np.reshape(weights, (K, -1)), None, d,
                                         model.is_convex, False, False,
                                         model.variant, warn_on_nok_weights)[0]

    model = DCFEstimatorModel(
        weights=weights, centers=model.centers,
        variant=model.variant, use_linear=model.use_linear,
        is_convex=model.is_convex, is_symmetrized=model.is_symmetrized,
    )
    if verbose > 1:
        logger(f"DCF({model.variant}), OPTIM, maxL:{model.get_maxL():.4f}")
    return model, DCFLocalOptStats(
        init_obj_val=init_fval,
        soln_obj_val=final_fval,
        opt_status=opt_res.status,
        niterations=opt_res.niter,
        nfun_evals=task.nfun_evals,
        ngrad_evals=task.njac_evals,
        runtime=opt_time,
    )


def dcf_phi(cellX, center, variant, use_linear):
    """Feature vector with bias term included."""
    diff = cellX - center
    if not use_linear:
        cellXe = np.linalg.norm(diff, axis=1,
                                ord=_get_norm_p(variant))
        return np.insert(cellXe[:, None], 0, 1.0, axis=1)
    d = cellX.shape[-1]
    d1 = d + 1
    nparams_pwf = _get_nparams_pwf(d, variant, use_linear)
    cellXe = np.empty((diff.shape[0], nparams_pwf))
    cellXe[:, 0] = 1.0  # bias
    cellXe[:, 1:d1] = diff
    if variant == '+':
        cellXe[:, d1:] = -cellXe[:, 1:d1]
        np.maximum(0.0, cellXe[:, 1:], out=cellXe[:, 1:])
    elif variant[-1] == 'q':
        norms = np.linalg.norm(cellXe[:, 1:d1], axis=1,
                               ord=_get_norm_p(variant))
        cellXe[:, d1] = norms
        cellXe[:, d1+1] = norms**2
    else:
        cellXe[:, d1] = np.linalg.norm(cellXe[:, 1:d1], axis=1,
                                       ord=_get_norm_p(variant))
    return cellXe


def _zero_square_mat(size):
    return coo_matrix(([], ([], [])), shape=(size, size))


def _stack_A(A, A_data, A_rows, A_cols, row_idx):
    if len(A_data) == 0:
        return A
    nvars = A.shape[1]
    if isinstance(A_data[0], np.ndarray):
        A_data = np.concatenate(A_data)
        A_rows = np.concatenate(A_rows)
        A_cols = np.concatenate(A_cols)
    return vstack([A, coo_matrix((A_data, (A_rows, A_cols)), shape=(row_idx, nvars)).tocsc()])


def _stack_b(b, b_mats):
    return np.hstack([b] + b_mats)


def _dcf_HgAb(
    X, y, partition, centers, variant, is_symmetrized, use_linear,
    L_regularizer, bias_regularizer, L_sum_regularizer,
    nparams_pwf, nparams_scaler, nparams_tot, nvars,
    scale_L_sum_regularizer_by_rec_cell_size,
):
    n = len(y)
    K = partition.ncells
    scaled_nparams_pwf = nparams_pwf * nparams_scaler

    regXX = None
    if bias_regularizer > 0.0 or L_sum_regularizer > 0.0:
        regXX = (n * L_sum_regularizer) * np.eye(scaled_nparams_pwf)
        regXX[0, 0] = n * bias_regularizer
        if is_symmetrized:
            regXX[1, 1] = regXX[0, 0]

    H_mats = []
    g_mats = []
    A = csc_matrix((0, nvars))
    b_mats = []
    Ks1 = K - 1
    nparams_range = np.arange(nparams_pwf)
    Ks1ones = np.ones((Ks1, 1), dtype=int)
    nparams_ones = np.ones(nparams_pwf, dtype=int)
    col_range_base = np.arange(nparams_pwf) * nparams_scaler
    row_range_base = np.kron(np.arange(K-1), np.ones(nparams_pwf + 1))
    row_idx = 0
    A_data = []
    A_rows = []
    A_cols = []
    for k, cell_k in enumerate(partition.cells):
        cell_size = len(cell_k)
        cellXk = dcf_phi(X[cell_k, :], centers[k, :], variant, use_linear)
        col_k = nparams_range + (k * nparams_pwf)
        if is_symmetrized:
            cellXke = np.empty((cell_size, 2*cellXk.shape[1]))
            cellXke[:, 0::2] = cellXk
            np.negative(cellXk, out=cellXk)
            cellXke[:, 1::2] = cellXk
        else:
            cellXke = cellXk
        cellXXke = np.dot(cellXke.transpose(), cellXke)
        if regXX is not None:
            if scale_L_sum_regularizer_by_rec_cell_size:
                cellXXke += regXX / cell_size
            else:
                cellXXke += regXX
        cellXy = np.dot(cellXke.transpose(), y[cell_k])
        cellXXke = triu(cellXXke).tocsc()
        H_mats.append(cellXXke)
        np.negative(cellXy, out=cellXy)
        g_mats.append(cellXy)

        cidx = np.concatenate([np.arange(k), np.arange(k+1, K)])
        data = np.hstack([
            dcf_phi(centers[k, :][None, :], centers[cidx, :], variant, use_linear),
            -Ks1ones,
        ]).flatten()
        col_pos = np.kron(cidx*scaled_nparams_pwf, nparams_ones).reshape(Ks1, -1)
        for shift in range(nparams_scaler):
            col_range = col_range_base + shift
            col_k = (k * scaled_nparams_pwf) + col_range[0]
            A_data.append(data)
            A_rows.append(row_idx + row_range_base)
            A_cols.append(np.hstack([
                col_pos + col_range,
                col_k * Ks1ones,
            ]).flatten())
            row_idx += Ks1
        if row_idx > 10000:
            A = _stack_A(A, A_data, A_rows, A_cols, row_idx)
            row_idx = 0
            A_data.clear()
            A_rows.clear()
            A_cols.clear()
            gc.collect()
    del cellXk
    if row_idx > 0:
        A = _stack_A(A, A_data, A_rows, A_cols, row_idx)
        del A_data
        del A_rows
        del A_cols
        gc.collect()
    if variant[-1] == 'w':
        H_mats.append(np.array([[np.sum(X**2)**2]]))
        # H matrix data is finalized later in _final_w_H
        g_mats.append(np.array([np.sum(y*np.sum(X**2, axis=1))]))

    if L_regularizer is not None:
        H_mats.append(np.array([[n * L_regularizer]]))
        g_mats.append(np.zeros(1))
    b_mats.append(np.zeros(A.shape[0]))
    b = _stack_b(np.array([]), b_mats)

    g = np.concatenate(g_mats)
    g /= n
    H = block_diag(H_mats, format='coo' if variant[-1] == 'w' else 'csc')
    if variant[-1] == 'w':
        H = _final_w_H(H, X, partition, centers, variant,
                       use_linear, nparams_tot, nparams_pwf * nparams_scaler)
    H /= n
    return H, g, A, b


def _add_cvx_or_mma_consraints_to_Ab(
    variant, K, d, nparams_pwf, A, b, cones,
    backend, is_mma, is_symmetrized,
):
    row_idx = 0
    A_data = []
    A_rows = []
    A_cols = []
    if variant == '+':
        assert not is_mma, 'DCF_+ models cannot be converted to MMA models!'
        # constraints: w[1:d1] >= -w[d1:]
        for k in range(K):
            A_data.append(-np.ones(2*d))
            rows = row_idx + np.arange(d)
            A_rows.append(rows)
            A_rows.append(rows)
            cols = k * nparams_pwf + 1 + np.arange(d)
            A_cols.append(cols)
            A_cols.append(cols + d)
            row_idx += d
        A_data = np.concatenate(A_data)
        A_rows = np.concatenate(A_rows)
        A_cols = np.concatenate(A_cols)
    else:
        if is_symmetrized:
            assert is_mma, 'Convex models cannot be symmetrized!'
            scaled_nparams_pwf = 2.0 * nparams_pwf
            for k in range(K):
                A_data += [1.0, 1.0]
                A_rows += [row_idx, row_idx+1]
                col = (k+1)*scaled_nparams_pwf
                A_cols += [col-2, col-1]
                row_idx += 2
        else:
            shifts = range(1, 2+int(variant[-1] == 'q'))
            sign = 1.0 if is_mma else -1.0
            # adding side constraint on the norm scaling variables
            for k in range(K):
                for shift in shifts:
                    A_data += [sign]
                    A_rows += [row_idx]
                    A_cols += [(k+1)*nparams_pwf - shift]
                    row_idx += 1
    A = _stack_A(A, A_data, A_rows, A_cols, row_idx)
    b = _stack_b(b, [np.zeros(row_idx)])
    cones = cones + [socp_nonnegative_cone(row_idx, backend)]
    return A, b, cones


def _add_regL_constraints_to_Ab(
    L_regularizer, L_regularizer_offset, A, b, cones, backend,
    K, nparams_pwf, nparams_scaler, nparams_reg, verbose,
):
    row_idx = 0
    A_data = []
    A_rows = []
    A_cols = []
    b_mats = []
    scaled_nparams_pwf = nparams_pwf * nparams_scaler
    col_Lreg = K * scaled_nparams_pwf + nparams_reg - 1
    reg_data = np.array([-1.0] + list(np.ones(nparams_pwf - 1)))
    row_range = np.arange(nparams_pwf)
    for shift in range(nparams_scaler):
        gc.collect()
        for k in range(K):
            A_data.append(reg_data)
            A_rows.append(row_idx + row_range)
            col_idx = np.arange(0, nparams_pwf)
            col_idx *= nparams_scaler
            col_idx[1:] += (k * scaled_nparams_pwf) + shift
            col_idx[0] = col_Lreg
            A_cols.append(col_idx)
            row_idx += nparams_pwf
    b_reg = np.zeros(row_idx)
    b_reg[::nparams_pwf] = L_regularizer_offset
    b_mats.append(b_reg)
    cones = cones + [socp_second_order_cone(nparams_pwf, backend)
                     for k in range(K * nparams_scaler)]
    del reg_data
    del row_range
    del col_idx
    gc.collect()
    if len(b_mats) > 0:
        b = _stack_b(b, b_mats)
        del b_mats
        gc.collect()
    if len(A_data) > 0:
        A_data = np.concatenate(A_data)
        A_rows = np.concatenate(A_rows)
        A_cols = np.concatenate(A_cols)
        gc.collect()
        nvars = A.shape[1]
        A_other = coo_matrix((A_data, (A_rows, A_cols)), shape=(row_idx, nvars))
        del A_data
        del A_rows
        del A_cols
        gc.collect()
        A_other = A_other.tocsc()
        gc.collect()
        A = vstack([A, A_other])
        gc.collect()
    return A, b, cones


def _final_w_H(H, X, partition, centers, variant,
               use_linear, nparams_tot, scaled_nparams_pwf):
    data = list(H.data)
    rows = list(H.row)
    cols = list(H.col)
    for k, cell_k in enumerate(partition.cells):
        cellXk = dcf_phi(X[cell_k, :], centers[k, :], variant, use_linear)
        cellXk *= np.sum(np.square(cellXk), axis=1)[:, None]
        rows += list(np.arange(scaled_nparams_pwf) + (k*scaled_nparams_pwf))
        cols += [nparams_tot-1] * scaled_nparams_pwf
        data += list(np.sum(cellXk, axis=0))
    return coo_matrix((data, (rows, cols)), shape=H.shape).tocsc()


def dcf_socp_data(
    X, y, partition, centers, variant, is_convex, is_mma, is_symmetrized,
    use_linear, L_regularizer, L_regularizer_offset,
    bias_regularizer, L_sum_regularizer, backend, verbose=0,
    scale_L_sum_regularizer_by_rec_cell_size=False,
):
    """
    >>> from ai.gandg.optim.socprog import SOCP_BACKEND__CLARABEL
    >>> from ai.gandg.common.partition import (
    ...     Partition, singleton_partition, find_min_dist_centers)

    >>> X = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]]).T
    >>> y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19])
    >>> partition = Partition(npoints=len(X), ncells=3,
    ...                       cells=[[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> center_idxs = [1, 4, 7]
    >>> H, g, A, b, cones, nparams_pwf, nparams_reg = dcf_socp_data(
    ...     X, y, partition, X[center_idxs, :], variant=2, is_convex=False,
    ...     is_mma=False, is_symmetrized=False, use_linear=True,
    ...     L_regularizer=None, L_regularizer_offset=0.0,
    ...     bias_regularizer=0.0, L_sum_regularizer=0.0,
    ...     backend=SOCP_BACKEND__CLARABEL,
    ... )
    >>> print(np.round(H.toarray(), decimals=3))
    [[0.333 0.    0.222 0.    0.    0.    0.    0.    0.   ]
     [0.    0.222 0.    0.    0.    0.    0.    0.    0.   ]
     [0.    0.    0.222 0.    0.    0.    0.    0.    0.   ]
     [0.    0.    0.    0.333 0.    0.222 0.    0.    0.   ]
     [0.    0.    0.    0.    0.222 0.    0.    0.    0.   ]
     [0.    0.    0.    0.    0.    0.222 0.    0.    0.   ]
     [0.    0.    0.    0.    0.    0.    0.333 0.    0.222]
     [0.    0.    0.    0.    0.    0.    0.    0.222 0.   ]
     [0.    0.    0.    0.    0.    0.    0.    0.    0.222]]
    >>> print(np.round(g, decimals=3)) # doctest: +NORMALIZE_WHITESPACE
    [-4.    -0.222 -2.667 -5.    -0.222 -3.333 -6.    -0.222 -4.   ]
    >>> print(np.round(A.toarray(), decimals=3))
    [[-1.  0.  0.  1. -3.  3.  0.  0.  0.]
     [-1.  0.  0.  0.  0.  0.  1. -6.  6.]
     [ 1.  3.  3. -1.  0.  0.  0.  0.  0.]
     [ 0.  0.  0. -1.  0.  0.  1. -3.  3.]
     [ 1.  6.  6.  0.  0.  0. -1.  0.  0.]
     [ 0.  0.  0.  1.  3.  3. -1.  0.  0.]]
    >>> X = np.array([[1.1, 1.1], [-1.2, 1.2], [-1.3, -1.3], [0.4, 0.4], [1.5, -1.5]])
    >>> y = np.array([1.1, 1.2, 1.3, 0.4, 0.5])
    >>> partition = singleton_partition(len(y))
    >>> center_idxs = find_min_dist_centers(X, partition)
    >>> H, g, A, b, cones, nparams_pwf, nparams_reg = dcf_socp_data(
    ...     X, y, partition, X[center_idxs, :], variant=2, is_convex=True,
    ...     is_mma=False, is_symmetrized=False, use_linear=True,
    ...     L_regularizer=3.0, L_regularizer_offset=5.0,
    ...     bias_regularizer=0.0, L_sum_regularizer=0.0,
    ...     backend=SOCP_BACKEND__CLARABEL,
    ... )
    >>> nparams_pwf
    4
    >>> H.nnz
    6
    >>> H.toarray()[:, :9]
    array([[0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.2],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]])
    >>> H.toarray()[-3:, -3:]
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 3.]])
    >>> g
    array([-0.22, -0.  , -0.  , -0.  , -0.24, -0.  , -0.  , -0.  , -0.26,
           -0.  , -0.  , -0.  , -0.08, -0.  , -0.  , -0.  , -0.1 , -0.  ,
           -0.  , -0.  ,  0.  ])

    >>> A.shape
    (45, 21)
    >>> np.linalg.matrix_rank(A.toarray())
    20
    >>> A.nnz
    125
    >>> A.toarray()[:25, :9]
    array([[-1.        ,  0.        ,  0.        ,  0.        ,  1.        ,
             2.3       , -0.1       ,  2.30217289,  0.        ],
           [-1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  1.        ],
           [-1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [-1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 1.        , -2.3       ,  0.1       ,  2.30217289, -1.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        , -1.        ,
             0.        ,  0.        ,  0.        ,  1.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        , -1.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        , -1.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 1.        , -2.4       , -2.4       ,  3.39411255,  0.        ,
             0.        ,  0.        ,  0.        , -1.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
            -0.1       , -2.5       ,  2.5019992 , -1.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        , -1.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        , -1.        ],
           [ 1.        , -0.7       , -0.7       ,  0.98994949,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
             1.6       , -0.8       ,  1.78885438,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  1.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 1.        ,  0.4       , -2.6       ,  2.63058929,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
             2.7       , -2.7       ,  3.81837662,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  1.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , -1.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        , -1.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ]])
    >>> np.vstack([A[-5:, :].T.toarray(), b[-5:]])
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  1.],
           [ 0., -1.,  0.,  0.,  0.],
           [ 0.,  5.,  0.,  0.,  0.]])
    >>> b.shape
    (45,)
    >>> np.sum(np.abs(b))
    25.0
    >>> len(cones)
    7
    """
    n, d = X.shape
    K = partition.ncells
    variant = str(variant)
    assert n == partition.npoints
    assert n >= K, 'Too few data points, n: {}, K: {}'.format(n, K)
    assert n > d, 'Too few data points, n: {}, d: {}'.format(n, d)
    if y.shape == (n, 1):
        y = y[:, 0]
    assert y.shape == (n,), 'Invalid y.shape: {}'.format(y.shape)

    nparams_pwf = _get_nparams_pwf(d, variant, use_linear)
    nparams_scaler = 1 + int(is_symmetrized)
    scaled_nparams_pwf = nparams_pwf * nparams_scaler
    nparams_tot = K * nparams_pwf * nparams_scaler
    if variant[-1] == 'w':
        nparams_tot += 1
    nparams_reg = int(L_regularizer is not None)
    nvars = nparams_tot + nparams_reg
    H, g, A, b = _dcf_HgAb(
        X, y, partition, centers, variant, is_symmetrized, use_linear,
        L_regularizer, bias_regularizer, L_sum_regularizer,
        nparams_pwf, nparams_scaler, nparams_tot, nvars,
        scale_L_sum_regularizer_by_rec_cell_size,
    )
    gc.collect()

    cones = [socp_nonnegative_cone(len(b), backend)]
    if is_convex or is_mma:
        A, b, cones = _add_cvx_or_mma_consraints_to_Ab(
            variant, K, d, nparams_pwf, A, b, cones,
            backend, is_mma, is_symmetrized,
        )
    if L_regularizer is not None:
        A, b, cones = _add_regL_constraints_to_Ab(
            L_regularizer, L_regularizer_offset, A, b, cones, backend,
            K, nparams_pwf, nparams_scaler, nparams_reg, verbose,
        )

    assert H.shape == (nvars, nvars), 'Invalid H.shape: {}'.format(H.shape)
    assert g.shape == (nvars,), 'Invalid g.shape: {}'.format(g.shape)
    assert A.shape[1] == nvars, 'Invalid A.shape: {}'.format(A.shape)
    assert len(b) == A.shape[0], 'Invalid len(b): {}'.format(b.shape)

    H = convert_matrix_to_socp_solver_format(H, backend)
    gc.collect()
    A = convert_matrix_to_socp_solver_format(A, backend)
    gc.collect()
    return H, g, A, b, cones, scaled_nparams_pwf, nparams_reg


def _dcf_l2_nolocal_clarabel_tests():
    """
    >>> from ai.gandg.optim.socprog import SOCP_BACKEND__CLARABEL
    >>> np.set_printoptions(legacy='1.25')
    >>> from ai.gandg.common.util import set_random_seed
    >>> set_random_seed(19)

    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2
    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])

    # L2-error of OLS is bigger than 6.
    >>> X_test = np.random.randn(500, 2)
    >>> y_test = regression_func(X_test)
    >>> ols_model = np.linalg.lstsq(X.T.dot(X), X.T.dot(y), rcond=-1)[0]
    >>> ols_yhat_test = np.sum(X_test * ols_model, axis=1)  # np.dot is not deterministic
    >>> np.round(np.sum(np.square(ols_yhat_test - y_test)) / len(y_test), decimals=4)  # OLS out-of-sample L2-error
    6.2752

    >>> est1 = DCFEstimator(variant=2, is_convex=True,
    ...                      train_args={'normalize': False,
    ...                                  'L_sum_regularizer': 0.0,
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'local_opt_maxiter': 0,
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model1 = est1.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model1.weights.shape
    (15, 4)
    >>> np.round([model1._socp_stats.obj_val, model1._socp_stats.proj_obj_val], decimals=4)
    array([0.0104, 0.0124])
    >>> np.round(np.max(np.linalg.norm(model1.weights[:, 1:], axis=1))**2, decimals=4)
    65.0408
    >>> yhat1 = est1.predict(model1, X)
    >>> np.round(np.mean(np.square(yhat1 - y)), decimals=4)  # in-sample L2-risk
    0.0248
    >>> yhat1_test = est1.predict(model1, X_test)
    >>> np.round(np.mean(np.square(yhat1_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.041

    >>> est2 = DCFEstimator(variant=2, is_convex=True,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_maxiter': 0,
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model2 = est2.train(X, y)
    >>> model2.weights.shape
    (15, 4)
    >>> np.round(model2._socp_stats.reg_var_value, decimals=4)
    0.0015
    >>> yhat2 = est2.predict(model2, X)
    >>> np.round(np.mean(np.square(yhat2 - y)), decimals=4)  # in-sample L2-risk
    0.0287
    >>> yhat2_test = est2.predict(model2, X_test)
    >>> np.round(np.mean(np.square(yhat2_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0433

    >>> est3 = DCFEstimator(variant=1, is_convex=True,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_maxiter': 0,
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model3 = est3.train(X, y, L_regularizer=None, L_regularizer_offset=0.0)
    >>> yhat3 = est3.predict(model3, X)
    >>> np.round(np.mean(np.square(yhat3 - y)), decimals=4)  # in-sample L2-risk
    0.0231
    >>> yhat3_test = est3.predict(model3, X_test)
    >>> np.round(np.mean(np.square(yhat3_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0393
    >>> np.round([model3._socp_stats.obj_val, model3._socp_stats.proj_obj_val], decimals=4)
    array([0.0103, 0.0116])
    >>> model3.weights.shape
    (15, 4)
    >>> model3._socp_stats.reg_var_value is None
    True

    >>> est4 = DCFEstimator(variant=np.inf, is_convex=True,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_maxiter': 0,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model4 = est4.train(X, y, afpc_q=2)
    >>> model4.variant
    'inf'
    >>> model4.weights.shape
    (7, 4)
    >>> np.round(model4._socp_stats.reg_var_value, decimals=4)
    0.0014
    >>> yhat4 = est4.predict(model4, X)
    >>> np.round(np.mean(np.square(yhat4 - y)), decimals=4)  # in-sample L2-risk
    0.133
    >>> yhat4_test = est4.predict(model4, X_test)
    >>> np.round(np.mean(np.square(yhat4_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.1321

    >>> est5 = DCFEstimator(variant=np.inf, is_convex=False,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_maxiter': 0,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model5 = est5.train(X, y)
    >>> model5.weights.shape
    (15, 4)
    >>> np.round(model5._socp_stats.reg_var_value, decimals=4)
    0.0003
    >>> yhat5 = est5.predict(model5, X)
    >>> np.round(np.mean(np.square(yhat5 - y)), decimals=4)  # in-sample L2-risk
    0.0266
    >>> yhat5_test = est5.predict(model5, X_test)
    >>> np.round(np.mean(np.square(yhat5_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.047

    >>> est6 = DCFEstimator(variant='+', is_convex=True,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_maxiter': 0,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model6 = est6.train(X, y, afpc_q=2)
    >>> model6.weights.shape
    (7, 5)
    >>> yhat6 = est6.predict(model6, X)
    >>> np.round(np.mean(np.square(yhat6 - y)), decimals=4)  # in-sample L2-risk
    0.1255
    >>> yhat6_test = est6.predict(model6, X_test)
    >>> np.round(np.mean(np.square(yhat6_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0992

    >>> est7 = DCFEstimator(variant='+', negate_y=True,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.01,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_maxiter': 0,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model7 = est7.train(X, y)
    >>> np.round([model7._socp_stats.obj_val, model7._socp_stats.proj_obj_val], decimals=4)
    array([0.6989, 0.7577])
    >>> model7.weights.shape
    (15, 5)
    >>> np.round(model7._socp_stats.reg_var_value, decimals=4)
    0.0
    >>> yhat7 = est7.predict(model7, X)
    >>> np.round(np.mean(np.square(yhat7 - y)), decimals=4)  # in-sample L2-risk
    0.8455
    >>> yhat7_test = est7.predict(model7, X_test)
    >>> np.round(np.mean(np.square(yhat7_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.8954

    >>> est8 = DCFEstimator(variant=np.inf, is_convex=False, is_symmetrized=True,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_maxiter': 0,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model8 = est8.train(X, y)
    >>> model8.weights.shape
    (15, 8)
    >>> np.round(model8._socp_stats.reg_var_value, decimals=4)
    0.0
    >>> np.round([model8._socp_stats.obj_val, model8._socp_stats.proj_obj_val], decimals=4)
    array([0.0079, 0.008 ])
    >>> yhat8 = est8.predict(model8, X)
    >>> np.round(np.mean(np.square(yhat8 - y)), decimals=4)  # in-sample L2-risk
    0.016
    >>> yhat8_test = est8.predict(model8, X_test)
    >>> np.round(np.mean(np.square(yhat8_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0554

    >>> est9 = DCFEstimator(variant='2q', is_convex=False,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_maxiter': 0,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model9 = est9.train(X, y)
    >>> model9.weights.shape
    (15, 5)
    >>> np.round(model9._socp_stats.reg_var_value, decimals=4)
    0.0002
    >>> np.round([model9._socp_stats.obj_val, model9._socp_stats.proj_obj_val], decimals=4)
    array([0.0078, 0.0121])
    >>> yhat9 = est9.predict(model9, X)
    >>> np.round(np.mean(np.square(yhat9 - y)), decimals=4)  # in-sample L2-risk
    0.0242
    >>> yhat9_test = est9.predict(model9, X_test)
    >>> np.round(np.mean(np.square(yhat9_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0755

    >>> est10 = DCFEstimator(variant='2q', is_convex=True,
    ...                      train_args={'normalize': False,
    ...                                  'L_sum_regularizer': 0.0,
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'local_opt_maxiter': 0,
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model10 = est10.train(X, y, afpc_q=2)
    >>> model10.weights.shape
    (7, 5)
    >>> yhat10 = est10.predict(model10, X)
    >>> np.round(np.mean(np.square(yhat10 - y)), decimals=4)  # in-sample L2-risk
    0.1912
    >>> yhat10_test = est6.predict(model10, X_test)
    >>> np.round(np.mean(np.square(yhat10_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.1824

    >>> est11 = DCFEstimator(variant='2w', is_convex=False,
    ...                      train_args={'normalize': False,
    ...                                  'L_sum_regularizer': 0.0,
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'local_opt_maxiter': 0,
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model11 = est11.train(X, y)
    >>> len(model11.weights)
    2
    >>> model11.weights[0].shape
    (15, 4)
    >>> np.round(model11.weights[1], decimals=6)
    -0.011034
    >>> np.round(model11._socp_stats.reg_var_value, decimals=4)
    0.0
    >>> yhat11 = est11.predict(model11, X)
    >>> np.round(np.mean(np.square(yhat11 - y)), decimals=4)  # in-sample L2-risk
    0.0228
    >>> yhat11_test = est11.predict(model11, X_test)
    >>> np.round(np.mean(np.square(yhat11_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0376

    >>> est12 = DCFEstimator(variant=2, is_convex=False,
    ...                      train_args={'normalize': False,
    ...                                  'use_linear': False,
    ...                                  'L_sum_regularizer': 0.0,
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'local_opt_maxiter': 0,
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model12 = est12.train(X, y)
    >>> model12.weights.shape
    (15, 2)
    >>> np.round(model12._socp_stats.reg_var_value, decimals=4)
    0.0
    >>> np.round([model12._socp_stats.obj_val, model12._socp_stats.proj_obj_val], decimals=4)
    array([0.4732, 0.5728])
    >>> yhat12 = est12.predict(model12, X)
    >>> np.round(np.mean(np.square(yhat12 - y)), decimals=4)  # in-sample L2-risk
    1.1456
    >>> yhat12_test = est12.predict(model12, X_test)
    >>> np.round(np.mean(np.square(yhat12_test - y_test)), decimals=4)  # out-of-sample L2-error
    1.3786

    >>> est13 = DCFEstimator(variant=2, is_convex=False,
    ...                      train_args={'normalize': False,
    ...                                  'L_sum_regularizer': 0.0,
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'local_opt_maxiter': 0,
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model13 = est13.train(X, y)
    >>> model13.weights.shape
    (15, 4)
    >>> np.round(model13._socp_stats.reg_var_value, decimals=4)
    0.0002
    >>> yhat13 = est13.predict(model13, X)
    >>> np.round(np.mean(np.square(yhat13 - y)), decimals=4)  # in-sample L2-risk
    0.0249
    >>> yhat13_test = est13.predict(model13, X_test)
    >>> np.round(np.mean(np.square(yhat13_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0418

    >>> est14 = DCFEstimator(variant=2, is_convex=False,
    ...                       train_args={'verbose': 0,
    ...                                   'normalize': False,
    ...                                   'L_sum_regularizer': 0.0,
    ...                                   'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                   'L_regularizer_offset': 'np.log(n)',
    ...                                   'local_opt_maxiter': 0,
    ...                                   'afpc_ntrials': 3,
    ...                                   'kmeans_objval': True,
    ...                                   'kmeans_kwargs': {},
    ...                                   'backend': SOCP_BACKEND__CLARABEL})
    >>> model14 = est14.train(X, y)
    >>> model14.weights.shape
    (16, 4)
    >>> np.round([model14._socp_stats.obj_val, model14._socp_stats.proj_obj_val], decimals=4)
    array([0.008 , 0.0107])
    >>> np.round(model14._socp_stats.reg_var_value, decimals=4)
    0.0008
    >>> yhat14 = est14.predict(model14, X)
    >>> np.round(np.mean(np.square(yhat14 - y)), decimals=4)  # in-sample L2-risk
    0.0213
    >>> yhat14_test = est14.predict(model14, X_test)
    >>> np.round(np.mean(np.square(yhat14_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0171

    >>> est15 = DCFEstimator(variant=2, is_convex=True,
    ...                      train_args={'normalize': False,
    ...                                  'L_sum_regularizer': 0.01,
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'local_opt_maxiter': 0,
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model15 = est15.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model15.weights.shape
    (15, 4)
    >>> np.round([model15._socp_stats.obj_val, model15._socp_stats.proj_obj_val], decimals=4)
    array([0.4792, 0.5389])
    >>> np.round(np.max(np.linalg.norm(model15.weights[:, 1:], axis=1))**2, decimals=4)
    12.9079
    >>> yhat15 = est15.predict(model15, X)
    >>> np.round(np.mean(np.square(yhat15 - y)), decimals=4)  # in-sample L2-risk
    0.3804
    >>> yhat15_test = est15.predict(model15, X_test)
    >>> np.round(np.mean(np.square(yhat15_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.3254

    >>> est16 = DCFEstimator(variant=2, is_convex=True,
    ...                      train_args={'normalize': False,
    ...                                  'L_sum_regularizer': '0.01/I_k',
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'local_opt_maxiter': 0,
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model16 = est16.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model16.weights.shape
    (15, 4)
    >>> np.round([model16._socp_stats.obj_val, model16._socp_stats.proj_obj_val], decimals=4)
    array([0.1657, 0.6008])
    >>> np.round(np.max(np.linalg.norm(model16.weights[:, 1:], axis=1))**2, decimals=4)
    19.3132
    >>> yhat16 = est16.predict(model16, X)
    >>> np.round(np.mean(np.square(yhat16 - y)), decimals=4)  # in-sample L2-risk
    0.1022
    >>> yhat16_test = est16.predict(model16, X_test)
    >>> np.round(np.mean(np.square(yhat16_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0859

    >>> est17 = DCFEstimator(variant=2, is_convex=True,
    ...                      train_args={'normalize': False,
    ...                                  'L_sum_regularizer': 0.01,
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'afpc_min_cell_size': 3,
    ...                                  'local_opt_maxiter': 0,
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model17 = est17.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model17.weights.shape
    (13, 4)
    >>> np.round([model17._socp_stats.obj_val, model17._socp_stats.proj_obj_val], decimals=4)
    array([0.4392, 0.4881])
    >>> np.round(np.max(np.linalg.norm(model17.weights[:, 1:], axis=1))**2, decimals=4)
    12.9377
    >>> yhat17 = est17.predict(model17, X)
    >>> np.round(np.mean(np.square(yhat17 - y)), decimals=4)  # in-sample L2-risk
    0.3274
    >>> yhat17_test = est17.predict(model17, X_test)
    >>> np.round(np.mean(np.square(yhat17_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.2615

    >>> est18 = DCFEstimator(variant=2, is_convex=True,
    ...                      train_args={'normalize': False,
    ...                                  'L_sum_regularizer': '(x_radius**2)/n',
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'local_opt_maxiter': 0,
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model18 = est18.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model18.weights.shape
    (15, 4)
    >>> np.round([model18._socp_stats.obj_val, model18._socp_stats.proj_obj_val], decimals=4)
    array([1.5026, 1.6325])
    >>> np.round(np.max(np.linalg.norm(model18.weights[:, 1:], axis=1))**2, decimals=4)
    2.8956
    >>> yhat18 = est18.predict(model18, X)
    >>> np.round(np.mean(np.square(yhat18 - y)), decimals=4)  # in-sample L2-risk
    1.8392
    >>> yhat18_test = est18.predict(model18, X_test)
    >>> np.round(np.mean(np.square(yhat18_test - y_test)), decimals=4)  # out-of-sample L2-error
    1.5557

    >>> est19 = DCFEstimator(variant=2, is_convex=True,
    ...                      train_args={'normalize': True,
    ...                                  'L_sum_regularizer': '(x_radius**2)/n',
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'local_opt_maxiter': 0,
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model19 = est19.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model19.weights.shape
    (15, 4)
    >>> np.round([model19._socp_stats.obj_val, model19._socp_stats.proj_obj_val], decimals=4)
    array([0.2393, 0.26  ])
    >>> np.round(np.max(np.linalg.norm(model19.weights[:, 1:], axis=1))**2, decimals=4)
    0.9064
    >>> yhat19 = est19.predict(model19, X)
    >>> np.round(np.mean(np.square(yhat19 - y)), decimals=4)  # in-sample L2-risk
    1.8392
    >>> yhat19_test = est19.predict(model19, X_test)
    >>> np.round(np.mean(np.square(yhat19_test - y_test)), decimals=4)  # out-of-sample L2-error
    1.5557
    """
    pass


def _dcf_l2_nolocal_lbfgs_tests():
    """
    >>> np.set_printoptions(legacy='1.25')
    >>> from ai.gandg.optim.socprog import SOCP_BACKEND__LBFGS
    >>> from ai.gandg.common.util import set_random_seed
    >>> set_random_seed(19)

    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2
    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])

    # L2-error of OLS is bigger than 6.
    >>> X_test = np.random.randn(500, 2)
    >>> y_test = regression_func(X_test)
    >>> ols_model = np.linalg.lstsq(X.T.dot(X), X.T.dot(y), rcond=-1)[0]
    >>> ols_yhat_test = np.sum(X_test * ols_model, axis=1)  # np.dot is not deterministic
    >>> np.round(np.sum(np.square(ols_yhat_test - y_test)) / len(y_test), decimals=4)  # OLS out-of-sample L2-error
    6.2752

    >>> est1 = DCFEstimator(variant=2, is_convex=True,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_maxiter': 0,
    ...                                 'backend': SOCP_BACKEND__LBFGS})
    >>> model1 = est1.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model1.weights.shape
    (15, 4)
    >>> np.round([model1._socp_stats.obj_val, model1._socp_stats.proj_obj_val], decimals=3)
    array([0.01 , 0.012])
    >>> v = np.max(np.linalg.norm(model1.weights[:, 1:], axis=1))**2
    >>> abs(v - 45) < 1 or v  # this test is instable over different archs (Intel vs Apple Mx)
    True
    >>> yhat1 = est1.predict(model1, X)
    >>> np.round(np.mean(np.square(yhat1 - y)), decimals=2)  # in-sample L2-risk
    0.02
    >>> yhat1_test = est1.predict(model1, X_test)
    >>> np.round(np.mean(np.square(yhat1_test - y_test)), decimals=2)  # out-of-sample L2-error
    0.04
    """
    pass


def _dcf_l2_local_nonsmooth_tests():
    """
    >>> from ai.gandg.optim.socprog import SOCP_BACKEND__CLARABEL
    >>> np.set_printoptions(legacy='1.25')
    >>> from ai.gandg.common.util import set_random_seed
    >>> set_random_seed(19)

    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2
    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])

    # L2-error of OLS is bigger than 6.
    >>> X_test = np.random.randn(500, 2)
    >>> y_test = regression_func(X_test)
    >>> ols_model = np.linalg.lstsq(X.T.dot(X), X.T.dot(y), rcond=-1)[0]
    >>> ols_yhat_test = np.sum(X_test * ols_model, axis=1)  # np.dot is not deterministic
    >>> np.round(np.sum(np.square(ols_yhat_test - y_test)) / len(y_test), decimals=4)  # OLS out-of-sample L2-error
    6.2752

    >>> est1 = DCFEstimator(variant=2, is_convex=False,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'nonsmooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'local_opt_noise_level': 0.0,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model1 = est1.train(X, y)
    >>> model1.weights.shape
    (11, 4)
    >>> np.round(model1._socp_stats.reg_var_value, decimals=4)
    0.0002
    >>> yhat1 = est1.predict(model1, X)
    >>> np.round(np.mean(np.square(yhat1 - y)), decimals=4)  # in-sample L2-risk
    0.0093
    >>> yhat1_test = est1.predict(model1, X_test)
    >>> np.round(np.mean(np.square(yhat1_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0271
    >>> np.round([model1._socp_stats.obj_val, model1._socp_stats.proj_obj_val,
    ...           model1._local_opt_stats.init_obj_val, model1._local_opt_stats.soln_obj_val],
    ...          decimals=4)
    array([0.0105, 0.0124, 0.0124, 0.0047])

    >>> est2 = DCFEstimator(variant=2, is_convex=False, is_symmetrized=True,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'nonsmooth',
    ...                                 'local_opt_maxiter': 20,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model2 = est2.train(X, y)
    >>> model2.weights.shape
    (15, 8)
    >>> np.round(model2._socp_stats.reg_var_value, decimals=4)
    0.0
    >>> yhat2 = est2.predict(model2, X)
    >>> np.round(np.mean(np.square(yhat2 - y)), decimals=4)  # in-sample L2-risk
    0.0143
    >>> yhat2_test = est2.predict(model2, X_test)
    >>> np.round(np.mean(np.square(yhat2_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0613
    >>> np.round([model2._socp_stats.obj_val, model2._socp_stats.proj_obj_val,
    ...           model2._local_opt_stats.init_obj_val, model2._local_opt_stats.soln_obj_val],
    ...          decimals=4)
    array([0.0073, 0.0076, 0.0076, 0.0071])

    >>> est3 = DCFEstimator(variant='+', is_convex=False,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'nonsmooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'local_opt_noise_level': 0.0,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model3 = est3.train(X, y)
    >>> model3.weights.shape
    (13, 5)
    >>> np.round(model3._socp_stats.reg_var_value, decimals=4)
    0.0039
    >>> yhat3 = est3.predict(model3, X)
    >>> np.round(np.mean(np.square(yhat3 - y)), decimals=4)  # in-sample L2-risk
    0.0098
    >>> yhat3_test = est3.predict(model3, X_test)
    >>> np.round(np.mean(np.square(yhat3_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0067
    >>> np.round([model3._socp_stats.obj_val, model3._socp_stats.proj_obj_val,
    ...           model3._local_opt_stats.init_obj_val, model3._local_opt_stats.soln_obj_val],
    ...          decimals=4)
    array([0.0059, 0.0101, 0.01  , 0.0049])

    >>> est4 = DCFEstimator(variant='+', is_convex=False, is_symmetrized=True,
    ...                     train_args={'verbose': 1,
    ...                                 'normalize': False,
    ...                                 'L_sum_regularizer': 0.01,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'nonsmooth',
    ...                                 'local_opt_maxiter': 20,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model4 = est4.train(X, y) # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    DCF(+), TRAIN, n: 200, d: 2
    DCF(+), CLUSTER, K: 15, max_epsilon: 0.95, avg_epsilon: 0.56
    DCF(+), DATA, n: 200, d: 2, x_radius: 3.5599, y_radius: 9.2333
    DCF(+), PARAMS, xscale: 1.0000, yscale: 1.0000, L_regularizer_offset: 5.2983, L_regularizer:...
    DCF(+), SOLVE, nvars: 151, nineqcons: 420, nsoccons: 30, etime: ...s
    DCF(+), SOLVED, etime: ...s
    DCF(+), OPTIM: DCFLocalOptStats(..., opt_status='MaxIterReached', niterations=20, nfun_evals=28, ngrad_evals=21, ...)
    >>> len(model4.weights), model4.weights[0].shape, model4.weights[1].shape
    (2, (5, 5), (2, 5))
    >>> np.round(model4._socp_stats.reg_var_value, decimals=4)
    0.0
    >>> yhat4 = est4.predict(model4, X)
    >>> np.round(np.mean(np.square(yhat4 - y)), decimals=4)  # in-sample L2-risk
    0.101
    >>> yhat4_test = est4.predict(model4, X_test)
    >>> np.round(np.mean(np.square(yhat4_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0862
    >>> np.round([model4._socp_stats.obj_val, model4._socp_stats.proj_obj_val,
    ...           model4._local_opt_stats.init_obj_val, model4._local_opt_stats.soln_obj_val],
    ...          decimals=4)
    array([0.3773, 0.3986, 0.3986, 0.2257])

    >>> est5 = DCFEstimator(variant=np.inf, is_convex=True,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'nonsmooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model5 = est5.train(X, y, afpc_q=2)
    >>> model5.weights.shape
    (7, 4)
    >>> np.min(model5.weights[:, -1]) >= -1e-5 or np.min(model5.weights[:, -1])
    True
    >>> np.round(model5._socp_stats.reg_var_value, decimals=4)
    0.0014
    >>> yhat5 = est5.predict(model5, X)
    >>> np.round(np.mean(np.square(yhat5 - y)), decimals=4)  # in-sample L2-risk
    0.0759
    >>> yhat5_test = est5.predict(model5, X_test)
    >>> np.round(np.mean(np.square(yhat5_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0711

    >>> est6 = DCFEstimator(variant='+', is_convex=True,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'nonsmooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'local_opt_noise_level': 1e-8,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model6 = est6.train(X, y, afpc_q=2)
    >>> model6.weights.shape
    (7, 5)
    >>> model6._local_opt_stats # doctest: +ELLIPSIS
    DCFLocalOptStats(..., opt...='MaxIterReached', niter...=10, nfun...=181, ngrad...=11, run...)
    >>> yhat6 = est6.predict(model6, X)
    >>> np.round(np.mean(np.square(yhat6 - y)), decimals=4)  # in-sample L2-risk
    0.0948
    >>> yhat6_test = est6.predict(model6, X_test)
    >>> np.round(np.mean(np.square(yhat6_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0602

    >>> est7 = DCFEstimator(variant=2, is_convex=False,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.01,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'nonsmooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model7 = est7.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model7.weights.shape
    (14, 4)
    >>> np.round([model7._socp_stats.obj_val, model7._socp_stats.proj_obj_val], decimals=4)
    array([0.3612, 0.4195])
    >>> np.round(np.max(np.linalg.norm(model7.weights[:, 1:], axis=1))**2, decimals=4)
    8.1722
    >>> yhat7 = est7.predict(model7, X)
    >>> np.round(np.mean(np.square(yhat7 - y)), decimals=4)  # in-sample L2-risk
    0.152
    >>> yhat7_test = est7.predict(model7, X_test)
    >>> np.round(np.mean(np.square(yhat7_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.2191

    >>> est8 = DCFEstimator(variant=2, is_convex=False,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': 0.01,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'nonsmooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'local_opt_L_regularizer_offset': 1.0,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model8 = est8.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model8.weights.shape
    (14, 4)
    >>> np.round([model8._socp_stats.obj_val, model8._socp_stats.proj_obj_val], decimals=4)
    array([0.3612, 0.4195])
    >>> np.round(np.max(np.linalg.norm(model8.weights[:, 1:], axis=1))**2, decimals=4)
    8.0068
    >>> yhat8 = est8.predict(model8, X)
    >>> np.round(np.mean(np.square(yhat8 - y)), decimals=4)  # in-sample L2-risk
    0.1532
    >>> yhat8_test = est8.predict(model8, X_test)
    >>> np.round(np.mean(np.square(yhat8_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.2209

    >>> est10 = DCFEstimator(variant=2, is_convex=False,
    ...                      train_args={'normalize': False,
    ...                                  'L_sum_regularizer': '(x_radius**2)/n',
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'local_opt_type': 'nonsmooth',
    ...                                  'local_opt_maxiter': 10,
    ...                                  'local_opt_L_regularizer_offset': 'np.sqrt(np.log(n))',
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model10 = est10.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model10.weights.shape
    (13, 4)
    >>> np.round([model10._socp_stats.obj_val, model10._socp_stats.proj_obj_val], decimals=4)
    array([1.1057, 1.301 ])
    >>> np.round(np.max(np.linalg.norm(model10.weights[:, 1:], axis=1))**2, decimals=4)
    2.9837
    >>> yhat10 = est10.predict(model10, X)
    >>> np.round(np.mean(np.square(yhat10 - y)), decimals=4)  # in-sample L2-risk
    1.123
    >>> yhat10_test = est10.predict(model10, X_test)
    >>> np.round(np.mean(np.square(yhat10_test - y_test)), decimals=4)  # out-of-sample L2-error
    1.115

    >>> est11 = DCFEstimator(variant=2, is_convex=False,
    ...                      train_args={'normalize': True,
    ...                                  'L_sum_regularizer': '(x_radius**2)/n',
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'local_opt_type': 'nonsmooth',
    ...                                  'local_opt_maxiter': 10,
    ...                                  'local_opt_L_regularizer_offset': 'np.sqrt(np.log(n))',
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model11 = est11.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model11.weights.shape
    (14, 4)
    >>> np.round([model11._socp_stats.obj_val, model11._socp_stats.proj_obj_val], decimals=4)
    array([0.1761, 0.2072])
    >>> np.round(np.max(np.linalg.norm(model11.weights[:, 1:], axis=1))**2, decimals=4)
    0.9658
    >>> yhat11 = est11.predict(model11, X)
    >>> np.round(np.mean(np.square(yhat11 - y)), decimals=4)  # in-sample L2-risk
    1.075
    >>> yhat11_test = est11.predict(model11, X_test)
    >>> np.round(np.mean(np.square(yhat11_test - y_test)), decimals=4)  # out-of-sample L2-error
    1.0707
    """
    pass


def _dcf_l2_local_smooth_tests():
    """
    >>> from ai.gandg.optim.socprog import SOCP_BACKEND__CLARABEL
    >>> np.set_printoptions(legacy='1.25')
    >>> from ai.gandg.common.util import set_random_seed
    >>> set_random_seed(19)

    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2
    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])

    # L2-error of OLS is bigger than 6.
    >>> X_test = np.random.randn(500, 2)
    >>> y_test = regression_func(X_test)
    >>> ols_model = np.linalg.lstsq(X.T.dot(X), X.T.dot(y), rcond=-1)[0]
    >>> ols_yhat_test = np.sum(X_test * ols_model, axis=1)  # np.dot is not deterministic
    >>> np.round(np.sum(np.square(ols_yhat_test - y_test)) / len(y_test), decimals=4)  # OLS out-of-sample L2-error
    6.2752

    >>> est1 = DCFEstimator(variant=2, is_convex=False,
    ...                     train_args={'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model1 = est1.train(X, y)
    >>> model1.weights.shape
    (12, 4)
    >>> np.round(model1._socp_stats.reg_var_value, decimals=4)
    0.0001
    >>> yhat1 = est1.predict(model1, X)
    >>> np.round(np.mean(np.square(yhat1 - y)), decimals=4)  # in-sample L2-risk
    0.0091
    >>> yhat1_test = est1.predict(model1, X_test)
    >>> np.round(np.mean(np.square(yhat1_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0273
    >>> np.round([model1._socp_stats.obj_val, model1._socp_stats.proj_obj_val,
    ...           model1._local_opt_stats.init_obj_val, model1._local_opt_stats.soln_obj_val],
    ...          decimals=4)
    array([0.0017, 0.002 , 0.002 , 0.0007])

    >>> est2 = DCFEstimator(variant=2, is_convex=False, is_symmetrized=True,
    ...                     train_args={'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_maxiter': 20,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model2 = est2.train(X, y)
    >>> model2.weights.shape
    (15, 8)
    >>> np.round(model2._socp_stats.reg_var_value, decimals=4)
    0.0
    >>> yhat2 = est2.predict(model2, X)
    >>> np.round(np.mean(np.square(yhat2 - y)), decimals=4)  # in-sample L2-risk
    0.0143
    >>> yhat2_test = est2.predict(model2, X_test)
    >>> np.round(np.mean(np.square(yhat2_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0624
    >>> np.round([model2._socp_stats.obj_val, model2._socp_stats.proj_obj_val,
    ...           model2._local_opt_stats.init_obj_val, model2._local_opt_stats.soln_obj_val],
    ...          decimals=4)
    array([0.0012, 0.0012, 0.0012, 0.0011])

    >>> est3 = DCFEstimator(variant='+', is_convex=False,
    ...                     train_args={'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model3 = est3.train(X, y)
    >>> model3.weights.shape
    (13, 5)
    >>> np.round(model3._socp_stats.reg_var_value, decimals=4)
    0.0022
    >>> yhat3 = est3.predict(model3, X)
    >>> np.round(np.mean(np.square(yhat3 - y)), decimals=4)  # in-sample L2-risk
    0.0099
    >>> yhat3_test = est3.predict(model3, X_test)
    >>> np.round(np.mean(np.square(yhat3_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0065
    >>> np.round([model3._socp_stats.obj_val, model3._socp_stats.proj_obj_val,
    ...           model3._local_opt_stats.init_obj_val, model3._local_opt_stats.soln_obj_val],
    ...          decimals=4)
    array([0.0009, 0.0016, 0.0016, 0.0008])

    >>> est4 = DCFEstimator(variant='+', is_convex=False, is_symmetrized=True,
    ...                     train_args={'verbose': 1,
    ...                                 'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_maxiter': 20,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model4 = est4.train(X, y) # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    DCF(+), TRAIN, n: 200, d: 2
    DCF(+), CLUSTER, K: 15, max_epsilon: 0.95, avg_epsilon: 0.56
    DCF(+), DATA, n: 200, d: 2, x_radius: 3.5599, y_radius: 9.2333
    DCF(+), PARAMS, xscale:..., yscale:..., L_regularizer_offset: 2.9644, L_regularizer: 0.4836...
    DCF(+), SOLVE, nvars: 151, nineqcons: 420, nsoccons: 30, etime: ...s
    DCF(+), SOLVED, etime: ...s
    DCF(+), OPTIM: DCFLocalOptStats(..., opt...='MaxIterReached', niter...=20, nfun...=28, ngrad...=21, run...)
    >>> model4.weights.shape
    (15, 10)
    >>> np.round(model4._socp_stats.reg_var_value, decimals=4)
    0.0
    >>> yhat4 = est4.predict(model4, X)
    >>> np.round(np.mean(np.square(yhat4 - y)), decimals=4)  # in-sample L2-risk
    0.0086
    >>> yhat4_test = est4.predict(model4, X_test)
    >>> np.round(np.mean(np.square(yhat4_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0477
    >>> np.round([model4._socp_stats.obj_val, model4._socp_stats.proj_obj_val,
    ...           model4._local_opt_stats.init_obj_val, model4._local_opt_stats.soln_obj_val],
    ...          decimals=4)
    array([0.0006, 0.0036, 0.0036, 0.0007])

    >>> est5 = DCFEstimator(variant=np.inf, is_convex=True,
    ...                     train_args={'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model5 = est5.train(X, y, afpc_q=2)
    >>> model5.weights.shape
    (7, 4)
    >>> np.min(model5.weights[:, -1]) >= -1e-5 or np.min(model5.weights[:, -1])
    True
    >>> np.round(model5._socp_stats.reg_var_value, decimals=4)
    0.0008
    >>> yhat5 = est5.predict(model5, X)
    >>> np.round(np.mean(np.square(yhat5 - y)), decimals=4)  # in-sample L2-risk
    0.0761
    >>> yhat5_test = est5.predict(model5, X_test)
    >>> np.round(np.mean(np.square(yhat5_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0712

    >>> est6 = DCFEstimator(variant='+', is_convex=True,
    ...                     train_args={'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'local_opt_noise_level': 1e-8,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model6 = est6.train(X, y, afpc_q=2)
    >>> model6.weights.shape
    (7, 5)
    >>> model6._local_opt_stats # doctest: +ELLIPSIS
    DCFLocalOptStats(..., opt...='MaxIterReached', nit...=10, nfun...=182, ngrad...=11, run...)
    >>> yhat6 = est6.predict(model6, X)
    >>> np.round(np.mean(np.square(yhat6 - y)), decimals=4)  # in-sample L2-risk
    0.0948
    >>> yhat6_test = est6.predict(model6, X_test)
    >>> np.round(np.mean(np.square(yhat6_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0602

    >>> est7 = DCFEstimator(variant=2, is_convex=False,
    ...                     train_args={'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'L_sum_regularizer': 0.01,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model7 = est7.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model7.weights.shape
    (13, 4)
    >>> np.round([model7._socp_stats.obj_val, model7._socp_stats.proj_obj_val], decimals=4)
    array([0.0575, 0.0668])
    >>> np.round(np.max(np.linalg.norm(model7.weights[:, 1:], axis=1))**2, decimals=4)
    2.647
    >>> yhat7 = est7.predict(model7, X)
    >>> np.round(np.mean(np.square(yhat7 - y)), decimals=4)  # in-sample L2-risk
    0.13
    >>> yhat7_test = est7.predict(model7, X_test)
    >>> np.round(np.mean(np.square(yhat7_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.186

    >>> est8 = DCFEstimator(variant=2, is_convex=False,
    ...                     train_args={'L_sum_regularizer': 0.01,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'local_opt_L_regularizer_offset': 2.0,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model8 = est8.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model8.weights.shape
    (13, 4)
    >>> np.round([model8._socp_stats.obj_val, model8._socp_stats.proj_obj_val], decimals=4)
    array([0.0575, 0.0668])
    >>> np.round(np.max(np.linalg.norm(model8.weights[:, 1:], axis=1))**2, decimals=4)
    2.647
    >>> yhat8 = est8.predict(model8, X)
    >>> np.round(np.mean(np.square(yhat8 - y)), decimals=4)  # in-sample L2-risk
    0.13
    >>> yhat8_test = est8.predict(model8, X_test)
    >>> np.round(np.mean(np.square(yhat8_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.186

    >>> est9 = DCFEstimator(variant=2, is_convex=False,
    ...                     train_args={'normalize': False,
    ...                                 'L_sum_regularizer': '(x_radius**2)/n',
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'local_opt_L_regularizer_offset': 'np.sqrt(np.log(n))',
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model9 = est9.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model9.weights.shape
    (15, 4)
    >>> np.round([model9._socp_stats.obj_val, model9._socp_stats.proj_obj_val], decimals=4)
    array([1.1057, 1.301 ])
    >>> np.round(np.max(np.linalg.norm(model9.weights[:, 1:], axis=1))**2, decimals=4)
    2.9284
    >>> yhat9 = est9.predict(model9, X)
    >>> np.round(np.mean(np.square(yhat9 - y)), decimals=4)  # in-sample L2-risk
    1.1377
    >>> yhat9_test = est9.predict(model9, X_test)
    >>> np.round(np.mean(np.square(yhat9_test - y_test)), decimals=4)  # out-of-sample L2-error
    1.1269

    >>> est10 = DCFEstimator(variant=2, is_convex=False,
    ...                      train_args={'normalize': True,
    ...                                  'L_sum_regularizer': '(x_radius**2)/n',
    ...                                  'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                  'L_regularizer_offset': 'np.log(n)',
    ...                                  'local_opt_type': 'smooth',
    ...                                  'local_opt_maxiter': 10,
    ...                                  'local_opt_L_regularizer_offset': 'np.sqrt(np.log(n))',
    ...                                  'backend': SOCP_BACKEND__CLARABEL})
    >>> model10 = est10.train(X, y, L_regularizer=0.0, L_regularizer_offset=0.0)
    >>> model10.weights.shape
    (15, 4)
    >>> np.round([model10._socp_stats.obj_val, model10._socp_stats.proj_obj_val], decimals=4)
    array([0.1761, 0.2072])
    >>> np.round(np.max(np.linalg.norm(model10.weights[:, 1:], axis=1))**2, decimals=4)
    1.0035
    >>> yhat10 = est10.predict(model10, X)
    >>> np.round(np.mean(np.square(yhat10 - y)), decimals=4)  # in-sample L2-risk
    1.0352
    >>> yhat10_test = est10.predict(model10, X_test)
    >>> np.round(np.mean(np.square(yhat10_test - y_test)), decimals=4)  # out-of-sample L2-error
    1.0365
    """
    pass


def _dcf_l2_local_smooth_mma_tests():
    """
    >>> from ai.gandg.optim.socprog import SOCP_BACKEND__CLARABEL
    >>> np.set_printoptions(legacy='1.25')
    >>> from ai.gandg.common.util import set_random_seed
    >>> set_random_seed(19)

    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2
    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])

    # L2-error of OLS is bigger than 6.
    >>> X_test = np.random.randn(500, 2)
    >>> y_test = regression_func(X_test)
    >>> ols_model = np.linalg.lstsq(X.T.dot(X), X.T.dot(y), rcond=-1)[0]
    >>> ols_yhat_test = np.sum(X_test * ols_model, axis=1)  # np.dot is not deterministic
    >>> np.round(np.sum(np.square(ols_yhat_test - y_test)) / len(y_test), decimals=4)  # OLS out-of-sample L2-error
    6.2752

    >>> est1 = DCFEstimator(variant='mma', is_convex=False,
    ...                     train_args={'L_sum_regularizer': 0.0,
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'warn_on_nok_weights': True,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model1 = est1.train(X, y)
    >>> model1.is_mma()
    True
    >>> np.round(model1.get_maxL(), decimals=2)
    4.01
    >>> model1.weights.shape
    (14, 4, 3)
    >>> np.round(model1._socp_stats.reg_var_value, decimals=4)
    0.0002
    >>> yhat1 = est1.predict(model1, X)
    >>> np.round(np.mean(np.square(yhat1 - y)), decimals=3)  # in-sample L2-risk
    0.009
    >>> yhat1_test = est1.predict(model1, X_test)
    >>> np.round(np.mean(np.square(yhat1_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0093
    >>> np.round([model1._socp_stats.obj_val, model1._socp_stats.proj_obj_val,
    ...           model1._local_opt_stats.init_obj_val, model1._local_opt_stats.soln_obj_val],
    ...          decimals=4)
    array([0.002 , 0.0021, 0.0021, 0.0007])

    >>> est2 = DCFEstimator(variant='mma', is_convex=False,
    ...                     train_args={'L_sum_regularizer': '(x_radius**2)/n',
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model2 = est2.train(X, y)
    >>> model2.is_mma()
    True
    >>> np.round(model2.get_maxL(), decimals=4)
    1.3728
    >>> model2.weights.shape
    (6, 4, 3)
    >>> np.round(model2._socp_stats.reg_var_value, decimals=4)
    0.0
    >>> yhat2 = est2.predict(model2, X)
    >>> np.round(np.mean(np.square(yhat2 - y)), decimals=4)  # in-sample L2-risk
    1.022
    >>> yhat2_test = est2.predict(model2, X_test)
    >>> np.round(np.mean(np.square(yhat2_test - y_test)), decimals=4)  # out-of-sample L2-error
    1.0078
    >>> np.round([model2._socp_stats.obj_val, model2._socp_stats.proj_obj_val,
    ...           model2._local_opt_stats.init_obj_val, model2._local_opt_stats.soln_obj_val],
    ...          decimals=4)
    array([0.186 , 0.2178, 0.515 , 0.1628])

    >>> est3 = DCFEstimator(variant='mma', is_convex=False, is_symmetrized=True,
    ...                     train_args={'L_sum_regularizer': '(x_radius**2)/n',
    ...                                 'L_regularizer': 'max(1.0, x_radius)**2 * (K/n)',
    ...                                 'L_regularizer_offset': 'np.log(n)',
    ...                                 'local_opt_type': 'smooth',
    ...                                 'local_opt_maxiter': 10,
    ...                                 'backend': SOCP_BACKEND__CLARABEL})
    >>> model3 = est3.train(X, y)
    >>> model3.is_mma()
    True
    >>> model3.is_symmetrized
    True
    >>> np.round(model3.get_maxL(), decimals=4)
    1.0964
    >>> len(model3.weights)
    2
    >>> model3.weights[0].shape
    (5, 4, 3)
    >>> model3.weights[1].shape
    (7, 4, 3)
    >>> np.round(model3._socp_stats.reg_var_value, decimals=4)
    0.0
    >>> yhat3 = est3.predict(model3, X)
    >>> np.round(np.mean(np.square(yhat3 - y)), decimals=4)  # in-sample L2-risk
    0.6714
    >>> yhat3_test = est3.predict(model3, X_test)
    >>> np.round(np.mean(np.square(yhat3_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.6812
    >>> np.round([model3._socp_stats.obj_val, model3._socp_stats.proj_obj_val,
    ...           model3._local_opt_stats.init_obj_val, model3._local_opt_stats.soln_obj_val],
    ...          decimals=4)
    array([0.1276, 0.1382, 0.3631, 0.1376])
    """
    pass
