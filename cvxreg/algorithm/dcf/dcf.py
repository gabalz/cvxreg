import gc
import copy
import warnings
import numpy as np
from functools import partial
from collections import namedtuple
from timeit import default_timer as timer
from scipy.sparse import csc_matrix, block_diag, coo_matrix, vstack, triu

from cvxreg.common.estimator import EstimatorModel, Estimator
from cvxreg.common.regression import prepare_prediction, postprocess_prediction
from cvxreg.optim.socprog import (
    socp_solve, SOCP_BACKEND__LBFGS, convert_matrix_to_socp_solver_format,
    socp_nonnegative_cone, socp_second_order_cone,
)
from cvxreg.common.distance import euclidean_distance
from cvxreg.common.partition import (
    Partition, voronoi_partition, cell_radii, find_closest_centers,
)
from cvxreg.algorithm.apcnls.fpc import (
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
    """Normalizing the bias terms without changing the training risk."""
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
        from cvxreg.algorithm.dcf.optim_task import SmoothMaxMinAffineOptimTask
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
            from cvxreg.algorithm.dcf.optim_task import NonSmoothDCFLocalOptimTask
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
            from cvxreg.algorithm.dcf.optim_task import SmoothDCFLocalOptimTask
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

    from cvxreg.optim.lbfgs import LBFGS
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
