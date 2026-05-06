import numpy as np
import scipy as sp
from functools import partial
from sklearn.model_selection import KFold

from cvxreg.common.estimator import Estimator
from cvxreg.algorithm.apcnls.fpc import (
    adaptive_farthest_point_clustering,
    FPC_FIRST_IDX__DEFAULT, FPC_FIRST_IDX__RANDOM,
)


def nearest_neighbors_train(X, y, **kwargs):
    from sklearn.neighbors import KNeighborsRegressor
    kwargs = dict(kwargs)
    verbose = kwargs.pop('verbose', False)
    if 'L' in kwargs:
        del kwargs['L']
    if 'n_neighbors' in kwargs and isinstance(kwargs['n_neighbors'], str):
        n, d = X.shape
        k_str = kwargs['n_neighbors']
        if k_str == 'AFPC':
            k = get_afpc_k(X,
                           kwargs.pop('afpc_ntrials', 1),
                           kwargs.pop('afpc_q', 1))
        else:
            k = int(np.ceil(eval(k_str)))
        if verbose:
            print(f'kNN, k: {k_str} -> {k}')
        if 'cv' in kwargs:
            k = get_cv_k(X, y, k, kwargs.pop('cv'), kwargs.pop('nkcands', 100))
            if verbose:
                print(f'kCV: {k}')
        kwargs['n_neighbors'] = k
    knn = KNeighborsRegressor(**kwargs)
    knn.fit(X, y)
    return knn


def knn_predict(model, X, nsplit=2500):
    n = X.shape[0]
    yhat = np.zeros(n)
    splits = list(range(0, n, nsplit))
    splits = zip(splits, splits[1:] + [n])
    for (split_start, split_end) in splits:
        Xsplit = X[split_start:split_end, :]
        yhat[split_start:split_end] = model.predict(Xsplit)
    return yhat


class NearestNeighborsEstimator(Estimator):
    def __init__(self, **kwargs):
        Estimator.__init__(
            self,
            train=partial(nearest_neighbors_train, **kwargs),
            predict=knn_predict,
        )


def get_afpc_k(X, afpc_ntrials, afpc_q):
    k = 1
    for afpc_trial in range(afpc_ntrials):
        partition = adaptive_farthest_point_clustering(
            data=X, q=afpc_q,
            first_idx=(FPC_FIRST_IDX__DEFAULT if afpc_trial == 0
                       else FPC_FIRST_IDX__RANDOM),
        )
        k = max(k, partition.ncells)
    return k


def _get_k_cv_mse(X, y, trees, k, cv_splits):
    mses = []
    for i, (_, test_inds) in enumerate(cv_splits):
        inds = trees[i].query(X[test_inds, :], k=k)[1]
        yhat = y[inds]
        if len(yhat.shape) > 1:
            yhat = yhat.mean(axis=1)
        elif len(test_inds) == 1:
            yhat = np.mean(yhat)
        mses.append(np.mean(np.square(yhat - y[test_inds])))
    return np.mean(mses)


def get_cv_k(X, y, k, cv, nkcands):
    trees = []
    cv_splits = list(KFold(n_splits=cv).split(X))
    for (train_inds, _) in cv_splits:
        trees.append(sp.spatial.cKDTree(X[train_inds, :], leafsize=100))
    if k <= nkcands:
        k_cands = list(range(1, int(k)))
    else:
        k_cands = sorted(set([int(np.round(_k))
                              for _k in np.linspace(1, k, nkcands)]))
    assert len(k_cands) > 0, f'No k-NN candidates, k:{k}, nkcands:{nkcands}'
    k_mse = np.inf
    for k_cand in k_cands:
        k_cand_mse = _get_k_cv_mse(X, y, trees, k_cand, cv_splits)
        # print(f'k_cand: {k_cand}, k_cand_mse: {k_cand_mse:.4f}')
        if k_cand_mse < k_mse:
            k = k_cand
            k_mse = k_cand_mse
    return k
