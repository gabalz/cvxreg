
import sys
import numpy as np

from functools import partial
from timeit import default_timer as timer
from scipy.sparse import block_diag, coo_matrix

from common.estimator import EstimatorModel, Estimator
from common.regression import max_affine_predict, partition_predict
from optim.quadprog import (
    qp_solve, convert_matrix_to_qp_solver_format, QP_BACKEND__DEFAULT,
)
from algorithm.apcnls.fpc import (
    adaptive_farthest_point_clustering, cell_radiuses, get_data_radius,
)


class APCNLSEstimator(Estimator):
    """Adaptively Partitioning Convex Nonparametric Least Squares (APCNLS) estimator.

    >>> from common.util import set_random_seed
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

    >>> apcnls1 = APCNLSEstimator()
    >>> model1 = apcnls1.train(X, y)  # missing Lipschitz constant
    >>> model1.weights.shape
    (9, 3)
    >>> np.round(model1.V, decimals=4)
    0.0384
    >>> model1.V0 is None
    True
    >>> model1.obj_val
    -925.3174570854741
    >>> model1.proj_obj_val
    -927.0106351998556
    >>> np.round(model1.train_diff, decimals=4)
    0.0058
    >>> np.round(model1.cell_diff_max, decimals=4)
    1.2003
    >>> np.round(model1.partition_radius, decimals=4)
    2.0427
    >>> np.round(model1.proj_diff_corr, decimals=4)
    -0.0007
    >>> yhat1 = apcnls1.predict(model1, X)
    >>> np.round(np.sum(np.square(yhat1 - y)) / len(y), decimals=4)  # in-sample L2-risk
    0.3683
    >>> yhat1_test = apcnls1.predict(model1, X_test)
    >>> np.round(np.sum(np.square(yhat1_test - y_test)) / len(y_test), decimals=4)  # out-of-sample L2-error
    0.4312

    >>> apcnls2 = APCNLSEstimator()
    >>> model2 = apcnls2.train(X, y, use_L=True, L=4.5, use_V0=True,
    ...                        L_regularizer=None, V_regularizer=1.0/X.shape[0])  # good Lipschitz constant
    >>> model2.weights.shape
    (9, 3)
    >>> np.round(model2.V, decimals=4)
    1.2982
    >>> np.round(model2.V0, decimals=4)
    18.3841
    >>> np.round(model2.partition_radius, decimals=4)
    2.0427
    >>> np.round(model2.proj_diff_corr, decimals=4)
    -0.0983
    >>> yhat2 = apcnls2.predict(model2, X)
    >>> np.round(np.sum(np.square(yhat2 - y)) / len(y), decimals=4)  # in-sample L2-risk
    0.1396
    >>> yhat2_test = apcnls2.predict(model2, X_test)
    >>> np.round(np.sum(np.square(yhat2_test - y_test)) / len(y_test), decimals=4)  # out-of-sample L2-error
    0.1452

    >>> apcnls3 = APCNLSEstimator()
    >>> model3 = apcnls3.train(X, y, use_L=True, L=4.5, use_V0=False,
    ...                        L_regularizer=None, V_regularizer=1.0)  # good Lipschitz constant
    >>> model3.weights.shape
    (9, 3)
    >>> np.round(model3.V, decimals=4)
    0.218
    >>> model3.V0 is None
    True
    >>> np.round(model3.partition_radius, decimals=4)
    2.0427
    >>> np.round(model3.proj_diff_corr, decimals=4)
    -0.0117
    >>> yhat3 = apcnls3.predict(model3, X)
    >>> np.round(np.sum(np.square(yhat3 - y)) / len(y), decimals=4)  # in-sample L2-risk
    0.2901
    >>> yhat3_test = apcnls3.predict(model3, X_test)
    >>> np.round(np.sum(np.square(yhat3_test - y_test)) / len(y_test), decimals=4)  # out-of-sample L2-error
    0.3243

    >>> set_random_seed(17)

    >>> def regression_func(X):
    ...     return 0.5 * np.sum(np.square(X), axis=1)
    >>> X = np.random.randn(300, 5)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])

    # L2-error of OLS is bigger than 9.
    >>> X_test = np.random.randn(500, 5)
    >>> y_test = regression_func(X_test)
    >>> ols_model = np.linalg.lstsq(X.T.dot(X), X.T.dot(y), rcond=-1)[0]
    >>> ols_yhat_test = np.sum(X_test * ols_model, axis=1)  # np.dot is not deterministic
    >>> np.round(np.sum(np.square(ols_yhat_test - y_test)) / len(y_test), decimals=4)  # OLS out-of-sample L2-error
    9.2304

    >>> apcnls4 = APCNLSEstimator()
    >>> model4 = apcnls4.train(X, y)  # good Lipschitz constant
    >>> model4.weights.shape
    (25, 6)
    >>> np.round(model4.obj_val, decimals=1)
    -1265.7
    >>> np.round(model4.proj_obj_val, decimals=1)
    -1265.7
    >>> np.round(model4.V, decimals=4)
    0.0017
    >>> np.round(model4.partition_radius, decimals=4)
    5.8376
    >>> np.round(model4.train_diff, decimals=4)
    0.0002
    >>> yhat4 = apcnls4.predict(model4, X)
    >>> np.round(np.sum(np.square(yhat4 - y)) / len(y), decimals=4)  # in-sample L2-risk
    0.2199
    >>> yhat4_test = apcnls4.predict(model4, X_test)
    >>> np.round(np.sum(np.square(yhat4_test - y_test)) / len(y_test), decimals=4)  # out-of-sample L2-error
    0.5371
    """
    def __init__(self, train_args={}, predict_args={}):
        Estimator.__init__(
            self,
            train=partial(apcnls_train, **train_args),
            predict=partial(max_affine_predict, **predict_args),
        )    


class APCNLSEstimatorModel(EstimatorModel):
    """The model of APCNLS estimators."""
    def __init__(
        self, weights, V, V0, L_est, partition_radius, proj_diff_corr, train_diff,
        cell_diff_min, cell_diff_max, cell_diff_mean, cell_diff_median, cell_diff_std,
        nqpiter, afpc_seconds, qp_seconds,
        obj_val, proj_obj_val, max_viol, regularizer, dual_vars,
    ):
        EstimatorModel.__init__(self, weights)
        self.V = V
        self.V0 = V0
        self.L_est = L_est
        self.partition_radius = partition_radius
        self.proj_diff_corr = proj_diff_corr
        self.train_diff = train_diff
        self.cell_diff_min = cell_diff_min
        self.cell_diff_max = cell_diff_max
        self.cell_diff_mean = cell_diff_mean
        self.cell_diff_median = cell_diff_median
        self.cell_diff_std = cell_diff_std
        self.regularizer = regularizer
        self.nqpiter = nqpiter
        self.afpc_seconds = afpc_seconds
        self.qp_seconds = qp_seconds
        self.obj_val = obj_val
        self.proj_obj_val = proj_obj_val
        self.max_viol = max_viol
        self.dual_vars = dual_vars


def apcnls_train(
    X, y,
    regularizer=0.0, use_L=False, L=None, L_regularizer=None,
    use_V0=False, V_regularizer='AUTO',
    backend=QP_BACKEND__DEFAULT,
    verbose=False, init_weights=None, init_dual_vars=None,
):
    """Training an APCNLS estimator.

    :param X: data matrix (each row is a sample)
    :param y: target vector
    :param partition: partition to be induced by the trained max-affine function
    :param regularizer: ridge regularization parameter on the gradients
    :param use_L: use L if provided
    :param L: maximum Lipschitz constant (as the max-norm of the gradients)
    :param L_regularizer: soft constraint scaler on Lipschitz constant
    :param use_V0: set a hard constraint on the maximum max-affine violation
    :param V_regularizer: svaling the L2-regularizer of V
    :param backend: quadratic programming solver
    :param verbose: whether to print verbose output
    :param init_weights: warm starting weights for QP
    :param init_dual_vars: warm starting dual variables for QP
    :return: APCNLSEstimatorModel object having the results
    """
    n, d = X.shape
    assert len(y) == n
    if len(y.shape) > 1:
        assert len(y.shape) == 2 and y.shape[1] == 1
        y = y.ravel()

    if verbose > 0:
        print('Training APCNLS, n: {}, d: {}, L: {}, regularizer: {}'.format(
            X.shape[0], X.shape[1], L, regularizer,
        ))

    if not use_L:
        L = None
    elif isinstance(L, str):
        L = eval(L)

    if isinstance(regularizer, str) and regularizer not in ('AUTO', 'DEFAULT'):
        regularizer = eval(regularizer)
    if isinstance(V_regularizer, str) and V_regularizer not in ('DEFAULT', 'AUTO'):
        V_regularizer = eval(V_regularizer)

    start = timer()
    partition = adaptive_farthest_point_clustering(data=X)
    afpc_seconds = timer() - start

    rad_scale = max(1.0, get_data_radius(X))
    if regularizer == 'AUTO':
        regularizer = 1.0
    if L_regularizer == 'AUTO':
        L_regularizer = (rad_scale**2) * d * float(partition.ncells)/n
    if V_regularizer == 'AUTO':
        V_regularizer = d * np.log(n)

    start = timer()
    partition_radius = max(cell_radiuses(X, partition))
    if regularizer == 'DEFAULT':
        regularizer = (partition_radius**2) / float(partition.ncells)
    V0 = None if L is None or not use_V0 else 2*L*partition_radius
    H, g, A, b, cell_idx = apcnls_qp_data(
        X, y, partition,
        regularizer=regularizer, L=L, L_regularizer=L_regularizer,
        V0=V0, V_regularizer=V_regularizer,
    )
    if init_weights is not None:
        init_weights = init_weights.ravel()

    nqpiter = 1
    res = qp_solve(
        H, g, A, b,
        x0=init_weights, y0=init_dual_vars,
        backend=backend, verbose=verbose,
    )
    weights = res.primal_soln
    dual_vars = res.dual_soln

    max_viol = max(0.0, np.max(A.dot(weights) - b))
    obj_val = 0.5*weights.dot(H.dot(weights)) + g.dot(weights)

    L_est = None
    if L_regularizer is not None:
        L_est = weights[-1]
        weights = weights[:-1]
    V = weights[-1]
    weights = np.reshape(weights[:-1], (partition.ncells, (1+X.shape[1])))
    yhat = max_affine_predict(weights, X)
    proj_obj_val = 0.5 * (np.sum(np.square(y - yhat)) - y.dot(y))

    yhat_partition = partition_predict(partition, weights, X)
    cell_diffs = [np.mean((yhat_partition - y)[cell]) for cell in partition.cells]
    proj_diff_corr = np.dot(yhat - yhat_partition, yhat_partition - y) / len(y)

    return APCNLSEstimatorModel(
        weights=weights,
        V=V, V0=V0, L_est=L_est,
        partition_radius=partition_radius,
        train_diff=np.mean(yhat-y),
        cell_diff_min=np.min(cell_diffs),
        cell_diff_max=np.max(cell_diffs),
        cell_diff_mean=np.mean(cell_diffs),
        cell_diff_std=np.std(cell_diffs),
        cell_diff_median=np.median(cell_diffs),
        proj_diff_corr=proj_diff_corr,
        nqpiter=nqpiter,
        afpc_seconds=afpc_seconds,
        qp_seconds=(timer() - start),
        obj_val=obj_val,
        proj_obj_val=proj_obj_val,
        max_viol=max_viol,
        regularizer=regularizer,
        dual_vars=dual_vars,
    )


def _add_L_V0_to_Ab(L, L_regularizer, V0, K, d, A_data, A_rows, A_cols, row_idx):
    """
    Adding the Lipschitz constraints and the upper bound on the constraint relaxation (V)
    to the end of the constraint parameters A and b.
    """
    d1 = d + 1
    if L is not None:
        for k in range(K):
            col0 = k * d1 + 1
            for l in range(d):
                A_data += [1.0, -1.0]
                A_rows += [row_idx, row_idx + 1]
                row_idx += 2
                col = col0 + l
                A_cols += [col, col]
    elif L_regularizer is not None:
        Kd1 = K * d1
        for k in range(K):
            col0 = k * d1 + 1
            for l in range(d):
                A_data += [1.0, -1.0, -1.0, -1.0]
                A_rows += [row_idx, row_idx, row_idx+1, row_idx+1]
                row_idx += 2
                col = col0 + l
                A_cols += [col, Kd1+1, col, Kd1+1]
    L_shift = 0
    if V0 is not None:
        L_shift = 1
        A_data += [1.0]
        A_rows += [row_idx]
        row_idx += 1
        A_cols += [K*(d+1)]

    b = np.zeros(row_idx)
    if L is not None:
        b[-2*K*d-L_shift:] = L
    if V0 is not None:
        b[-1] = V0

    return A_data, A_rows, A_cols, row_idx, b


def apcnls_qp_data(
    X, y, partition,
    regularizer=0.0, L=None, L_regularizer=None,
    V0=None, V_regularizer=1.0,
    backend=QP_BACKEND__DEFAULT,
):
    """Constructing max-affine convex regression matrices for quadratic programming (QP).
    QP format: 0.5*(w'*H*w) + g'*w + 0.5*regularizer*(w'*w), s.t. A*w <= b and max_i|w[i]| <= L.

    :param X: data matrix (each row is a sample, without augmented leading 1s)
    :param y: target vector
    :param partition: induced partition by the considered max-affine functions
    :param regularizer: ridge regression regularizer
    :param L: maximum Lipschitz constant (as the max-norm of the gradients)
    :param L_regularizer: scaler for soft L regularization
    :param V0: maximum max-affine constraint violation
    :param V_regularizer: scaler for the L2-regularizer of V
    :param backend: quadratic programming solver
    :return: QP parameters H, g, A, b, and the constraint row index for each cell

    >>> from common.partition import singleton_partition
    >>> X = np.array([[1.1, 1.1], [-1.2, 1.2], [-1.3, -1.3], [0.4, 0.4], [1.5, -1.5]])
    >>> y = np.array([1.1, 1.2, 1.3, 0.4, 0.5])
    >>> p = singleton_partition(len(y))


    >>> H, g, A, b, cell_idx = apcnls_qp_data(X, y, p, regularizer=0.1)
    >>> cell_idx
    array([ 0,  4,  8, 12, 16])
    >>> H.shape
    (16, 16)
    >>> np.linalg.matrix_rank(H.toarray())
    16
    >>> H.nnz
    46
    >>> H.toarray()[:, :9]
    array([[ 1.  ,  1.1 ,  1.1 ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 1.1 ,  1.31,  1.21,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 1.1 ,  1.21,  1.31,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  1.  , -1.2 ,  1.2 ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  , -1.2 ,  1.54, -1.44,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  1.2 , -1.44,  1.54,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  , -1.3 , -1.3 ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -1.3 ,  1.79,  1.69],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -1.3 ,  1.69,  1.79],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]])
    >>> H.toarray()[-3:, -3:]
    array([[ 2.35, -2.25,  0.  ],
           [-2.25,  2.35,  0.  ],
           [ 0.  ,  0.  ,  5.  ]])
    >>> g
    array([-1.1 , -1.21, -1.21, -1.2 ,  1.44, -1.44, -1.3 ,  1.69,  1.69,
           -0.4 , -0.16, -0.16, -0.5 , -0.75,  0.75,  0.  ])
    >>> A.shape
    (20, 16)
    >>> np.linalg.matrix_rank(A.toarray())
    13
    >>> A.toarray()[:, :9]
    array([[-1. , -1.1, -1.1,  1. ,  1.1,  1.1,  0. ,  0. ,  0. ],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  1. ,  1.1,  1.1],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 1. , -1.2,  1.2, -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  1. , -1.2,  1.2],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 1. , -1.3, -1.3,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  1. , -1.3, -1.3, -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 1. ,  0.4,  0.4,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ,  0.4,  0.4,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0.4,  0.4],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 1. ,  1.5, -1.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ,  1.5, -1.5,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  1.5, -1.5],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
    >>> A.nnz
    140
    >>> b.shape
    (20,)
    >>> np.sum(np.abs(b))
    0.0

    >>> H, g, A, b, cell_idx = apcnls_qp_data(X, y, p, L=5.0)
    >>> A.shape
    (40, 16)
    >>> A.toarray()[:33, :9]
    array([[-1. , -1.1, -1.1,  1. ,  1.1,  1.1,  0. ,  0. ,  0. ],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  1. ,  1.1,  1.1],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 1. , -1.2,  1.2, -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  1. , -1.2,  1.2],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 1. , -1.3, -1.3,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  1. , -1.3, -1.3, -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 1. ,  0.4,  0.4,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ,  0.4,  0.4,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0.4,  0.4],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 1. ,  1.5, -1.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ,  1.5, -1.5,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  1.5, -1.5],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
    >>> A.toarray()[:10, -5:]
    array([[ 0. ,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ],
           [ 1.1,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  1. ,  1.1,  1.1, -1. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ],
           [ 1.2,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  1. , -1.2,  1.2, -1. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ]])
    >>> A.toarray()[-5:, :]
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,
             0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0., -1.,  0.]])
    >>> b
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
           5., 5., 5., 5., 5., 5.])

    >>> H, g, A, b, cell_idx = apcnls_qp_data(X, y, p, L=5.0, V0=0.1)
    >>> A.shape
    (41, 16)
    >>> A.toarray()[:33, :9]
    array([[-1. , -1.1, -1.1,  1. ,  1.1,  1.1,  0. ,  0. ,  0. ],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  1. ,  1.1,  1.1],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 1. , -1.2,  1.2, -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  1. , -1.2,  1.2],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 1. , -1.3, -1.3,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  1. , -1.3, -1.3, -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 1. ,  0.4,  0.4,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ,  0.4,  0.4,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0.4,  0.4],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 1. ,  1.5, -1.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ,  1.5, -1.5,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  1.5, -1.5],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
    >>> A.toarray()[:10, -5:]
    array([[ 0. ,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ],
           [ 1.1,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  1. ,  1.1,  1.1, -1. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ],
           [ 1.2,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  1. , -1.2,  1.2, -1. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ]])
    >>> A.toarray()[-5:, :]
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  1.]])
    >>> b
    array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
           0. , 0. , 0. , 0. , 0. , 0. , 0. , 5. , 5. , 5. , 5. , 5. , 5. ,
           5. , 5. , 5. , 5. , 5. , 5. , 5. , 5. , 5. , 5. , 5. , 5. , 5. ,
           5. , 0.1])

    >>> H, g, A, b, cell_idx = apcnls_qp_data(X, y, p, L_regularizer=7.0, V0=0.1)
    >>> A.shape
    (41, 17)
    >>> A.toarray()[:33, :9]
    array([[-1. , -1.1, -1.1,  1. ,  1.1,  1.1,  0. ,  0. ,  0. ],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  1. ,  1.1,  1.1],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 1. , -1.2,  1.2, -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  1. , -1.2,  1.2],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 1. , -1.3, -1.3,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  1. , -1.3, -1.3, -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 1. ,  0.4,  0.4,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ,  0.4,  0.4,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0.4,  0.4],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 1. ,  1.5, -1.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ,  1.5, -1.5,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  1.5, -1.5],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
    >>> A.toarray()[:10, -5:]
    array([[ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 1. ,  1.1,  1.1, -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 1. , -1.2,  1.2, -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ]])
    >>> A.toarray()[-5:, :]
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             1.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  1.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0., -1.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  1.,  0.]])

    >>> H, g, A, b, cell_idx = apcnls_qp_data(X, y, p, L_regularizer=7.0, V0=0.1)
    >>> A.shape
    (41, 17)
    >>> A.toarray()[:33, :9]
    array([[-1. , -1.1, -1.1,  1. ,  1.1,  1.1,  0. ,  0. ,  0. ],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  1. ,  1.1,  1.1],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 1. , -1.2,  1.2, -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  1. , -1.2,  1.2],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 1. , -1.3, -1.3,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  1. , -1.3, -1.3, -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 1. ,  0.4,  0.4,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ,  0.4,  0.4,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0.4,  0.4],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 1. ,  1.5, -1.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ,  1.5, -1.5,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  1.5, -1.5],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
    >>> A.toarray()[:10, -5:]
    array([[ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 1. ,  1.1,  1.1, -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 1. , -1.2,  1.2, -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ]])
    >>> A.toarray()[-5:, :]
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             1.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  1.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0., -1.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  1.,  0.]])

    >>> H, g, A, b, cell_idx = apcnls_qp_data(X, y, p, L_regularizer=7.0, V0=0.1)
    >>> A.shape
    (41, 17)
    >>> A.toarray()[:33, :9]
    array([[-1. , -1.1, -1.1,  1. ,  1.1,  1.1,  0. ,  0. ,  0. ],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  1. ,  1.1,  1.1],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [-1. , -1.1, -1.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 1. , -1.2,  1.2, -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  1. , -1.2,  1.2],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  1.2, -1.2,  0. ,  0. ,  0. ],
           [ 1. , -1.3, -1.3,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  1. , -1.3, -1.3, -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  1.3,  1.3],
           [ 1. ,  0.4,  0.4,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ,  0.4,  0.4,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0.4,  0.4],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 1. ,  1.5, -1.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ,  1.5, -1.5,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  1.5, -1.5],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
    >>> A.toarray()[:10, -5:]
    array([[ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 1. ,  1.1,  1.1, -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 1. , -1.2,  1.2, -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ],
           [ 0. ,  0. ,  0. , -1. ,  0. ]])
    >>> A.toarray()[-5:, :]
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             1.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  1.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0., -1.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  0.,  1.,  0.]])
    """
    n, d = X.shape
    K = partition.ncells
    assert n == partition.npoints
    assert n >= K, 'Too few data points, n: {}, K: {}'.format(n, K)
    assert n > d, 'Too few data points, n: {}, d: {}'.format(n, d)
    if y.shape == (n, 1):
        y = y[:, 0]
    assert y.shape == (n,), 'Invalid y.shape: {}'.format(y.shape)
    assert L is None or L_regularizer is None

    X = np.insert(X, 0, 1.0, axis=1)
    d1 = d+1  # bias extended input dimension
    assert X.shape == (n, d1)
    Kd1 = K*d1
    nvars = Kd1 + 1 + int(L_regularizer is not None)  # number of variables of QP

    regmat = None
    if regularizer > 0.0:
        regmat = regularizer * np.eye(d1)
        regmat[0, 0] = 0.0

    H_mats = []
    g_mats = []
    A_data = []
    A_rows = []
    A_cols = []
    row_idx = 0
    cell_idx = []
    ones_per_row = np.ones(2*d1 + 1)
    for j, cell_j in enumerate(partition.cells):
        cell_idx.append(row_idx)
        cellX = X[cell_j, :]
        cell_size = cellX.shape[0]
        cellXX = np.dot(cellX.transpose(), cellX)
        if regmat is not None:
            cellXX += regmat
        H_mats.append(cellXX)
        g_mats.append(-np.dot(cellX.transpose(), y[cell_j]))
        data = list(np.hstack([cellX, -cellX, -np.ones((cell_size, 1))]).flatten())
        row_offsets = np.kron(range(cell_size), ones_per_row)
        col_j = [j*d1+offset for offset in range(d1)]
        for k in range(K):
            if k == j:
                continue
            rows = [row_idx + row for row in row_offsets]
            row_idx += cell_size
            col_k = [k*d1+offset for offset in range(d1)]
            A_data += data
            A_rows += rows
            A_cols += list(np.kron(col_k + col_j + [Kd1], np.ones((cell_size, 1))).flatten())
    A_data, A_rows, A_cols, row_idx, b = _add_L_V0_to_Ab(
        L, L_regularizer, V0, K, d, A_data, A_rows, A_cols, row_idx,
    )

    H_mats.append(np.array([[n*V_regularizer]]))
    g_mats.append(np.array([0.0]))
    if L_regularizer is not None:
        H_mats.append(np.array([[n*L_regularizer]]))
        g_mats.append(np.array([0.0]))

    H = block_diag(H_mats, format='csc')
    H = (H + H.transpose()) / 2.  # numerical stabilization of symmetry
    g = np.concatenate(g_mats)
    A = coo_matrix((A_data, (A_rows, A_cols)), shape=(row_idx, nvars)).tocsc()

    assert H.shape == (nvars, nvars), 'Invalid H.shape: {}'.format(H.shape)
    assert g.shape == (nvars,), 'Invalid g.shape: {}'.format(g.shape)
    assert A.shape == (row_idx, nvars), 'Invalid A.shape: {}'.format(A.shape)
    assert b.shape == (row_idx,), 'Invalid b.shape: {}'.format(b.shape)

    H = convert_matrix_to_qp_solver_format(H, backend)
    A = convert_matrix_to_qp_solver_format(A, backend)
    return H, g, A, b, np.array(cell_idx)
