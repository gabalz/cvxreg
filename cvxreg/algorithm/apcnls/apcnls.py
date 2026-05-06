import numpy as np
from functools import partial
from timeit import default_timer as timer
from scipy.sparse import block_diag, coo_matrix, triu

from cvxreg.common.estimator import EstimatorModel, Estimator
from cvxreg.common.regression import max_affine_predict, partition_predict
from cvxreg.optim.quadprog import (
    qp_solve, convert_matrix_to_qp_solver_format, QP_BACKEND__DEFAULT,
)
from cvxreg.common.partition import cell_radii
from cvxreg.algorithm.apcnls.fpc import (
    adaptive_farthest_point_clustering, get_data_radius,
)


class APCNLSEstimator(Estimator):
    """Adaptively Partitioning Convex Nonparametric Least Squares (APCNLS) estimator."""
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
        obj_val, proj_obj_val, max_viol, L_sum_regularizer, dual_vars,
        xmean, xscale, yscale, ymean,
    ):
        EstimatorModel.__init__(self, weights, xmean, xscale, yscale, ymean)
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
        self.L_sum_regularizer = L_sum_regularizer
        self.nqpiter = nqpiter
        self.afpc_seconds = afpc_seconds
        self.qp_seconds = qp_seconds
        self.obj_val = obj_val
        self.proj_obj_val = proj_obj_val
        self.max_viol = max_viol
        self.dual_vars = dual_vars


def _data_preprocess(X, y):
    xmean = np.mean(X)
    X = X - xmean
    xscale = np.sqrt(np.max(np.sum(np.square(X), axis=1)))
    X /= xscale
    ymean = np.mean(y)
    y = y - ymean
    yscale = np.max(abs(y))
    y /= yscale
    return X, y, xmean, xscale, ymean, yscale


def _parse_params(
    L_sum_regularizer, L_regularizer,
    L_regularizer_offset, v_regularizer,
    X, K, afpc_eps,
):
    n, d = X.shape
    afpc_eps  # might be used within the following evals

    x_radius = get_data_radius(X)
    if L_sum_regularizer == 'AUTO':
        L_sum_regularizer = x_radius**2 * (d/n)
    elif isinstance(L_sum_regularizer, str):
        L_sum_regularizer = eval(L_sum_regularizer)
    if L_regularizer == 'AUTO':
        L_regularizer = max(1.0, x_radius)**2 * (K/n)
    elif isinstance(L_regularizer, str):
        L_regularizer = eval(L_regularizer)
    if L_regularizer_offset == 'AUTO':
        L_regularizer_offset = np.log(n)
    elif isinstance(L_regularizer_offset, str):
        L_regularizer_offset = eval(L_regularizer_offset)
    if v_regularizer == 'AUTO':
        v_regularizer = d * np.log(n)
    elif isinstance(v_regularizer, str):
        v_regularizer = eval(v_regularizer)

    return L_sum_regularizer, L_regularizer, L_regularizer_offset, v_regularizer


def apcnls_train(
    X, y,
    L_sum_regularizer=0.0, use_L=False, L=None, override_L=None,
    L_regularizer=None, L_regularizer_offset='AUTO',
    afpc_q=1, data_preprocess=False,
    use_V0=False, v_regularizer='AUTO',
    backend=QP_BACKEND__DEFAULT,
    verbose=False, init_weights=None, init_dual_vars=None,
):
    """Training an APCNLS estimator.

    :param X: data matrix (each row is a sample)
    :param y: target vector
    :param partition: partition to be induced by the trained max-affine function
    :param L_sum_regularizer: ridge regularization parameter on the gradients (sum-grad)
    :param use_L: use L if provided
    :param L: maximum Lipschitz constant (as the max-norm of the gradients)
    :param override_L: use this L instead of the provided one
    :param L_regularizer: soft constraint scaler on Lipschitz constant (max-grad)
    :param L_regularizer_offset: until this value the regularization should be zero
    :param afpc_q: scaling parameter for AFPC stopping rule
    :param data_preprocess: perform recommended data preprocessing
    :param use_V0: set a hard constraint on the maximum max-affine violation
    :param v_regularizer: svaling the L2-regularizer of V
    :param backend: quadratic programming solver
    :param verbose: whether to print verbose output
    :param init_weights: warm starting weights for QP
    :param init_dual_vars: warm starting dual variables for QP
    :return: APCNLSEstimatorModel object having the results
    """
    n, d = X.shape
    assert len(y) == n
    assert n > d
    if len(y.shape) > 1:
        assert len(y.shape) == 2 and y.shape[1] == 1
        y = y.ravel()

    if verbose > 0:
        print('Training APCNLS, n: {}, d: {}, L: {}'.format(
            X.shape[0], X.shape[1], L,
        ))

    if not use_L:
        L = None
    elif override_L is not None:
        L = override_L
    if isinstance(L, str):
        L = eval(L)

    if data_preprocess:
        X, y, xmean, xscale, ymean, yscale = _data_preprocess(X, y)
    else:
        xmean = xscale = None
        ymean = yscale = None

    start = timer()
    partition = adaptive_farthest_point_clustering(data=X, q=afpc_q)
    K = float(partition.ncells)
    afpc_seconds = timer() - start

    partition_radius = max(cell_radii(X, partition))
    (L_sum_regularizer, L_regularizer,
     L_regularizer_offset, v_regularizer) = _parse_params(
         L_sum_regularizer, L_regularizer,
         L_regularizer_offset, v_regularizer,
         X, K, afpc_eps=partition_radius,
    )

    start = timer()
    V0 = None if L is None or not use_V0 else 2*L*partition_radius
    H, g, A, b, cell_idx = apcnls_qp_data(
        X, y, partition,
        L_sum_regularizer=L_sum_regularizer, L=L,
        L_regularizer=L_regularizer,
        L_regularizer_offset=L_regularizer_offset,
        V0=V0, v_regularizer=v_regularizer,
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
    obj_val = (
        0.5 * (weights.dot(H.dot(weights)) + weights.dot(triu(H, 1).dot(weights)))
        + g.dot(weights)
    )

    L_est = None
    if L_regularizer is not None:
        L_est = weights[-1]
        weights = weights[:-1]
    V = weights[-1]
    weights = np.reshape(weights[:-1], (partition.ncells, (1+X.shape[1])))
    yhat = max_affine_predict(weights, X)
    proj_obj_val = 0.5 * (np.mean(np.square(y - yhat)) - y.dot(y)/n)

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
        L_sum_regularizer=L_sum_regularizer,
        dual_vars=dual_vars,
        xmean=xmean,
        xscale=xscale,
        yscale=yscale,
        ymean=ymean,
    )


def _add_L_V0_to_Ab(L, L_regularizer, L_regularizer_offset,
                    V0, K, d, A_data, A_rows, A_cols, row_idx):
    """
    Adding the Lipschitz constraints and the upper bound on the constraint relaxation (V)
    to the end of the constraint parameters A and b.
    """
    d1 = d + 1
    row_idx_1 = row_idx
    if L is not None:
        for k in range(K):
            col0 = k * d1 + 1
            for ll in range(d):
                A_data += [1.0, -1.0]
                A_rows += [row_idx, row_idx + 1]
                row_idx += 2
                col = col0 + ll
                A_cols += [col, col]
    elif L_regularizer is not None:
        Kd1 = K * d1
        for k in range(K):
            col0 = k * d1 + 1
            for ll in range(d):
                A_data += [1.0, -1.0, -1.0, -1.0]
                A_rows += [row_idx, row_idx, row_idx+1, row_idx+1]
                row_idx += 2
                col = col0 + ll
                A_cols += [col, Kd1+1, col, Kd1+1]
    L_shift = 0
    row_idx_2 = row_idx
    if V0 is not None:
        L_shift = 1
        A_data += [1.0]
        A_rows += [row_idx]
        row_idx += 1
        A_cols += [K*(d+1)]

    b = np.zeros(row_idx)
    if L is not None:
        b[-2*K*d-L_shift:] = L
    elif L_regularizer is not None:
        b[row_idx_1:row_idx_2] = L_regularizer_offset
    if V0 is not None:
        b[-1] = V0

    return A_data, A_rows, A_cols, row_idx, b


def apcnls_qp_data(
    X, y, partition,
    L_sum_regularizer=0.0, L=None,
    L_regularizer=None,
    L_regularizer_offset=0.0,
    V0=None, v_regularizer=1.0,
    backend=QP_BACKEND__DEFAULT,
):
    """Constructing max-affine convex regression matrices for quadratic programming (QP).
    QP format: 0.5*(w'*H*w) + g'*w + 0.5*regularizer*(w'*w), s.t. A*w <= b and max_i|w[i]| <= L.

    :param X: data matrix (each row is a sample, without augmented leading 1s)
    :param y: target vector
    :param partition: induced partition by the considered max-affine functions
    :param L_sum_regularizer: ridge regression regularizer
    :param L: maximum Lipschitz constant (as the max-norm of the gradients)
    :param L_regularizer: scaler for soft L regularization
    :param L_regularizer_offset: up to this point L_regularization should be zero
    :param V0: maximum max-affine constraint violation
    :param v_regularizer: scaler for the L2-regularizer of V
    :param backend: quadratic programming solver
    :return: QP parameters H, g, A, b, and the constraint row index for each cell
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
    if L_sum_regularizer > 0.0:
        regmat = L_sum_regularizer * np.eye(d1)
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
        cellXX = triu(cellXX).tocsc()
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
        L, L_regularizer, L_regularizer_offset,
        V0, K, d, A_data, A_rows, A_cols, row_idx,
    )

    H_mats.append(np.array([[n*v_regularizer]]))
    g_mats.append(np.array([0.0]))
    if L_regularizer is not None:
        H_mats.append(np.array([[n*L_regularizer]]))
        g_mats.append(np.array([0.0]))

    H = block_diag(H_mats, format='csc')
    H /= n
    g = np.concatenate(g_mats)
    g /= n
    A = coo_matrix((A_data, (A_rows, A_cols)), shape=(row_idx, nvars)).tocsc()

    assert H.shape == (nvars, nvars), 'Invalid H.shape: {}'.format(H.shape)
    assert g.shape == (nvars,), 'Invalid g.shape: {}'.format(g.shape)
    assert A.shape == (row_idx, nvars), 'Invalid A.shape: {}'.format(A.shape)
    assert b.shape == (row_idx,), 'Invalid b.shape: {}'.format(b.shape)

    H = convert_matrix_to_qp_solver_format(H, backend)
    A = convert_matrix_to_qp_solver_format(A, backend)
    return H, g, A, b, np.array(cell_idx)
