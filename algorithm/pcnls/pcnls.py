import sys
import numpy as np

from timeit import default_timer as timer
from scipy.sparse import block_diag, coo_matrix

from common.estimator import EstimatorModel
from common.regression import max_affine_predict
from optim.quadprog import qp_solve, convert_matrix_to_qp_solver_format, QP_BACKEND__DEFAULT


class PCNLSEstimatorModel(EstimatorModel):
    """The model of PCNLS estimators."""
    def __init__(
        self, weights, nqpiter, seconds,
        obj_val, proj_obj_val, max_viol, regularizer, dual_vars,
    ):
        EstimatorModel.__init__(self, weights)
        self.regularizer = regularizer
        self.nqpiter = nqpiter
        self.seconds = seconds
        self.obj_val = obj_val
        self.proj_obj_val = proj_obj_val
        self.max_viol = max_viol
        self.dual_vars = dual_vars


def pcnls_train(
    X, y, partition,
    regularizer=0.0, use_L=True, scale_L=1.0, override_L=None, L=None, L_regularizer=None,
    backend=QP_BACKEND__DEFAULT,
    verbose=False, init_weights=None, init_dual_vars=None,
):
    """Training a PCNLS estimator.

    :param X: data matrix (each row is a sample)
    :param y: target vector
    :param partition: partition to be induced by the trained max-affine function
    :param regularizer: ridge regularization parameter on the gradients
    :param use_L: use L if provided
    :param L: maximum Lipschitz constant (as the max-norm of the gradients)
    :param override_L: use this L instead of the provided one
    :param L_regularizer: soft constraint scaler on Lipschitz constant
    :param backend: quadratic programming solver
    :param verbose: whether to print verbose output
    :param init_weights: warm starting weights for QP
    :param init_dual_vars: warm starting dual variables for QP
    :return: PCNLSEstimatorModel object having the results
    """
    n, d = X.shape
    assert len(y) == n
    if len(y.shape) > 1:
        assert len(y.shape) == 2 and y.shape[1] == 1
        y = y.ravel()

    if verbose > 0:
        print('Training PCNLS, n: {}, K: {}, d: {}, L: {}, regularizer: {}'.format(
            partition.npoints, partition.ncells, X.shape[1], L, regularizer,
        ))

    if not use_L:
        L = None
    elif override_L is not None:
        L = override_L
    if isinstance(L, str):
        L = eval(L)

    if isinstance(regularizer, str):
        regularizer = eval(regularizer)
    if isinstance(L_regularizer, str):
        L_regularizer = eval(L_regularizer)
    if isinstance(scale_L, str):
        scale_L = eval(scale_L)
    if L is not None and scale_L is not None:
        L *= scale_L

    start = timer()
    H, g, A, b, cell_idx = pcnls_qp_data(
        X, y, partition,
        regularizer=regularizer, L=L, L_regularizer=L_regularizer,
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

    if L_regularizer is not None:
        L_est = weights[-1]
        weights = weights[:-1]
    weights = np.reshape(weights, (partition.ncells, (1+X.shape[1])))
    yhat = max_affine_predict(weights, X)
    proj_obj_val = 0.5 * (np.sum(np.square(y - yhat)) - y.dot(y))

    model = PCNLSEstimatorModel(
        weights=weights,
        nqpiter=nqpiter,
        seconds=(timer() - start),
        obj_val=obj_val,
        proj_obj_val=proj_obj_val,
        max_viol=max_viol,
        regularizer=regularizer,
        dual_vars=dual_vars,
    )
    if L_regularizer is not None:
        model.L_est = L_est
    return model


def _add_L_to_Ab(L, L_regularizer, K, d, A_data, A_rows, A_cols, row_idx):
    """Adding the Lipschitz constraints to the end of the constraint parameters A and b."""
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
        Kd1 = K*d1
        for k in range(K):
            col0 = k * d1 + 1
            for l in range(d):
                A_data += [1.0, -1.0, -1.0, -1.0]
                A_rows += [row_idx, row_idx, row_idx+1, row_idx+1]
                row_idx += 2
                col = col0 + l
                A_cols += [col, Kd1, col, Kd1]

    b = np.zeros(row_idx)
    if L is not None:
        b[-2*K*d:] = L

    return A_data, A_rows, A_cols, row_idx, b


def pcnls_qp_data(
    X, y, partition,
    regularizer=0.0, L=None, L_regularizer=None,
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
    :param backend: quadratic programming solver
    :return: QP parameters H, g, A, b, and the constraint row index for each cell

    >>> np.set_printoptions(legacy='1.25')
    >>> from common.partition import singleton_partition
    >>> X = np.array([[1.1, 1.1], [-1.2, 1.2], [-1.3, -1.3], [0.4, 0.4], [1.5, -1.5]])
    >>> y = np.array([1.1, 1.2, 1.3, 0.4, 0.5])
    >>> p = singleton_partition(len(y))
    >>> H, g, A, b, cell_idx = pcnls_qp_data(X, y, p, regularizer=0.1)
 
    >>> cell_idx
    array([ 0,  4,  8, 12, 16])
        
    >>> H.shape
    (15, 15)
    >>> np.linalg.matrix_rank(H.toarray())
    15
    >>> H.nnz
    45
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
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]])
    >>> g
    array([-1.1 , -1.21, -1.21, -1.2 ,  1.44, -1.44, -1.3 ,  1.69,  1.69,
           -0.4 , -0.16, -0.16, -0.5 , -0.75,  0.75])

    >>> A.shape
    (20, 15)
    >>> np.linalg.matrix_rank(A.toarray())
    12
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
    120
    >>> b.shape
    (20,)
    >>> np.sum(np.abs(b))
    0.0

    >>> H, g, A, b, cell_idx = pcnls_qp_data(X, y, p, L=5.0)
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
    >>> b
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
           5., 5., 5., 5., 5., 5.])

    >>> H, g, A, b, cell_idx = pcnls_qp_data(X, y, p, L_regularizer=7.0)
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
    >>> A.toarray()[-5:, :]
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,
             0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             1.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            -1.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0.,  1., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
             0., -1., -1.]])
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
    nvars = K*d1 + int(L_regularizer is not None)  # number of variables of QP

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
    for j, cell_j in enumerate(partition.cells):
        cell_idx.append(row_idx)
        cellX = X[cell_j, :]
        cell_size = cellX.shape[0]
        cellXX = np.dot(cellX.transpose(), cellX)
        if regmat is not None:
            cellXX += regmat
        H_mats.append(cellXX)
        g_mats.append(-np.dot(cellX.transpose(), y[cell_j]))
        data = list(np.hstack((cellX, -cellX)).flatten())
        row_offsets = np.kron(range(cell_size), np.ones(2*d1))
        col_j = [j*d1+offset for offset in range(d1)]
        for k in range(K):
            if k == j:
                continue
            rows = [row_idx + row for row in row_offsets]
            row_idx += cell_size
            col_k = [k*d1+offset for offset in range(d1)]
            A_data += data
            A_rows += rows
            A_cols += list(np.kron(col_k + col_j, np.ones((cell_size, 1))).flatten())
    A_data, A_rows, A_cols, row_idx, b = _add_L_to_Ab(
        L, L_regularizer, K, d, A_data, A_rows, A_cols, row_idx,
    )

    if L_regularizer is not None:
        H_mats.append(np.array([[L_regularizer]]))
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
