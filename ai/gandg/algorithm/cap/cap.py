import numpy as np
from functools import partial

from ai.gandg.common.util import rand_direction
from ai.gandg.common.distance import squared_distance
from ai.gandg.common.estimator import EstimatorModel, Estimator
from ai.gandg.common.partition import Partition, max_affine_partition
from ai.gandg.common.regression import max_affine_fit_partition, max_affine_predict


class CAPEstimatorModel(EstimatorModel):
    """The model of a CAP estimator."""
    def __init__(self, weights, gcv_error, niter):
        EstimatorModel.__init__(self, weights)
        self.gcv_error = gcv_error
        self.niter = niter


def _cap_loss(y, yhat):
    """Squared loss between two vectors.

    :param y: target vector
    :param yhat: estimate vector
    :return: squared loss
    """
    return squared_distance(y, yhat, axis=0)


def _cap_fit(X, y, betaI):
    """Ridge regression fitting.

    :param X: data matrix (each row is a sample augmented with a leading 1)
    :param y: target vector
    :betaI: identity matrix scaled by the regularization constant
    :return: weight vector of the ridge regressor
    """
    assert len(y) > 0
    assert X.shape[0] == len(y)
    return np.linalg.solve(X.T.dot(X) + betaI, X.T.dot(y))


def _cap_split(cell, xarray, b, mincellsize):
    cell_le = cell[xarray <= b]
    cell_gt = cell[xarray > b]
    return (min(len(cell_le), len(cell_gt)) >= mincellsize), cell_le, cell_gt


def _eval_cap_split(
    P, X1, y, betaI, k, cell_le, cell_gt,
    cand_W, cand_XW, best_cand_err, best_cand_W, best_cand_XW, best_cand_P,
):
    n = P.npoints
    K = P.ncells

    cand_Wk = cand_W[:, k]
    cand_W[:, k] = _cap_fit(X1[cell_le, :], y[cell_le], betaI)
    cand_W[:, K] = _cap_fit(X1[cell_gt, :], y[cell_gt], betaI)

    cand_XWk = cand_XW[:, k]
    cand_XW[:, k] = X1.dot(cand_W[:, k])
    cand_XW[:, K] = X1.dot(cand_W[:, K])
                    
    err = _cap_loss(y, np.max(cand_XW, axis=1))
    if err < best_cand_err:
        best_cand_err = err
        best_cand_W = cand_W.copy()
        best_cand_XW = cand_XW.copy()
        best_cand_cells = list(P.cells)
        best_cand_cells[k] = cell_le
        best_cand_cells += [cell_gt]
        best_cand_P = Partition(npoints=n, ncells=K+1, cells=best_cand_cells)

        cand_W[:, k] = cand_Wk
        cand_XW[:, k] = cand_XWk

    return cand_W, cand_XW, best_cand_err, best_cand_W, best_cand_XW, best_cand_P


def _cap_gcv_err(XW, y, d, P):
    n = XW.shape[0]
    err = _cap_loss(y, np.max(XW, axis=1))
    return err / (n * (1.0 - float(P.ncells*(d+1))/n)**2)


def _cap_jitter_fix(xarray, xmin, xmax, jitter_tol):
    if xmax - xmin < jitter_tol:
        xarray = xarray + jitter_tol * np.random.randn(*xarray.shape)
        xmin = min(xarray)
        xmax = max(xarray)
    return xarray, xmin, xmax


def cap_train(
    X,
    y,
    nknots=10,
    nranddirs=0,
    mincellsize='max(2*(d+1), n/(3*np.log(n)))',
    ridge_regularizer=1e-6,
    jitter_tol=1e-4,
    gcv_wait=None,
    **kwargs
):
    """Convex Adaptive Partitioning (CAP).

    :param X: data matrix (each row is a sample)
    :param y: target vector
    :param nknots: number of knots used for splitting
    :param nranddirs: number of random directions (greater than 0 enables FastCAP)
    :param mincellsize: minimum cell size
    :param ridge_regularizer: ridge regression parameter for cell fitting
    :param jitter_tol: jitter tolerance and seeding standard deviation for splitting
    :param gcv_wait: number of iteration to wait for improvement in the GCV error
    :return: weights, GCV error, number of iterations
    """
    n, d = X.shape

    if isinstance(nknots, str):
        nknots = int(np.ceil(eval(nknots)))
    if isinstance(mincellsize, str):
        mincellsize = int(np.ceil(eval(mincellsize)))
    if isinstance(ridge_regularizer, str):
        ridge_regularizer = int(np.ceil(eval(ridge_regularizer)))

    X1 = np.insert(X, 0, 1.0, axis=1)  # augmenting the data with leading 1s
    betaI = ridge_regularizer * np.eye(d+1)
    betaI[0,0] = 0.0  # do not regularize the bias term

    P = Partition(npoints=n, ncells=1, cells=(np.array(range(n)),))
    W = _cap_fit(X1, y, betaI)[:, np.newaxis]
    XW = X1.dot(W)

    model = W
    model_gcv_err = _cap_gcv_err(XW, y, d, P)
    gcv_miss = 0
    niter = 0
    while max(P.cell_sizes()) >= 2*mincellsize and n > (P.ncells+1)*(d+1):
        niter += 1

        # Generate candidates and select the one with smallest MSE.
        best_cand_err = np.inf
        best_cand_W = None
        best_cand_XW = None
        best_cand_P = None
        cand_W = np.insert(W, W.shape[1], 0.0, axis=1)
        cand_XW = np.insert(XW, XW.shape[1], 0.0, axis=1)
        for k, cell in enumerate(P.cells):
            if len(cell) < 2*mincellsize:
                continue  # splitting this cell is not possible

            Xcell = X[cell, :]
            ycell = y[cell]
            
            if nranddirs <= 0:  # CAP splitting along canonical directions
                for dim in range(d):
                    Xcelld = Xcell[:, dim]
                    Xcelld, xmin, xmax = _cap_jitter_fix(
                        Xcelld, np.min(Xcelld), np.max(Xcelld), jitter_tol,
                    )
                    has_found_valid_knot = False
                    for knot in range(nknots):
                        b = xmin + (float(knot)/(nknots-1)) * (xmax - xmin)
                        is_valid, cell_le, cell_gt = _cap_split(cell, Xcelld, b, mincellsize)
                        if not is_valid:
                            continue

                        has_found_valid_knot = True
                        (cand_W, cand_XW, best_cand_err,
                         best_cand_W, best_cand_XW, best_cand_P) = _eval_cap_split(
                             P, X1, y, betaI, k, cell_le, cell_gt,
                             cand_W, cand_XW, best_cand_err,
                             best_cand_W, best_cand_XW, best_cand_P,
                        )
                    if not has_found_valid_knot:  # split by the median if nothing else worked
                        b = np.median(Xcelld)
                        is_valid, cell_le, cell_gt = _cap_split(cell, Xcelld, b, mincellsize)
                        if not is_valid:
                            continue

                        (cand_W, cand_XW, best_cand_err,
                         best_cand_W, best_cand_XW, best_cand_P) = _eval_cap_split(
                             P, X1, y, betaI, k, cell_le, cell_gt,
                             cand_W, cand_XW, best_cand_err,
                             best_cand_W, best_cand_XW, best_cand_P,
                         )
            else:  # FastCAP splitting along random directions
                G = rand_direction(nranddirs, d)
                for iranddir in range(nranddirs):
                    g = G[iranddir, :]
                    Xcellg = Xcell.dot(g)
                    Xcellg, xmin, xmax = _cap_jitter_fix(
                        Xcellg, np.min(Xcellg), np.max(Xcellg), jitter_tol,
                    )
                    has_found_valid_knot = False
                    for knot in range(nknots):
                        b = xmin + (float(knot)/(nknots-1)) * (xmax - xmin)
                        is_valid, cell_le, cell_gt = _cap_split(cell, Xcellg, b, mincellsize)
                        if not is_valid:
                            continue

                        has_found_valid_knot = True
                        (cand_W, cand_XW, best_cand_err,
                         best_cand_W, best_cand_XW, best_cand_P) = _eval_cap_split(
                             P, X1, y, betaI, k, cell_le, cell_gt,
                             cand_W, cand_XW, best_cand_err,
                             best_cand_W, best_cand_XW, best_cand_P,
                        )
                    if not has_found_valid_knot:  # split by the median if nothing else worked
                        b = np.median(Xcellg)
                        is_valid, cell_le, cell_gt = _cap_split(cell, Xcellg, b, mincellsize)
                        if not is_valid:
                            continue

                        (cand_W, cand_XW, best_cand_err,
                         best_cand_W, best_cand_XW, best_cand_P) = _eval_cap_split(
                             P, X1, y, betaI, k, cell_le, cell_gt,
                             cand_W, cand_XW, best_cand_err,
                             best_cand_W, best_cand_XW, best_cand_P,
                         )

        # Attempt to refit the best candidate.
        P = best_cand_P
        W = best_cand_W
        XW = best_cand_XW
        cand_P = max_affine_partition(X1, W.T)
        if cand_P.ncells == P.ncells and min(cand_P.cell_sizes()) >= mincellsize:
            P = cand_P
            W = max_affine_fit_partition(cand_P, X1, y, extend_X1=False).T
            XW = X1.dot(W)

        # Save the best final model based on CAP's GCV estimate.
        gcv_err = _cap_gcv_err(XW, y, d, P)
        if gcv_err < model_gcv_err:
            model = W
            model_gcv_err = gcv_err
            gcv_miss = 0
        elif gcv_err > model_gcv_err:
            gcv_miss += 1
        if gcv_wait is not None and gcv_wait <= gcv_miss:
            break  # stop if GCV does not improve for a while

    return CAPEstimatorModel(weights=model.T, gcv_error=model_gcv_err, niter=niter)


class CAPEstimator(Estimator):
    """The CAP estimator.

    >>> np.set_printoptions(legacy='1.25')
    >>> from ai.gandg.common.util import set_random_seed
    >>> set_random_seed(19)

    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2

    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])

    >>> X_test = np.random.randn(500, 2)
    >>> y_test = regression_func(X_test)
    >>> ols_model = np.linalg.lstsq(X, y, rcond=-1)[0]
    >>> ols_yhat_test = np.sum(X_test * ols_model, axis=1)  # np.dot is not deterministic
    >>> np.round(np.sum(np.square(ols_yhat_test - y_test)) / len(y_test), decimals=4)  # OLS out-of-sample L2-error
    6.2752

    >>> cap = CAPEstimator()
    >>> model = cap.train(X, y)
    >>> model.weights.shape
    (7, 3)
    >>> yhat = cap.predict(model, X)
    >>> np.round(np.sum(np.square(yhat - y)) / len(y), decimals=4)  # in-sample L2-risk
    0.0102
    >>> yhat_test = cap.predict(model, X_test)
    >>> np.round(np.sum(np.square(yhat_test - y_test)) / len(y_test), decimals=4)  # out-of-sample L2-error
    0.0046

    >>> fastcap = CAPEstimator(train_args={'nranddirs': 3})
    >>> model = fastcap.train(X, y)
    >>> model.weights.shape
    (7, 3)
    >>> yhat = fastcap.predict(model, X)
    >>> np.round(np.sum(np.square(yhat - y)) / len(y), decimals=4)  # in-sample L2-risk
    0.0107
    >>> yhat_test = fastcap.predict(model, X_test)
    >>> np.round(np.sum(np.square(yhat_test - y_test)) / len(y_test), decimals=4)  # out-of-sample L2-error
    0.0063
    """
    def __init__(self, train_args={}, predict_args={}):
        Estimator.__init__(
            self,
            train=partial(cap_train, **train_args),
            predict=partial(max_affine_predict, **predict_args),
        )
