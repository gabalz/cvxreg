import numpy as np

from ai.gandg.cvxreg.common.estimator import EstimatorModel


def prepare_prediction(model, X, extend_X1):
    if isinstance(model, EstimatorModel):
        weights = model.weights
        if model.xmean is not None:
            X = X - model.xmean
        if model.xscale is not None:
            X = X / model.xscale
    else:
        weights = model
    if extend_X1:
        X = np.insert(X, 0, 1.0, axis=1)
    return weights, X


def postprocess_prediction(model, yhat):
    if isinstance(model, EstimatorModel):
        if model.yscale is not None:
            yhat *= model.yscale
        if model.ymean is not None:
            yhat += model.ymean
    return yhat


def partition_predict(partition, model, X, extend_X1=True):
    """Prediction by a non-continuous piecewise-linear model.

    :param partition: Partition object
    :param model: weights for each cell (each row is a hyperplane), or EstimatorModel having the weights
    :param X: data matrix (each row is a sample)
    :param extend_X1: whether or not to extend the data with leading 1s
    :return: predicted vector (one value for each sample)
    """
    assert partition.npoints == X.shape[0]
    weights, X = prepare_prediction(model, X, extend_X1)
    assert partition.ncells == weights.shape[0]
    yhat = np.zeros(X.shape[0])
    for i, cell in enumerate(partition.cells):
        yhat[cell] = X[cell, :].dot(weights[i, :])
    return yhat


def max_affine_predict(model, X, extend_X1=True):
    """Prediction by a max-affine model.

    :param model: max-affine weights (each row is a hyperplane), or EstimatorModel having the weights
    :param X: data matrix (each row is a sample)
    :param extend_X1: whether or not to extend the data with leading 1s
    :return: predicted vector (one value for each sample)
    """
    weights, X = prepare_prediction(model, X, extend_X1)
    yhat = np.max(X.dot(weights.T), axis=1)
    return postprocess_prediction(model, yhat)


def max_affine_fit_partition(partition, X, y, extend_X1=True, rcond=None):
    """OLS fitting each cell within a partition.

    :param partition: Partition object of which cells to be fitted by OLS
    :param X: data matrix (each row is a sample)
    :param y: target vector
    :param extend_X1: whether or not to extend the data with leading 1s
    :param rcond: cut-off ratio for small singular values (see np.linalg.lstsq)
    :return: weight matrix (each row represents an OLS fit)
    """
    if extend_X1:
        X = np.insert(X, 0, 1.0, axis=1)
    weights = np.empty((partition.ncells, X.shape[1]))
    for i, cell in enumerate(partition.cells):
        weights[i, :] = np.linalg.lstsq(X[cell, :], y[cell], rcond=rcond)[0]
    return weights


def cv_indices(npoints, ncvfolds):
    """Calculating cross-validation indices.

    :param npoints: number of data points
    :param ncvfolds: number of CV-folds
    :return sample indices of the test and train sets, respectively
    """
    assert npoints >= ncvfolds
    idxset = set(range(npoints))
    test = []
    train = []
    val = 0
    step = npoints / ncvfolds
    for f in range(ncvfolds):
        next_val = val + step if f < ncvfolds-1 else npoints
        idx = np.array(range(int(val), int(next_val)), dtype=int)
        test.append(idx)
        train.append(np.array(sorted(idxset - set(idx)), dtype=int))
        val = next_val
    return test, train
