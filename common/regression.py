import numpy as np

from common.estimator import EstimatorModel


def max_affine_predict(weights, X, extend_X1=True):
    """Prediction by a max-affine model.

    :param weights: max-affine weights (each row is a hyperplane), or EstimatorModel having the weights
    :param X: data matrix (each row is a sample)
    :param extend_X1: whether or not to extend the data with leading 1s
    :return: predicted vector (one value for each sample)

    >>> X = np.array([[1.1, 1.1], [-1.2, 1.2], [-1.3, -1.3], [0.4, 0.4], [1.5, -1.5]])

    >>> W = np.array([[1.0, 2.0, 3.0]])
    >>> max_affine_predict(W, X)
    array([ 6.5,  2.2, -5.5,  3. , -0.5])

    >>> W = np.array([[1.0, 2.0, 3.0], [-1., -2., -3.]])
    >>> max_affine_predict(W, X)
    array([6.5, 2.2, 5.5, 3. , 0.5])
    """
    if isinstance(weights, EstimatorModel):
        weights = weights.weights
    if extend_X1:
        X = np.insert(X, 0, 1.0, axis=1)
    return np.max(X.dot(weights.T), axis=1)


def max_affine_fit_partition(partition, X, y, extend_X1=True, rcond=None):
    """OLS fitting each cell within a partition.

    :param partition: Partition object of which cells to be fitted by OLS
    :param X: data matrix (each row is a sample)
    :param y: target vector
    :param extend_X1: whether or not to extend the data with leading 1s
    :param rcond: cut-off ratio for small singular values (see np.linalg.lstsq)
    :return: weight matrix (each row represents an OLS fit)

    >>> from common.distance import squared_distance
    >>> from common.partition import Partition

    >>> X = np.array([[1., 1.], [-1., 1.], [0., 1.], [-1., -1.], [-2., -1], [0., 1.]])
    >>> y = 0.5 * np.sum(np.square(X), axis=1)
    >>> p = Partition(npoints=6, ncells=2, cells=(np.array([0, 3, 4]), np.array([1, 2, 5])))
    >>> weights = max_affine_fit_partition(p, X, y)
    >>> weights.shape
    (2, 3)
    >>> np.round(weights, decimals=4)
    array([[ 1.  , -1.5 ,  1.5 ],
           [ 0.25, -0.5 ,  0.25]])

    >>> yhat = max_affine_predict(weights, X)
    >>> np.round(squared_distance(yhat, y, axis=0) / len(y), decimals=4)
    2.8333
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

    >>> test, train = cv_indices(10, 4)
    >>> test
    [array([0, 1]), array([2, 3, 4]), array([5, 6]), array([7, 8, 9])]
    >>> train  # doctest:+NORMALIZE_WHITESPACE
    [array([2, 3, 4, 5, 6, 7, 8, 9]), 
     array([0, 1, 5, 6, 7, 8, 9]), 
     array([0, 1, 2, 3, 4, 7, 8, 9]), 
     array([0, 1, 2, 3, 4, 5, 6])]
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
