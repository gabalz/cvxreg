import numpy as np


def shuffle_data(X, y):
    """Shuffle the samples.

    :param X: data matrix (each row is a sample)
    :param y: target vector
    :return: shuffled X, shuffled y, permutation

    >>> from common.util import set_random_seed
    >>> set_random_seed(19)
    >>> X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> pX, py, perm = shuffle_data(X, y)
    >>> pX
    array([[4, 4],
           [2, 2],
           [1, 1],
           [3, 3],
           [5, 5]])
    >>> py
    array([4, 2, 1, 3, 5])
    >>> perm
    array([3, 1, 0, 2, 4])
    """
    assert X.shape[0] == len(y)
    perm = np.random.permutation(len(y))
    return X[perm, :], y[perm], perm


def normalize_data(X, y, tol=1e-6):
    """Centering and scaling the data, and dropping dependent columns.

    :param X: data matrix (each row is a sample)
    :param y: target vector
    :param tol: tolerance for dropping dependent columns
    :return: transformed X, transformed y,
             eigenvector matrix,
             mean vector of X, scaling vector of X,
             mean of y, scaler of y

    >>> X = np.array([[1, 2, 1], [1, 3, 1], [1, 4, 1], [1, 5, 1], [1, 6, 1]])
    >>> y = 2 + 3*X[:, 1]
    >>> y
    array([ 8, 11, 14, 17, 20])
    >>> w1 = np.linalg.solve(X.T.dot(X) + 1e-4*np.eye(3), X.T.dot(y))
    >>> np.round(w1, decimals=4)
    array([1., 3., 1.])
    >>> np.round(X.dot(w1), decimals=4)
    array([ 8., 11., 14., 17., 20.])

    >>> sX, sy, V, x_mean, x_scale, y_mean, y_scale = normalize_data(X, y)
    >>> V.shape
    (1, 3)
    >>> x_mean
    array([1., 4., 1.])
    >>> np.round(y_mean, decimals=6)
    14.0
    >>> w2 = np.linalg.solve(sX.T.dot(sX), sX.T.dot(sy))
    >>> w2
    array([1.58113883])
    >>> w2 = w2.dot(V) * (y_scale / x_scale)
    >>> w2
    array([0., 3., 0.])
    >>> b = y_mean - w2.dot(x_mean)
    >>> np.round(b, decimals=6)
    2.0
    >>> X.dot(w2) + b
    array([ 8., 11., 14., 17., 20.])
    """
    n, d = X.shape
    assert len(y) == n

    # Centralizing:
    x_mean = np.mean(X, 0)
    y_mean = np.mean(y)
    X = X - x_mean
    y = y - y_mean

    # SVD: X == np.dot(u[:, :V.shape[0]] * s, V)
    U, s, V = np.linalg.svd(X, full_matrices=False)

    # Downscaling:
    x_scale = max(1, s[0])
    s /= x_scale
    y_scale = max(1, np.max(np.abs(y)))
    y /= y_scale

    # Dropping dependent columns:
    depcols = np.where(s < tol)
    if len(depcols) > 0:
        U = np.delete(U, depcols, axis=1)
        s = np.delete(s, depcols)
        V = np.delete(V, depcols, axis=0)
    X = U*s

    return X, y, V, x_mean, x_scale, y_mean, y_scale
