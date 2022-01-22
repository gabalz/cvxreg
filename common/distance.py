import numpy as np


def squared_distance(left, right, axis=1):
    """Calculates the squared Euclidean norm of each row of a matrix.
    
    :param left: left data matrix (or vector)
    :param right: right data matrix (or vector)
    :param axis: distance calculation along this axis (defaults to distances along rows)
    :returns: vector of distances for each row of the matrix

    >>> mat = np.array([[1, 2, 3], [2, 3, 4], [0, 1, 2]])
    >>> squared_distance(mat, 0)
    array([14, 29,  5])
    """
    diff = np.square(left - right)
    if len(diff.shape) == 1:
        return np.sum(diff)
    return np.sum(diff, axis=axis)
