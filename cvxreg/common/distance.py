import numpy as np


def euclidean_distance(left, right, axis=1):
    """Calculates the Euclidean norm of each row of the matrix left-right.

    :param left: left data matrix (or vector)
    :param right: right data matrix (or vector)
    :param axis: distance calculation along this axis (defaults to distances along rows)
    :returns: vector of distances for each row of the matrix
    """
    return np.sqrt(squared_distance(left, right, axis))


def squared_distance(left, right, axis=1):
    """Calculates the squared Euclidean norm of each row of a matrix.

    :param left: left data matrix (or vector)
    :param right: right data matrix (or vector)
    :param axis: distance calculation along this axis (defaults to distances along rows)
    :returns: vector of distances for each row of the matrix
    """
    diff = np.square(left - right)
    if len(diff.shape) == 1:
        return np.sum(diff)
    return np.sum(diff, axis=axis)
