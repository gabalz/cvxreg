from __future__ import print_function

import sys
import random
import numpy as np


def set_random_seed(seed):
    """Sets the random seed.

    :param seed: new random seed

    >>> set_random_seed(19)
    >>> random.randint(0, 10000)
    708
    >>> np.random.rand(3, 2)
    array([[0.6356515 , 0.15946741],
           [0.42432349, 0.93350408],
           [0.20335322, 0.5258474 ]])
    """
    random.seed(seed)
    np.random.seed(random.randint(0, int(1e8)))


def rand_direction(n, d):
    """Generates uniformly random directions.

    :param n: number of random directions to generate
    :param d: dimension of the random direction vectors
    :return: matrix of random directions (size: n x d)

    >>> set_random_seed(19)
    >>> x = rand_direction(10, 3)
    >>> x
    array([[-0.78538771,  0.31285999,  0.53412056],
           [-0.08505102,  0.08648851, -0.99261577],
           [-0.03835023,  0.14032477, -0.98936253],
           [ 0.44378856,  0.20424592, -0.87254531],
           [-0.13436984, -0.8220843 , -0.55328307],
           [ 0.41833227,  0.30613646,  0.85514828],
           [-0.77293566, -0.57197324,  0.2746217 ],
           [-0.7169452 ,  0.44005241,  0.54068794],
           [ 0.99017269,  0.1185976 ,  0.07411245],
           [-0.33350422,  0.42052367, -0.84376227]])
    >>> np.linalg.norm(x, axis=1)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    """
    x = np.random.randn(n, d)
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    return x


def eprint(*args, **kwargs):
    """Printing to standard error, useful for debugging (as standard error is not buffered)."""
    print(*args, file=sys.stderr, **kwargs)
