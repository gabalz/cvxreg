from __future__ import print_function

import sys
import random
import numpy as np


def set_random_seed(seed):
    """Sets the random seed.

    :param seed: new random seed
    """
    random.seed(seed)
    np.random.seed(random.randint(0, int(1e8)))


def rand_direction(n, d):
    """Generates uniformly random directions.

    :param n: number of random directions to generate
    :param d: dimension of the random direction vectors
    :return: matrix of random directions (size: n x d)
    """
    x = np.random.randn(n, d)
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    return x


def eprint(*args, **kwargs):
    """Printing to standard error, useful for debugging (as standard error is not buffered)."""
    print(*args, file=sys.stderr, **kwargs)
