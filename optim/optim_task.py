
import numpy as np


class OptimTask:
    """
    Function definition of an optimization task.
    """
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.nfun_evals = 0
        self.njac_evals = 0

    def fun(self, x):
        self.nfun_evals += 1

    def jac(self, x):
        self.njac_evals += 1

    def jac_finite_difference(self, x, eps=1e-6):
        grad = np.zeros_like(x)
        x = x.copy()
        eps2 = 2.0 * eps
        for i in range(len(grad)):
            xi = x[i]
            x[i] = xi + eps
            grad[i] = self.fun(x)
            x[i] = xi - eps
            grad[i] -= self.fun(x)
            x[i] = xi
            grad[i] /= eps2
        return grad
