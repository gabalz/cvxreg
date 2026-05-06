from functools import partial

from cvxreg.common.estimator import Estimator
from cvxreg.common.partition import singleton_partition
from cvxreg.common.regression import max_affine_predict
from cvxreg.algorithm.pcnls.pcnls import pcnls_train


def _cnls_train(X, y, **kwargs):
    return pcnls_train(
        X, y,
        partition=singleton_partition(len(y)),
        **kwargs
    )


class CNLSEstimator(Estimator):
    """Convex Nonparametric Least Squares (CNLS) estimator."""
    def __init__(self, train_args={}, predict_args={}):
        Estimator.__init__(
            self,
            train=partial(_cnls_train, **train_args),
            predict=partial(max_affine_predict, **predict_args),
        )
