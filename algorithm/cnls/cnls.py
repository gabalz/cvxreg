import numpy as np

from functools import partial

from common.estimator import Estimator
from common.partition import singleton_partition
from common.regression import max_affine_predict
from algorithm.pcnls.pcnls import pcnls_train


def _cnls_train(X, y, **kwargs):
    return pcnls_train(
        X, y,
        partition=singleton_partition(len(y)),
        **kwargs
    )


class CNLSEstimator(Estimator):
    """Convex Nonparametric Least Squares (CNLS) estimator.

    >>> from common.util import set_random_seed
    >>> set_random_seed(19)

    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2

    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])

    >>> X_test = np.random.randn(500, 2)
    >>> y_test = regression_func(X_test)
    >>> ols_model = np.linalg.lstsq(X, y, rcond=-1)[0]
    >>> ols_yhat_test = np.sum(X_test * ols_model, axis=1)  # np.dot is not deterministic
    >>> np.round(np.mean(np.square(ols_yhat_test - y_test)), decimals=4)  # OLS out-of-sample L2-error
    6.2752

    >>> cnls = CNLSEstimator()
    >>> model = cnls.train(X, y)
    >>> model.weights.shape
    (200, 3)
    >>> yhat = cnls.predict(model, X)
    >>> np.round(np.mean(np.square(yhat - y)), decimals=4)  # in-sample L2-risk
    0.0078
    >>> yhat_test = cnls.predict(model, X_test)
    >>> np.round(np.mean(np.square(yhat_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.0094

    """
    def __init__(self, train_args={}, predict_args={}):
        Estimator.__init__(
            self,
            train=partial(_cnls_train, **train_args),
            predict=partial(max_affine_predict, **predict_args),
        )
