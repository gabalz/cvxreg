
import numpy as np
from functools import partial

from ai.gandg.common.estimator import Estimator, EstimatorModel
from ai.gandg.common.regression import max_affine_predict


def _ols_train(X, y, **kwargs):
    X = np.insert(X, 0, 1.0, axis=1)
    return EstimatorModel(weights=np.linalg.lstsq(X, y, rcond=-1)[0][:, np.newaxis].T)


class OLSEstimator(Estimator):
    """Ordinary Least-Squares (OLS) estimator.

    >>> np.set_printoptions(legacy='1.25')
    >>> from ai.gandg.common.util import set_random_seed
    >>> set_random_seed(19)

    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2

    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])
    >>> X_test = np.random.randn(500, 2)
    >>> y_test = regression_func(X_test)

    >>> ols = OLSEstimator()
    >>> model = ols.train(X, y)
    >>> yhat = ols.predict(model, X)
    >>> np.round(np.sum(np.square(yhat - y)) / len(y), decimals=4)  # in-sample L2-risk
    2.189
    >>> yhat_test = ols.predict(model, X_test)
    >>> np.round(np.sum(np.square(yhat_test - y_test)) / len(y_test), decimals=4)  # out-of-sample L2-error
    1.8673
    """
    def __init__(self, train_args={}, predict_args={}):
        Estimator.__init__(
            self,
            train=partial(_ols_train, **train_args),
            predict=partial(max_affine_predict, **predict_args),
        )
