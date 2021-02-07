import numpy as np

from functools import partial

from common.estimator import Estimator
from common.partition import rand_voronoi_partition
from common.regression import max_affine_predict
from y2015_pcnls.pcnls import pcnls_train


def _pcnls_voronoi_train(X, y, **kwargs):
    n, d = X.shape
    ncenters = int(np.ceil(n**(d/(d+4))))
    return pcnls_train(
        X, y,
        partition=rand_voronoi_partition(ncenters, X),
        **kwargs
    )


class PCNLSVoronoiEstimator(Estimator):
    """PCNLS with uniformly drawn random Voronoi partition with K^{d/(d+4)} centers.

    >>> from common.util import set_random_seed
    >>> set_random_seed(19)

    # L2-error of OLS is bigger than 6.
    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2
    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])

    >>> X_test = np.random.randn(500, 2)
    >>> y_test = regression_func(X_test)
    >>> ols_model = np.linalg.lstsq(X.T.dot(X), X.T.dot(y), rcond=-1)[0]
    >>> ols_yhat_test = np.sum(X_test * ols_model, axis=1)  # np.dot is not deterministic
    >>> np.round(np.sum(np.square(ols_yhat_test - y_test)) / len(y_test), decimals=4)  # OLS out-of-sample L2-error
    6.2752

    >>> pcnlsv = PCNLSVoronoiEstimator()
    >>> model = pcnlsv.train(X, y)
    >>> model.weights.shape
    (6, 3)
    >>> yhat = pcnlsv.predict(model, X)
    >>> np.round(np.sum(np.square(yhat - y)) / len(y), decimals=4)  # in-sample L2-risk
    0.3661
    >>> yhat_test = pcnlsv.predict(model, X_test)
    >>> np.round(np.sum(np.square(yhat_test - y_test)) / len(y_test), decimals=4)  # out-of-sample L2-error
    0.3686
    """
    def __init__(self, train_args={}, predict_args={}):
        Estimator.__init__(
            self,
            train=partial(_pcnls_voronoi_train, **train_args),
            predict=partial(max_affine_predict, **predict_args),
        )
