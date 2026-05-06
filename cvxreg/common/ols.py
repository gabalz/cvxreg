
import numpy as np
from functools import partial

from cvxreg.common.estimator import Estimator, EstimatorModel
from cvxreg.common.regression import max_affine_predict


def _ols_train(X, y, **kwargs):
    X = np.insert(X, 0, 1.0, axis=1)
    return EstimatorModel(weights=np.linalg.lstsq(X, y, rcond=-1)[0][:, np.newaxis].T)


class OLSEstimator(Estimator):
    """Ordinary Least-Squares (OLS) estimator."""
    def __init__(self, train_args={}, predict_args={}):
        Estimator.__init__(
            self,
            train=partial(_ols_train, **train_args),
            predict=partial(max_affine_predict, **predict_args),
        )
