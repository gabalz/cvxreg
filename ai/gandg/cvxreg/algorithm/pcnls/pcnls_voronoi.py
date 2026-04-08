import numpy as np
from functools import partial

from ai.gandg.cvxreg.common.estimator import Estimator
from ai.gandg.cvxreg.common.partition import rand_voronoi_partition
from ai.gandg.cvxreg.common.regression import max_affine_predict
from ai.gandg.cvxreg.algorithm.pcnls.pcnls import pcnls_train


def _pcnls_voronoi_train(X, y, **kwargs):
    n, d = X.shape
    ncenters = int(np.ceil(n**(d/(d+4))))
    return pcnls_train(
        X, y,
        partition=rand_voronoi_partition(ncenters, X),
        **kwargs
    )


class PCNLSVoronoiEstimator(Estimator):
    """PCNLS with uniformly drawn random Voronoi partition with K^{d/(d+4)} centers."""
    def __init__(self, train_args={}, predict_args={}):
        Estimator.__init__(
            self,
            train=partial(_pcnls_voronoi_train, **train_args),
            predict=partial(max_affine_predict, **predict_args),
        )
