import numpy as np
from functools import partial

from ai.gandg.cvxreg.common.estimator import EstimatorModel, Estimator
from ai.gandg.cvxreg.common.partition import rand_voronoi_partition, max_affine_partition
from ai.gandg.cvxreg.common.distance import squared_distance
from ai.gandg.cvxreg.common.regression import max_affine_predict, max_affine_fit_partition


class LSPAEstimatorModel(EstimatorModel):
    """The model of a LSPA estimator."""
    def __init__(self, weights, niters):
        EstimatorModel.__init__(self, weights)
        self.niters = niters


def lspa_train(
    X,
    y,
    ncenters,
    nrestarts=1,
    nfinalsteps=None,
    obj_tol=1e-6,
):
    """Least-Squares Partition Algorithm (LSPA).

    :param X: data matrix (each row is a sample)
    :param y: target vector
    :param ncenters: number of centers to draw for initialization
    :param nrestarts: number of restarts
    :param nfinalsteps: number of iterations after a non-decreasing objective (None means number of samples)
    :param obj_tol: tolerance to determine whether the objective is decreasing
    :return: LSPA model
    """
    n, d = X.shape

    if isinstance(ncenters, str):
        ncenters = int(np.ceil(eval(ncenters)))
    if isinstance(nrestarts, str):
        nrestarts = int(np.ceil(eval(nrestarts)))
    if nfinalsteps is None:
        nfinalsteps = n
    elif isinstance(nfinalsteps, str):
        nfinalsteps = int(np.ceil(eval(nfinalsteps)))

    X1 = np.insert(X, 0, 1.0, axis=1)

    niters = []
    best_err = np.inf
    best_weights = None
    for restart in range(nrestarts):
        partition = rand_voronoi_partition(ncenters, X)

        niter = 0
        maxiter = nfinalsteps
        while niter < maxiter:
            niter += 1

            # fit the partition
            weights = max_affine_fit_partition(partition, X1, y, extend_X1=False)

            # evaluate the model
            yhat = max_affine_predict(weights, X1, extend_X1=False)
            err = squared_distance(yhat, y, axis=0)

            # save the best model
            if err < best_err - obj_tol:
                maxiter = niter + nfinalsteps
                best_err = err
                best_weights = weights

            # calculate the induced partition
            induced_partition = max_affine_partition(X1, weights)
            if partition == induced_partition:
                break  # terminate when converged
            partition = induced_partition

        niters.append(niter)

    return LSPAEstimatorModel(
        weights=best_weights,
        niters=niters,
    )


class LSPAEstimator(Estimator):
    """The LSPA estimator."""
    def __init__(self, train_args={}, predict_args={}):
        Estimator.__init__(
            self,
            train=partial(lspa_train, **train_args),
            predict=partial(max_affine_predict, **predict_args),
        )
