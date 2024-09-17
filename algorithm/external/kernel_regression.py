
import numpy as np
from functools import partial

from common.estimator import Estimator
from algorithm.apcnls.fpc import get_data_radius


def kernel_reg_train(X, y, kernel, **kwargs):
    from sklearn.model_selection import GridSearchCV
    from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
    from skfda.ml.regression import KernelRegression
    from skfda.misc.kernels import epanechnikov, normal, uniform, tri_weight
    kwargs = dict(kwargs)
    if 'L' in kwargs:
        del kwargs['L']
    if isinstance(kernel, str):
        kernel = kernel.lower()
        if kernel == 'normal':
            kernel = normal
        elif kernel == 'uniform':
            kernel = uniform
        elif kernel == 'epanechnikov':
            kernel = epanechnikov
        elif kernel == 'tri_weight':
            kernel = tri_weight
        else:
            raise Exception(f'Not supported kernel: {kernel}')
    n, d = X.shape
    X_range = 2*get_data_radius(X)
    y_range = max(y) - min(y)
    X_range, y_range
    bandwidth_max = X_range**(d/(2+d)) * ((y_range**2)/n)**((1/(2+d)))
    bandwidths = np.linspace(0, bandwidth_max, 101)[1:]
    kreg = GridSearchCV(
        KernelRegression(kernel_estimator=NadarayaWatsonHatMatrix(kernel=kernel)),
        param_grid={'kernel_estimator__bandwidth': bandwidths},
        cv=5, refit=True,
    )
    kreg.fit(X, y)
    return kreg


def kernel_reg_predict(model, X, nsplit=2500):
    n = X.shape[0]
    yhat = np.zeros(n)
    splits = list(range(0, n, nsplit))
    splits = zip(splits, splits[1:] + [n])
    for (split_start, split_end) in splits:
        Xsplit = X[split_start:split_end, :]
        yhat[split_start:split_end] = model.predict(Xsplit)
    return yhat


class KernelRegEstimator(Estimator):
    def __init__(self, kernel, **kwargs):
        Estimator.__init__(
            self,
            train=partial(kernel_reg_train, kernel=kernel, **kwargs),
            predict=kernel_reg_predict,
        )
