
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


def _kernel_reg_l2_loss_tests():
    """
    >>> from common.util import set_random_seed
    >>> set_random_seed(19)

    >>> def regression_func(X):
    ...     return 1.0 - 2.0*X[:, 0] + X[:, 1]**2
    >>> X = np.random.randn(200, 2)
    >>> y = regression_func(X) + 0.1 * np.random.randn(X.shape[0])

    # L2-error of OLS is bigger than 6.
    >>> X_test = np.random.randn(500, 2)
    >>> y_test = regression_func(X_test)
    >>> ols_model = np.linalg.lstsq(X.T.dot(X), X.T.dot(y), rcond=-1)[0]
    >>> ols_yhat_test = np.sum(X_test * ols_model, axis=1)  # np.dot is not deterministic
    >>> np.round(np.sum(np.square(ols_yhat_test - y_test)) / len(y_test), decimals=4)  # OLS out-of-sample L2-error
    6.2752

    >>> kreg1 = KernelRegEstimator('normal')
    >>> model1 = kreg1.train(X, y)
    >>> yhat1 = kreg1.predict(model1, X)
    >>> np.round(np.mean(np.square(yhat1 - y)), decimals=4)  # in-sample L2-risk
    0.033
    >>> yhat1_test = kreg1.predict(model1, X_test)
    >>> np.round(np.mean(np.square(yhat1_test - y_test)), decimals=4)  # out-of-sample L2-error
    0.1667
    """
    pass
