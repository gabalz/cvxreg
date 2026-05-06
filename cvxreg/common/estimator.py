
import numpy as np


class Estimator:
    """Abstract class of an estimator."""
    def __init__(self, train, predict):
        self.train = train
        self.predict = predict


class EstimatorModel:
    """Abstract class of the model of an estimator.

    :param weights: the learned weights of the model
    :param xmean: offset vector to be subtracted from the x sample vectors
    :param xscale: xscale scalar to be multiplied to the x sample vectors
    :param yscale: yscale scalar to be multiplied to the yhat estimate values
    :param ymean: offset scalar to be added to the yhat estimate values
    """
    def __init__(self, weights, xmean=None, xscale=None, yscale=None, ymean=None):
        self.weights = weights
        self.xmean = xmean
        self.xscale = xscale
        self.yscale = yscale
        self.ymean = ymean


def _const_train(X, y, **kwargs):
    ymean = np.mean(y, axis=0)
    assert len(ymean.shape) <= 2
    return EstimatorModel(weights=None, ymean=ymean)


def _const_predict(model, X, **kwargs):
    ymean = model.ymean
    assert len(ymean.shape) <= 2
    if len(ymean.shape) == 2:
        yhat = np.ones((X.shape[0], ymean.shape[1])) * ymean[None, :]
    else:
        yhat = np.ones(X.shape[0]) * ymean
    return yhat


class ConstEstimator(Estimator):
    def __init__(self, train_args={}, predict_args={}):
        Estimator.__init__(self, _const_train, _const_predict)
