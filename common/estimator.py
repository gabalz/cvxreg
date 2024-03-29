
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
