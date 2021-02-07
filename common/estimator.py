
class Estimator:
    """Abstract class of an estimator."""
    def __init__(self, train, predict):
        self.train = train
        self.predict = predict


class EstimatorModel:
    """Abstract class of the model of an estimator."""
    def __init__(self, weights):
        self.weights = weights
