
from functools import partial
from sklearn.ensemble import RandomForestRegressor

from common.estimator import Estimator


def random_forest_train(X, y, **kwargs):
    kwargs = dict(kwargs)
    if 'L' in kwargs:
        del kwargs['L']
    rf = RandomForestRegressor(**kwargs)
    rf.fit(X, y)
    return rf


class RandomForestEstimator(Estimator):
    def __init__(self, **kwargs):
        Estimator.__init__(
            self,
            train=partial(random_forest_train, **kwargs),
            predict=lambda model, X: model.predict(X),
        )
