
from functools import partial

from common.estimator import Estimator


def random_forest_train(X, y, **kwargs):
    from sklearn.ensemble import RandomForestRegressor
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
