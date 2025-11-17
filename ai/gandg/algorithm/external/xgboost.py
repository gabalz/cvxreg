from functools import partial

from ai.gandg.common.estimator import Estimator


def xgb_train(X, y, **kwargs):
    from xgboost import XGBRegressor
    kwargs = dict(kwargs)
    if 'L' in kwargs:
        del kwargs['L']
    xgb = XGBRegressor(**kwargs)
    xgb.fit(X, y)
    return xgb


class XgbEstimator(Estimator):
    def __init__(self, **kwargs):
        Estimator.__init__(
            self,
            train=partial(xgb_train, **kwargs),
            predict=lambda model, X: model.predict(X),
        )
