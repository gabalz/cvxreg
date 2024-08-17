
from functools import partial

from common.estimator import Estimator


def nearest_neighbors_train(X, y, **kwargs):
    from sklearn.neighbors import KNeighborsRegressor
    kwargs = dict(kwargs)
    if 'L' in kwargs:
        del kwargs['L']
    if 'n_neighbors' in kwargs and isinstance(kwargs['n_neighbors'], str):
        n, d = X.shape
        k_str = kwargs['n_neighbors']
        k = eval(k_str)
        print(f'KNN, k: {k_str} -> {k}')
        kwargs['n_neighbors'] = k
    knn = KNeighborsRegressor(**kwargs)
    knn.fit(X, y)
    return knn


class NearestNeighborsEstimator(Estimator):
    def __init__(self, **kwargs):
        Estimator.__init__(
            self,
            train=partial(nearest_neighbors_train, **kwargs),
            predict=lambda model, X: model.predict(X),
        )
