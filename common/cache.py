
import os
import joblib


class ResultCache:
    def __init__(self, is_enabled, project_path, experiment_id):
        self.cache_dir = None
        if is_enabled:
            self.cache_dir = (
                os.path.join(project_path, '_result_cache', experiment_id))
            print(f'cache_dir: {self.cache_dir}')
            self.persister_dict = {}

    def is_enabled(self):
        return (self.cache_dir is not None)

    def _ensure_estimator_persister(self, estimator_name):
        if not self.is_enabled() or estimator_name in self.persister_dict:
            return
        estimator_cache_dir = os.path.join(self.cache_dir, estimator_name)
        os.makedirs(estimator_cache_dir, exist_ok=True)
        self.persister_dict[estimator_name] = joblib.Memory(estimator_cache_dir,
                                                            verbose=2)

    def cached_func(self, func, estimator_name, is_jupyter_func=False):
        if not self.is_enabled():
            return func
        self._ensure_estimator_persister(estimator_name)
        if is_jupyter_func:
            old_module = func.__module__
            func.__module__ = 'jupyter_notebook'
            func.__qualname__ = func.__name__
        func_cached = self.persister_dict[estimator_name].cache(func)
        if is_jupyter_func:
            func.__module__ = old_module
            func_cached.__module__ = old_module
        return func_cached

    def get_hash(self, func, params):
        return joblib.hashing.hash(joblib.func_inspect.filter_args(
            func, [], (), params,
        ))
