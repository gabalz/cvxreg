
import time
import numpy as np
import pandas as pd
from joblib import delayed
from collections import OrderedDict

from ai.gandg.common.util import set_random_seed
from ai.gandg.common.estimator import EstimatorModel


def loss_l1(yhat, y):  # L1-error
    return np.mean(np.abs(yhat - y))


def loss_l2(yhat, y):  # L2-error
    return np.mean(np.square(yhat - y))


def loss_linf(yhat, y):  # Linf-error
    return np.max(np.abs(yhat - y))


def get_random_seed_offset(d, n, run):
    return d * n + run


def _prepare_experiment_params(n, estimator_name, run,
                               data_random_seed, training_random_seed,
                               d=None):
    return OrderedDict(
        ([] if d is None else [('d', d)])
        + [
            ('n', n),
            ('estimator_name', estimator_name),
            ('run', run),
            ('data_random_seed', data_random_seed),
            ('training_random_seed', training_random_seed),
        ]
    )


def experiment_runner(estimator_name, params, result_cache, run_experiment_func):
    exp_hash = result_cache.get_hash(run_experiment_func, params)
    print(f'experiment_runner ({exp_hash}): {list(params.items())}')
    return result_cache.cached_func(run_experiment_func, estimator_name,
                                    is_jupyter_func=True)(
        *list(params.values())
    )


def _get_experiment_log_message(d, n, estimator_name, run,
                                X_train, y_train,
                                X_test, y_test, L_true=None):
    assert (n, d) == X_train.shape
    assert n == len(y_train)
    assert d == X_test.shape[1]
    assert X_test.shape[0] == len(y_test)
    X_train_norms = np.linalg.norm(X_train, axis=1)
    X_test_norms = np.linalg.norm(X_test, axis=1)
    return (
        '\nExperiment, d: {}, n: {}, estimator: {}, {}run: {},\n'
        'train data, minX: {:.2f}, maxX: {:.2f}, minXnorm: {:.4f}, maxXnorm: {:.2f},\n'
        '            miny: {:.2f}, meany: {:.4f}, stdy: {:.4f}, maxy: {:.2f},\n'
        ' test data, minX: {:.2f}, maxX: {:.2f}, minXnorm: {:.4f}, maxXnorm: {:.2f},\n'
        '            miny: {:.2f}, meany: {:.4f}, stdy: {:.4f}, maxy: {:.2f},\n'
    ).format(
        d, n, estimator_name,
        '' if L_true is None else f'L_true: {L_true:.1f}, ',
        run,
        np.min(X_train), np.max(X_train), np.min(X_train_norms), np.max(X_train_norms),
        np.min(y_train), np.mean(y_train), np.std(y_train), np.max(y_train),
        np.min(X_test), np.max(X_test), np.min(X_test_norms), np.max(X_test_norms),
        np.min(y_test), np.mean(y_test), np.std(y_test), np.max(y_test),
    )


def _init_experiment_result(**kwargs):
    result = OrderedDict()
    for k, v in kwargs.items():
        if k.startswith('L_') and (v is None or not np.isfinite(v)):
            continue
        result[k] = v
    return result


def _calc_experiment_result(result, estimator, run,
                            stat_losses, training_random_seed,
                            X_train, y_train, X_test, y_test,
                            y_train_noiseless=None, train_args={}):
    n, d = X_train.shape
    set_random_seed(training_random_seed + get_random_seed_offset(d, n, run))

    real_time, cpu_time = time.time(), time.perf_counter()
    model = estimator.train(X_train, y_train, **train_args)
    result['model'] = model
    if isinstance(result, EstimatorModel):
        result['nweights'] = model.weights.shape[0]
        result['max_weight_norm'] = max(np.linalg.norm(model.weights, axis=1))
    yhat_train = estimator.predict(model, X_train)
    for loss_name, loss in stat_losses.items():
        result[f'train_{loss_name}-risk'] = loss(yhat_train, y_train)
        if y_train_noiseless is not None:
            result[f'train_{loss_name}-error'] = loss(yhat_train, y_train_noiseless)
    result['train_diff_mean'] = np.mean(yhat_train - y_train)
    result['train_diff_median'] = np.median(yhat_train - y_train)
    result['train_cpu_time'] = time.perf_counter() - cpu_time
    result['train_real_time'] = time.time() - real_time

    real_time, cpu_time = time.time(), time.perf_counter()
    yhat_test = estimator.predict(model, X_test)
    test_err_name = 'risk' if y_train_noiseless is None else 'error'
    for loss_name, loss in stat_losses.items():
        result[f'test_{loss_name}-{test_err_name}'] = loss(yhat_test, y_test)
    result['test_cpu_time'] = time.perf_counter() - cpu_time
    result['test_real_time'] = time.time() - real_time

    if 'l2' in stat_losses.keys():
        result['train_fvu'] = result['train_l2-risk'] / np.var(y_train)
        result['test_fvu'] = result[f'test_l2-{test_err_name}'] / np.var(y_test)


def _get_result_log_message(d, n, estimator_name, run,
                            result, report_loss_name):
    message = (
        f'\nResult, d: {d}, n: {n}, estimator: {estimator_name}'
        f', run: {run}, loss: {report_loss_name}\n'
    )
    if f'train_{report_loss_name}-error' in result:
        message += (
            ' train, error: {:.4f}, risk: {:.4f}, real_time: {}s,\n'
            '  test, error: {:.4f}, real_time: {}s'
        ).format(
            result[f'train_{report_loss_name}-error'],
            result[f'train_{report_loss_name}-risk'],
            int(np.ceil(result['train_real_time'])),
            result[f'test_{report_loss_name}-error'],
            int(np.ceil(result['test_real_time'])),
        )
    else:
        message += (
            ' train, risk: {:.4f}, real_time: {}s,\n'
            '  test, risk: {:.4f}, real_time: {}s'
        ).format(
            result[f'train_{report_loss_name}-risk'],
            int(np.ceil(result['train_real_time'])),
            result[f'test_{report_loss_name}-risk'],
            int(np.ceil(result['test_real_time'])),
        )
    return message


def calc_experiment_result(n, estimator_name, run,
                           get_data_func, get_estimator_func,
                           stat_losses, report_loss_name,
                           data_random_seed, training_random_seed,
                           log_func=None, d=None, L=None, L_scaler=1.0):
    if d is None:
        X_train, y_train, X_test, y_test = get_data_func(
            n, run, data_random_seed,
        )
        d = X_train.shape[1]
        y_train_noiseless = None
    else:
        X_train, y_train, X_test, y_test, y_train_noiseless = get_data_func(
            d, n, run, data_random_seed,
        )
    L_true = None
    L_est = np.inf
    if L is not None:
        L_true = max(L(X_train), L(X_test)) if callable(L) else L
        Lscaler = eval(L_scaler) if isinstance(L_scaler, str) else L_scaler
        L_est = (L_true * Lscaler) if np.isfinite(L_true) else np.inf
    if log_func is not None:
        log_func(_get_experiment_log_message(d, n, estimator_name, run,
                                             X_train, y_train, X_test, y_test,
                                             L_true=L_true))
    train_args = OrderedDict()
    if np.isfinite(L_est):
        train_args['L'] = L_est

    result = _init_experiment_result(
        d=d, n=n, estimator_name=estimator_name, run=run,
        data_random_seed=data_random_seed,
        training_random_seed=training_random_seed,
        L_est=L_est, L_true=L_true,
    )
    estimator = get_estimator_func(estimator_name)
    _calc_experiment_result(
        result=result, estimator=estimator, run=run,
        stat_losses=stat_losses, training_random_seed=training_random_seed,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        y_train_noiseless=y_train_noiseless, train_args=train_args,
    )
    if log_func is not None:
        log_func(_get_result_log_message(
            d=d, n=n, estimator_name=estimator_name, run=run,
            result=result, report_loss_name=report_loss_name,
        ))
    return result


def prepare_experiment_calc_funcs(nsamples, estimators, nruns,
                                  data_random_seed, training_random_seed,
                                  result_cache, run_experiment_func,
                                  domain_dims=(None,)):
    delayed_funcs = []
    for d in domain_dims:
        for n in nsamples:
            for estimator_name in estimators.keys():
                for run in range(nruns):
                    params = _prepare_experiment_params(
                        d=d, n=n, run=run,
                        estimator_name=estimator_name,
                        data_random_seed=data_random_seed,
                        training_random_seed=training_random_seed,
                    )
                    delayed_funcs.append(delayed(experiment_runner)(
                        estimator_name, params, result_cache, run_experiment_func,
                    ))
    return delayed_funcs


def _collect_stat_keys_and_values(results, estimator_name):
    stat_keys = set()
    stat_values = OrderedDict()
    for k, r in results.items():
        if k[-2] != estimator_name:
            continue
        stat_values.setdefault(k[:-2], []).append(r)
        for sk in r.keys():
            stat_keys.add(sk)
    return stat_keys, stat_values


def collect_estimator_stats(
    estimator_name, results,
    skipped_stats = (
        'd', 'n', 'run', 'estimator_name',
        'model', 'L_true', 'L_est',
    ),
    stat_funcs = OrderedDict((
        ('mean', np.mean),
        ('std', np.std),
        ('min', np.min),
        ('median', np.median),
        ('max', np.max),
    )),    
):
    stat_keys, stat_values = _collect_stat_keys_and_values(results, estimator_name)
    stat = {}
    for svk, svv in stat_values.items():
        ss = OrderedDict()
        for sk in stat_keys:
            if sk in skipped_stats:
                continue
            for sf_name, sf in stat_funcs.items():
                ss[sk + '__' + sf_name] = sf([v[sk] for v in svv])
        stat[svk] = ss
    stat = pd.DataFrame(stat)
    stat.columns.names = ('d', 'n')[-len(next(iter(results.keys()))[:-2]):]
    return stat


def collect_stats_by_name(
    all_stats, stat_name, skipped_estimators=(),
):
    result = {}
    for estimator_name, stats in all_stats.items():
        if estimator_name in skipped_estimators:
            continue
        result[estimator_name] = stats.loc[stat_name, :]
    return pd.DataFrame(result)
