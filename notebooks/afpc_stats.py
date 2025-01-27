
import numpy as np
import pandas as pd
import seaborn as sns

from common.partition import cell_radiuses
from algorithm.apcnls.fpc import adaptive_farthest_point_clustering, get_data_radius


def get_afpc_stats(nsamples, nruns, data_random_seed,
                   get_data_func, report_loss,
                   q=1, domain_dims=(None,)):
    stats = []
    is_d = domain_dims[0] is not None
    for d in domain_dims:
        for n in nsamples:
            K_vals = []
            train_eps_vals = []
            test_eps_vals = []
            for run in range(nruns):
                args = ([d] if is_d else []) + [n, run, data_random_seed]
                X_train, y_train, X_test, y_test = get_data_func(*args)[:4]
                partition, center_idxs = adaptive_farthest_point_clustering(
                    data=X_train, q=q, return_center_idxs=True,
                )
                centers = X_train[center_idxs, :]
                stats.append((
                    get_data_radius(X_train),
                    get_data_radius(X_test),
                    np.max(y_test - np.mean(y_test)),
                    report_loss(np.mean(y_test), y_test),
                    partition.ncells,
                    np.max(cell_radiuses(X_train, partition, centers)),
                    np.max(cell_radiuses(X_test, partition, centers)),
                ))
    return pd.DataFrame(
        stats,
        index=pd.MultiIndex.from_product(
            ([domain_dims] if is_d else []) + [nsamples, range(nruns)],
            names=(['d'] if is_d else []) + ['n', 'nrun'],
        ),
        columns=['rad(X_train)', 'rad(X_test)', 'rad(y)',
                 'loss(ybar)', 'K', 'train_eps', 'test_eps'],
    )


def _prepare_plotting(d, nsamples, afpc_stats):
    is_d = afpc_stats.index.names[0] == 'd'
    if is_d:
        afpc_stats = afpc_stats.xs(d, level='d', drop_level=True)
    assert tuple(afpc_stats.index.names) == ('n', 'nrun')
    sns.set(font_scale=2.0, rc={'text.usetex' : True})
    sns.set_theme(style='white')
    return afpc_stats, sorted(nsamples), is_d, sns.color_palette()


def plot_partition_size(ax, d, nsamples, afpc_stats, q=1):
    afpc_stats, nsamples, is_d, colors = _prepare_plotting(d, nsamples, afpc_stats)
    K_stats = afpc_stats[['K']].groupby(level=0).agg(['mean', 'std']).values
    K_means, K_stds = K_stats[:, 0], K_stats[:, 1]
    Kmax = np.array(nsamples)**(d/(2*q+d))
    ax.vlines(x=nsamples, ymin=0.0, ymax=Kmax,
              color='gray', linestyle='--', zorder=1, alpha=0.5)
    ax.fill_between(nsamples, K_means - K_stds, K_means + K_stds,
                    alpha=0.1, color=colors[1])
    sns.lineplot(ax=ax, x=nsamples, y=Kmax,
                 linewidth=3, label='$n^{d/(d+2)}$', color=colors[0])
    sns.lineplot(ax=ax, x=nsamples, y=K_means,
                 linewidth=3, label='$K$', color=colors[1])
    ax.set(xlabel='number of samples ($n$)', ylim=(0.0, None))
    ax.set(title='AFPC partition size' + (f' (d = {d})' if is_d else ''))
    return ax


def plot_partition_epsilon(ax, d, nsamples, afpc_stats):
    afpc_stats, nsamples, is_d, colors = _prepare_plotting(d, nsamples, afpc_stats)
    stats = afpc_stats[['train_eps', 'test_eps']].groupby(level=0).agg(['mean', 'std']).values
    ax.vlines(x=nsamples, ymin=0.0, ymax=np.max(stats[:, [0, 2]], axis=1),
              color='gray', linestyle='--', zorder=1, alpha=0.5)
    for ci, (i, label) in enumerate(zip([0, 2], ['train', 'test'])):
        color = colors[1+ci]
        ax.fill_between(nsamples,
                        stats[:, i] - stats[:, i+1],
                        stats[:, i] + stats[:, i+1],
                        alpha=0.1, color=color)
        sns.lineplot(ax=ax, x=nsamples, y=stats[:, i],
                     linewidth=3, label='$\epsilon$ ('+label+')', color=color)
    ax.set(xlabel='number of samples ($n$)', ylim=(0.0, None))
    ax.set(title='AFPC covering radius' + (f' (d = {d})' if is_d else ''))
    return ax
