import numpy as np
import pandas as pd
import seaborn as sns

from ai.gandg.common.partition import cell_radii, voronoi_partition
from ai.gandg.algorithm.apcnls.fpc import get_data_radius


def get_clustering_stats(nsamples, nruns, data_random_seed,
                         get_data_func, get_cluster_func, report_loss,
                         domain_dims=(None,)):
    stats = []
    is_d = domain_dims[0] is not None
    for d in domain_dims:
        for n in nsamples:
            for run in range(nruns):
                args = ([d] if is_d else []) + [n, run, data_random_seed]
                X_train, y_train, X_test, y_test = get_data_func(*args)[:4]
                partition, centers = get_cluster_func(data=X_train)
                test_partition = voronoi_partition(centers, X_test)
                stats.append((
                    get_data_radius(X_train),
                    get_data_radius(X_test),
                    np.max(y_test - np.mean(y_test)),
                    report_loss(np.mean(y_test), y_test),
                    partition.ncells,
                    np.max(cell_radii(X_train, partition, centers)),
                    np.max(cell_radii(X_test, test_partition)),
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


def get_clustering_cell_size_distribution(n, nruns, data_random_seed,
                                          get_data_func, get_cluster_func, d=None):
    stats = []
    for run in range(nruns):
        csizes = {}
        args = ([] if d is None else [d]) + [n, run, data_random_seed]
        X_train = get_data_func(*args)[0]
        partition, _ = get_cluster_func(X_train)
        for cell in partition.cells:
            csize = len(cell)
            if csize not in csizes:
                csizes[csize] = 0
            csizes[csize] += 1
        stats.append(pd.Series(csizes).sort_index())
    return pd.DataFrame(stats).T


def _init_sns():
    sns.set(font_scale=2.0, rc={'text.usetex': True})
    sns.set_theme(style='white')


def _prepare_plotting(ax, d, nsamples, afpc_stats):
    is_d = afpc_stats.index.names[0] == 'd'
    if is_d:
        afpc_stats = afpc_stats.xs(d, level='d', drop_level=True)
    assert tuple(afpc_stats.index.names) == ('n', 'nrun')
    _init_sns()
    ax.set_xticks(nsamples)
    return afpc_stats, sorted(nsamples), is_d, sns.color_palette()


def plot_partition_size(ax, d, nsamples, afpc_stats, q=1):
    afpc_stats, nsamples, is_d, colors = _prepare_plotting(ax, d, nsamples, afpc_stats)
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
    afpc_stats, nsamples, is_d, colors = _prepare_plotting(ax, d, nsamples, afpc_stats)
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
                     linewidth=3, label=r'$\epsilon$ ('+label+')', color=color)
    ax.set(xlabel='number of samples ($n$)', ylim=(0.0, None))
    ax.set(title='AFPC covering radius' + (f' (d = {d})' if is_d else ''))
    return ax


def plot_partition_cell_size_distribution(ax, d, cs_stats, n=None):
    ymax = np.nanmax(cs_stats.values)
    _init_sns()
    sns.lineplot(cs_stats, ax=ax, linewidth=1, dashes=False, legend=False)
    ax.set(xlabel='cell size', ylabel='number of cells')
    title = 'cell size distribution over runs'
    ax.set(title=title + ('' if n is None else f' ($n = {n}$)'))
    ax.axvline(x=d, color='gray', linestyle='--', zorder=1, alpha=0.5)
    ax.text(x=d+0.5, y=ymax*0.95, s=f'$d = {d}$', fontsize=14, color='gray')
    return ax
