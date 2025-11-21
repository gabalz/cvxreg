import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from collections import OrderedDict

from ai.gandg.common.experiment import collect_stats_by_name


# color-blind-friendly palette: https://mk.bcgsc.ca/colorblind/palettes.mhtml
blind_friendly_colors = [
    (34/255, 113/255, 178/255),
    (213/255, 94/255, 0),
    (53/255, 155/255, 115/255),
    (240/255, 228/255, 66/255),
]


def bar_plot(
    data_df, estimator_names, hue,
    bbox_to_anchor=None, output=None, title='',
    xlabel=None, ylabel=None, log_scale=None, yscale=None,
    colors=blind_friendly_colors,
):
    df = data_df[data_df['method'].isin(estimator_names)].sort_values('method')
    df['method'] = df['method'].map(estimator_names)

    sns.set(font_scale=2.0, rc={'text.usetex': True})
    sns.set_theme(
        style="whitegrid", palette="colorblind",
        rc={'figure.figsize': (5, 3), 'axes.titlesize': 24,
            'axes.linewidth': 1.0, 'axes.labelsize': 24})

    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.major.width'] = 1

    nc = len(np.unique(df[hue]))
    ax = sns.barplot(data=df, x='method', y='y',
                     hue=hue, palette=colors[:nc],
                     order=list(estimator_names.values()),
                     estimator='mean', errorbar=('sd', 1),
                     legend=True, log_scale=log_scale)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if yscale is not None:
        ax.set_yscale(yscale)
    plt.legend(title=None)
    if bbox_to_anchor is not None:
        sns.move_legend(ax, "upper right", bbox_to_anchor=bbox_to_anchor)
    plt.setp(ax.get_legend().get_texts(), fontsize=18)
    plt.xticks(rotation=75)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=18)
    ax.spines[['right', 'top']].set_visible(False)
    if output is not None:
        ax.figure.savefig(output, bbox_inches='tight', pad_inches=0.0)
    return ax


def plot_bar_perf(
    all_stats, perf, estimator_names,
    domain_dims=None, colors=blind_friendly_colors,
    title='', ylabel=None, yscale=None, data_scale=1.0, nscale=None, output=None,
):
    if isinstance(perf, dict):
        vals = perf
    else:
        print(perf)
        vals = {k: r[perf] for k, r in all_stats.items()}
    vals = pd.Series(vals) * data_scale
    if nscale is not None:
        for n, scale in nscale.items():
            vals[vals.index.get_level_values(0) == n] *= scale
    vals = vals.reset_index()
    if domain_dims is None:
        columns = {
            'level_0': '$n$', 'level_1': 'method', 'level_2': 'run', 0: 'y'}
    else:
        columns = {
            'level_0': '$d$', 'level_1': '$n$', 'level_2': 'method', 'level_3': 'run', 0: 'y'}
    vals.rename(inplace=True, columns=columns)
    if domain_dims is None:
        bar_plot(vals, estimator_names, hue='$n$', title=title,
                 ylabel=ylabel, yscale=yscale, output=output, colors=colors)
    else:
        for d in domain_dims:
            vals_d = vals.loc[vals['$d$'] == d, :]
            bar_plot(vals_d, estimator_names, hue='$n$', title=title + f' (d={d})',
                     ylabel=ylabel, yscale=yscale, output=output, colors=colors)
            plt.show()


def plot_stats(
    ax, all_stats, stat_name,
    stat_dev_name=None, d=None, skipped_estimators=(),
    ylabel='', title='', scaler=1.0, colors=None,
):
    sns.set(font_scale=2.0, rc={'text.usetex': True})
    sns.set_theme(style='white')
    if colors is None:
        colors = sns.color_palette()
    cstats = collect_stats_by_name(all_stats, stat_name, skipped_estimators) * scaler
    if d is not None:
        cstats = cstats.xs(d, level='d', drop_level=True)
    nsamples = sorted(cstats.index.get_level_values(0))
    color_dict = {estimator_name: colors[i % len(colors)]
                  for i, estimator_name in enumerate(cstats.keys())}
    if stat_dev_name is not None:
        cstats_dev = collect_stats_by_name(all_stats, stat_dev_name, skipped_estimators) * scaler
        if d is not None:
            cstats_dev = cstats_dev.xs(d, level='d', drop_level=True)
        for estimator_name, estats_dev in cstats_dev.items():
            if estimator_name in skipped_estimators:
                continue
            estats = cstats[estimator_name]
            ax.fill_between(
                nsamples, estats-estats_dev, estats+estats_dev,
                alpha=0.1, color=color_dict[estimator_name],
            )
    for estimator_name, estats in cstats.items():
        if estimator_name in skipped_estimators:
            continue
        sns.lineplot(ax=ax, x=nsamples, y=estats,
                     linewidth=3, label=estimator_name, color=color_dict[estimator_name])
    ymin = ax.get_ylim()[0]
    ax.vlines(x=nsamples, ymin=ymin, ymax=np.max(cstats.values, axis=1),
              color='gray', linestyle='--', zorder=1, alpha=0.5)
    if d is not None:
        title = title + f' (d = {d})'
    ax.set_xticks(nsamples)
    ax.set(xlabel='number of samples ($n$)',
           ylabel=ylabel, title=title, ylim=(ymin, None))


def plot_standard_stats(
    all_stats, report_loss_name,
    d=None, skipped_estimators=(), colors=None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    train_perf_name = f'train_{report_loss_name}-risk'
    plot_stats(
        ax=ax1, all_stats=all_stats,
        ylabel='$L_2$-risk', title='Performance on train samples',
        stat_name=train_perf_name+'__mean',
        stat_dev_name=train_perf_name+'__std',
        d=d, skipped_estimators=skipped_estimators, colors=colors,
    )
    test_perf_type = 'risk' if d is None else 'error'
    test_perf_name = f'test_{report_loss_name}-{test_perf_type}'
    plot_stats(
        ax=ax2, all_stats=all_stats,
        ylabel='$L_2$-risk', title='Performance on test samples',
        stat_name=test_perf_name+'__mean',
        stat_dev_name=test_perf_name+'__std',
        d=d, skipped_estimators=skipped_estimators, colors=colors,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    train_time_name = 'train_real_time'
    plot_stats(
        ax=ax1, all_stats=all_stats, scaler=1.0/60,
        ylabel='minutes', title='Training time',
        stat_name=train_time_name+'__mean',
        stat_dev_name=train_time_name+'__std',
        d=d, skipped_estimators=skipped_estimators, colors=colors,
    )
    test_time_name = 'test_real_time'
    plot_stats(
        ax=ax2, all_stats=all_stats, scaler=1.0/60,
        ylabel='minutes', title='Test (prediction) time',
        stat_name=test_time_name+'__mean',
        stat_dev_name=test_time_name+'__std',
        d=d, skipped_estimators=skipped_estimators, colors=colors,
    )


def plot_dcf_nparams(all_stats, dcf_estimator_names, max_n,
                     max_d=None, output=None):
    nparams = {}
    for key, result in all_stats.items():
        if max_d is None:
            n, estimator, run = key
        else:
            d, n, estimator, run = key
            if d != max_d:
                continue
        if estimator not in dcf_estimator_names or n != max_n:
            continue
        model = result['model']
        nparams[('init', estimator, run)] = model._socp_stats.nparams_w_centers
        nparams[('final', estimator, run)] = model.get_nparams(include_centers=True)
    if len(nparams) > 0:
        print('Data of storing the center points is included.')
        plot_bar_perf(None, nparams, dcf_estimator_names,
                      yscale='log', ylabel='param.s (log scale)', output=output)
        plt.show()


def plot_dcf_training_times(all_stats, dcf_estimator_names, max_n,
                            max_d=None, output=None):
    train_times = {}
    for key, result in all_stats.items():
        if max_d is None:
            n, estimator, run = key
        else:
            d, n, estimator, run = key
            if d != max_d:
                continue
        if estimator not in dcf_estimator_names or n != max_n:
            continue
        model = result['model']
        train_times[('AFPC', estimator, run)] = model._clustering_stats.runtime
        train_times[('data', estimator, run)] = model._socp_stats.runtime_data
        train_times[('SOCP', estimator, run)] = model._socp_stats.runtime_solve
        train_times[('FT', estimator, run)] = model._local_opt_stats.runtime
    if len(train_times) > 0:
        plot_bar_perf(None, train_times, dcf_estimator_names,
                      ylabel='training (seconds)', output=output)
        plt.show()


def plot_dcf_niterations(all_stats, dcf_estimator_names, max_n,
                         max_d=None, output=None):
    niterations = {}
    for key, result in all_stats.items():
        if max_d is None:
            n, estimator, run = key
        else:
            d, n, estimator, run = key
            if d != max_d:
                continue
        if estimator not in dcf_estimator_names or n != max_n:
            continue
        model = result['model']
        niterations[('SOCP', estimator, run)] = model._socp_stats.niterations
        niterations[('FT', estimator, run)] = model._local_opt_stats.niterations
    if len(niterations) > 0:
        plot_bar_perf(None, niterations, dcf_estimator_names,
                      ylabel='num. of iterations', output=output)
        plt.show()


def print_dcf_lipschitz_constants(all_stats):
    from ai.gandg.algorithm.dcf.dcf import DCFEstimatorModel
    maxLs = OrderedDict()
    for key, result in all_stats.items():
        model = result['model']
        if not isinstance(model, DCFEstimatorModel):
            continue
        maxLs.setdefault(key[:-1], []).append(model.get_maxL())
    for key, values in maxLs.items():
        print(f"{key}: {np.min(values):.2f} <= {np.mean(values):.2f}"
              f" +- {np.std(values):.2f} <= {np.max(values):.2f}")
