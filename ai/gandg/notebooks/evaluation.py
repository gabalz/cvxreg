import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from ai.gandg.common.experiment import collect_stats_by_name


def plot_stats(
    ax, all_stats, stat_name,
    stat_dev_name=None, d=None, skipped_estimators=(),
    ylabel='', title='', scaler=1.0,
):
    sns.set(font_scale=2.0, rc={'text.usetex': True})
    sns.set_theme(style='white')
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
    d=None, skipped_estimators=(),
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    train_perf_name = f'train_{report_loss_name}-risk'
    plot_stats(
        ax=ax1, all_stats=all_stats,
        ylabel='$L_2$-risk', title='Performance on train samples',
        stat_name=train_perf_name+'__mean',
        stat_dev_name=train_perf_name+'__std',
        d=d, skipped_estimators=skipped_estimators,
    )
    test_perf_type = 'risk' if d is None else 'error'
    test_perf_name = f'test_{report_loss_name}-{test_perf_type}'
    plot_stats(
        ax=ax2, all_stats=all_stats,
        ylabel='$L_2$-risk', title='Performance on test samples',
        stat_name=test_perf_name+'__mean',
        stat_dev_name=test_perf_name+'__std',
        d=d, skipped_estimators=skipped_estimators,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    train_time_name = 'train_real_time'
    plot_stats(
        ax=ax1, all_stats=all_stats, scaler=1.0/60,
        ylabel='minutes', title='Training time',
        stat_name=train_time_name+'__mean',
        stat_dev_name=train_time_name+'__std',
        d=d, skipped_estimators=skipped_estimators,
    )
    test_time_name = 'test_real_time'
    plot_stats(
        ax=ax2, all_stats=all_stats, scaler=1.0/60,
        ylabel='minutes', title='Test (prediction) time',
        stat_name=test_time_name+'__mean',
        stat_dev_name=test_time_name+'__std',
        d=d, skipped_estimators=skipped_estimators,
    )
