"""Plot SRBench snapshot and empirical-overlap portfolio curves as separate figures.

Run: python figures/solve_over_time/solve_over_time.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, StrMethodFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LEFT = ROOT/'figures/srbench_10seed_snapshot_solve_rate/data.json'
RIGHT = ROOT/'figures/srbench2_1m_spliced_snap5/curve.csv'
RECOVERY = ROOT/'runs/srbench2_9-17_1m_spliced_terra_recovery/first_recovery.json'
METHODS = [('baseline', 'Baseline', 'PySR', '#3975b7'),
           ('evolved', '709715', 'Evolved PySR', '#db7825')]


def style(ax):
    ax.set(ylim=(0, 100), ylabel='Cumulative recovery rate (%)', yticks=range(0,101,20))
    ax.yaxis.set_minor_locator(FixedLocator(range(10,100,20)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='y', which='minor', length=0)
    ax.grid(which='major', alpha=.22)
    ax.grid(axis='y', which='minor', alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False, loc='lower right')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT/name)
    plt.close(fig)
    print(OUT/name)


def plotted_seconds(record):
    """Same rule as figures/plot_srbench2_spliced_portfolio.py."""
    if not record['final_positive']:
        return None
    if record.get('first_solve_kind') == 'first_restart_snapshot':
        return min(float(record['first_solve_scheduled_seconds']), 3600.)
    return record['first_solve_budget_seconds']


def empirical_with_seed_sd(right_rows, figsize):
    """Empirical curve as mean over 10 seeds (each seed = % of 9 problems), with +/-1 SD band."""
    records = json.loads(RECOVERY.read_text())
    fig, ax = plt.subplots(figsize=figsize)
    for key, method, label, color in METHODS:
        rows = [r for r in right_rows if r['method'] == method]
        xs = np.array([float(r['seconds']) for r in rows])
        ref = np.array([float(r['solve_rate_percent']) for r in rows])
        recs = [r for r in records if r['method'] == method]
        seeds = sorted({r['seed'] for r in recs})
        assert len(seeds) == 10
        per_seed = []
        for seed in seeds:
            times = [t for r in recs if r['seed'] == seed and (t := plotted_seconds(r)) is not None]
            n = sum(r['seed'] == seed for r in recs)
            assert n == 9
            per_seed.append([100*sum(t <= x for t in times)/n for x in xs])
        per_seed = np.array(per_seed)
        mean, sd = per_seed.mean(0), per_seed.std(0, ddof=1)
        assert np.allclose(mean, ref)
        total = round(mean[-1]*90/100)
        ax.fill_between(xs/60, np.maximum(0, mean-sd), np.minimum(100, mean+sd), step='post',
                        color=color, alpha=.18, linewidth=0)
        ax.step(xs/60, mean, where='post', color=color, linewidth=2.2,
                label=f'{label} ({total}/90)')
        print(f'{label}: final {mean[-1]:.1f}% +/- {sd[-1]:.1f} (seed SD)')
    ax.set(xscale='log', xlim=(1/60, 60), xlabel='Search time (min)')
    ax.set_xticks([.02, .1, .5, 1, 5, 15, 60])
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    style(ax)
    return fig


def main():
    table = json.loads(LEFT.read_text())['tables']['all']
    with RIGHT.open() as f:
        right_rows = list(csv.DictReader(f))
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'savefig.facecolor': 'white'})
    scale = 0.6
    figsize = (7.6*scale, 5*scale)  # one panel of the former 15.2x5 two-panel figure
    left_fig, left = plt.subplots(figsize=figsize)
    right_fig, right = plt.subplots(figsize=figsize)
    times = sorted(map(int, table))
    assert max(times) == 90
    for key, method, label, color in METHODS:
        mean = np.array([table[str(t)][key]['mean'] for t in times])
        sd = np.array([table[str(t)][key]['sd'] for t in times])
        left.fill_between(times, np.maximum(0, mean-sd), np.minimum(100, mean+sd),
                          color=color, alpha=.18, linewidth=0)
        left.plot(times, mean, color=color, marker='o', markersize=4.2,
                  linewidth=2.2, label=label)
        rows = [r for r in right_rows if r['method'] == method]
        xs = [float(r['seconds'])/60 for r in rows]
        ys = [float(r['solve_rate_percent']) for r in rows]
        total = round(ys[-1]*90/100)
        assert total == (72 if method == 'Baseline' else 74)
        right.step(xs, ys, where='post', color=color, linewidth=2.2,
                   label=f'{label} ({total}/90)')
    left.set(xlim=(9, 90), xlabel='Search time (sec)')
    left.set_xticks(range(10, 91, 10))
    right.set(xscale='log', xlim=(1/60, 60), xlabel='Search time (min)')
    right.set_xticks([.02, .1, .5, 1, 5, 15, 60])
    right.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    right.xaxis.set_minor_formatter(NullFormatter())
    style(left)
    style(right)
    save(left_fig, 'srbench_solve_over_time.pdf')
    save(right_fig, 'empirical_solve_over_time.pdf')
    save(empirical_with_seed_sd(right_rows, figsize), 'empirical_solve_over_time_errorbars.pdf')


if __name__ == '__main__':
    main()
