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


if __name__ == '__main__':
    main()
