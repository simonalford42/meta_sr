"""Plot cumulative symbolic recovery from completed 5-second snapshots.

Run from any directory: python figures/plot_empiricalbench_snapshot_solve_rate.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/empiricalbench_90s_snapshot_solve_rate'
SOURCES = {
    'Base PySR': 'empiricalbench_baseline_9-15_10seed_90s_snap5',
    'Evolved 709715': '709715-empiricalbench_9-15_10seed_90s_snap5',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    table = []
    for (label, dirname), color in zip(SOURCES.items(), ['#2878B5', '#D55E00']):
        records = json.loads((ROOT / 'runs' / dirname / 'snapshot_solve_times.json').read_text())['records']
        assert len(records) == 90 and all(r['status'] == 'complete' for r in records)
        assert len({(r['dataset'], r['seed']) for r in records}) == 90
        times = list(range(0, 91, 5))
        counts = [sum(any(o.get('matched_equation') and not o.get('final')
                          and o.get('scheduled_seconds') is not None
                          and o['scheduled_seconds'] <= t
                          for o in r['observations']) for r in records) for t in times]
        rates = [100 * n / len(records) for n in counts]
        ax.step(times, rates, where='post', color=color, lw=2.3, label=label)
        ax.plot(times[1:], rates[1:], 'o', color=color, ms=3)
        ax.annotate(f'{rates[-1]:.1f}% ({counts[-1]}/90)', (90, rates[-1]),
                    xytext=(-8, 9), textcoords='offset points', ha='right', color=color, weight='bold')
        table.extend(dict(method=label, snapshot_seconds=t, solved=n, total=90, solve_rate_percent=v)
                     for t, n, v in zip(times, counts, rates))
    ax.set(xlim=(0, 92), ylim=(0, 100), xlabel='Elapsed fit time (seconds)',
           ylabel='Cumulative symbolic solve rate', title='EmpiricalBench · 9 problems × 10 seeds')
    ax.set_xticks(range(0, 91, 10))
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(axis='y', alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', frameon=False)
    fig.text(.5, .025, '5-second snapshot checkpoints; includes fit startup. Later final frontiers excluded.\n'
             'Symbolic matching; Planck/Rydberg clean-grid checks are not applied.', ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .075, 1, 1))
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'solve_rate.{ext}', dpi=200)
    plt.close(fig)
    with (OUT / 'solve_rate.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (OUT / 'README.md').write_text('# EmpiricalBench 90-second snapshot comparison\n\n'
        'Reproduce with `python figures/plot_empiricalbench_snapshot_solve_rate.py`.\n\n'
        'Cumulative confirmed symbolic matches at nominal snapshot checkpoints. Each of 90 problem–seed trials '
        'has equal weight. Times include fit startup; later final frontiers are excluded. '
        'Unresolved checks are not counted as solves. This uses symbolic matching without the separate '
        'Planck/Rydberg clean-grid check.\n\nSources:\n\n' + ''.join(
            f'- {label}: `runs/{dirname}/snapshot_solve_times.json`\n' for label, dirname in SOURCES.items()))
    print(OUT)


if __name__ == '__main__':
    main()
