"""Plot Terra-reviewed EmpiricalBench recovery against logarithmic seconds.

Reproduce: python figures/plot_empiricalbench_portfolio_log_seconds.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/empiricalbench_9-16_portfolio90s_terra_recovery/first_recovery.json'
OUT = ROOT / 'figures/empiricalbench_portfolio_log_seconds'


def main():
    records = json.loads(SOURCE.read_text())
    assert len(records) == 180
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    exported = []
    for method, label, color, style in [
        ('Baseline', 'Base PySR', '#2878B5', '-'),
        ('709715', 'Evolved 709715', '#D55E00', '--'),
    ]:
        rows = [r for r in records if r['method'] == method]
        assert len(rows) == 90 and len({(r['dataset'], r['seed']) for r in rows}) == 90
        counts = Counter(r['first_solve_budget_seconds'] for r in rows
                         if r['first_solve_budget_seconds'] is not None)
        assert all(1 <= t <= 3600 for t in counts)
        xs, ys, solved = [1.0], [0.0], 0
        for seconds, count in sorted(counts.items()):
            solved += count
            xs.append(seconds)
            ys.append(100 * solved / len(rows))
        xs.append(3600)
        ys.append(100 * solved / len(rows))
        ax.step(xs, ys, where='post', color=color, ls=style, lw=2.4, label=label)
        exported.extend({'method': label, 'seconds': t, 'solve_rate_percent': rate}
                        for t, rate in zip(xs, ys))
    ax.set(xscale='log', xlim=(1, 3600), ylim=(0, 100),
           xlabel='Elapsed search time (seconds, log scale)', ylabel='Cumulative recovery rate',
           title='EmpiricalBench · one-hour portfolios · 10 seeds per problem')
    ax.set_xticks([1, 5, 10, 30, 90, 300, 900, 3600])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(which='major', alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper left')
    ax.annotate('Both: 71/90 (78.9%)', xy=(3600, 100*71/90), xytext=(-8, 12),
                textcoords='offset points', ha='right', fontsize=10)
    fig.text(.5, .025,
        'Terra-reviewed fitted-family recovery; 9 problems × 10 seeds. Warm-up excluded.\n'
        'Approximate binary-search timings; transient recoveries can be missed.',
        ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .075, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'solve_rate.{ext}', dpi=200)
    plt.close(fig)
    with (OUT/'curve.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(exported)
    (OUT/'README.md').write_text(
        '# EmpiricalBench recovery on a logarithmic seconds axis\n\n'
        'Reproduce: `python figures/plot_empiricalbench_portfolio_log_seconds.py`.\n\n'
        f'Source: `{SOURCE.relative_to(ROOT)}`.\n\n'
        'Curves step at the per-trial first recovery times from Terra binary-search reviews, '
        'rather than interpolating the sampled summary table. Each trial has equal weight. '
        'Near matches are excluded; accepted empirical-family labels are normalized as documented '
        'in the review results. Search time excludes warm-up; the final overshoot is clipped to '
        'the 3600-second budget, matching the source budget-time column. The axis begins at one '
        'second because zero cannot appear on a logarithmic axis.\n\n'
        'Binary search assumes recovery persists on cumulative Pareto frontiers; temporary '
        'recoveries and final-negative trials with earlier matches can be missed.\n')
    print(OUT)


if __name__ == '__main__':
    main()
