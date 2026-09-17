"""Plot Terra-reviewed synthetic portfolios without assuming final solve counts."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, PercentFormatter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('review_dir', type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('figures/srbench2_1m_spliced_snap5'))
    a = p.parse_args()
    records = json.loads((a.review_dir/'first_recovery.json').read_text())
    if len(records) != 180:
        raise ValueError('Expected 180 reviewed trials')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9,5.6))
    table = []
    for method, label, color, style in [('Baseline','Base PySR','#2878B5','-'),('709715','Evolved 709715','#D55E00','--')]:
        rows = [r for r in records if r['method']==method]
        assert len(rows)==90
        events = Counter(r['first_solve_budget_seconds'] for r in rows if r['final_positive'])
        xs, ys, n = [1.], [0.], 0
        for t, count in sorted(events.items()):
            assert 1 <= t <= 3600
            n += count; xs.append(t); ys.append(n/90*100)
        xs.append(3600); ys.append(n/90*100)
        ax.step(xs, ys, where='post', lw=2.3, color=color, ls=style,
                label=f'{label}: {n}/90 ({n/90*100:.1f}%)')
        table.extend(dict(method=method, seconds=t, solve_rate_percent=y) for t,y in zip(xs,ys))
    ax.set(xscale='log', xlim=(1,3600), ylim=(0,100), xlabel='Cumulative search time (seconds, log scale)',
           ylabel='Cumulative observed recovery rate', title='SRBench2 empirical overlap · synthetic 1M-evaluation portfolios')
    ax.set_xticks([1,5,10,30,90,300,900,3600]); ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter()); ax.grid(alpha=.22)
    ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,loc='upper left')
    fig.text(.5,.025,'New first restart: 5-second captures; later restarts: archived endpoints. Warm-up excluded.\n'
             'Terra binary-search approximation; broad Bode family criterion; 9 tasks × 10 seeds.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.08,1,1))
    for ext in ['png','pdf']: fig.savefig(a.output_dir/f'solve_rate.{ext}',dpi=200)
    plt.close(fig)
    with (a.output_dir/'curve.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    (a.output_dir/'README.md').write_text(f'# Synthetic SRBench2 portfolio plot\n\nSource: `{a.review_dir}`.\n\n'
        'New first restarts replace archived first restarts; later fits are reused and timestamps shifted. '
        'Final totals are computed, never fixed to historical counts. See the review README for scoring and timing caveats.\n')
    print(a.output_dir/'solve_rate.png')


if __name__ == '__main__': main()
