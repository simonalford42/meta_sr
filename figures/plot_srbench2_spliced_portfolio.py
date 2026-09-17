"""Plot Terra-reviewed synthetic portfolios without assuming final solve counts."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, FixedLocator, NullFormatter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('review_dir', type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('figures/srbench2_1m_spliced_snap5'))
    a = p.parse_args()
    records = json.loads((a.review_dir/'first_recovery.json').read_text())
    if len(records) != 180:
        raise ValueError('Expected 180 reviewed trials')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    # Match the reference overall_linear.pdf page dimensions (points / 72).
    fig, ax = plt.subplots(figsize=(537.890625/72,350.1809375/72))
    def plotted_seconds(record):
        if not record['final_positive']:
            return None
        if record.get('first_solve_kind') == 'first_restart_snapshot':
            scheduled = record['first_solve_scheduled_seconds']
            assert scheduled is not None and scheduled > 0
            return min(float(scheduled), 3600.)
        return record['first_solve_budget_seconds']

    # Use the scheduled capture clock for snapshots, retaining the original
    # per-trial actual capture times in the review data. Restart ends have no
    # shared scheduled checkpoint, so retain their recorded recovery times.
    checkpoints = {3600.}
    for path in (a.review_dir/'snapshots').glob('*.json'):
        for observation in json.loads(path.read_text()):
            scheduled = observation.get('scheduled_seconds')
            if scheduled is not None and 0 < scheduled <= 3600:
                checkpoints.add(float(scheduled))
    checkpoints.update(t for r in records if (t := plotted_seconds(r)) is not None)
    xs = sorted(checkpoints)
    table = []
    for method, label, color in [('Baseline','PySR','#2878B5'),('709715','Evolved PySR','#D55E00')]:
        rows = [r for r in records if r['method']==method]
        assert len(rows)==90
        times = [t for r in rows if (t := plotted_seconds(r)) is not None]
        assert all(1 <= t <= 3600 for t in times)
        n = len(times)
        ys = [sum(t <= checkpoint for t in times)/90*100 for checkpoint in xs]
        assert len(xs) == len(set(xs)) and ys[-1] == n/90*100
        ax.plot([t/60 for t in xs], ys, color=color, linestyle="-", linewidth=2.3,
                label=f'{label} ({n}/90)')
        table.extend(dict(method=method, seconds=t, solve_rate_percent=y) for t,y in zip(xs,ys))
    ax.set(xscale='log', xlim=(1/60,60), ylim=(0,100), xlabel='Search time (min)',
           ylabel='Cumulative recovery rate (%)')
    ax.set_xticks([0.02,0.1,0.5,1,5,15,60]); ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(0,101,20))
    ax.yaxis.set_minor_locator(FixedLocator(range(10,100,20)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(alpha=.22)
    ax.grid(axis="y", which="minor", alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,loc='upper left')
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(a.output_dir/f'solve_rate.{ext}',dpi=200)
    plt.close(fig)
    with (a.output_dir/'curve.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    (a.output_dir/'README.md').write_text(f'# Synthetic SRBench2 portfolio plot\n\nSource: `{a.review_dir}`.\n\n'
        'New first restarts replace archived first restarts; later fits are reused and timestamps shifted. '
        'Solid lines connect cumulative recovery values at the shared checkpoints, including checkpoints with no increase. Five-second captures use their scheduled times (5, 10, 15 seconds, etc.), not the slightly delayed actual read times. The common checkpoint grid combines scheduled captures and recorded restart-end recovery times, plus the one-hour endpoint. Raw review timings remain unchanged. The logarithmic axis is in minutes; CSV times are in seconds. '
        'Final totals are computed, never fixed to historical counts. See the review README for scoring and timing caveats.\n')
    print(a.output_dir/'solve_rate.png')


if __name__ == '__main__': main()
