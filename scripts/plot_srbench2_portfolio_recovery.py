#!/usr/bin/env python3
"""Plot completed recovery records, with an optional EmpiricalBench overlap filter.

No API calls, restart processing, or SLURM jobs.
python scripts/plot_srbench2_portfolio_recovery.py --empirical-only
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EMPIRICAL_EXACT_OVERLAP = {
    'hubble', 'ideal_gas', 'kepler', 'leavitt', 'newton', 'planck', 'rydberg', 'schechter',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--empirical-only', action='store_true')
    parser.add_argument('--input', type=Path, default=ROOT/'reports/srbench2_portfolio_solve_over_time/first_recovery.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    records = json.loads(args.input.read_text())
    if args.empirical_only:
        records = [r for r in records if r['dataset'].removeprefix('first_principles_') in EMPIRICAL_EXACT_OVERLAP]
    tasks = sorted({r['dataset'] for r in records})
    n_tasks = len(tasks)
    assert n_tasks == (8 if args.empirical_only else 10)
    out = args.output or ROOT/'figures'/('empiricalbench_overlap_portfolio_solve_over_time' if args.empirical_only else 'srbench2_portfolio_solve_over_time')
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    table = []
    for method, color in [('Baseline', '#3264ad'), ('709715', '#d45b24')]:
        rows = [r for r in records if r['method'] == method]
        seeds = {r['seed'] for r in rows}
        assert len(seeds) == 10 and len(rows) == n_tasks*10
        assert len({(r['dataset'], r['seed']) for r in rows}) == len(rows)
        times = sorted(r['first_solve_budget_seconds']/60 for r in rows if r['first_solve_budget_seconds'] is not None)
        xs = [0] + times + [60]
        ys = [0] + [i/len(seeds) for i in range(1, len(times)+1)] + [len(times)/len(seeds)]
        ax.step(xs, ys, where='post', color=color, lw=2, label=f'{method} ({ys[-1]:.1f}/{n_tasks} at 60 min)')
        for minute in range(61):
            count = sum(t <= minute for t in times)
            table.append(dict(method=method, minutes=minute, solved_trials=count, total_trials=len(rows), mean_tasks_solved=count/len(seeds)))
    title = 'EmpiricalBench overlap in SRBench2' if args.empirical_only else 'SRBench2'
    ax.set(xlim=(0, 60), ylim=(0, n_tasks), xlabel='Cumulative search time (minutes)',
           ylabel=f'Mean tasks solved out of {n_tasks}', title=f'{title} · one-hour portfolios · 10 seeds')
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    note = ('8 shared reference-family tasks; Bode excluded (phenomenological in SRBench2).'
            if args.empirical_only else '10 reference-family tasks; absorption and Bode excluded.')
    fig.text(.5, .025, note+'\nApproximate cumulative-frontier binary search; restart-end timing; final negatives treated as unsolved.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, 1))
    for suffix in ['png', 'pdf']:
        fig.savefig(out/f'solve_rate.{suffix}', dpi=180)
    ax.set(xscale='log', xlim=(.05, 60), xlabel='Cumulative search time (minutes, log scale)')
    ax.set_xticks([.1, .5, 1, 5, 15, 60], labels=['0.1', '0.5', '1', '5', '15', '60'])
    for suffix in ['png', 'pdf']:
        fig.savefig(out/f'solve_rate_log.{suffix}', dpi=180)
    plt.close(fig)
    with (out/'solve_rate.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (out/'README.md').write_text(
        f'# {title}: portfolio recovery\n\n'
        'Plots use the completed SRBench2 recovery records; no new reviews were made. '
        'This is a subset of SRBench2 runs, not separate EmpiricalBench evaluations.\n\n'
        +note+'\n\nIncluded tasks: '+', '.join(t.removeprefix('first_principles_') for t in tasks)+'.\n\n'
        'Mean tasks solved is the number of recovered task–seed trials divided by ten seeds. '
        'All tasks/seeds remain in the denominator, including final negatives. '
        'The binary-search approximation can miss temporary recoveries.\n\n'
        'Reproduce: `python scripts/plot_srbench2_portfolio_recovery.py'+(' --empirical-only' if args.empirical_only else '')+'`\n')
    print(out)
    print([r for r in table if r['minutes'] == 60])


if __name__ == '__main__':
    main()
