#!/usr/bin/env python3
"""Compact MIPS recovery table, generated from all three raw ten-seed evaluations.

Run: python figures/plot_mips_task_checkmarks.py
Output: figures/mips_task_checkmarks.pdf

Recovery pools components across seeds. Paper-reported successes are excluded,
including Parity_Last2 and Previous_Equals_Current (Unique2), which our raw
checkpoint reproduction did not recover.
709714 evolved on MIPS from scratch; it is not an SRBench-bundle fine-tune,
so its column is labeled MIPS-evolved. Row order follows the discussion's
approximate difficulty order; the last three tasks are unordered failures.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MarkerPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_mips_results import DEFAULTS, inspect

PAPER_SOLVED_NOT_REPRODUCED = {
    'rnn_parity_last2_numerical',
    'rnn_unique2_numerical',
}

TASKS = [
    ('rnn_min_numerical', 'Minimum'),
    ('rnn_max_numerical', 'Maximum'),
    ('rnn_base_3_addition', 'Base-3 addition'),
    ('rnn_base_6_addition', 'Base-6 addition'),
    ('rnn_alternating_last4_numerical', 'Alternating-last4'),
    ('rnn_base_4_addition', 'Base-4 addition'),
    ('rnn_base_5_addition', 'Base-5 addition'),
    ('rnn_base_7_addition', 'Base-7 addition'),
    ('rnn_parity_last4_numerical', 'Parity-last4'),
    ('rnn_div_7_numerical', 'Divisibility by 7'),
    ('rnn_div_5_numerical', 'Divisibility by 5'),
    ('rnn_alternating_last3_numerical', 'Alternating-last3'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style', choices=['plain', 'ruled'], default='plain')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'figures/mips_task_checkmarks.pdf')
    args = parser.parse_args()
    reproduction = json.loads((ROOT / 'outputs/mips_reproduction_all/summary.json').read_text())
    original = {r['task'] for r in reproduction['tasks'] if r['independent_success']}
    paper_solved = original | PAPER_SOLVED_NOT_REPRODUCED
    assert len(paper_solved) == 32
    methods = [inspect(ROOT / p, excluded_tasks=paper_solved) for p in DEFAULTS.values()]
    for m in methods:
        assert not m['missing'] and len(m['runs']) == 10
        assert set(m['groups']) == {t for t, _ in TASKS}
    solved = [m['assembled'] for m in methods]
    totals = [len(s) for s in solved]
    assert totals == [5, 7, 9], f'Paper-unsolved comparison changed: {totals}'

    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11.5,
                         'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(4.3, 4.15))
    fig.subplots_adjust(left=.025, right=.98, bottom=.02, top=.98)
    ax.set_xlim(0, 4.3)
    n_tasks = len(TASKS)
    ax.set_ylim(-.8, n_tasks + 1.8)
    ax.axis('off')
    ink, gray = '#24333D', '#89949C'
    colors = ['#317DA5', '#BB7A20', '#C33D3D']
    # Marker coordinates preserve a 90-degree bend regardless of axis scaling.
    checkmark = MarkerPath([(-.60, .10), (-.15, -.35), (.75, .55)])
    xs = [2.35, 3.13, 3.94]
    ax.text(.12, n_tasks + 1.05, 'Task', fontweight='bold', color=ink, va='center')
    for x, label, color in zip(xs, ['Base', 'Evolved', 'MIPS-\nevolved'], colors):
        ax.text(x, n_tasks + 1.05, label, fontweight='bold', ha='center', va='center', color=color, linespacing=1.1)
    ax.plot([.08, 4.27], [n_tasks + .38, n_tasks + .38], color=ink, lw=.9)
    for i, (task, label) in enumerate(TASKS):
        y = n_tasks - .25 - i
        if args.style == 'ruled' and i < len(TASKS) - 1:
            ax.plot([.08, 4.27], [y-.5, y-.5], color='#D7DCE0', lw=.4, zorder=0)
        ax.text(.12, y, label, color=ink, va='center')
        for x, successes, color in zip(xs, solved, colors):
            if task in successes:
                # Draw a checkmark as vectors, independent of symbol font support.
                ax.plot([x], [y], linestyle='none', marker=checkmark,
                        markersize=13, markerfacecolor='none',
                        markeredgecolor=color, markeredgewidth=1.65)
            else:
                ax.text(x, y, '–', ha='center', va='center', color=gray)
    ax.plot([.08, 4.27], [.13, .13], color=ink, lw=.9)
    ax.text(.12, -.43, f'Solved / {n_tasks}', fontweight='bold', color=ink, va='center')
    for x, total, color in zip(xs, totals, colors):
        ax.text(x, -.43, str(total), ha='center', va='center', fontweight='bold', color=color, fontsize=12.5)
    output = args.output
    fig.savefig(output, metadata={'Title': 'MIPS task recovery',
        'Subject': 'Base PySR vs SRBench-evolved 709715 vs MIPS-evolved 709714; pooled ten-seed exact transition-table recovery on 12 paper-unsolved tasks'})
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
