#!/usr/bin/env python3
"""Compact MIPS recovery table, generated from all three raw ten-seed evaluations.

Run: python figures/plot_mips_task_checkmarks.py
Output: figures/mips_task_checkmarks.pdf

Recovery pools components across seeds. Original successes are excluded.
709714 evolved on MIPS from scratch; it is not an SRBench-bundle fine-tune,
so its column is labeled MIPS-evolved. Row order follows the discussion's
approximate difficulty order; the last three tasks are unordered failures.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_mips_results import DEFAULTS, inspect

TASKS = [
    ('rnn_min_numerical', 'Minimum'),
    ('rnn_max_numerical', 'Maximum'),
    ('rnn_parity_last2_numerical', 'Parity-last2'),
    ('rnn_unique2_numerical', 'Unique2'),
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
    reproduction = json.loads((ROOT / 'outputs/mips_reproduction_all/summary.json').read_text())
    original = {r['task'] for r in reproduction['tasks'] if r['independent_success']}
    methods = [inspect(ROOT / p, excluded_tasks=original) for p in DEFAULTS.values()]
    for m in methods:
        assert not m['missing'] and len(m['runs']) == 10
        assert set(m['groups']) == {t for t, _ in TASKS}
    solved = [m['assembled'] for m in methods]
    totals = [len(s) for s in solved]
    assert totals == [7, 9, 11], f'Historical comparison changed: {totals}'

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(4.5, 4.15))
    fig.subplots_adjust(left=.025, right=.98, bottom=.02, top=.98)
    ax.set_xlim(0, 4.5)
    ax.set_ylim(-.8, 15.8)
    ax.axis('off')
    ink, gray = '#24333D', '#89949C'
    colors = ['#317DA5', '#BB7A20', '#22826A']
    xs = [2.50, 3.27, 4.07]
    ax.text(.12, 15.05, 'Task', fontweight='bold', color=ink, va='center')
    for x, label, color in zip(xs, ['Base', 'Evolved', 'MIPS-\nevolved'], colors):
        ax.text(x, 15.05, label, fontweight='bold', ha='center', va='center', color=color, linespacing=1.1)
    ax.plot([.08, 4.47], [14.38, 14.38], color=ink, lw=.9)
    for i, (task, label) in enumerate(TASKS):
        y = 13.75 - i
        if i % 2 == 0:
            ax.add_patch(Rectangle((.08, y-.48), 4.39, .96, facecolor='#F3F5F6', edgecolor='none'))
        ax.text(.12, y, label, color=ink, va='center')
        for x, successes, color in zip(xs, solved, colors):
            if task in successes:
                # Draw a checkmark as vectors, independent of symbol font support.
                ax.plot([x-.09, x-.025, x+.11], [y, y-.12, y+.17],
                        lw=1.65, color=color, solid_capstyle='round', solid_joinstyle='round')
            else:
                ax.text(x, y, '–', ha='center', va='center', color=gray)
    ax.plot([.08, 4.47], [.13, .13], color=ink, lw=.9)
    ax.text(.12, -.43, 'Solved / 14', fontweight='bold', color=ink, va='center')
    for x, total, color in zip(xs, totals, colors):
        ax.text(x, -.43, str(total), ha='center', va='center', fontweight='bold', color=color, fontsize=11)
    output = ROOT / 'figures/mips_task_checkmarks.pdf'
    fig.savefig(output, metadata={'Title': 'MIPS task recovery',
        'Subject': 'Base PySR vs SRBench-evolved 709715 vs MIPS-evolved 709714; pooled ten-seed exact transition-table recovery'})
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
