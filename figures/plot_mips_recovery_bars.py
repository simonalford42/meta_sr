#!/usr/bin/env python3
"""Plot successful scalar fits and pooled task recovery from ten-seed results.

Run: python figures/plot_mips_recovery_bars.py
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_mips_results import DEFAULTS, inspect

OUT = ROOT / 'figures/mips_recovery_bars'


def main():
    reproduction = json.loads((ROOT / 'outputs/mips_reproduction_all/summary.json').read_text())
    excluded = {r['task'] for r in reproduction['tasks'] if r['independent_success']}
    excluded.update(('rnn_parity_last2_numerical', 'rnn_unique2_numerical'))
    assert len(excluded) == 32
    labels = ['Base', 'Evolved', 'MIPS-evolved']
    colors = ['#317DA5', '#BB7A20', '#C33D3D']
    records = []
    for label, path in zip(labels, DEFAULTS.values()):
        method = inspect(ROOT / path, excluded_tasks=excluded)
        assert not method['missing']
        assert len(method['components']) == 39 and len(method['runs']) == 10
        assert len(method['groups']) == 12 and len(method['records']) == 390
        records.append({
            'method': label, 'source': path,
            'successful_fits': sum(r.get('gt_match_score') == 1 for r in method['records']),
            'total_fits': len(method['records']),
            'tasks_solved': len(method['assembled']), 'total_tasks': len(method['groups']),
            'task_criterion': 'Every component recovered in at least one seed; seeds may differ',
        })
    assert [r['successful_fits'] for r in records] == [203, 230, 245]
    assert [r['tasks_solved'] for r in records] == [5, 7, 9]
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11,
                         'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    fig.subplots_adjust(left=.13, right=.98, bottom=.14, top=.83)
    width = .30
    for i, (record, color) in enumerate(zip(records, colors)):
        for offset, numerator, denominator, hatch in [
            (-.18, record['successful_fits'], record['total_fits'], None),
            (.18, record['tasks_solved'], record['total_tasks'], '///'),
        ]:
            height = numerator / denominator
            ax.bar(i + offset, height, width, color=color if hatch is None else 'white',
                   edgecolor=color, hatch=hatch, linewidth=1.4, zorder=3)
            ax.text(i + offset, height + .025, f'{numerator}/{denominator}',
                    ha='center', va='bottom', fontsize=10, color=color, weight='bold')
    ax.set_xticks(range(3), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Recovery rate')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', alpha=.2, zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(handles=[Patch(facecolor='#687580', edgecolor='#687580', label='Successful fits'),
                       Patch(facecolor='white', edgecolor='#687580', hatch='///', label='Tasks solved')],
              loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False)
    fig.savefig(OUT.with_suffix('.pdf'), metadata={
        'Title': 'MIPS successful fits and tasks solved',
        'Subject': 'Paper-unsolved scope; 390 scalar fits and 12 tasks; tasks pool ten seeds'})
    plt.close(fig)
    OUT.with_suffix('.json').write_text(json.dumps(records, indent=2) + '\n')
    OUT.with_name(OUT.name + '_caption.txt').write_text(r'''\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/mips_recovery_bars.pdf}
\caption{MIPS recovery on twelve tasks not solved in the original paper. Solid bars show exact scalar-fit successes over 39 subproblems and ten seeds (390 fits); hatched bars show algorithmic tasks for which every component is recovered in at least one seed, allowing different seeds for different components (twelve tasks). Heights are percentages and annotations give raw counts. Base, Evolved, and MIPS-evolved denote baseline PySR, SRBench-evolved run 709715, and MIPS-evolved run 709714, respectively. The five paper-solved training task families are excluded. Each fit has a budget of $10^6$ evaluations and a 500-second timeout. Task recovery refers to the recorded component relations, without separate end-to-end validation of assembled programs.}
\label{fig:mips-recovery-bars}
\end{figure}
''')
    print(OUT.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
