#!/usr/bin/env python3
"""Render the refined Parity-Last-4 example from artifacts and raw evaluations.

Run: python figures/plot_mips_parity_last4_example.py
Outputs: PDF, LaTeX caption, and a compact audit record in figures/.
"""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mips_tasks import load_component_artifact
from scripts.inspect_mips_results import DEFAULTS, inspect

TASK = 'rnn_parity_last4_numerical'
OUT = ROOT / 'figures/mips_parity_last4_example'
ARTIFACTS = ROOT / 'outputs/mips_evolution_51_artifacts'
COMPONENTS = [f'mips:{TASK}:{suffix}' for suffix in
              ('hidden:0', 'hidden:1', 'hidden:2', 'output:0')]


def main():
    methods = [inspect(ROOT / path) for path in DEFAULTS.values()]
    assert all(not m['missing'] and len(m['runs']) == 10 for m in methods)
    artifacts = [load_component_artifact(c, root=ARTIFACTS) for c in COMPONENTS]
    rows = [len(a['X_full']) for a in artifacts]
    assert rows == [32, 32, 32, 16]
    counts = [[len(m['exact'][c]) for m in methods] for c in COMPONENTS]
    totals = [sum(bool(m['exact'][c]) for c in COMPONENTS) for m in methods]
    assert totals == [3, 3, 4]
    assert [len(m['joint'][TASK]) for m in methods] == [0, 0, 1]
    old_path = ROOT / f'outputs/mips_transition_tables/tasks/{TASK}/diagnostic.json'
    old = json.loads(old_path.read_text())
    old_states = next(c['unique_input_count'] for c in old['components']
                      if c['kind'] == 'output')
    assert old_states == 12
    assert np.unique(artifacts[0]['X_full'][:, :3], axis=0).shape[0] == 16
    assert all(a['metadata']['deterministic'] for a in artifacts)
    assert all(a['metadata']['provenance']['scale'] == 4 for a in artifacts)

    ink, muted = '#24333D', '#5E6B73'
    colors = ['#317DA5', '#BB7A20', '#C33D3D']
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'pdf.fonttype': 42, 'mathtext.fontset': 'dejavuserif'})
    fig = plt.figure(figsize=(8.5, 5.25))
    ax = fig.add_axes([.025, .03, .95, .94])
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis('off')
    ax.text(0, 5.8, 'Parity-Last-4: one algorithmic task, four SR subproblems',
            fontsize=13, weight='bold', color=ink)
    ax.text(0, 5.42, r'$y_t=(x_t+x_{t-1}+x_{t-2}+x_{t-3})\;\mathrm{mod}\;2$',
            fontsize=12, color=ink)
    ax.text(0, 5.1, 'Return 1 when the last four bits contain an odd number of ones.',
            fontsize=9.5, color=muted)
    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    parity = [sum(bits[max(0, i-3):i+1]) % 2 for i in range(len(bits))]
    for y, label, values in [(4.58, 'Input', bits), (4.03, 'Output', parity)]:
        ax.text(0, y+.08, label, va='center', color=ink)
        for i, value in enumerate(values):
            x = 1.05 + .45*i
            selected = y == 4.58 and i >= 4
            ax.add_patch(Rectangle((x, y-.13), .37, .42,
                         facecolor='#DDEAF2' if selected else '#F0F2F3',
                         edgecolor=colors[0] if selected else '#D1D7DB', lw=.7))
            ax.text(x+.185, y+.08, str(value), ha='center', va='center', color=ink)
    ax.text(0, 3.63, 'Missing preceding bits are zero; the final four inputs are highlighted.',
            fontsize=8, color=muted)

    ax.plot([5.05, 5.05], [3.65, 4.95], color='#D1D7DB', lw=.8)
    ax.text(5.35, 4.85, 'Refine the learned integer encoding', weight='bold', color=ink)
    ax.text(5.35, 4.42, r'$\mathrm{round}(Z)$', color=ink, fontsize=12)
    ax.text(8.00, 4.42, r'$\mathrm{round}(4Z)$', color=ink, fontsize=12)
    ax.annotate('', xy=(7.75, 4.5), xytext=(7.10, 4.5),
                arrowprops={'arrowstyle': '->', 'color': muted, 'lw': 1.2})
    ax.text(5.35, 4.02, '12 observed states', color=ink)
    ax.text(8.00, 4.02, '16 observed states', color=ink)
    ax.text(5.35, 3.73, 'Conflicting transitions', fontsize=8.5, color=muted)
    ax.text(8.00, 3.73, 'All relations deterministic', fontsize=8.5, color=muted)

    ax.text(0, 3.14, 'Three state updates + one output function', weight='bold', color=ink)
    ax.text(6.00, 3.14, 'Exact recovery: successful seeds / 10', fontsize=9.5, color=muted)
    xs = [6.50, 7.95, 9.40]
    ax.text(.10, 2.66, 'Scalar relation', weight='bold', color=ink)
    ax.text(4.25, 2.66, 'Input-table size', weight='bold', ha='center', color=ink)
    for x, label, color in zip(xs, ['Base\nPySR', 'SRBench-evolved\n709715', 'MIPS-evolved\n709714'], colors):
        ax.text(x, 2.73, label, ha='center', va='center', fontsize=9, weight='bold', color=color)
    ax.plot([0, 10], [2.40, 2.40], color=ink, lw=.8)
    for i, y in enumerate([2.12, 1.66, 1.20, .74]):
        formula = (rf'$h_{{t,{i}}}=f_{i}(h_{{t-1}},x_t)$' if i < 3
                   else r'$y_t=g(h_t)$')
        ax.text(.10, y, formula, va='center', color=ink, fontsize=11)
        size = '32 state-input pairs' if i < 3 else '16 states'
        ax.text(4.25, y, size, va='center', ha='center', color=ink, fontsize=9)
        for x, count, color in zip(xs, counts[i], colors):
            ax.text(x, y, f'{count}/10', va='center', ha='center',
                    color=color if count else muted, weight='bold' if count else 'normal')
        if i < 3:
            ax.plot([0, 10], [y-.24, y-.24], color='#E2E6E9', lw=.6)
    ax.plot([0, 10], [.46, .46], color=ink, lw=.8)
    ax.text(.10, .18, 'Subproblems recovered across seeds', va='center', weight='bold', color=ink)
    for x, count, color in zip(xs, totals, colors):
        ax.text(x, .18, f'{count}/4', ha='center', va='center', weight='bold', color=color)
    fig.savefig(OUT.with_suffix('.pdf'), metadata={
        'Title': 'Refined Parity-Last-4: task, state space, and scalar recovery',
        'Subject': 'Observed relation sizes and exact recovery over ten seeds; MIPS run 709714 evolved from scratch'})
    plt.close(fig)

    caption = r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/mips_parity_last4_example.pdf}
\caption{Parity-Last-4 as a symbolic-regression task. The algorithm outputs the parity of the most recent four input bits, treating missing preceding bits as zero. Refining the learned encoding from $\mathrm{round}(Z)$ to $\mathrm{round}(4Z)$ increases the observed hidden-state set from 12 to 16 states and removes transition conflicts. The three hidden coordinates produce three scalar transition problems, each with 32 observed state--input pairs ($16$ states $\times$ two input bits); the output function produces a fourth problem with 16 state inputs. These are observed relation sizes, not Cartesian products of coordinate ranges. Across ten search seeds, baseline PySR and SRBench-evolved PySR (709715) each recover three of the four subproblems, while MIPS-evolved PySR (709714) recovers all four. Entries give exact-recovery counts out of ten seeds. Task recovery in our comparison permits different components to be recovered in different seeds; for this example, the MIPS-evolved method also recovers all four components together in one of ten seeds. Each fit has a budget of $10^6$ evaluations and a 500-second timeout. Run 709714 is evolved directly on MIPS rather than fine-tuned from 709715. Recovery denotes exact agreement with the complete recorded relations; it is not a separate end-to-end validation of an assembled program.}
\label{fig:mips-parity-last4-example}
\end{figure*}
'''
    OUT.with_name(OUT.name + '_caption.txt').write_text(caption)
    records = []
    for label, method in zip(DEFAULTS, methods):
        relevant = [r for r in method['records'] if r['dataset_name'] in COMPONENTS]
        records.append({'method': label, 'evaluation_path': str(method['path'].relative_to(ROOT)),
                        'exact_run_indices': {c: sorted(method['exact'][c]) for c in COMPONENTS},
                        'joint_run_indices': sorted(method['joint'][TASK]),
                        'raw_results': [{'path': str(Path(r['source']).relative_to(ROOT)),
                                         'sha256': hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()}
                                        for r in relevant]})
    OUT.with_suffix('.json').write_text(json.dumps({
        'task': TASK, 'original_states': old_states, 'refined_states': rows[-1],
        'scale': 4, 'input_table_sizes': dict(zip(COMPONENTS, rows)),
        'example_input': bits, 'example_output': parity, 'methods': records,
    }, indent=2) + '\n')
    print(OUT.with_suffix('.pdf'))
    print(OUT.with_name(OUT.name + '_caption.txt'))


if __name__ == '__main__':
    main()
