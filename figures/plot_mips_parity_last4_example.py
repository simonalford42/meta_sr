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
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from domains import MIPSTransitionDomain
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

    # Use one recovered witness per component, allowing different seeds as in
    # the pooled comparison. Display regrouping/rounding is checked below.
    witness_runs = [8, 2, 5, 8]
    displayed = [
        'mips_min(((mips_max(x1,x0)-x3)*x1-x3)*x1-x3+2, '
        '4*mips_abs(mips_xor(x1*(0.227249-x3),x1)+x3-mips_zero(x1)))',
        'mips_floordiv(mips_min(mips_min(2.98567*x3+0.660363*x1,2.32073)'
        '+x3*mips_max(3.00001,x1),9.32075),1)',
        '4*mips_eq(mips_lt(x1,2*x0),x3)',
        'mips_min(1,x2)',
    ]
    equation_text = [
        r'$\min(((\max(x_1,x_0)-x_3)x_1-x_3)x_1-x_3+2,$' + '\n' +
        r'$\quad 4\,|\mathrm{xor}(x_1(0.227249-x_3),x_1)+x_3-\mathrm{zero}(x_1)|)$',
        r'$\lfloor\min(\min(2.98567x_3+0.660363x_1,\,2.32073)$' + '\n' +
        r'$\quad +\,x_3\max(3.00001,x_1),\,9.32075)\rfloor$',
        r'$4\,\mathrm{eq}(\mathrm{lt}(x_1,2x_0),x_3)$',
        r'$\min(1,x_2)$',
    ]
    witnesses = []
    namespace = MIPSTransitionDomain().predict_namespace()
    for c, run, display in zip(COMPONENTS, witness_runs, displayed):
        record = next(r for r in methods[-1]['records']
                      if r['dataset_name'] == c and r['run_index'] == run)
        assert record['gt_match_score'] == 1
        errors = {}
        for split, root in [('train', ARTIFACTS),
                            ('heldout', ROOT / 'outputs/mips_refined_six_artifacts/heldout')]:
            artifact = load_component_artifact(c, root=root)
            env = dict(namespace, **{f'x{i}': artifact['X_full'][:, i]
                                     for i in range(artifact['X_full'].shape[1])})
            prediction = eval(display, {'__builtins__': {}}, env)
            raw_prediction = eval(record['gt_matched_equation'], {'__builtins__': {}}, env)
            error = float(np.max(np.abs(prediction-artifact['y_full'])))
            assert error <= 1e-9, (c, split, error)
            assert np.max(np.abs(prediction-raw_prediction)) <= 1e-9
            errors[split] = error
        witnesses.append({'component': c, 'run_index': run,
                          'raw_equation': record['gt_matched_equation'],
                          'displayed_expression': display, 'display_max_error': errors})

    ink, muted = '#24333D', '#5E6B73'
    colors = ['#317DA5', '#BB7A20', '#C33D3D']
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'pdf.fonttype': 42, 'mathtext.fontset': 'dejavuserif'})
    fig = plt.figure(figsize=(13.6, 4.0))
    ax = fig.add_axes([.02, .055, .96, .91])
    ax.set(xlim=(0, 16.5), ylim=(0, 4.8))
    ax.axis('off')
    ax.text(8.25, 4.50, 'Parity-last 4', ha='center', fontsize=15, weight='bold', color=ink)
    xs = [4.80, 6.10, 7.55]
    ax.text(.10, 3.88, 'Scalar relation', weight='bold', color=ink, va='center')
    ax.text(3.20, 3.88, 'Input-table size', weight='bold', ha='center', color=ink, va='center')
    for x, label, color in zip(xs, ['Base', 'Evolved', 'MIPS-evolved'], colors):
        ax.text(x, 3.88, label, ha='center', va='center', fontsize=9, weight='bold', color=color)
    ax.text(9.00, 3.88, 'Discovered equation (MIPS-evolved)', va='center', weight='bold', color=ink)
    ax.plot([0, 16.5], [3.52, 3.52], color=ink, lw=.8)
    for i, y in enumerate([3.00, 2.08, 1.28, .70]):
        formula = (rf'$h_{{t,{i}}}=f_{i}(h_{{t-1}},x_t)$' if i < 3
                   else r'$y_t=g(h_t)$')
        ax.text(.10, y, formula, va='center', color=ink, fontsize=10.5)
        size = '32 state-input pairs' if i < 3 else '16 states'
        ax.text(3.20, y, size, va='center', ha='center', color=ink, fontsize=9)
        for x, count, color in zip(xs, counts[i], colors):
            ax.text(x, y, f'{count}/10', va='center', ha='center',
                    color=color if count else muted, weight='bold' if count else 'normal')
        ax.text(9.00, y, equation_text[i], va='center', color=ink, fontsize=10, linespacing=1.8)
        if i < 3:
            boundary = [2.53, 1.60, .98][i]
            ax.plot([0, 16.5], [boundary, boundary], color='#E2E6E9', lw=.6)
    ax.plot([0, 16.5], [.39, .39], color=ink, lw=.8)
    ax.text(.10, .12, 'Subproblems recovered across seeds', va='center', weight='bold', color=ink)
    for x, count, color in zip(xs, totals, colors):
        ax.text(x, .12, f'{count}/4', ha='center', va='center', weight='bold', color=color)
    fig.savefig(OUT.with_suffix('.pdf'), metadata={
        'Title': 'Parity-last 4',
        'Subject': 'Observed relation sizes, recovered equations, and exact recovery over ten seeds'})
    plt.close(fig)

    caption = r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/mips_parity_last4_example.pdf}
\caption{Four scalar regression problems for Parity-last 4 after refining the integer encoding by a factor of four. The observed state set has 16 states, yielding 32 state--input pairs for each transition relation and 16 inputs for the output relation. Entries give exact-recovery counts out of ten search seeds. Baseline PySR and SRBench-evolved PySR (709715) each recover three of four subproblems across seeds; MIPS-evolved PySR (709714) recovers all four. The last column shows recovered MIPS-evolved expressions from run indices 8, 2, 5, and 8, respectively. In transition expressions, $(x_0,x_1,x_2)=h_{t-1}$ and $x_3$ is the current input bit; in the output expression, $(x_0,x_1,x_2)=h_t$. Here $\mathrm{lt}(a,b)=\mathbf{1}[a<b]$, $\mathrm{eq}(a,b)=\mathbf{1}[|a-b|<0.5]$, $\mathrm{zero}(a)=\mathbf{1}[|a|<0.5]$, and $\mathrm{xor}(a,b)=(\mathrm{round}(a)+\mathrm{round}(b))\bmod 2$. Expressions are algebraically regrouped and constants rounded to six significant digits for display; the displayed forms are verified against the complete recorded training and held-out relations within absolute tolerance $10^{-9}$. Task recovery pools component successes across seeds; MIPS-evolved PySR also recovers all four components in the same run index (8). Each fit uses at most $10^6$ evaluations and a 500-second timeout. Run 709714 was evolved directly on MIPS.}
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
        'displayed_witnesses': witnesses, 'methods': records,
    }, indent=2) + '\n')
    print(OUT.with_suffix('.pdf'))
    print(OUT.with_name(OUT.name + '_caption.txt'))


if __name__ == '__main__':
    main()
