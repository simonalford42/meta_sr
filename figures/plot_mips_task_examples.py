#!/usr/bin/env python3
"""Per-task MIPS example tables (same layout as the Parity-last-4 example).

Covers Alternating-last 4 and Base-4/5/7 addition, which use the original
(unrefined) integer encoding with a 2-D hidden state.

Run: python figures/plot_mips_task_examples.py
Outputs: figures/mips_<short>_example.{pdf,json} and _caption.txt per task.
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

ARTIFACTS = ROOT / 'outputs/mips_evolution_51_artifacts'
SUFFIXES = ('hidden:0', 'hidden:1', 'output:0')

# Each component: (MIPS-evolved run index of the displayed witness,
# display expression in h0/h1/x or h0/h1/a/b, LaTeX of that expression).
# Display forms are regrouped, constants rounded to six significant digits, and
# simplified using the observed value sets; each is verified on the full table.
TASKS = [
    dict(task='rnn_alternating_last4_numerical', short='alternating_last4',
         title='Alternating-last 4', inputs=['x'],
         input_desc=r'$x\in\{0,1\}$ (input bit)', output_desc='(output bit)',
         components=[
             (8, 'mips_floordiv(mips_eq(mips_not(h0),x)+30*mips_not(x)-0.203372*h0+7.82772,1)',
              r'$\lfloor\mathrm{eq}(\mathrm{not}(h_0),x)+30\,\mathrm{not}(x)-0.203372\,h_0+7.82772\rfloor$'),
             (1, 'mips_zero(mips_xor(mips_lt(x,h0),x))*mips_max(h0-37,mips_max(mips_zero(x),h1))',
              r'$\mathrm{zero}(\mathrm{xor}(\mathrm{lt}(x,h_0),x))\cdot\max(h_0-37,\,\max(\mathrm{zero}(x),h_1))$'),
             (0, 'h1', r'$h_1$'),
         ]),
    dict(task='rnn_base_4_addition', short='base4_addition',
         title='Base-4 addition', inputs=['a', 'b'],
         input_desc=r'$a,b\in\{0,\ldots,3\}$ (input digits)', output_desc='(sum digit)',
         components=[
             (4, '3.25534-mips_min(mips_mod(-2.74466,a+b+h1-1),a+b+h1-1.74466)',
              r'$3.25534-\min(\mathrm{mod}(-2.74466,\,a+b+h_1-1),\;a+b+h_1-1.74466)$'),
             (3, 'mips_eq(mips_min((a+b-1.91107)*(h1+0.432432),1.07176),1)',
              r'$\mathrm{eq}(\min((a+b-1.91107)(h_1+0.432432),\,1.07176),\,1)$'),
             (2, '3-h0+2*mips_zero(h1)', r'$3-h_0+2\,\mathrm{zero}(h_1)$'),
         ]),
    dict(task='rnn_base_5_addition', short='base5_addition',
         title='Base-5 addition', inputs=['a', 'b'],
         input_desc=r'$a,b\in\{0,\ldots,4\}$ (input digits)', output_desc='(sum digit)',
         components=[
             (5, 'mips_floordiv(mips_max(6.06243-(a+b+h1),mips_mod(0.243552-(a+b+h1),4.38026)),1)',
              r'$\lfloor\max(6.06243-(a+b+h_1),\;\mathrm{mod}(0.243552-(a+b+h_1),\,4.38026))\rfloor$'),
             (3, 'mips_eq(1,mips_max(0.119279*(a+b),mips_mod(-1.03325,0.25413*(a+b)*h1)))',
              r'$\mathrm{eq}(1,\,\max(0.119279(a+b),\;\mathrm{mod}(-1.03325,\,0.25413(a+b)h_1)))$'),
             (2, 'mips_abs(h0+h1-6)-h1', r'$|h_0+h_1-6|-h_1$'),
         ]),
    dict(task='rnn_base_7_addition', short='base7_addition',
         title='Base-7 addition', inputs=['a', 'b'],
         input_desc=r'$a,b\in\{0,\ldots,6\}$ (input digits)', output_desc='(sum digit)',
         components=[
             (5, 'mips_eq(0.0810486*(a+b+h0),1)', r'$\mathrm{eq}(0.0810486(a+b+h_0),\,1)$'),
             (2, 'mips_floordiv(mips_max(8.00009-(a+b+h0),mips_mod(13.0001,a+b+h0)),1)',
              r'$\lfloor\max(8.00009-(a+b+h_0),\;\mathrm{mod}(13.0001,\,a+b+h_0))\rfloor$'),
             (4, '8-h1-2*h0', r'$8-h_1-2h_0$'),
         ]),
]

INK, MUTED = '#24333D', '#5E6B73'
COLORS = ['#317DA5', '#BB7A20', '#C33D3D']


def fmt_set(vals):
    vals = sorted(vals)
    if len(vals) > 4 and vals == list(range(vals[0], vals[-1] + 1)):
        return rf'\{{{vals[0]},\ldots,{vals[-1]}\}}'
    return r'\{' + ','.join(map(str, vals)) + r'\}'


def render(spec, methods, namespace):
    task, inputs = spec['task'], spec['inputs']
    comps = [f'mips:{task}:{s}' for s in SUFFIXES]
    artifacts = [load_component_artifact(c, root=ARTIFACTS) for c in comps]
    names = ['h0', 'h1', *inputs]
    for a in artifacts:
        assert a['metadata']['deterministic'] and a['metadata']['hidden_dim'] == 2
        assert a['metadata']['input_dim'] == len(inputs)
    witnesses = []
    for c, a, (run, display, _) in zip(comps, artifacts, spec['components']):
        record = next(r for r in methods[-1]['records']
                      if r['dataset_name'] == c and r['run_index'] == run)
        assert record['gt_match_score'] == 1, (c, run)
        X = a['X_full']
        env = dict(namespace, **{n: X[:, i] for i, n in enumerate(names[:X.shape[1]])})
        raw_env = dict(namespace, **{f'x{i}': X[:, i] for i in range(X.shape[1])})
        pred = eval(display, {'__builtins__': {}}, env)
        raw = eval(record['gt_matched_equation'], {'__builtins__': {}}, raw_env)
        err = float(np.max(np.abs(pred - a['y_full'])))
        assert err <= 1e-9 and np.max(np.abs(pred - raw)) <= 1e-9, (c, err)
        witnesses.append({'component': c, 'run_index': run,
                          'raw_equation': record['gt_matched_equation'],
                          'displayed_expression': display, 'display_max_error': err})

    X = artifacts[0]['X_full']
    hsets = [sorted({int(v) for v in X[:, i]}) for i in range(2)]
    for i in range(2):
        assert sorted({int(v) for v in artifacts[i]['y_full']}) == hsets[i]
    ysets = sorted({int(v) for v in artifacts[2]['y_full']})
    n_states = len(np.unique(artifacts[2]['X_full'], axis=0))
    rows = [len(a['X_full']) for a in artifacts]
    counts = [[len(m['exact'][c]) for m in methods] for c in comps]
    totals = [sum(bool(m['exact'][c]) for c in comps) for m in methods]
    joint = [len(m['joint'][task]) for m in methods]
    args = 'h_0,h_1,' + ','.join(inputs)

    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'pdf.fonttype': 42, 'mathtext.fontset': 'dejavuserif'})
    fig = plt.figure(figsize=(13.6, 3.5))
    ax = fig.add_axes([.02, .055, .96, .91])
    ax.set(xlim=(0, 16.5), ylim=(0.35, 4.8))
    ax.axis('off')
    ax.text(8.25, 4.62, spec['title'], ha='center', fontsize=15, weight='bold', color=INK)
    ax.text(8.25, 4.22,
            rf'$h_0\in{fmt_set(hsets[0])}$,   $h_1\in{fmt_set(hsets[1])}$'
            rf'   (integer hidden-state coordinates; ${n_states}$ joint states),   '
            + spec['input_desc'] + rf',   $y\in{fmt_set(ysets)}$ ' + spec['output_desc'],
            ha='center', fontsize=9.5, color=MUTED)
    xs = [4.80, 6.10, 7.55]
    ax.text(.10, 3.88, 'Scalar relation', weight='bold', color=INK, va='center')
    ax.text(3.20, 3.88, 'Input-table size', weight='bold', ha='center', color=INK, va='center')
    for x, label, color in zip(xs, ['Base', 'Evolved', 'MIPS-evolved'], COLORS):
        ax.text(x, 3.88, label, ha='center', va='center', fontsize=9, weight='bold', color=color)
    ax.text(9.00, 3.88, 'Discovered equation (MIPS-evolved)', va='center', weight='bold', color=INK)
    ax.plot([0, 16.5], [3.52, 3.52], color=INK, lw=.8)
    ys = [3.12, 2.42, 1.72]
    for i, y in enumerate(ys):
        formula = rf"$h_{i}'=f_{i}({args})$" if i < 2 else r'$y=g(h_0,h_1)$'
        ax.text(.10, y, formula, va='center', color=INK, fontsize=10.5)
        size = f'{rows[i]} state-input pairs' if i < 2 else f'{rows[i]} states'
        ax.text(3.20, y, size, va='center', ha='center', color=INK, fontsize=9)
        for x, count, color in zip(xs, counts[i], COLORS):
            ax.text(x, y, f'{count}/10', va='center', ha='center',
                    color=color if count else MUTED, weight='bold' if count else 'normal')
        ax.text(9.00, y, spec['components'][i][2], va='center', color=INK, fontsize=10)
        if i < 2:
            ax.plot([0, 16.5], [y - .35, y - .35], color='#E2E6E9', lw=.6)
    ax.plot([0, 16.5], [1.30, 1.30], color=INK, lw=.8)
    ax.text(.10, 1.00, 'Subproblems recovered across seeds', va='center', weight='bold', color=INK)
    ax.text(.10, .66, 'All subproblems in the same seed', va='center', weight='bold', color=INK)
    for x, total, j, color in zip(xs, totals, joint, COLORS):
        ax.text(x, 1.00, f'{total}/3', ha='center', va='center', weight='bold', color=color)
        ax.text(x, .66, f'{j}/10', ha='center', va='center',
                color=color if j else MUTED, weight='bold' if j else 'normal')
    out = ROOT / f"figures/mips_{spec['short']}_example"
    fig.savefig(out.with_suffix('.pdf'), metadata={
        'Title': spec['title'],
        'Subject': 'Observed relation sizes, recovered equations, and exact recovery over ten seeds'})
    plt.close(fig)

    input_text = (r'$x\in\{0,1\}$ is the scalar input bit' if inputs == ['x']
                  else r'$a,b$ are the two input digits')
    runs = ', '.join(str(c[0]) for c in spec['components'])
    caption = (
        rf"""\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/mips_{spec['short']}_example.pdf}}
\caption{{Three scalar regression problems for {spec['title']} (original integer encoding). The observed state set has {n_states} states, yielding {rows[0]} state--input pairs for each transition relation and {rows[2]} inputs for the output relation. Entries give exact-recovery counts out of ten search seeds; the bottom row counts seeds in which all three subproblems are recovered together. The last column shows recovered MIPS-evolved expressions from run indices {runs}, respectively. $h=(h_0,h_1)$ is the integer-encoded hidden state, {input_text}, primes denote the next state, and $y$ is read from the (new) hidden state. Here $\mathrm{{lt}}(a,b)=\mathbf{{1}}[a<b]$, $\mathrm{{eq}}(a,b)=\mathbf{{1}}[|a-b|<0.5]$, $\mathrm{{zero}}(a)=\mathrm{{not}}(a)=\mathbf{{1}}[|a|<0.5]$, $\mathrm{{mod}}(a,b)=a\bmod|b|$, and $\mathrm{{xor}}(a,b)=(\mathrm{{round}}(a)+\mathrm{{round}}(b))\bmod 2$. Expressions are algebraically regrouped, simplified using the observed value sets, and constants rounded to six significant digits; the displayed forms are verified against the complete recorded relations within absolute tolerance $10^{{-9}}$. Each fit uses at most $10^6$ evaluations and a 500-second timeout. Run 709714 was evolved directly on MIPS.}}
\label{{fig:mips-{spec['short'].replace('_', '-')}-example}}
\end{{figure*}}
""")
    out.with_name(out.name + '_caption.txt').write_text(caption)
    records = []
    for label, method in zip(DEFAULTS, methods):
        relevant = [r for r in method['records'] if r['dataset_name'] in comps]
        records.append({'method': label, 'evaluation_path': str(method['path'].relative_to(ROOT)),
                        'exact_run_indices': {c: sorted(method['exact'][c]) for c in comps},
                        'joint_run_indices': sorted(method['joint'][task]),
                        'raw_results': [{'path': str(Path(r['source']).relative_to(ROOT)),
                                         'sha256': hashlib.sha256(Path(r['source']).read_bytes()).hexdigest()}
                                        for r in relevant]})
    out.with_suffix('.json').write_text(json.dumps({
        'task': task, 'states': n_states, 'input_table_sizes': dict(zip(comps, rows)),
        'displayed_witnesses': witnesses, 'methods': records}, indent=2) + '\n')
    print(out.with_suffix('.pdf'))


def main():
    methods = [inspect(ROOT / path) for path in DEFAULTS.values()]
    assert all(not m['missing'] and len(m['runs']) == 10 for m in methods)
    namespace = MIPSTransitionDomain().predict_namespace()
    for spec in TASKS:
        render(spec, methods, namespace)


if __name__ == '__main__':
    main()
