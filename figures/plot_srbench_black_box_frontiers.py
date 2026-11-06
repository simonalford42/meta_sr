#!/usr/bin/env python3
"""Plot existing local test-R2 envelopes and SRBench 2021 reference models.

Run: python figures/plot_srbench_black_box_frontiers.py
No searches, model evaluations, or SLURM submissions are performed.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/srbench_black_box_frontiers'
LOCAL = {'Base PySR': '290227', 'Evolved BasicSR (GT-R2)': '271625'}
COLORS = ['#2475B0', '#D35432']
GRID = np.arange(1, 41)
PAPER = 'https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/c0c7c76d30bd3dcaefc96f40275bdc0a-Paper-round1.pdf'


def envelope(frontier):
    values = np.array([max((p['test_r2'] for p in frontier
                           if p['complexity'] <= c and np.isfinite(p['test_r2'])),
                          default=np.nan) for c in GRID])
    assert np.isfinite(values).all(), 'Missing trial at a complexity budget'
    assert (np.diff(values) >= -1e-12).all()
    return values


def summarize(values, statistic, bootstrap):
    fn = np.median if statistic == 'median' else np.mean
    center = fn(values, axis=0)
    low, high = np.quantile(fn(values[bootstrap], axis=1), [.025, .975], axis=0)
    return center, low, high


def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / f'runs/{run}/srbench_black_box_results.json' for run in LOCAL.values()]
    original = ROOT / 'srbench/docs/csv/blackbox_results_datasets.csv'
    ref = pd.read_csv(original)
    ref['method'] = ref.algorithm.str.lstrip('*')
    datasets = [json.loads(p.read_text())['datasets'] for p in paths]
    names = sorted(datasets[0])
    assert len(names) == 122 and set(names) == set(datasets[1]) == set(ref.dataset)
    assert not ref.duplicated(['method', 'dataset']).any()
    bootstrap = np.random.default_rng(20260911).integers(122, size=(2000, 122))
    curves, trial_rows, local_rows = {}, [], []
    for (label, run), data in zip(LOCAL.items(), datasets):
        task_curves = []
        for name in names:
            assert len(data[name]) == 10
            trial_curves = np.array([envelope(f) for f in data[name]])
            task_curves.append(np.median(trial_curves, axis=0))
            for trial, (frontier, curve) in enumerate(zip(data[name], trial_curves)):
                for c, r2 in zip(GRID, curve):
                    trial_rows.append(dict(method=label, dataset=name, trial=trial,
                                           complexity=c, best_test_r2=r2))
            best = [max(f, key=lambda p: (p['test_r2'], -p['complexity'])) for f in data[name]]
            local_rows.append(dict(method=label, dataset=name,
                                   r2_test=np.median([p['test_r2'] for p in best]),
                                   model_size=np.median([p['complexity'] for p in best])))
        curves[label] = np.array(task_curves)
    pd.DataFrame(trial_rows).to_csv(OUT / 'local_trial_envelopes.csv.gz', index=False)
    pd.DataFrame(local_rows).to_csv(OUT / 'local_selected_models.csv', index=False)
    # Only symbolic methods with all 122 tasks; AIFeynman has only 107.
    symbolic = ref[ref.algorithm.str.startswith('*')].copy()
    counts = symbolic.groupby('method').dataset.nunique()
    included = counts[counts == 122].index
    symbolic = symbolic[symbolic.method.isin(included)]
    symbolic[['method', 'dataset', 'r2_test', 'model_size']].to_csv(
        OUT / 'reference_dataset_medians.csv', index=False)
    summary_rows, reference_rows = [], []
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    for statistic in ['median', 'mean']:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={'width_ratios': [1, 1.35]})
        for (label, values), color in zip(curves.items(), COLORS):
            center, low, high = summarize(values, statistic, bootstrap)
            for ax in axes:
                ax.step(GRID, center, where='post', color=color, lw=2.3, label=label, zorder=4)
                ax.fill_between(GRID, low, high, step='post', color=color, alpha=.13)
            for c, y, lo, hi in zip(GRID, center, low, high):
                summary_rows.append(dict(statistic=statistic, method=label, complexity=c,
                                         r2=y, ci_low=lo, ci_high=hi, n_datasets=122, n_trials=1220))
        references = []
        for method in included:
            data = symbolic[symbolic.method == method].set_index('dataset').loc[names]
            center, low, high = summarize(data[['model_size', 'r2_test']].to_numpy(), statistic, bootstrap)
            x, y = center
            references.append((method, x, y))
            reference_rows.append(dict(statistic=statistic, method=method, size=x, r2=y,
                                       size_low=low[0], size_high=high[0], r2_low=low[1],
                                       r2_high=high[1], n_datasets=122))
        # Label references in a separate key to avoid colliding text on dense points.
        references.sort(key=lambda row: row[2], reverse=True)
        palette = plt.get_cmap('tab20')
        for i, (method, x, y) in enumerate(references):
            axes[1].scatter(x, y, s=65, marker='D', color=palette(i), edgecolor='white',
                            linewidth=.6, zorder=5, label=method)
        axes[0].set(xlim=(1, 40), ylim=(0, 1), title='Best equation with complexity ≤ C',
                    xlabel='Maximum equation complexity C')
        axes[0].legend(loc='lower right', frameon=False)
        axes[1].set_xscale('log')
        axes[1].set(xlim=(1, 22000), title='Published SR methods as reference points',
                    xlabel='C for curves; aggregate model size for points')
        axes[1].set_ylim((-.05, 1) if statistic == 'median' else (-.85, 1))
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles[2:], labels[2:], loc='upper left', bbox_to_anchor=(1.01, 1),
                       frameon=False, fontsize=9, title='SRBench 2021')
        for ax in axes:
            ax.set_ylabel(f'{statistic.capitalize()} test $R^2$ across datasets')
            ax.grid(alpha=.18)
        fig.suptitle('SRBench black-box accuracy vs. equation complexity', fontsize=15)
        fig.text(.06, .04, '122 tasks · median over trials within each task · local bands: 95% dataset-bootstrap CI\n'
                 'Curves: test-selected envelopes. Diamonds: published selected models, not size-constrained frontiers.\n'
                 'Native complexity conventions and evaluation protocols differ; AIFeynman omitted (15 missing tasks).',
                 fontsize=9, va='bottom')
        fig.tight_layout(rect=(0, .16, 1, .94))
        save(fig, f'black_box_frontiers_{statistic}')
    pd.DataFrame(summary_rows).to_csv(OUT / 'frontier_summary.csv', index=False)
    pd.DataFrame(reference_rows).to_csv(OUT / 'reference_summary.csv', index=False)
    provenance = {'paper': PAPER, 'local_runs': {}, 'reference_source': str(original.relative_to(ROOT)),
                  'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in paths + [original]},
                  'aggregation': 'Median over 10 trials per task; then median or mean over 122 tasks.',
                  'bootstrap': '2000 paired dataset resamples, seed 20260911; pointwise 95% intervals.',
                  'excluded_reference_methods': {'AIFeynman': 'Only 107 of 122 tasks available.'}}
    for label, run in LOCAL.items():
        m = json.loads((ROOT / f'runs/{run}/manifest.json').read_text())
        provenance['local_runs'][label] = dict(run=run, source=m.get('method_meta', {}).get('source'),
                                             max_evals=m['max_evals'], black_box=m['black_box'])
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (OUT / 'README.md').write_text(f'''# SRBench black-box frontiers

Reproduce with `python figures/plot_srbench_black_box_frontiers.py`.

- Base PySR: runs/290227. Evolved BasicSR GT-R2: runs/271625, sourced from runs/150815.
- Both have 122 tasks × 10 trials, 1M evaluation budgets and 1500-second black-box timeouts.
- At each integer C from 1 to 40, take the maximum finite saved test R² among equations with native complexity ≤ C, independently per trial. Median over trials, then median (main figure) or mean (companion) over tasks. No clipping, missing-trial omission, or extrapolation beyond C=40. All trials have a complexity-1 candidate. The mean version is a mean of trial medians, not a mean of all trials.
- Bands are pointwise 95% intervals from 2000 task-bootstrap samples. They describe variation across tasks, not uncertainty of selecting an equation.
- Reference diamonds use the published per-task trial medians in srbench/docs/csv/blackbox_results_datasets.csv, then the same across-task statistic. All 13 complete symbolic methods are included. AIFeynman is excluded because it lacks 15 tasks; non-symbolic ML methods are outside this equation-complexity comparison.
- Reference x coordinates aggregate each method's chosen model sizes; they are NOT a bound applying to every task. Published data has selected models, not within-trial frontiers. Therefore these points cannot establish dominance against the local size-constrained curves.
- Local curves select using held-out test R² (an oracle envelope, as requested); published models use the original selection protocol. Search budgets, operators and split protocols also differ. The overlay is descriptive, not a controlled method ranking.
- The paper defines complexity as operators + features + constants. These are conceptually tree sizes, but native binary operators, powers, simplification and scaling can change counts. Local saved native complexity and released SRBench model_size are retained without claiming exact equivalence; recounting only the saved test-Pareto models could miss discarded candidates under a new measure.
- The main median aggregation follows the local SRBench Figure 1 plotting notebook (estimator=np.median); its paper caption instead says mean of medians, which the companion implements. This is a replacement for the requested rank plot, not a reproduction of Figures 1–2 or their training-time panel.

Source paper: {PAPER}

The CSVs retain exact values (including negatives), sample counts and reference confidence intervals. The left panels zoom to R² ∈ [0, 1]; the right panels show all reference point estimates. Source hashes and evaluation metadata are in provenance.json.
''')
    print(pd.DataFrame(summary_rows).query('complexity in [10, 20, 40]')[['statistic', 'method', 'complexity', 'r2']].to_string(index=False))
    print(f'Wrote figures and data to {OUT}')


if __name__ == '__main__':
    main()
