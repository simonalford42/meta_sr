#!/usr/bin/env python3
"""Build MIPS figure candidates from complete archived evaluations; no jobs run."""
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/mips_figure_mockups'
SOURCES = {
    'PySR': 'runs/709714/final_eval_baseline_3seed_1h',
    'PySR++': 'runs/709715/final_eval_mips_native_3seed_1h',
    'MIPS evolved from scratch': 'runs/709714/final_eval_mips_3seed_1h',
}
COLORS = ['#687887', '#2586ad', '#d88a25', '#8d62a8']
LABELS = ['Original\nmethod', 'PySR', 'PySR++', 'PySR++\nfine-tuned']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = []
    def read(rel):
        p = ROOT / rel
        b = p.read_bytes()
        sources.append({'path': str(rel), 'sha256': hashlib.sha256(b).hexdigest()})
        return json.loads(b)
    reproduction = read('outputs/mips_reproduction_all/summary.json')
    components = (ROOT / 'splits/mips_sr_targets_plus_refined.txt').read_text().splitlines()
    assert len(components) == len(set(components)) == 51
    groups = defaultdict(list)
    for c in components:
        groups[c.split(':')[1]].append(c)
    assert len(groups) == 17
    original = {t['task'] for t in reproduction['tasks'] if t['independent_success']}
    assert len(reproduction['tasks']) == 62 and len(original) == 30
    raw, aggregates, matrices, task_matrices = [], {}, {}, {}
    for method, folder in SOURCES.items():
        summary = read(folder + '/eval_summary.json')
        tasks = read(folder + '/slurm_pysr/eval_0000/tasks.json')
        assert len(tasks) == 153 and summary['n_runs'] == 3
        indexed = {}
        for i, task in enumerate(tasks):
            rel = folder + f'/slurm_pysr/eval_0000/results/task_{i:06d}.json'
            r = read(rel)
            assert (r['dataset_name'], r['run_index']) == (task['dataset_name'], task['run_index'])
            key = (r['dataset_name'], r['run_index'])
            assert key not in indexed
            indexed[key] = int(r.get('gt_match_score') == 1)
            raw.append(dict(method=method, dataset=r['dataset_name'], run_index=r['run_index'],
                            seed=task['seed'], exact=indexed[key], gt_match_score=r.get('gt_match_score'),
                            error=r.get('error'), timed_out=r.get('timed_out'),
                            equation=r.get('gt_matched_equation'), source=rel))
        assert set(indexed) == {(c, s) for c in components for s in range(3)}
        matrix = np.array([[indexed[c, s] for s in range(3)] for c in components])
        task_matrix = {t: np.array([all(indexed[c, s] for c in cs) for s in range(3)]) for t, cs in groups.items()}
        overall = [len(original | {t for t, v in task_matrix.items() if v[s]}) for s in range(3)]
        details = summary['mips_sr_targets_plus_refined']['result_details']
        assert sum(matrix.flat) == sum(sum(x == 1 for x in d['run_gt_scores']) for d in details)
        aggregates[method] = dict(scalar_exact_runs=int(matrix.sum()), scalar_total_runs=153,
            scalar_per_seed=matrix.sum(axis=0).tolist(), overall_coverage_per_seed=overall,
            evaluated_groups_per_seed=np.array(list(task_matrix.values())).sum(axis=0).tolist(),
            components_solved_any_seed=int(matrix.any(axis=1).sum()),
            groups_solved_same_seed_at_least_once=sum(bool(v.any()) for v in task_matrix.values()),
            command=summary['command'])
        matrices[method], task_matrices[method] = matrix, task_matrix
    (OUT / 'summary.json').write_text(json.dumps(aggregates, indent=2) + '\n')
    (OUT / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    (OUT / 'original_62_results.json').write_text(json.dumps(reproduction, indent=2) + '\n')
    with (OUT / 'all_scalar_results.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0])); w.writeheader(); w.writerows(raw)
    with (OUT / 'all_62_task_results.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['task', 'original_status', 'original_full_validation', *[f'{m}_seed{s}_all_components_exact' for m in SOURCES for s in range(3)]])
        for t in reproduction['tasks']:
            w.writerow([t['task'], t['status'], t['independent_success'], *[int(task_matrices[m][t['task']][s]) if t['task'] in groups else '' for m in SOURCES for s in range(3)]])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    footer = ('Overall: original successes retained + newly exact task groups; new groups lack full-program validation.\n'
              'SR: 51 distinct subtasks × 3 seeds = 153 fits per method; 1 h/fit. Fine-tuned full evaluation pending.')
    vals = [[30, np.mean(aggregates['PySR']['overall_coverage_per_seed']), np.mean(aggregates['PySR++']['overall_coverage_per_seed']), np.nan],
            [np.nan, aggregates['PySR']['scalar_exact_runs']/3, aggregates['PySR++']['scalar_exact_runs']/3, np.nan]]
    titles = ['Overall problem coverage / 62', 'Exact SR subtasks / 51 (mean over seeds)']
    def save(fig, name):
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(OUT / f'{name}.{ext}', dpi=190, facecolor='white')
        plt.close(fig)
    for style in ['bars', 'dots']:
        fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.7))
        for j, ax in enumerate(axs):
            ax.set_title(titles[j], loc='left', fontweight='bold', pad=18)
            ax.set_ylim(0, 62 if j == 0 else 51); ax.set_xlim(-.6, 3.6)
            ax.set_xticks(range(4), LABELS); ax.grid(axis='y', alpha=.16); ax.set_axisbelow(True)
            for i, y in enumerate(vals[j]):
                if np.isnan(y):
                    ax.text(i, 13 if j == 0 else 11, 'Pending' if i == 3 else 'Not measured', ha='center', color='#777777', fontsize=9)
                    continue
                if style == 'bars':
                    ax.bar(i, y, color=COLORS[i], width=.58)
                else:
                    ax.vlines(i, 0, y, color=COLORS[i], alpha=.22, lw=3)
                    ax.scatter(i, y, color=COLORS[i], s=110, zorder=3)
                label = f'{y:g}/62' if j == 0 else f'{y:.2f}/51\n({aggregates[LABELS[i]]["scalar_exact_runs"]}/153 fits)'
                ax.annotate(label, (i, y), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=10)
                if i in (1, 2):
                    key = 'overall_coverage_per_seed' if j == 0 else 'scalar_per_seed'
                    yy = aggregates[LABELS[i]][key]
                    ax.scatter(np.array([-.1, 0, .1])+i, yy, s=13, c='black', zorder=4)
        fig.suptitle('MIPS: original method → PySR → PySR++ → fine-tuned', x=.07, ha='left', fontweight='bold', fontsize=15)
        fig.text(.07, .025, footer + '\nBlack dots: individual seeds.', fontsize=8.5, color='#555555')
        fig.subplots_adjust(left=.07, right=.98, top=.8, bottom=.27, wspace=.27)
        save(fig, '01_bars' if style == 'bars' else '02_dots')
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 9), gridspec_kw={'width_ratios': [1, 1.25]})
    tasks_order = [t['task'] for t in reproduction['tasks']]
    a = np.full((62, 4), -1.0)
    a[:, 0] = [t in original for t in tasks_order]
    for j, m in enumerate(['PySR', 'PySR++'], 1):
        a[:, j] = [1 if t in original else np.mean(task_matrices[m][t]) if t in groups else 0 for t in tasks_order]
    b = np.full((51, 4), -1.0)
    for j, m in enumerate(['PySR', 'PySR++'], 1): b[:, j] = matrices[m].mean(axis=1)
    cmap = ListedColormap(['#e4e7eb', '#ffffff', '#b9d8cf', '#66ad99', '#16745b'])
    norm = BoundaryNorm([-1.5, -.5, .16, .5, .84, 1.1], 5)
    for ax, data, title in zip(axs, [a, b], ['All 62 overall problems', 'All 51 evaluated SR subtasks']):
        ax.imshow(data, aspect='auto', cmap=cmap, norm=norm)
        ax.set_xticks(range(4), LABELS, fontsize=9); ax.xaxis.tick_top()
        ax.set_title(title, loc='left', fontweight='bold', pad=44)
    axs[0].set_yticks(range(62), [str(i+1) for i in range(62)], fontsize=6)
    axs[0].set_ylabel('Benchmark task index (names in all_62_task_results.csv)')
    start = 0
    for t, cs in groups.items():
        axs[1].axhline(start-.5, color='#aaaaaa', lw=.5)
        start += len(cs)
    centers=[]; start=0
    for t, cs in groups.items(): centers.append(start+(len(cs)-1)/2); start+=len(cs)
    axs[1].set_yticks(centers, [t.removeprefix('rnn_').removesuffix('_numerical') for t in groups], fontsize=8)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, edgecolor='#cccccc', label=l) for c,l in zip(cmap.colors,['Unavailable / pending', '0/3 seeds (or original unsolved)', '1/3 seeds', '2/3 seeds', '3/3 seeds (or original success)'])], loc='lower center', bbox_to_anchor=(.5,.078), ncol=3, fontsize=8)
    fig.text(.06,.02,footer,fontsize=8.5,color='#555555')
    fig.subplots_adjust(left=.07,right=.97,top=.87,bottom=.17,wspace=.75)
    save(fig, '03_task_matrix')
    print(json.dumps(aggregates, indent=2))


if __name__ == '__main__':
    main()
