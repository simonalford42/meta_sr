#!/usr/bin/env python3
"""Snapshot --official data and render Table 1 and SRBench-style Figures 2/3.

Run from any directory. Uses existing local artifacts only; never submits jobs.
The snapshot and CSVs permit rendering without rereading mutable run directories.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import srbench_official_results as official
import srbench_results_io as srio

OUT = ROOT / 'reports' / 'srbench_table1'
METHODS = ['PySR++', 'BasicSR++', 'HPO', 'PySR', 'BasicSR', 'MDLformer']
FAMILIES = ['pysrpp', 'basicsrpp', 'hpo', 'pysr_baseline', 'basicsr_baseline']


def snapshot(args):
    # Reuse the exact discovery/selection logic; capture its selected paths too.
    original = official._column_from_records
    def capture(records, merged_records, training, project_root):
        result = original(records, merged_records, training, project_root)
        for kind in ['gt', 'black_box']:
            record = official._pick_evaluation(records, official.ONE_MILLION, kind)
            result[kind + '_path'] = str(record['run_dir'].relative_to(ROOT)) if record else None
        return result
    official._column_from_records = capture
    try:
        columns = official.build_official_columns(ROOT / 'runs', ROOT)
    finally:
        official._column_from_records = original
    (OUT / 'official.txt').write_text(official.format_official_table(columns) + '\n')
    by_key = {c['key']: c for c in columns}
    bb, gt, selected = [], [], []
    for method, family in zip(METHODS, FAMILIES):
        keys = {kind: family if family.endswith('baseline') else family + '_' + (
            ('r2' if kind == 'black_box' else 'gt') if args.objective == 'specific' else args.objective)
            for kind in ['black_box', 'gt']}
        item = {'method': method, 'keys': keys}
        for kind in keys:
            col = by_key[keys[kind]]
            path = ROOT / col[kind + '_path']
            if method == 'PySR' and kind == 'gt' and args.pysr_gt == 'original':
                path = ROOT / 'runs/290227'
            manifest = json.loads((path / 'manifest.json').read_text())
            item[kind] = {'path': str(path.relative_to(ROOT)), 'manifest': manifest,
                          'official_path': col[kind + '_path']}
            if kind == 'black_box':
                result_path = path / 'srbench_black_box_results.json'
                data = json.loads(result_path.read_text())['datasets']
                for dataset, frontiers in data.items():
                    if frontiers and isinstance(frontiers[0], dict):
                        frontiers = [frontiers]
                    for trial, frontier in enumerate(frontiers):
                        valid = [p for p in frontier if p.get('test_r2') is not None]
                        if not valid:
                            raise ValueError(f'Missing R2: {path}/{dataset}/{trial}')
                        # Same oracle maximum as --official; smallest size breaks exact ties.
                        best = max(valid, key=lambda p: (p['test_r2'], -p['complexity']))
                        bb.append(dict(method=method, dataset=dataset, trial=trial,
                                       r2=best['test_r2'], size=best['complexity']))
            else:
                result_path = path / 'srbench_full_results.json'
                keyed = srio.load_keyed_results(path)
                if keyed is None:
                    keyed = srio.build_keyed_results(path, manifest)
                for row in keyed.values():
                    if row.get('present') and row.get('error') is None:
                        gt.append(dict(method=method, dataset=row['dataset'], family=row['family'],
                                       seed=row['seed'], noise=float(row['noise']), solved=int(bool(row['solved']))))
            if result_path.exists():
                item[kind]['sha256'] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        selected.append(item)
    pd.DataFrame(bb).to_csv(OUT / 'black_box_trials.csv', index=False)
    pd.DataFrame(gt).to_csv(OUT / 'ground_truth_trials.csv', index=False)
    data = dict(created_utc=datetime.now(timezone.utc).isoformat(), official_columns=columns,
                selected=selected, options=vars(args))
    (OUT / 'provenance.json').write_text(json.dumps(data, indent=2) + '\n')


def ci(values, statistic, rng):
    a = np.asarray(values)
    samples = a[rng.integers(len(a), size=(10000, len(a)))]
    return np.quantile(statistic(samples, axis=1), [.025, .975])


def figures(bb, gt):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    rng = np.random.default_rng(20260908)
    data = bb.groupby(['dataset', 'method'])[['r2', 'size']].median().reset_index()
    if not (data.groupby('dataset').method.nunique() == 5).all():
        raise ValueError('Black-box ranks require all five methods on every dataset')
    for metric in ['r2', 'size']:
        data[metric + '_rank'] = data.groupby('dataset')[metric].transform(
            lambda s: s.round(3).rank(ascending=metric != 'r2', method='average'))
    data.to_csv(OUT / 'black_box_dataset_ranks.csv', index=False)
    points = data.groupby('method')[['r2_rank', 'size_rank']].median().reindex(METHODS[:-1])
    coords = points.to_numpy()
    ranks = np.zeros(5, dtype=int)
    left = set(range(5)); level = 1
    while left:
        front = [i for i in left if not any(np.all(coords[j] <= coords[i]) and
                 np.any(coords[j] < coords[i]) for j in left)]
        for i in front: ranks[i] = level
        left.difference_update(front); level += 1
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    colors = ['#2166ac', '#d6604d', '#7b3294', '#4d9221', '#666666']
    offsets = {'PySR++': (15, -12), 'BasicSR++': (10, -18), 'HPO': (15, 16),
               'PySR': (-12, -19), 'BasicSR': (-12, 10)}
    summaries = []
    for i, method in enumerate(METHODS[:-1]):
        x, y = coords[i]; sub = data[data.method == method]
        cx = ci(sub.r2_rank, np.median, rng); cy = ci(sub.size_rank, np.median, rng)
        color = colors[ranks[i]-1]
        ax.errorbar(x, y, xerr=[[x-cx[0]], [cx[1]-x]], yerr=[[y-cy[0]], [cy[1]-y]],
                    fmt='o', color=color, capsize=3, ms=7, zorder=3)
        dx, dy = offsets[method]
        ax.annotate(method, (x,y), xytext=(dx,dy), textcoords='offset points',
                    ha='left' if dx > 0 else 'right', bbox=dict(fc='white', ec='none', alpha=.85, pad=1))
        summaries.append(dict(method=method, r2_rank=x, size_rank=y,
                              r2_low=cx[0], r2_high=cx[1], size_low=cy[0], size_high=cy[1], pareto_rank=int(ranks[i])))
    for r in sorted(set(ranks)):
        pts = coords[ranks == r]; pts = pts[np.argsort(pts[:,0])]
        ax.plot(pts[:,0], pts[:,1], marker='o', color=colors[r-1], alpha=.6, label=f'Pareto rank {r}')
    ax.set(xlim=(.5,5.7), ylim=(.5,5.7), xticks=range(1,6), yticks=range(1,6),
           xlabel=r'Test $R^2$ rank (lower is better)', ylabel='Model complexity rank (lower is better)')
    ax.grid(alpha=.18); ax.legend(loc='upper left', frameon=False, fontsize=9)
    fig.text(.5,.02,'122 black-box datasets · median ranks with 95% bootstrap CIs\nMDLformer: no official results', ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.065,1,1)); save(fig, 'srbench_figure2_black_box')
    pd.DataFrame(summaries).to_csv(OUT / 'figure2_summary.csv', index=False)

    rates = gt.groupby(['method','family','dataset','noise']).solved.mean().reset_index()
    order = rates.groupby('method').solved.mean().sort_values(ascending=False).index.tolist() + ['MDLformer']
    fig, axes = plt.subplots(1,2,figsize=(7.4,4.6), sharey=True)
    noise_levels = [.0,.001,.01,.1]; palette = ['#2166ac','#4393c3','#e08214','#b2182b']
    summaries = []
    for ax, family in zip(axes, ['Feynman','Strogatz']):
        for n, (noise,color,marker) in enumerate(zip(noise_levels,palette,['o','s','^','D'])):
            for i, method in enumerate(order[:-1]):
                values = rates[(rates.method == method) & (rates.family == family) & (rates.noise == noise)].solved.to_numpy()*100
                if not len(values): continue
                mean = values.mean(); low,high = ci(values,np.mean,rng)
                ax.errorbar(mean,i+(n-1.5)*.16,xerr=[[mean-low],[high-mean]],fmt=marker,
                            color=color,ms=4,capsize=2,label=f'{noise:g}' if i == 0 else None)
                summaries.append(dict(method=method,family=family,noise=noise,rate=mean,low=low,high=high,n_datasets=len(values)))
        ax.text(50,len(order)-1,'—',ha='center',va='center',color='.5')
        ax.set(title=family,xlabel='Solution rate (%)',xlim=(-3,103),xticks=[0,25,50,75,100])
        ax.set_yticks(range(len(order)),order); ax.grid(axis='x',alpha=.2)
    axes[0].set_ylim(len(order)-.5, -.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,title='Target noise',loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.57,.045))
    fig.text(.5,.012,'Means over datasets; 95% dataset-bootstrap CIs · MDLformer: no official results',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.19,1,1)); save(fig,'srbench_figure3_ground_truth')
    pd.DataFrame(summaries).to_csv(OUT / 'figure3_summary.csv', index=False)


def save(fig, stem):
    fig.savefig(OUT / (stem+'.pdf'))
    fig.savefig(OUT / (stem+'.png'),dpi=180)
    plt.close(fig)


def table(bb, gt, provenance):
    r2 = bb.groupby('method').r2.mean(); solves = gt.groupby('method').solved.mean()*100
    options = provenance['options']
    def cell(v, percent=False):
        if abs(v)>1000:
            mantissa, exponent = f'{v:.2e}'.split('e')
            return rf'${mantissa}\!\times\!10^{{{int(exponent)}}}$'
        return f'{v:.1f}' if percent else f'{v:.3f}'
    rows = []
    for title, vals, percent in [(r'Black box $R^2$',r2,False),(r'GT solve (\%), 1M evals.',solves,True)]:
        entries = []
        for m in METHODS:
            if m not in vals or (percent and options['budget_label']=='empty-1.5min'):
                entries.append('---'); continue
            s=cell(vals[m],percent)
            if vals[m] == vals.max(): s=r'\textbf{'+s+'}'
            entries.append(s)
        if percent and options['budget_label']=='empty-1.5min': title=r'GT solve (\%), 1.5 min'
        rows.append(title+' & '+' & '.join(entries)+r' \\')
    for label in [r'GT solve (\%), 15 min',r'GT solve (\%), 15 min + restarts']:
        rows.append(label+' & '+' & '.join(['---']*6)+r' \\')
    caption = (r'SRBench results. Black-box $R^2$ is the mean over trials of the maximum test $R^2$ '
               r'on each stored frontier (the \texttt{--official} convention). GT is the per-run solve '
               r'percentage pooled over all datasets, seeds, and four noise levels. ')
    caption += {'specific':r'For the first three methods, black box uses $R^2$-trained variants and GT uses GT-trained variants. ',
                'gt':'The first three methods use GT-trained variants. ',
                'r2':r'The first three methods use $R^2$-trained variants. '}[options['objective']]
    caption += r'--- denotes unavailable or unreported results; MDLformer has no official entry. '
    if options['budget_label']!='empty-1.5min':
        caption += r'1M denotes an evaluation cap, not a 1.5-minute time budget; wall-clock limits differ. '
    if options['pysr_gt']=='original':
        caption += r'PySR GT uses the complete original 1M run (290227); the current official selector instead picks an incomplete 15-minute restart portfolio. '
    else:
        caption += r'PySR GT follows the official selector but is an incomplete 15-minute restart portfolio, not a comparable single 1M run. '
    caption += r'BasicSR has catastrophic negative-$R^2$ outliers; its mean is reported without clipping.'
    tex = ('% Paste into ICLR Overleaf; requires \\usepackage{booktabs,graphicx}.\n'
           '\\begin{table}[t]\n\\centering\n\\caption{'+caption+'}\n\\label{tab:srbench-table1}\n'
           '\\small\n\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\linewidth}{!}{%\n'
           '\\begin{tabular}{lrrrrrr}\n\\toprule\nMetric & '+' & '.join(METHODS)+r' \\'+'\n\\midrule\n'+
           '\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n')
    (OUT/'srbench_table1.tex').write_text(tex)
    wrapper = (r'\documentclass{article}'+'\n'+r'\usepackage{iclr2027_conference,times,booktabs,graphicx}'+'\n'+
               r'\iclrfinalcopy'+'\n'+r'\begin{document}\pagestyle{empty}\null'+'\n'+
               r'\input{srbench_table1.tex}'+'\n'+r'\end{document}'+'\n')
    (OUT/'srbench_table1_document.tex').write_text(wrapper)
    # The official ICLR style is vendored alongside the report directory.
    subprocess.run(['tectonic','-Z', 'search-path='+str(ROOT/'reports/iclr2027'),
                    'srbench_table1_document.tex'],cwd=OUT,check=True)
    (OUT/'srbench_table1_document.pdf').replace(OUT/'srbench_table1.pdf')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--refresh',action='store_true',help='Replace snapshot from live official results')
    p.add_argument('--objective',choices=['specific','gt','r2'],default='specific')
    p.add_argument('--pysr-gt',choices=['original','official'],default='original')
    p.add_argument('--budget-label',choices=['1M','empty-1.5min'],default='1M')
    args=p.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.refresh or not (OUT/'provenance.json').exists(): snapshot(args)
    bb=pd.read_csv(OUT/'black_box_trials.csv'); gt=pd.read_csv(OUT/'ground_truth_trials.csv')
    if not np.isfinite(bb[['r2','size']].to_numpy()).all(): raise ValueError('Nonfinite BB results')
    provenance=json.loads((OUT/'provenance.json').read_text())
    figures(bb,gt); table(bb,gt,provenance)
    print('Artifacts:',OUT)

if __name__=='__main__': main()
