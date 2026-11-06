#!/usr/bin/env python3
"""Export unsampled W&B ablation histories and plot them; --refresh refetches data."""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent / 'ablations_generations10'
METRICS = {
    'train': ('val_eval/train_avg_score', 'val_eval/train_reeval_gen_submitted'),
    'val': ('val_eval/avg_score', 'val_eval/gen_submitted'),
}
GROUPS = {
    'model_quality': ('Model quality', ['standard', 'llm-small2', 'llm-best2']),
    'operators': ('Operator ablations', ['standard', 'no-mut', 'no-loss', 'no-select', 'no-survive', 'no-data-mut']),
    'meta_mutations': ('Meta-mutation ablations', ['standard', 'no-explore', 'no-refine', 'no-cross', 'no-simplify']),
    'prompt_feedback': ('Prompt feedback ablations', ['standard', 'no-feedback', 'uninfo-no-fb']),
    'reevaluation': ('Reevaluation ablations', ['standard', 'nruns1', 'nruns3', 'nruns10', 'reeval1to3', 'reeval-dyn']),
    'task_vs_topk': ('Task vs. top-k population', ['standard', 'pop-topk']),
}
LABELS = {
    'standard': 'Standard', 'llm-small2': 'Small (cheap2)', 'llm-best2': 'Best (best2)',
    'no-mut': 'No mutation', 'no-loss': 'No loss', 'no-select': 'No selection',
    'no-survive': 'No survival', 'no-data-mut': 'No data-aware mutation',
    'no-explore': 'No explore', 'no-refine': 'No refine', 'no-cross': 'No crossover',
    'no-simplify': 'No simplify', 'no-feedback': 'No execution feedback',
    'uninfo-no-fb': 'Uninformative prompts + no feedback',
    'nruns1': '1 run, no reevaluation', 'nruns3': '3 runs, no reevaluation',
    'nruns10': '10 runs, no reevaluation', 'reeval1to3': 'Population reevaluation: 1 → 3',
    'reeval-dyn': 'Dynamic TTTS (top-k)', 'pop-topk': 'Top-k',
}


def fetch():
    import wandb
    names = set(n for _, group in GROUPS.values() for n in group)
    filters = {'config.generations': 10, 'createdAt': {'$gte': '2026-09-04T00:00:00Z'},
               'displayName': {'$in': sorted(names)}}
    runs = list(wandb.Api(timeout=90).runs('simon-alford/meta-sr', filters=filters))
    metadata, records = [], []
    for run in runs:
        print(f'Fetching {run.name}: {run.id} ({run.state})', flush=True)
        metadata.append({'id': run.id, 'name': run.name, 'state': run.state,
                         'url': run.url, 'created_at': run.created_at, 'config': run.config})
        # Scan each pair separately: train and validation complete at different steps.
        for split, (metric, gen_key) in METRICS.items():
            for row in run.scan_history(keys=[metric, gen_key], page_size=1000):
                score, gen = row.get(metric), row.get(gen_key)
                if score is None:
                    continue
                if gen is None or not math.isfinite(float(score)):
                    raise ValueError(f'Invalid metric row: {run.id}, {row}')
                records.append({'ablation': run.name, 'run_id': run.id, 'state': run.state,
                                'split': split, 'generation': int(gen), 'score': float(score),
                                'metric': metric, 'generation_key': gen_key, 'step': row.get('_step')})
    payload = {'project': 'simon-alford/meta-sr', 'fetched_at': datetime.now(timezone.utc).isoformat(),
               'filters': filters, 'runs': metadata, 'records': records}
    (OUT / 'wandb_export.json').write_text(json.dumps(payload, indent=2) + '\n')
    return payload


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / 'wandb_export.json'
    data = fetch() if args.refresh or not cache.exists() else json.loads(cache.read_text())
    records = data['records']
    write_csv(OUT / 'scores.csv', records,
              ['ablation', 'run_id', 'state', 'split', 'generation', 'score', 'metric', 'generation_key', 'step'])
    runs = {r['name']: r for r in data['runs']}
    if len(runs) != len(data['runs']):
        raise ValueError('Multiple runs per ablation: select explicitly before plotting.')
    if 'standard' not in runs:
        raise ValueError('Standard run is missing.')
    series, coverage, final = {}, [], []
    for name, run in runs.items():
        final_row = {'ablation': name, 'run_id': run['id'], 'url': run['url']}
        for split in METRICS:
            rows = sorted([r for r in records if r['run_id'] == run['id'] and r['split'] == split],
                          key=lambda r: r['generation'])
            gens = [r['generation'] for r in rows]
            if len(gens) != len(set(gens)) or any(g < 0 or g > 10 for g in gens):
                raise ValueError(f'Unexpected generation sequence: {name}/{split}: {gens}')
            if not rows:
                raise ValueError(f'No scores for {name}/{split}')
            series[name, split] = rows
            missing = sorted(set(range(11)) - set(gens))
            coverage.append(f'| {name} | {split} | {len(rows)} | {", ".join(map(str, missing)) or "none"} |')
            final_row[f'{split}_generation'] = rows[-1]['generation']
            final_row[f'{split}_score'] = rows[-1]['score']
        final.append(final_row)
    write_csv(OUT / 'final_scores.csv', sorted(final, key=lambda r: r['ablation']),
              ['ablation', 'run_id', 'url', 'train_generation', 'train_score', 'val_generation', 'val_score'])
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.dpi': 180})
    missing_names = sorted(set(n for _, group in GROUPS.values() for n in group) - runs.keys())
    with PdfPages(OUT / 'all_ablations.pdf') as pdf:
        for slug, (title, names) in GROUPS.items():
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
            for index, name in enumerate(names):
                if name not in runs:
                    continue
                label = LABELS[name]
                if name == 'standard':
                    label = {'model_quality': 'Standard (medium2)', 'reevaluation': 'Standard (population: 3 → 10)',
                             'task_vs_topk': 'Standard (task)'}.get(slug, label)
                for ax, split in zip(axes, METRICS):
                    rows = series[name, split]
                    ax.plot([r['generation'] for r in rows], [r['score'] for r in rows],
                            label=label, color='black' if index == 0 else f'C{index-1}',
                            linewidth=2.5 if index == 0 else 1.7, marker='o' if index == 0 else ['s','^','D','v','P'][index-1],
                            markersize=4, zorder=10 if index == 0 else 3)
            values = [r['score'] for name in names if name in runs for split in METRICS for r in series[name, split]]
            low, high = min(values), max(values)
            for ax, label in zip(axes, ['Train score', 'Validation score']):
                ax.set(title=label, xlabel='Generation', ylabel=label, xlim=(-0.2, 10.2),
                       ylim=(max(-0.025, low-0.04), min(1.025, high+0.04)))
                ax.set_xticks(range(11))
                ax.grid(alpha=0.2)
            fig.suptitle(title, fontsize=16)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=False, fontsize=10)
            note = 'Markers show logged evaluations; lines connect available generations.'
            absent = [n for n in names if n not in runs]
            if absent:
                note += ' Missing run: ' + ', '.join(absent) + '.'
            fig.text(0.5, 0.008, note, ha='center', fontsize=8, color='#555555')
            fig.tight_layout(rect=(0, 0.19, 1, 0.94))
            fig.savefig(OUT / f'{slug}.png')
            fig.savefig(OUT / f'{slug}.pdf')
            pdf.savefig(fig)
            plt.close(fig)
    readme = f'''# Generation-10 ablations

Fetched {data['fetched_at']} from `{data['project']}`. Includes {len(runs)} runs and {len(records)} metric observations.

Train: `val_eval/train_avg_score`, aligned to `val_eval/train_reeval_gen_submitted`.
Validation: `val_eval/avg_score`, aligned to `val_eval/gen_submitted`.
Generation 0 is the initial population. These are unsmoothed individual-run scores, without confidence intervals.
Missing generations are not imputed; markers identify actual evaluations and lines connect them.
Each pair shares y limits; limits differ between groups so all values, including zeros, remain visible.
Standard is medium2 / task selection / population reevaluation from 3 to 10 runs.
Small is the cheap2 preset. Dynamic TTTS also changes population selection to top-k.
The 25-generation cooldown ablation is outside the requested generation-10 groups.

Missing named runs: {', '.join(missing_names) or 'none'}.

Reproduce from cached data: `python figures/plot_ablations.py`.
Refresh W&B data: `python figures/plot_ablations.py --refresh`.
`wandb_export.json` includes run configs, provenance, and metric observations;
`scores.csv` contains all observations; `final_scores.csv` contains the last observed scores.
`all_ablations.pdf` contains all six figures, also available separately as PNG/PDF.

## Coverage

| Ablation | Split | Points | Missing generations (0–10) |
|---|---|---:|---|
''' + '\n'.join(sorted(coverage)) + '\n'
    (OUT / 'README.md').write_text(readme)
    print(f'Saved six plot pairs, combined PDF, data, and coverage to {OUT}')
    print(f'Missing runs: {missing_names}')


if __name__ == '__main__':
    main()
