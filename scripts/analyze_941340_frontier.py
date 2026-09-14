#!/usr/bin/env python3
"""Export run 941340's final population Pareto frontier and audit noise scores.

Read-only analysis of existing evaluations; never submits jobs. Run from the repo
root with `python scripts/analyze_941340_frontier.py`. Includes the zero-score
endpoint in the mathematical frontier. No traces or full run data are exported.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evolution_helpers import code_loc

KINDS = ('mutation', 'survival', 'selection', 'loss')
RUN = ROOT / 'runs/941340'
OUT = RUN / 'best_bundles/final_pareto'


def name(bundle):
    return ' | '.join((bundle['operators'].get(k) or {}).get('name', 'default') for k in KINDS)


def loc(bundle):
    return sum(code_loc(o['code']) for o in bundle['operators'].values() if o)


def noise_stats(bundle):
    """Equal-weight task means, keeping infrastructure errors as recorded zeros."""
    records = []
    for detail in bundle.get('result_details') or []:
        for index, levels in enumerate(detail.get('run_noise_results') or []):
            for level in levels:
                records.append(dict(dataset=detail['dataset'], run_index=index,
                                    noise=float(level['target_noise']),
                                    score=level.get('gt_match_score'), error=level.get('error')))
    stats = {}
    for noise in sorted({r['noise'] for r in records}):
        subset = [r for r in records if r['noise'] == noise]
        task_means = {
            ds: mean(r['score'] for r in subset if r['dataset'] == ds and r['score'] is not None)
            for ds in sorted({r['dataset'] for r in subset})
        }
        stats[str(noise)] = dict(score=mean(task_means.values()), observations=len(subset),
                                 errors=sum(bool(r['error']) for r in subset),
                                 task_means=task_means)
    return stats, records


def main():
    data = json.loads((RUN / 'run_data.json').read_text())
    final = data['generations'][-1]
    population = final['population']
    front, best_score = [], float('-inf')
    for b in sorted(population, key=lambda b: (loc(b), -b['score'])):
        if b['score'] > best_score:
            front.append(b)
            best_score = b['score']
    front.reverse()
    OUT.mkdir(parents=True, exist_ok=True)
    old_eval = json.loads((ROOT / 'runs/709715/train_val_90s_10seed/eval_summary.json').read_text())
    old_train = {d['dataset']: d for d in old_eval['barely_unsolvable']['result_details']}
    old_names = ' | '.join(o['name'] for o in old_eval['operators'])
    latest = {}
    for generation in data['generations']:
        for key in ('offspring', 'population'):
            for b in generation[key]:
                latest[name(b)] = b
    rows, all_records = [], []
    for b in front:
        filename = f'frontier_loc{loc(b):03d}.jl'
        pieces = [f'# Run 941340 final generation {final["generation"]} Pareto frontier\n'
                  f'# Code LOC: {loc(b)}; all-noise training GT score: {b["score"]}\n'
                  f'# Seeds: {b["seeds_evaluated"]}; exact recorded function bodies.\n'
                  '# See comparison.md and metrics.json in this directory.\n']
        for kind in KINDS:
            op = b['operators'].get(kind)
            if op:
                pieces.append(f'# === {kind}: {op["name"]} ===\n' + op['code'].strip() + '\n')
        text = '\n'.join(pieces)
        assert code_loc(text) == loc(b)
        (OUT / filename).write_text(text)
        # JSON sidecar preserves operator weights and enables load_bundle().
        (OUT / filename.replace('.jl', '.json')).write_text(json.dumps(
            {'best_bundle': {k: b[k] for k in ('operators', 'score', 'seeds_evaluated', 'best_hparams')},
             'source': str(RUN.relative_to(ROOT)), 'generation': final['generation']}, indent=2) + '\n')
        stats, records = noise_stats(b)
        assert len(stats) == 4
        assert abs(mean(v['score'] for v in stats.values()) - b['score']) < 1e-10
        zero = stats['0.0']
        overlap = sorted(set(zero['task_means']) & set(old_train))
        assert len(overlap) == 20
        good = [ds for ds in overlap if not any(r['error'] for r in records if r['noise'] == 0 and r['dataset'] == ds)
                and not old_train[ds].get('errors')]
        row = dict(loc=loc(b), score=b['score'], seeds=b['seeds_evaluated'], filename=filename,
                   name=name(b), component_loc={k: code_loc(o['code']) for k,o in b['operators'].items() if o},
                   noise=stats, validation=data['val_results'].get(name(b)),
                   comparison_90s=dict(tasks=len(overlap),
                       old=mean(old_train[ds]['avg_gt'] for ds in overlap),
                       new=mean(zero['task_means'][ds] for ds in overlap),
                       error_free_tasks=len(good),
                       old_error_free=mean(old_train[ds]['avg_gt'] for ds in good) if good else None,
                       new_error_free=mean(zero['task_means'][ds] for ds in good) if good else None),
                   error_messages=dict(Counter(r['error'] for r in records if r['error'])))
        rows.append(row)
        all_records.extend(dict(loc=loc(b), **r) for r in records)
    summary = dict(source='runs/941340/run_data.json', generation=final['generation'],
                   definition='Nondominated final population: minimize code LOC, maximize stored score; includes zero-score endpoint.',
                   config=data['config'], frontier=rows,
                   dominated_final=[dict(loc=loc(b), score=b['score'], name=name(b)) for b in population if b not in front],
                   identification=data.get('identification'),
                   original_bundle_in_new_run=None)
    # Fresh identification outcomes are stored separately from run_data's bundles.
    # Join by explicit task-file index; combined.json does not retain noise labels.
    fresh_dir = RUN / 'slurm_pysr/eval_0484'
    tasks = json.loads((fresh_dir / 'tasks.json').read_text())
    first_tasks = {t['config_id']: t for t in tasks}
    for row, bundle in zip(rows, front):
        ids = []
        for config_id, task in first_tasks.items():
            if all(bundle['operators'][k]['name'] in str(task[f'custom_{k}_code']) for k in KINDS):
                ids.append(config_id)
        if not ids:
            row['fresh_identification'] = None
            continue
        assert len(ids) == 1
        details = {}
        for index, task in enumerate(tasks):
            if task['config_id'] != ids[0]:
                continue
            result = json.loads((fresh_dir / 'results' / f'task_{index:06d}.json').read_text())
            assert all(result[k] == task[k] for k in ('config_id', 'dataset_name', 'run_index'))
            details.setdefault(task['dataset_name'], []).append([dict(
                target_noise=task['target_noise'], gt_match_score=result['gt_match_score'], error=result['error'])])
        stats, fresh_records = noise_stats({'result_details':[dict(dataset=ds,run_noise_results=levels) for ds,levels in details.items()]})
        score = mean(v['score'] for v in stats.values())
        record = next(r for r in data['identification']['records'] if r['bundle_name'] == row['name'])
        assert abs(score-record['fresh_score']) < 1e-10
        row['fresh_identification'] = dict(score=score, seeds=data['identification']['n_fresh_runs'], noise=stats,
                                            source='runs/941340/slurm_pysr/eval_0484')
        bad = {r['dataset'] for r in fresh_records if r['noise'] == 0 and r['error']}
        good = sorted(set(stats['0.0']['task_means']) & set(old_train) - bad)
        row['fresh_identification']['comparison_90s_error_free'] = dict(
            tasks=good, n_tasks=len(good), old=mean(old_train[ds]['avg_gt'] for ds in good),
            new=mean(stats['0.0']['task_means'][ds] for ds in good))
    if old_names in latest:
        b = latest[old_names]
        baseline_stats, baseline_records = noise_stats(b)
        summary['original_bundle_in_new_run'] = dict(score=b['score'], seeds=b['seeds_evaluated'], noise=baseline_stats)
        baseline_bad = {(r['dataset'], r['noise']) for r in baseline_records if r['error']}
        baseline_grid = {(ds,float(n)):s for n,v in baseline_stats.items() for ds,s in v['task_means'].items()}
        for row in rows:
            records = [r for r in all_records if r['loc'] == row['loc']]
            bad = {(r['dataset'],r['noise']) for r in records if r['error']}
            grid = {(ds,float(n)):s for n,v in row['noise'].items() for ds,s in v['task_means'].items()}
            cells = sorted(set(grid) & set(baseline_grid) - bad - baseline_bad)
            row['same_run_error_free_comparison'] = dict(
                task_noise_cells=len(cells), old=mean(baseline_grid[c] for c in cells),
                new=mean(grid[c] for c in cells), cells=cells)
        # Fixed common subset across baseline and the three largest frontier points.
        top = {r['loc'] for r in rows[:3]}
        bad_top = {(r['dataset'],r['noise']) for r in all_records if r['loc'] in top and r['error']}
        common = sorted(set(baseline_grid) - baseline_bad - bad_top)
        summary['top_three_common_error_free'] = dict(
            cells=common, task_noise_cells=len(common), baseline=mean(baseline_grid[c] for c in common),
            frontier={r['loc']:mean(r['noise'][str(n)]['task_means'][ds] for ds,n in common) for r in rows[:3]})
    (OUT / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Recorded frontier scores', '',
             'Final generation 30 population, filtered for exact LOC/score nondominance. '
             'Scores are percentages of ground-truth matches; errors remain failures. '
             'Noise levels are relative target noise. See comparison.md for interpretation.', '',
             '| LOC / Julia bundle | Seeds | All noise | 0 | 0.001 | 0.01 | 0.1 | Errors at noise 0 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        values = [f'{100*r["noise"][n]["score"]:.2f}%' for n in ['0.0','0.001','0.01','0.1']]
        zero = r['noise']['0.0']
        lines.append(f'| [{r["loc"]}]({r["filename"]}) | {r["seeds"]} | {100*r["score"]:.3f}% | '
                     + ' | '.join(values) + f' | {zero["errors"]}/{zero["observations"]} |')
    lines += ['', '## Ten-seed fresh identification', '',
              'Only these final-frontier bundles were included in the fresh identification pass. '
              'The original frontier continues to use final-population scores.', '',
              '| LOC | Fresh all-noise | Noise 0 | Noise 0.001 | Noise 0.01 | Noise 0.1 | Errors at noise 0 |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        fresh = r['fresh_identification']
        if fresh:
            zero = fresh['noise']['0.0']
            values = [f'{100*fresh["noise"][n]["score"]:.2f}%' for n in ['0.0','0.001','0.01','0.1']]
            lines.append(f'| {r["loc"]} | {100*fresh["score"]:.3f}% | ' + ' | '.join(values)
                         + f' | {zero["errors"]}/{zero["observations"]} |')
    lines += ['', '## Matched zero-noise tasks without recorded errors', '',
              'Each row uses its own subset of the 20 training tasks: retain a task only '
              'if all recorded evaluations for both bundles are free of recorded errors. '
              'The reference is the separate 709715 90-second, ten-seed evaluation. '
              'These subsets differ by row and are not a new ranking.', '',
              '| LOC | Shared tasks | 709715 (261 LOC) | 941340 candidate |',
              '| ---: | ---: | ---: | ---: |']
    for r in rows:
        c = r['comparison_90s']
        lines.append(f'| {r["loc"]} | {c["error_free_tasks"]} | {100*c["old_error_free"]:.2f}% | {100*c["new_error_free"]:.2f}% |')
    lines += ['', '## Same-run reference, shared error-free task/noise cells', '',
              'The original 709715 validation winner was also evaluated inside 941340. '
              'Each row below retains only task/noise cells for which both candidates '
              'have no recorded errors across their recorded seeds. Each cell is equally weighted. '
              'These are sensitivity checks, not corrected population scores.', '',
              '| LOC | Shared cells (of 80) | Original 261 LOC | Candidate |',
              '| ---: | ---: | ---: | ---: |']
    for r in rows:
        c = r['same_run_error_free_comparison']
        lines.append(f'| {r["loc"]} | {c["task_noise_cells"]} | {100*c["old"]:.2f}% | {100*c["new"]:.2f}% |')
    (OUT / 'scores.md').write_text('\n'.join(lines) + '\n')
    with (OUT / 'per_task_noise.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['loc', 'dataset', 'run_index', 'noise', 'score', 'error'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(all_records)
    with (OUT / 'frontier.csv').open('w') as f:
        fields = ['loc', 'score', 'seeds', 'noise0', 'noise0001', 'noise001', 'noise01', 'zero_noise_errors', 'filename', 'name']
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(loc=r['loc'], score=r['score'], seeds=r['seeds'],
                                 **{k:r['noise'][n]['score'] for k,n in zip(fields[3:7], ['0.0','0.001','0.01','0.1'])},
                                 zero_noise_errors=r['noise']['0.0']['errors'], filename=r['filename'], name=r['name']))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('config','identification','frontier')}, indent=2))
    for r in rows:
        print(r['loc'], r['score'], r['seeds'], {n:(v['score'],v['errors'],v['observations']) for n,v in r['noise'].items()},r['component_loc'])
        print(r['name'])
        print('90s comparison',r['comparison_90s'])


if __name__ == '__main__':
    main()
