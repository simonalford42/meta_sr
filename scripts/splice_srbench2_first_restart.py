"""Replace first restarts without modifying archives; no searches or API calls."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_aggregation import merge_frontiers

DATASETS = {'first_principles_' + s for s in
            ('bode', 'hubble', 'ideal_gas', 'kepler', 'leavitt', 'newton', 'planck', 'rydberg', 'schechter')}


def snapshots(result):
    first = result['portfolio']['restarts'][0]
    duration = first['search_runtime_seconds']
    trace = first.get('execution_trace')
    if not trace:
        raise ValueError('Missing first-restart execution trace')
    observations, missing, seen = [], [], set()
    for o in sorted(trace, key=lambda x: x['elapsed_seconds']):
        scheduled = o.get('scheduled_seconds')
        if o.get('final') or scheduled is None:
            continue
        seen.add(scheduled)
        if o.get('status') != 'ok' or not o.get('equations'):
            missing.append({'scheduled_seconds': scheduled, 'status': o.get('status')})
            continue
        if not 0 < o['elapsed_seconds'] <= duration:
            raise ValueError('Snapshot outside first restart clock')
        observations.append({'seconds': o['elapsed_seconds'], 'scheduled_seconds': scheduled,
                             'kind': 'first_restart_snapshot', 'frontier': o['equations']})
    missing.extend({'scheduled_seconds': t, 'status': 'missing'}
                   for t in range(5, math.ceil(duration), 5) if t not in seen)
    cumulative, elapsed = [], 0.
    for i, restart in enumerate(result['portfolio']['restarts']):
        dt = restart['search_runtime_seconds']
        if restart.get('error') or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Failed restart or invalid clock')
        elapsed += dt
        cumulative = merge_frontiers([cumulative, restart['pareto_frontier']])
        observations.append({'seconds': elapsed, 'kind': 'restart_end', 'restart_index': i,
                             'frontier': cumulative})
    if any(a['seconds'] > b['seconds'] for a, b in zip(observations, observations[1:])):
        raise ValueError('Nonmonotone snapshot clock')
    return observations, missing


def replace(original, fresh):
    if original.get('error') or fresh.get('error') or not fresh.get('present'):
        raise ValueError('Missing or failed trial')
    for k in ('dataset', 'seed', 'run_index', 'noise', 'config_id'):
        if original[k] != fresh[k]:
            raise ValueError(f'Trial mismatch: {k}')
    op, np = original['portfolio'], fresh['portfolio']
    for p in (op, np):
        if (p['restart_max_evals'] != 1000000 or p['restart_timeout_seconds'] is not None
                or not p['warmup_excluded_from_budget']):
            raise ValueError('Wrong portfolio protocol')
    if len(np['restarts']) != 1 or np['restarts'][0]['seed'] != op['restarts'][0]['seed']:
        raise ValueError('Expected exactly the original first restart seed')
    row = copy.deepcopy(original)
    row['portfolio']['restarts'][0] = copy.deepcopy(np['restarts'][0])
    # Retain every archived later restart. The synthetic clock shifts by the
    # actual difference in first-fit duration; raw final time may exceed 3600.
    row['portfolio']['search_runtime_seconds'] = sum(r['search_runtime_seconds'] for r in row['portfolio']['restarts'])
    row['portfolio']['first_restart_snapshot_seconds'] = 5
    row['portfolio']['warmup_seconds'] = np['warmup_seconds']
    row['pareto_frontier'] = merge_frontiers([r['pareto_frontier'] for r in row['portfolio']['restarts']])
    for k in ('solved', 'gt_match_score', 'test_r2', 'runtime_seconds', 'solve_time',
              'best_equation', 'best_loss'):
        row[k] = None  # Old scores must not masquerade as recomputed scores.
    row['solve_time_source'] = 'pending_terra_review_synthetic_portfolio'
    row['synthetic_first_restart'] = {
        'old_seconds': op['restarts'][0]['search_runtime_seconds'],
        'new_seconds': np['restarts'][0]['search_runtime_seconds'],
        'later_restarts': 'unchanged; cumulative native-loss frontier rebuilt',
    }
    snapshots(row)  # Validate before writing.
    return row


def splice(source, replacement, output):
    def read(d, name):
        return json.loads((d/name).read_text())
    old, new = read(source, 'manifest.json'), read(replacement, 'manifest.json')
    for k in ('srbench_edition', 'ground_truth_protocol', 'mode', 'backend', 'max_samples',
              'cpus_per_task', 'baseline_l1_loss', 'maxsize_warmup', 'seeds', 'noise_levels', 'method_meta'):
        if old[k] != new[k]:
            raise ValueError(f'Configuration differs: {k}')
    a, b = read(source, 'srbench_full_results.json'), read(replacement, 'srbench_full_results.json')
    # Compare actual serialized worker settings as well as human-facing metadata.
    # This also detects edits to the evolved operators or data-seed defaults.
    def tasks(directory, manifest):
        return {(t['dataset_name'], t['seed']): t
                for batch in manifest['batches']
                for t in read(directory / batch['batch_dir'], 'tasks.json')}
    old_tasks, new_tasks = tasks(source, old), tasks(replacement, new)
    for key, t in new_tasks.items():
        if key[0] not in DATASETS:
            continue
        for field in ('pysr_kwargs', 'mutation_weights', 'custom_mutation_code',
                      'custom_selection_code', 'custom_survival_code', 'custom_loss_code',
                      'data_seed', 'data_split_seed', 'max_samples', 'domain', 'run_index'):
            # The observer interval is recorded in kwargs for cache identity only.
            x, y = copy.deepcopy(old_tasks[key].get(field)), copy.deepcopy(t.get(field))
            if field == 'pysr_kwargs':
                x.pop('_frontier_snapshot_seconds', None)
                y.pop('_frontier_snapshot_seconds', None)
            if x != y:
                raise ValueError(f'Worker configuration differs: {key}, {field}')
    expected = {(d, s) for d in DATASETS for s in range(10000,10010)}
    rows = {k: replace(r, b['results'][k]) for k, r in a['results'].items() if r['dataset'] in DATASETS}
    if {(r['dataset'],r['seed']) for r in rows.values()} != expected or len(rows) != 90:
        raise ValueError('Incomplete 9 x 10 grid')
    provenance = {'synthetic': True, 'original': str(source.resolve()), 'replacement': str(replacement.resolve()),
                  'source_sha256': {str(d.resolve()): hashlib.sha256((d/'srbench_full_results.json').read_bytes()).hexdigest()
                                    for d in (source, replacement)},
                  'clock': 'New first-fit duration plus archived later durations; plot clips final overshoot at 3600s.'}
    output.mkdir(parents=True, exist_ok=True)
    old.update(datasets=sorted(DATASETS), n_datasets=9, batches=[], derived_from=provenance)
    for name, data in [('manifest.json', old), ('srbench_full_results.json', {'meta': provenance, 'results': rows})]:
        path = output/name
        content = json.dumps(data, indent=2)+'\n'
        if path.exists() and path.read_text() != content:
            raise ValueError(f'Refusing to change existing derived data: {path}')
        path.write_text(content)
    print(f'Validated and wrote {len(rows)} synthetic trials: {output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path); p.add_argument('replacement', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); splice(a.source, a.replacement, a.output)
