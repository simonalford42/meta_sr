#!/usr/bin/env python3
"""Score saved expressions on held-out rows with frozen train-fitted affine maps.

Reports the existing best-test-candidate metric and training-selected controls.
No symbolic search or SLURM submission. Original benchmark files stay unchanged.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from reevaluate_black_box_affine import (
    affine_metrics, get_domain, predict, split_data, write_csv,
)


def test_score(y, prediction, scaler):
    prediction = scaler.inverse_transform(np.asarray(prediction).reshape(-1, 1)).ravel()
    # Preserve the benchmark's prediction clipping and denominator convention.
    prediction = np.clip(prediction, -1e10, 1e10)
    return float(1 - np.sum((y-prediction)**2)/(np.sum((y-y.mean())**2)+1e-10))


def frozen_predictions(p, metrics):
    return metrics['mean_target'] + metrics['slope']*(p-metrics['mean_prediction'])


def summarize(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in [
        'stored_best_test_r2', 'raw_best_test_r2', 'affine_best_test_r2',
        'raw_winner_affine_test_r2', 'train_selected_raw_test_r2',
        'train_selected_fixed_affine_test_r2', 'train_selected_affine_test_r2',
    ]}


def run(source, output, limit=None, train_replay=None):
    manifest = json.loads((source/'manifest.json').read_text())
    batch = source/manifest['black_box']['batch_dir']
    combined_path = batch/'combined.json'
    digest = hashlib.sha256(combined_path.read_bytes()).hexdigest()
    tasks = json.loads((batch/'tasks.json').read_text())
    tasks = {(t['dataset_name'], t['run_index']): t for t in tasks}
    raw = json.loads(combined_path.read_text())
    coefficients = {}
    if train_replay:
        previous = json.loads((train_replay/'summary.json').read_text())
        assert previous['combined_sha256'] == digest, 'Training replay source changed'
        with (train_replay/'candidates.csv').open() as f:
            for row in csv.DictReader(f):
                coefficients[row['dataset'], int(row['run_index']), int(row['pysr_index'])] = {
                    k: float(row[k]) for k in ['train_mse', 'affine_train_mse', 'slope',
                                              'mean_prediction', 'mean_target']}
    cache, candidates, trials, failures, mismatches = {}, [], [], [], []
    output.mkdir(parents=True, exist_ok=True)
    for i, result in enumerate(raw[:limit]):
        name, index = result['dataset_name'], result['run_index']
        task = tasks[name, index]
        assert task['black_box'] and not task.get('target_noise') and not task.get('target_noise_levels')
        data_key = (name, task['data_seed'])
        if data_key not in cache:
            cache.clear()
            cache[data_key] = get_domain(task.get('domain', 'srbench')).load_dataset(
                name, max_samples=None, data_seed=task['data_seed'])[:2]
        X, y, Xt, yt, scaler = split_data(*cache[data_key], task)
        rows = []
        for candidate_index, c in enumerate(result.get('pareto_frontier') or []):
            pysr_index = c.get('pysr_index', candidate_index)
            row = dict(dataset=name, run_index=index, seed=task['seed']+index,
                       pysr_index=pysr_index, complexity=c['complexity'],
                       equation=c['equation'], stored_test_r2=c['test_r2'])
            try:
                fitted = coefficients.get((name, index, pysr_index))
                if fitted is None:
                    fitted = affine_metrics(y, predict(c['equation'], X))
                p = predict(c['equation'], Xt)
                row.update(train_mse=fitted['train_mse'], affine_train_mse=fitted['affine_train_mse'],
                           raw_test_r2=test_score(yt, p, scaler),
                           affine_test_r2=test_score(yt, frozen_predictions(p, fitted), scaler),
                           slope=fitted['slope'], mean_prediction=fitted['mean_prediction'],
                           mean_target=fitted['mean_target'])
                row['train_parity'] = (bool(np.isclose(fitted['train_mse'], c['train_mse'], rtol=1e-6, atol=1e-8))
                                       if 'train_mse' in c else None)
                row['test_parity'] = bool(np.isclose(row['raw_test_r2'], c['test_r2'], rtol=1e-6, atol=1e-8))
                if row['train_parity'] is False or not row['test_parity']:
                    mismatches.append(row)
                candidates.append(row)
                rows.append(row)
            except Exception as exc:
                failures.append(dict(**row, error=str(exc)))
        if not rows:
            raise RuntimeError(f'No scored candidates: {name}/{index}')
        tie = lambda r: (r['complexity'], r['pysr_index'])
        best_raw = min(rows, key=lambda r: (-r['raw_test_r2'], *tie(r)))
        best_affine = min(rows, key=lambda r: (-r['affine_test_r2'], *tie(r)))
        train_raw = min(rows, key=lambda r: (r['train_mse'], *tie(r)))
        train_affine = min(rows, key=lambda r: (r['affine_train_mse'], *tie(r)))
        chosen = [best_raw, best_affine, train_raw, train_affine]
        trials.append(dict(dataset=name, run_index=index, seed=task['seed']+index,
            n_candidates=len(rows), stored_best_test_r2=max(c['test_r2'] for c in result['pareto_frontier']),
            raw_best_test_r2=best_raw['raw_test_r2'], affine_best_test_r2=best_affine['affine_test_r2'],
            raw_winner_affine_test_r2=best_raw['affine_test_r2'],
            train_selected_raw_test_r2=train_raw['raw_test_r2'],
            train_selected_fixed_affine_test_r2=train_raw['affine_test_r2'],
            train_selected_affine_test_r2=train_affine['affine_test_r2'],
            best_raw_index=best_raw['pysr_index'], best_affine_index=best_affine['pysr_index'],
            train_raw_index=train_raw['pysr_index'], train_affine_index=train_affine['pysr_index'],
            selected_parity_failures=sum(not r['test_parity'] or r['train_parity'] is False for r in chosen)))
        if (i+1)%50 == 0:
            print(f'{source.name}: {i+1}/{len(raw[:limit])}; errors={len(failures)}, '
                  f'mismatches={len(mismatches)}', flush=True)
    write_csv(output/'candidates.csv', candidates)
    write_csv(output/'trials.csv', trials)
    write_csv(output/'failures.csv', failures)
    write_csv(output/'parity_failures.csv', mismatches)
    report = dict(source=str(source), combined_sha256=digest,
                  training_replay=str(train_replay) if train_replay else None,
                  max_evals=manifest['max_evals'], black_box_timeout=manifest['black_box'].get('timeout_in_seconds'),
                  seed=manifest['seed'], n_trials=len(trials), n_candidates=len(candidates),
                  n_replay_errors=len(failures), n_parity_failures=len(mismatches),
                  n_candidates_with_stored_train_mse=sum(r['train_parity'] is not None for r in candidates),
                  n_selected_parity_failures=sum(r['selected_parity_failures'] for r in trials),
                  protocol=dict(calibration='OLS slope/intercept fitted on original training rows only',
                                test_scoring='original target units; predictions clipped to +/-1e10; denominator epsilon 1e-10; negative R2 retained',
                                aggregation='mean of per-trial scores; best_test columns select on test; train_selected columns select on train'),
                  scores=summarize(trials))
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-replay', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    report = run(args.run_dir, args.output_dir, args.limit, args.train_replay)
    if report['n_replay_errors'] or report['n_selected_parity_failures']:
        raise SystemExit('Replay diagnostics require review before interpreting results.')
