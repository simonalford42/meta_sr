#!/usr/bin/env python3
"""Replay saved PySR black-box candidates and fit affine maps on training rows.

No SR search, test-set scoring, or SLURM submission is performed. Dataset splits,
row sampling (including replacement), and scaling reproduce parallel_eval_pysr.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from domains import get_domain


def training_data(X, y, task):
    X_train, y_train, _, _, _ = split_data(X, y, task)
    return X_train, y_train


def split_data(X, y, task):
    """Reconstruct train/test rows and train-fitted scalers from a saved task."""
    finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[finite], y[finite]
    seed = task.get('data_split_seed')
    if seed is None:
        seed = task['seed'] + task['run_index']
    X, X_test, y, y_test = train_test_split(X, y, train_size=.75, test_size=.25,
                                 random_state=seed)
    cap = task['max_samples']
    if cap and len(y) > cap:
        keep = np.random.RandomState(seed).choice(len(y), cap)
        X, y = X[keep], y[keep]
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    return X, y, x_scaler.transform(X_test), y_test, y_scaler


def predict(equation, X):
    # Match pysr.export_sympy/pysr.export_numpy without initializing Julia.
    symbols = sp.symbols(f'x0:{X.shape[1]}')
    mappings = {str(s): s for s in symbols}
    mappings.update(square=lambda x: x**2, sin=sp.sin, cos=sp.cos,
                    exp=sp.exp, log=sp.log, sqrt=sp.sqrt)
    expr = sp.sympify(equation, locals=mappings, evaluate=False)
    with np.errstate(all='ignore'):
        pred = np.asarray(sp.lambdify(symbols, expr)(*X.T)) * np.ones(len(X))
    if np.iscomplexobj(pred) or not np.isfinite(pred).all():
        raise ValueError('Nonfinite/complex training predictions')
    return pred.astype(float)


def affine_metrics(y, p):
    mean_p, mean_y = np.mean(p), np.mean(y)
    pc, yc = p - mean_p, y - mean_y
    # Scale the centered predictor before OLS to avoid tiny/huge variances.
    scale = np.max(np.abs(pc))
    if scale == 0:
        slope = 0.0
        fitted = np.full_like(y, mean_y)
    else:
        z = pc / scale
        coefficient = np.dot(z, yc) / np.dot(z, z)
        slope = coefficient / scale
        fitted = mean_y + coefficient * z
    mse = float(np.mean((y - p)**2))
    affine_mse = float(np.mean((y - fitted)**2))
    variance = float(np.mean(yc**2))
    if variance == 0:
        raise ValueError('Constant training target; R2 undefined')
    if not np.isfinite([mse, affine_mse, slope]).all():
        raise ValueError('Nonfinite calibration or residual')
    if affine_mse > mse + 1e-10 * max(1, mse):
        raise AssertionError('OLS increased training error')
    return dict(train_mse=mse, train_r2=1-mse/variance,
                affine_train_mse=affine_mse, affine_train_r2=1-affine_mse/variance,
                slope=float(slope), intercept=float(mean_y-slope*mean_p),
                mean_prediction=float(mean_p), mean_target=float(mean_y))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows):
    before = np.array([r['train_r2'] for r in rows])
    after = np.array([r['affine_train_r2'] for r in rows])
    return dict(n_trials=len(rows), before_mean=float(before.mean()),
                after_mean=float(after.mean()), mean_gain=float((after-before).mean()),
                before_median=float(np.median(before)), after_median=float(np.median(after)),
                median_gain=float(np.median(after-before)),
                n_improved_over_001=int(np.sum(after-before > .01)))


def audit_selected(output):
    """Check replay parity for every candidate contributing to reported scores."""
    with (output/'selected.csv').open() as f:
        rows = list(csv.DictReader(f))
    mismatches = [r for r in rows if not np.isclose(
        float(r['train_mse']), float(r['stored_train_mse']), rtol=1e-6, atol=1e-8)]
    path = output/'summary.json'
    result = json.loads(path.read_text())
    result['n_selected_parity_failures'] = len(mismatches)
    result['protocol'] = {
        'data': 'original training rows from each 75/25 black-box trial',
        'calibration': 'unrestricted OLS slope and intercept fitted on training only',
        'aggregation': 'unclipped training R2, equal weight per dataset/seed trial',
        'selection': 'fixed candidate within each before/after pair',
        'replay_tolerance': {'rtol': 1e-6, 'atol': 1e-8},
    }
    path.write_text(json.dumps(result, indent=2)+'\n')
    return result


def run(run_dir, output, limit=None):
    manifest = json.loads((run_dir/'manifest.json').read_text())
    batch = run_dir/manifest['black_box']['batch_dir']
    tasks = json.loads((batch/'tasks.json').read_text())
    tasks = {(t['dataset_name'], t['run_index']): t for t in tasks}
    combined_path = batch/'combined.json'
    combined = json.loads(combined_path.read_text())
    candidates, selected, failures, parity_failures = [], [], [], []
    cache = {}
    for i, trial in enumerate(combined[:limit]):
        name, index = trial['dataset_name'], trial['run_index']
        task = tasks[name, index]
        assert task['black_box'] and not task.get('target_noise') and not task.get('target_noise_levels')
        domain = get_domain(task.get('domain', 'srbench'))
        data_key = (name, task['data_seed'])
        if data_key not in cache:
            # Only one dataset in memory; adjacent trials share the full data.
            cache.clear()
            cache[data_key] = domain.load_dataset(name, max_samples=None, data_seed=task['data_seed'])[:2]
        X, y = training_data(*cache[data_key], task)
        rows = []
        for candidate in trial.get('pareto_frontier') or []:
            row = dict(dataset=name, run_index=index, seed=task['seed']+index,
                       pysr_index=candidate['pysr_index'], complexity=candidate['complexity'],
                       equation=candidate['equation'], native_loss=candidate['loss'])
            try:
                row.update(affine_metrics(y, predict(candidate['equation'], X)))
                stored = candidate['train_mse']
                row['stored_train_mse'] = stored
                row['mse_absolute_difference'] = abs(row['train_mse']-stored)
                row['mse_relative_difference'] = abs(row['train_mse']-stored)/max(abs(stored), 1e-12)
                if not np.isclose(row['train_mse'], stored, rtol=1e-6, atol=1e-8):
                    parity_failures.append(row)
                candidates.append(row)
                rows.append(row)
            except Exception as exc:
                failures.append(dict(**row, error=str(exc)))
        if not rows:
            raise RuntimeError(f'No replayable candidates for {name}/{index}')
        for selection, key in [
            ('native_loss', lambda r: (r['native_loss'], r['complexity'], r['pysr_index'])),
            ('raw_train_mse', lambda r: (r['train_mse'], r['complexity'], r['pysr_index'])),
            ('affine_train_mse', lambda r: (r['affine_train_mse'], r['complexity'], r['pysr_index'])),
        ]:
            selected.append(dict(**min(rows, key=key), selection=selection))
        if (i+1) % 50 == 0:
            print(f'{run_dir.name}: {i+1}/{len(combined[:limit])} trials; '
                  f'{len(failures)} replay errors; {len(parity_failures)} parity mismatches', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'candidates.csv', candidates)
    write_csv(output/'selected.csv', selected)
    write_csv(output/'failures.csv', failures)
    write_csv(output/'parity_failures.csv', parity_failures)
    result = dict(source=str(run_dir), combined_sha256=hashlib.sha256(combined_path.read_bytes()).hexdigest(),
                  n_candidates=len(candidates), n_replay_errors=len(failures),
                  n_parity_failures=len(parity_failures),
                  max_mse_absolute_difference=max(r['mse_absolute_difference'] for r in candidates),
                  selections={name: summary([r for r in selected if r['selection']==name])
                              for name in ['native_loss','raw_train_mse','affine_train_mse']})
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    result = audit_selected(output)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    result = run(args.run_dir, args.output_dir, args.limit)
    if result['n_replay_errors'] or result['n_selected_parity_failures']:
        raise SystemExit('Replay validation failed; inspect saved diagnostics before interpreting scores.')
    if result['n_parity_failures']:
        print('WARNING: unselected candidates have replay mismatches; see parity_failures.csv.',
              file=sys.stderr)


if __name__ == '__main__':
    main()
