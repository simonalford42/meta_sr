#!/usr/bin/env python3
"""Reconstruct cumulative SRBench recovery from saved serial-restart frontiers.

No searches or SLURM jobs are launched. Symbolic checks are cached per dataset,
and a trial stops at its first recovered restart. This measures ever recovered,
not recovery on the final native-loss merged frontier. Within-restart discovery
times are unavailable, so recovery is placed at the restart's completion.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = {
    "Base PySR": ROOT / "runs/srbench_gt_baseline_15m_portfolio_1e6",
    "709715": ROOT / "runs/709715/srbench_gt_15m_portfolio_1e6",
}


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def index_inputs():
    grouped = defaultdict(list)
    for label, root in RUNS.items():
        manifest = json.loads((root / 'manifest.json').read_text())
        for batch in manifest['batches']:
            directory = root / batch['batch_dir']
            tasks = json.loads((directory / 'tasks.json').read_text())
            for i, task in enumerate(tasks):
                grouped[task['dataset_name']].append({
                    'method': label, 'dataset': task['dataset_name'],
                    'seed': task['seed'] + task.get('run_index', 0),
                    'noise': float(task.get('target_noise', batch['noise'])),
                    'path': str(directory / 'results' / f'task_{i:06d}.json'),
                })
    return grouped


def inventory(grouped, output):
    """Audit completeness/timing without running any symbolic checks."""
    counters = defaultdict(Counter)
    max_overshoot = defaultdict(float)
    for specs in grouped.values():
        for spec in specs:
            key = f"{spec['method']}|{spec['noise']:g}"
            counts = counters[key]
            counts['expected'] += 1
            path = Path(spec['path'])
            if not path.exists():
                counts['missing'] += 1
                continue
            raw = json.loads(path.read_text())
            counts['errors' if raw.get('error') else 'complete'] += 1
            counts['final_merged_solved'] += raw.get('gt_match_score') == 1
            portfolio = raw.get('portfolio')
            if not portfolio:
                counts['no_portfolio'] += 1
                continue
            restarts = portfolio['restarts']
            counts['restarts'] += len(restarts)
            counts['restart_frontier_rows'] += sum(len(r.get('pareto_frontier') or []) for r in restarts)
            counts['restart_traces'] += sum(bool(r.get('execution_trace')) for r in restarts)
            counts['failed_restarts'] += sum(bool(r.get('error')) for r in restarts)
            max_overshoot[key] = max(max_overshoot[key],
                portfolio['search_runtime_seconds'] - portfolio['total_search_budget_seconds'])
    result = {k: dict(v, max_search_budget_overshoot_seconds=max_overshoot[k])
              for k, v in counters.items()}
    write_json(output / 'inventory.json', result)
    print(json.dumps(result, indent=2))


def analyze_dataset(dataset, specs, output):
    from evaluation import check_pysr_symbolic_match, get_dataset_var_names
    from parallel_eval_pysr import _remap_formula_variables
    from utils import get_dataset_gt_formula

    output = Path(output)
    cache_path = output / 'cache' / f'{dataset}.json'
    result_path = output / 'datasets' / f'{dataset}.json'
    signature = hashlib.sha256(json.dumps([
        (s, Path(s['path']).stat().st_mtime_ns if Path(s['path']).exists() else None)
        for s in specs
    ], sort_keys=True).encode()).hexdigest()
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old.get('signature') == signature:
            return old
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    names = get_dataset_var_names(dataset)
    variables = [f'x{i}' for i in range(len(names))]
    target = _remap_formula_variables(get_dataset_gt_formula(dataset), names, variables)
    if not target:
        raise ValueError(f'Missing ground truth: {dataset}')
    counters = Counter()
    records = []
    # Existing *positive* checks are reusable. False row flags are NOT negative
    # evidence: the original evaluator stops after one match.
    known = set()
    for spec in specs:
        path = Path(spec['path'])
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        if raw.get('gt_match_score') == 1 and raw.get('gt_matched_equation'):
            known.add(raw['gt_matched_equation'])
        known.update(r['equation'] for r in raw.get('pareto_frontier') or []
                     if r.get('solved'))
    for equation in known:
        cache[equation] = {'match': True, 'source': 'saved_positive'}
    for spec in specs:
        record = {k: v for k, v in spec.items() if k != 'path'}
        record.update(first_solve_seconds=None, first_solve_budget_seconds=None,
                      first_solve_restart=None, first_solve_equation=None,
                      unresolved_checks_before_solve=0)
        path = Path(spec['path'])
        if not path.exists():
            record.update(status='missing', final_merged_solved=False)
            records.append(record)
            continue
        raw = json.loads(path.read_text())
        portfolio = raw.get('portfolio')
        if not portfolio:
            raise ValueError(f'No saved portfolio: {path}')
        record.update(status='error' if raw.get('error') else 'complete',
                      final_merged_solved=raw.get('gt_match_score') == 1,
                      restart_count=len(portfolio['restarts']),
                      total_search_seconds=portfolio['search_runtime_seconds'])
        elapsed = 0.0
        for restart in portfolio['restarts']:
            elapsed += max(0.0, float(restart.get('search_runtime_seconds') or 0))
            if restart.get('error'):
                counters['failed_restarts'] += 1
                continue
            rows = sorted(restart.get('pareto_frontier') or [],
                          key=lambda r: -r['complexity'])
            # Prefer an already-verified equation in this restart; its timestamp
            # is the same regardless of which frontier equation matches first.
            rows.sort(key=lambda r: not cache.get(r['equation'], {}).get('match'))
            for row in rows:
                r2 = row.get('r2')
                if r2 is None or not math.isfinite(r2) or r2 < 0.5:
                    counters['r2_gate_skips'] += 1
                    continue
                equation = row['equation']
                if equation in cache:
                    checked = cache[equation]
                    counters['cache_hits'] += 1
                else:
                    checked_raw = check_pysr_symbolic_match(
                        equation, target, var_names=variables, timeout_seconds=3)
                    checked = {'match': bool(checked_raw.get('match')),
                               'error': checked_raw.get('error'), 'source': 'sympy'}
                    cache[equation] = checked
                    counters['new_checks'] += 1
                    if counters['new_checks'] % 100 == 0:
                        write_json(cache_path, cache)
                if checked.get('error'):
                    counters['unresolved_check_uses'] += 1
                    record['unresolved_checks_before_solve'] += 1
                if checked['match']:
                    record.update(
                        first_solve_seconds=elapsed,
                        first_solve_budget_seconds=min(
                            elapsed, portfolio['total_search_budget_seconds']),
                        first_solve_restart=restart['restart_index'] + 1,
                        first_solve_equation=equation)
                    break
            if record['first_solve_seconds'] is not None:
                counters['later_restarts_skipped'] += (
                    len(portfolio['restarts']) - record['first_solve_restart'])
                break
        records.append(record)
    write_json(cache_path, cache)
    result = {'dataset': dataset, 'signature': signature, 'records': records,
              'counters': dict(counters)}
    write_json(result_path, result)
    return result


def render(output, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    records = [r for d in results for r in d['records']]
    write_json(output / 'first_recovery.json', records)
    noises = [0.0, 0.001, 0.01, 0.1]
    table = []
    for noise in noises + ['all']:
        for minute in range(16):
            for method in RUNS:
                subset = [r for r in records if r['method'] == method and
                          (noise == 'all' or r['noise'] == noise)]
                solved = sum(r['first_solve_budget_seconds'] is not None and
                             r['first_solve_budget_seconds'] <= minute * 60
                             for r in subset)
                table.append({'noise': noise, 'minutes': minute, 'method': method,
                              'solved': solved, 'expected': len(subset),
                              'percent_solved': 100 * solved / len(subset)})
    with (output / 'solve_rate.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    colors = {'Base PySR': '#3264ad', '709715': '#d45b24'}
    for ax, noise in zip(axes.flat, noises):
        for method in RUNS:
            subset = [r for r in records if r['method'] == method and r['noise'] == noise]
            times = sorted(r['first_solve_budget_seconds'] / 60 for r in subset
                           if r['first_solve_budget_seconds'] is not None)
            xs = [0] + times + [15]
            ys = [0] + [100 * i / len(subset) for i in range(1, len(times) + 1)]
            ys.append(ys[-1])
            ax.step(xs, ys, where='post', label=method, color=colors[method], linewidth=2)
        ax.set_title(f'Target noise: {noise:g}')
        ax.set_xlim(0, 15)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.set_xticks([0, 3, 6, 9, 12, 15])
    axes[0, 0].legend(frameon=False)
    fig.supxlabel('Cumulative search time (minutes)')
    fig.supylabel('Trials recovered at least once (%)')
    n_datasets = len({r['dataset'] for r in records})
    fig.suptitle(f'15-minute restart portfolios · {n_datasets} tasks × 10 seeds per noise level')
    fig.text(0.5, 0.005, 'Recovery credited at restart completion; warm-up excluded; final budget overshoot mapped to 15 min.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0.02, 0.025, 1, 0.97))
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'solve_rate.{ext}', dpi=180)
    totals = Counter()
    for d in results:
        totals.update(d['counters'])
    report = ['# SRBench 15-minute portfolio recovery over time', '',
              'Metric: percentage of task–seed trials with a symbolic recovery on any completed restart frontier. '
              f'All {n_datasets} selected tasks remain in the denominator; each method has '
              f'{n_datasets * 10:,} trials per noise level.', '',
              'The existing SRBench symbolic checker is used with a 3-second expression timeout and the saved '
              'held-out R² ≥ 0.5 gate. Positive final checks are reused; unchecked `solved=False` flags are never '
              'treated as negative checks. Results are cached by dataset and exact equation text across seeds, '
              'noise levels, and methods; checking stops after first recovery.', '',
              'Only restart-end frontiers are available, so discovery time is an upper bound at restart resolution. '
              'Warm-up and scoring are excluded. Small search-budget overshoots at the last restart are mapped '
              'to the nominal 15-minute endpoint; raw times are retained in first_recovery.json. '
              'Cumulative recovery can exceed the final merged-frontier score, because a later native-loss '
              'frontier can discard an earlier matching equation. Timeouts/parsing failures are unresolved '
              'and treated as non-matches, as in the evaluator; the curve is conservative for such checks.', '']
    for noise in noises + ['all']:
        report.extend([f'## Noise {noise}', '', '| Minutes | Base PySR | 709715 |',
                       '|---:|---:|---:|'])
        for minute in (1, 3, 5, 10, 15):
            vals = [next(r for r in table if r['noise'] == noise and r['minutes'] == minute
                         and r['method'] == m) for m in RUNS]
            report.append(f"| {minute} | " + ' | '.join(
                f"{v['percent_solved']:.2f}% ({v['solved']}/{v['expected']})" for v in vals) + ' |')
        report.append('')
    report.extend(['## Validation and accounting', '', '```json', json.dumps({
        'trials': len(records), 'status': dict(Counter(r['status'] for r in records)),
        'trials_with_unresolved_checks': sum(bool(r['unresolved_checks_before_solve']) for r in records),
        'final_positive_without_recovery': sum(r['final_merged_solved'] and
            r['first_solve_seconds'] is None for r in records),
        'counts': dict(totals),
        'final_merged_recoveries': {m: sum(r['final_merged_solved'] for r in records
                                        if r['method'] == m) for m in RUNS},
    }, indent=2), '```', ''])
    (output / 'README.md').write_text('\n'.join(report))
    print('\n'.join(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/portfolio_solve_over_time')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--datasets', nargs='+', help='Optional subset for verification')
    parser.add_argument('--array-task', action='store_true',
                        help='Analyze just the dataset indexed by SLURM_ARRAY_TASK_ID')
    parser.add_argument('--render-only', action='store_true',
                        help='Render completed dataset checkpoints; requires every selected dataset')
    parser.add_argument('--inventory-only', action='store_true',
                        help='Audit saved trials and restart timestamps without symbolic checks')
    args = parser.parse_args()
    for sub in ('cache', 'datasets'):
        (args.output / sub).mkdir(parents=True, exist_ok=True)
    grouped = index_inputs()
    if args.datasets:
        grouped = {k: v for k, v in grouped.items() if k in args.datasets}
    if args.inventory_only:
        inventory(grouped, args.output)
        return
    if args.array_task:
        dataset = sorted(grouped)[int(os.environ['SLURM_ARRAY_TASK_ID'])]
        result = analyze_dataset(dataset, grouped[dataset], args.output)
        print(dataset, result['counters'], flush=True)
        return
    if args.render_only:
        render(args.output, [json.loads((args.output / 'datasets' / f'{ds}.json').read_text())
                             for ds in sorted(grouped)])
        return
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze_dataset, ds, specs, str(args.output)): ds
                   for ds, specs in grouped.items()}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{len(results)}/{len(grouped)} {result['dataset']} "
                  f"{result['counters']} elapsed={time.monotonic()-started:.1f}s", flush=True)
    render(args.output, results)


if __name__ == '__main__':
    main()
