#!/usr/bin/env python3
"""Score saved native frontiers offline; report first observed recovery times.

Times include fit startup and are observation times, not exact discovery times.
No searches or SLURM submissions are performed. Symbolic errors and unavailable
snapshots remain explicit. Planck/Rydberg additionally use the clean-grid check.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def score_trace(trace, check, interval=10, budget=90):
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('Snapshot interval must be positive and finite')
    cache = {}
    observations = []
    first = None
    first_robust = None
    for snapshot in sorted(trace, key=lambda s: s['elapsed_seconds']):
        scheduled = snapshot.get('scheduled_seconds')
        # Keep the requested grid and the final fit frontier. Do not assign a
        # later frontier to an earlier deadline or invent an unavailable read.
        if not snapshot.get('final') and (scheduled is None or scheduled > budget):
            continue
        matched = robust = None
        errors = 0
        if snapshot.get('status') == 'ok':
            for row in snapshot.get('equations', []):
                equation = row['equation']
                if equation not in cache:
                    cache[equation] = check(equation)
                decision = cache[equation]
                errors += bool(decision.get('error'))
                if decision.get('match') and matched is None:
                    matched = equation
                if decision.get('robust_match') and robust is None:
                    robust = equation
        entry = {
            'scheduled_seconds': scheduled,
            'elapsed_seconds': snapshot['elapsed_seconds'],
            'source_updated_elapsed_seconds': snapshot.get('source_updated_elapsed_seconds'),
            'final': snapshot.get('final', False),
            'status': snapshot.get('status'),
            'matched_equation': matched, 'robust_matched_equation': robust,
            'unresolved_checks': errors,
        }
        observations.append(entry)
        if matched and first is None:
            first = entry
        if robust and first_robust is None:
            first_robust = entry
    present = {s['scheduled_seconds'] for s in observations if not s['final']}
    return {
        'first_solve_seconds': first['elapsed_seconds'] if first else None,
        'first_solve_scheduled_seconds': first['scheduled_seconds'] if first else None,
        'first_solve_equation': first['matched_equation'] if first else None,
        'first_robust_solve_seconds': first_robust['elapsed_seconds'] if first_robust else None,
        'first_robust_solve_equation': first_robust['robust_matched_equation'] if first_robust else None,
        'missing_scheduled_seconds': [i * interval for i in range(1, math.floor(budget / interval) + 1)
                                      if i * interval not in present],
        'observations': observations, 'equation_checks': cache,
    }


def score_task(item):
    task, result_path = item
    from evaluation import check_pysr_symbolic_match, get_dataset_var_names
    from parallel_eval_pysr import _remap_formula_variables
    from utils import get_dataset_gt_formula

    dataset = task['dataset_name']
    record = {'dataset': dataset, 'seed': task['seed'] + task.get('run_index', 0),
              'result_path': str(result_path)}
    if not result_path.exists():
        return dict(record, status='missing', error='Missing worker result')
    raw = json.loads(result_path.read_text())
    names = get_dataset_var_names(dataset)
    variables = [f'x{i}' for i in range(len(names))]
    target = _remap_formula_variables(get_dataset_gt_formula(dataset), names, variables)

    def check(equation):
        if not target:
            return {'match': False, 'error': 'Missing ground truth'}
        decision = check_pysr_symbolic_match(equation, target, var_names=variables, timeout_seconds=3)
        result = {'match': bool(decision.get('match')), 'error': decision.get('error')}
        if dataset in ('empirical_planck', 'empirical_rydberg'):
            from scripts.empbench_lib import numeric_recovery
            robust = numeric_recovery(equation, dataset)
            result.update(robust_match=robust['match'], robust_error=robust.get('error'))
        return result

    trace = raw.get('execution_trace') or []
    summary = score_trace(trace, check, interval=task.get('frontier_snapshot_seconds') or 10)
    return dict(record, status='error' if raw.get('error') else ('complete' if trace else 'missing_trace'),
                error=raw.get('error'), **summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    for root in args.run_dirs:
        inputs = []
        for tasks_path in sorted(root.glob('slurm_pysr/eval_*/tasks.json')):
            for i, task in enumerate(json.loads(tasks_path.read_text())):
                inputs.append((task, tasks_path.parent/'results'/f'task_{i:06d}.json'))
        if not inputs:
            raise ValueError(f'No saved tasks in {root}')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(score_task, inputs))
        payload = {'clock': 'fit_wall_time',
                   'criterion': 'SRBench symbolic equivalence; separate clean-grid recovery for Planck/Rydberg',
                   'note': 'First observed match, not exact discovery time. Final snapshots may exceed 90 seconds.',
                   'records': records}
        output = root/'snapshot_solve_times.json'
        output.write_text(json.dumps(payload, indent=2) + '\n')
        fields = ['dataset', 'seed', 'status', 'first_solve_seconds', 'first_solve_scheduled_seconds',
                  'first_solve_equation', 'first_robust_solve_seconds', 'first_robust_solve_equation', 'error']
        with (root/'snapshot_solve_times.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
        print(f'Wrote {output}: {sum(r.get("first_solve_seconds") is not None for r in records)}/{len(records)} observed symbolic recoveries', flush=True)


if __name__ == '__main__':
    main()
