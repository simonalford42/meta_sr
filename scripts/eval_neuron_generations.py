#!/usr/bin/env python3
"""Evaluate the recorded generation winners in one 450-task NeuronBench batch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_full_eval import WORLDS, _write_json_atomic
from operator_types import OperatorBundle
from parallel_eval_pysr import PySRSlurmEvaluator


def prepare(source):
    data = json.loads(source.read_text())
    configs, winners = [], []
    for generation in range(1, 16):
        entry = next(g for g in data['generations'] if g['generation'] == generation)
        matches = [b for b in entry['population']
                   if OperatorBundle.from_dict(b).display_name == entry['best_name']]
        if len(matches) != 1:
            raise ValueError(f'Generation {generation}: ambiguous recorded winner')
        bundle = OperatorBundle.from_dict(matches[0])
        config = bundle.to_pysr_config(dict(data['config']['pysr_kwargs']))
        config.name = f'generation_{generation:02d}'
        configs.append(config)
        winners.append({'generation': generation, 'name': entry['best_name'],
                        'training_score': entry['best_score'], 'bundle': matches[0]})
    manifest = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    worlds=list(WORLDS), train_worlds=data['config']['dataset_names'],
                    seeds=list(range(10000, 10005)), expected=450, threshold=1e-6,
                    selection='Recorded best_name at the end of each generation',
                    generations=winners, max_samples=data['config']['max_samples'],
                    configs=[c.to_json_dict() for c in configs])
    return configs, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'runs/708907/run_data.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Write manifest only; never submit')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    configs, manifest = prepare(args.source.resolve())
    if (output / 'batch.json').exists():
        raise SystemExit('Batch already submitted; use the plotting script to inspect results.')
    _write_json_atomic(output / 'manifest.json', manifest)
    print('15 recorded winners x 6 worlds x 5 seeds = 450 fits; seeds 10000--10004', flush=True)
    if args.dry_run:
        print(f'Manifest: {output / "manifest.json"}; no jobs submitted')
        return
    evaluator = PySRSlurmEvaluator(
        results_dir=str(output), partition='default_partition', time_limit='00:15:00',
        mem_per_cpu='8G', cpus_per_task=1, dataset_max_samples=manifest['max_samples'],
        data_seed=260809696, job_timeout=86400, max_concurrent_jobs=450,
        target_noise=0.0, repo_root=str(ROOT), hof_n_steps=0, use_cache=False,
        pysr_wall_limit=600, domain='neuron', black_box=False,
        retain_pareto_frontier=True, max_retries=1,
    )
    handle = evaluator.submit_configs(configs=configs, dataset_names=list(WORLDS),
                                      seed=10000, n_runs=5, fitness_metric='gt')
    _write_json_atomic(output / 'batch.json', dict(batch_dir=str(handle.batch_dir),
                                                 job_ids=handle.job_ids, n_tasks=handle.n_tasks))
    try:
        evaluator.collect_batch(handle)
    finally:
        subprocess.run([sys.executable, str(ROOT / 'figures/plot_neuron_generations.py'),
                        str(output)], check=True)


if __name__ == '__main__':
    main()
