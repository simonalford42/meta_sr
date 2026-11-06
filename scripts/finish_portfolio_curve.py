#!/usr/bin/env python3
"""Finish outstanding seed checkpoints inside an existing CPU allocation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from analyze_portfolio_seed_shards import run_seed, collect_seed_groups
from analyze_portfolio_solve_over_time import ROOT, collect_groups, index_inputs, render
from validate_portfolio_curve import main as validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    output = ROOT / 'reports/portfolio_solve_over_time'
    plan = json.loads((output / 'seed_plan.json').read_text())
    missing = [i for i, p in enumerate(plan) if not
               (output / 'seed_shards/datasets' / f'{p["cache_key"]}.json').exists()]
    print('Remaining seed indices:', missing, flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_seed, output, i): i for i in missing}
        for future in as_completed(futures):
            future.result()
            print('Completed seed', futures[future], flush=True)
    print('Collected seeds:', collect_seed_groups(output), flush=True)
    collect_groups(output)
    render(output, [json.loads((output / 'datasets' / f'{ds}.json').read_text())
                    for ds in sorted(index_inputs())])
    validate()


if __name__ == '__main__':
    main()
