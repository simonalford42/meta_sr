#!/usr/bin/env python3
"""Split only the slowest recovery groups into independent seed workers."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path

from analyze_portfolio_solve_over_time import (
    ROOT, analyze_dataset, input_signature, write_json,
)


def prepare(output, limit=32, extend=False, all_remaining=False):
    destination = output / 'seed_plan.json'
    if destination.exists() and not extend:
        raise ValueError('Seed plan already exists; do not reorder live workers')
    plan = json.loads(destination.read_text()) if destination.exists() else []
    covered = {p['parent_index'] for p in plan}
    original_count = len(plan)
    groups = json.loads((output / 'group_plan.json').read_text())
    remaining = []
    for index, item in enumerate(groups):
        if index in covered or (output / 'group_shards/datasets' / f'{item["cache_key"]}.json').exists():
            continue
        if all_remaining:
            remaining.append((0, index))
            continue
        path = output / 'group_shards/cache' / f'{item["cache_key"]}.json'
        cache = json.loads(path.read_text()) if path.exists() else {}
        unknown = set()
        for spec in item['specs']:
            raw = json.loads(Path(spec['path']).read_text())
            for restart in raw['portfolio']['restarts']:
                rows = [r for r in restart.get('pareto_frontier') or []
                        if r.get('r2') is not None and math.isfinite(r['r2']) and r['r2'] >= .5]
                if any(cache.get(r['equation'], {}).get('match') for r in rows):
                    break
                unknown.update(r['equation'] for r in rows if r['equation'] not in cache)
        remaining.append((len(unknown), index))
    selected = ([i for _, i in remaining] if all_remaining else
                [i for n, i in sorted(remaining, reverse=True)[:limit] if n >= 2000])
    for index in selected:
        for spec in groups[index]['specs']:
            plan.append({'parent_index': index, 'parent': groups[index], 'specs': [spec],
                         'cache_key': f'seed_{len(plan):04d}'})
    write_json(destination, plan)
    print(f'Prepared {len(plan) - original_count} new seed jobs, indices {original_count}–{len(plan)-1}, '
          f'for {len(selected)} groups: {selected}', flush=True)


def identity(record):
    return tuple(record[k] for k in ('method', 'dataset', 'seed', 'noise'))


def collect_seed_groups(output):
    plan = json.loads((output / 'seed_plan.json').read_text())
    grouped = defaultdict(list)
    for item in plan:
        grouped[item['parent_index']].append(item)
    completed = 0
    for items in grouped.values():
        parent = items[0]['parent']
        destination = output / 'group_shards/datasets' / f'{parent["cache_key"]}.json'
        signature = input_signature(parent['specs'])
        if destination.exists() and json.loads(destination.read_text())['signature'] == signature:
            completed += 1
            continue
        paths = [output / 'seed_shards/datasets' / f'{item["cache_key"]}.json' for item in items]
        if not all(path.exists() for path in paths):
            continue
        records = []
        counts = Counter()
        for item, path in zip(items, paths):
            result = json.loads(path.read_text())
            if result['signature'] != input_signature(item['specs']):
                raise ValueError(f'Stale seed checkpoint: {path}')
            records.extend(result['records'])
            counts.update(result['counters'])
        expected = {identity(s) for s in parent['specs']}
        if len(records) != len(expected) or {identity(r) for r in records} != expected:
            raise ValueError(f'Duplicate/missing seed records for {parent["cache_key"]}')
        write_json(destination, {'dataset': parent['dataset'], 'signature': signature,
                                 'records': records, 'counters': dict(counts)})
        completed += 1
    return completed, len(grouped)


def run_seed(output, index):
    plan = json.loads((output / 'seed_plan.json').read_text())
    item = plan[index]
    parent = item['parent']
    shard_output = output / 'seed_shards'
    for sub in ('cache', 'datasets'):
        (shard_output / sub).mkdir(parents=True, exist_ok=True)
    parent_result = output / 'group_shards/datasets' / f'{parent["cache_key"]}.json'
    if parent_result.exists():
        saved = json.loads(parent_result.read_text())
        if saved['signature'] == input_signature(parent['specs']):
            record = next(r for r in saved['records'] if identity(r) == identity(item['specs'][0]))
            write_json(shard_output / 'datasets' / f'{item["cache_key"]}.json', {
                'dataset': parent['dataset'], 'signature': input_signature(item['specs']),
                'records': [record], 'counters': {'completed_parent_reused': 1}})
            return
    group_plan = json.loads((output / 'group_plan.json').read_text())
    siblings = [output / 'group_shards/cache' / f'{p["cache_key"]}.json'
                for p in group_plan if p['dataset'] == parent['dataset']]
    siblings += [shard_output / 'cache' / f'{p["cache_key"]}.json'
                 for p in plan if p['parent']['dataset'] == parent['dataset']
                 and p['cache_key'] != item['cache_key']]
    result = analyze_dataset(parent['dataset'], item['specs'], shard_output,
                             item['cache_key'], parent['seed_cache'], siblings)
    print(item['cache_key'], result['counters'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/portfolio_solve_over_time')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--extend', action='store_true', help='Append; preserve all existing worker indices')
    parser.add_argument('--all-remaining', action='store_true')
    parser.add_argument('--index-offset', type=int, default=0)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output, args.limit, args.extend, args.all_remaining)
    elif args.collect:
        print('Collected seed groups:', collect_seed_groups(args.output), flush=True)
    else:
        run_seed(args.output, int(os.environ['SLURM_ARRAY_TASK_ID']) + args.index_offset)


if __name__ == '__main__':
    main()
