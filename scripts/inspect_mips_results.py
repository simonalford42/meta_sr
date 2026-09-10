#!/usr/bin/env python3
"""Inspect raw MIPS results without running searches or modifying files.

Default: compare the three ten-seed evaluations behind the MIPS figures.
Successful components are identified by recorded gt_match_score == 1.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'PySR': 'runs/709714/final_eval_baseline_10seed/slurm_pysr/eval_0000',
    'SRBench': 'runs/709715/final_eval_mips_native_10seed/slurm_pysr/eval_0000',
    'MIPS': 'runs/709714/final_eval/slurm_pysr/eval_0000',
}


def inspect(path):
    path = path.resolve()
    if not (path / 'tasks.json').exists():
        path = path / 'slurm_pysr/eval_0000'
    tasks = json.loads((path / 'tasks.json').read_text())
    if not tasks:
        raise ValueError(f'Empty task manifest: {path}')
    configs = {t.get('config_id') for t in tasks}
    if len(configs) != 1:
        raise ValueError(f'Expected one configuration, found {configs}: {path}')
    groups = defaultdict(set)
    exact = defaultdict(set)
    records = []
    present = set()
    expected = set()
    missing = []
    runs = set()
    for i, task in enumerate(tasks):
        ds, run = task['dataset_name'], task['run_index']
        key = (ds, run)
        if key in expected:
            raise ValueError(f'Duplicate dataset/run: {key}')
        expected.add(key)
        if not ds.startswith('mips:'):
            raise ValueError(f'Not a MIPS dataset: {ds}')
        groups[ds.split(':')[1]].add(ds)
        runs.add(run)
        file = path / f'results/task_{i:06d}.json'
        if not file.exists():
            missing.append(str(file))
            continue
        r = json.loads(file.read_text())
        if (r['dataset_name'], r['run_index']) != key:
            raise ValueError(f'Result does not match manifest: {file}')
        present.add(key)
        if r.get('gt_match_score') == 1:
            exact[ds].add(run)
        records.append(dict(r, source=str(file), seed=task['seed']))
    components = set().union(*groups.values())
    if expected != {(ds, r) for ds in components for r in runs}:
        raise ValueError(f'Nonrectangular dataset/run manifest: {path}')
    joint = {g: set.intersection(*(exact[c] for c in cs)) for g, cs in groups.items()}
    assembled = {g for g, cs in groups.items() if all(exact[c] for c in cs)}
    return dict(path=path, groups=groups, exact=exact, records=records, missing=missing,
                runs=runs, joint=joint, assembled=assembled, components=components,
                present=present, expected=expected)


def table(headers, rows):
    rows = [[str(x) for x in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print('  '.join(h.ljust(w) for h, w in zip(headers, widths)))
    print('  '.join('-' * w for w in widths))
    for row in rows:
        print('  '.join(x.ljust(w) for x, w in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', type=Path, help='Inspect one raw eval_0000 directory or its evaluation root.')
    parser.add_argument('--task', help='Filter task names by substring, e.g. base_6 or alternating_last4.')
    parser.add_argument('--components', action='store_true', help='Show per-component exact run indices.')
    parser.add_argument('--equations', action='store_true', help='Show one recorded exact witness per component and its raw path.')
    parser.add_argument('--original-summary', type=Path, default=ROOT / 'outputs/mips_reproduction_all/summary.json')
    args = parser.parse_args()
    original = json.loads(args.original_summary.read_text())
    original_solved = {t['task'] for t in original['tasks'] if t['independent_success']}
    paths = {str(args.eval_dir): args.eval_dir} if args.eval_dir else {k: ROOT / p for k, p in DEFAULTS.items()}
    methods = {k: inspect(p) for k, p in paths.items()}
    rows = []
    for name, m in methods.items():
        candidates = set(m['groups']) - original_solved
        scalar = sum(bool(m['exact'][c]) for c in m['components'])
        rows.append([name, f"{len(m['present'])}/{len(m['expected'])}",
                     f"{sum(r.get('gt_match_score') == 1 for r in m['records'])}/{len(m['expected'])}",
                     f"{scalar}/{len(m['components'])}",
                     f"{sum(bool(m['joint'][g]) for g in candidates)}/{len(candidates)}",
                     f"{len(m['assembled'] & candidates)}/{len(candidates)}"])
        print(f"{name}: {m['path']}")
        if m['missing']:
            print(f"  INCOMPLETE: {len(m['missing'])} missing results; counts are lower bounds.")
        errors = sum(bool(r.get('error')) for r in m['records'])
        timeouts = sum(bool(r.get('timed_out')) for r in m['records'])
        print(f'  {len(m["runs"])} seeds; {errors} recorded errors; {timeouts} recorded timeouts')
    print('\nSummary (counts always cover the full evaluated split):')
    table(['Method', 'Present fits', 'Exact fits', 'Subtasks any seed', 'New groups same seed', 'New groups across seeds'], rows)
    print('\nSame seed: at least one run_index solves every component together.')
    print('Across seeds: every component has a success, possibly in different run_indices.')
    print('New groups exclude the original reproduction successes; assembled programs are not revalidated here.')
    all_groups = sorted(set().union(*(set(m['groups']) for m in methods.values())))
    selected = [g for g in all_groups if not args.task or args.task in g]
    if not selected:
        parser.error(f'No task matches {args.task!r}')
    print('\nPer-task success: number of joint exact seeds / total seeds; A = all components found across seeds.')
    table(['Task', 'Original', *methods], [[g, 'solved' if g in original_solved else 'unsolved',
          *[f"{len(m['joint'][g])}/{len(m['runs'])}; A={'yes' if g in m['assembled'] else 'no'}" if g in m['groups'] else 'not evaluated' for m in methods.values()]] for g in selected])
    names = list(methods)
    for before, after in zip(names, names[1:]):
        a, b = methods[before], methods[after]
        common = a['components'] & b['components']
        sa = {c for c in common if a['exact'][c]}
        sb = {c for c in common if b['exact'][c]}
        print(f'\nDistinct component changes: {before} -> {after} (shared components only)')
        for label, diff in [('Gained', sb-sa), ('Lost', sa-sb)]:
            filtered = sorted(c for c in diff if not args.task or args.task in c)
            print(f'  {label}: ' + (', '.join(filtered) or 'none'))
    if args.components or args.equations:
        print('\nComponent exact run_indices (0-based; - = no successes, ? = missing result):')
        for g in selected:
            components = sorted(set().union(*(m['groups'].get(g, set()) for m in methods.values())))
            table(['Component', *methods], [[c, *[(','.join(map(str, sorted(m['exact'][c]))) or '-') + (' ?' if any((c,r) not in m['present'] for r in m['runs']) else '') if c in m['components'] else 'not evaluated' for m in methods.values()]] for c in components])
    if args.equations:
        print('\nFirst recorded exact witness per component (not a new verification):')
        for name, m in methods.items():
            seen = set()
            for r in m['records']:
                ds = r['dataset_name']
                if ds.split(':')[1] not in selected or ds in seen or r.get('gt_match_score') != 1:
                    continue
                seen.add(ds)
                print(f"{name} | {ds} | run_index={r['run_index']} seed={r['seed']}")
                print(f"  {r.get('gt_matched_equation') or r.get('best_equation')}")
                print(f"  {r['source']}")


if __name__ == '__main__':
    main()
