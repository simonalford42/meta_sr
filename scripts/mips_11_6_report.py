#!/usr/bin/env python3
"""Write results/mips-11-6/report.md: MIPS training-set-size ablation (11/6/26).

Every method is scored on all 51 MIPS scalar problems (17 tasks), 10 seeds,
eval seed 192, 1e6 evals / 500 s per fit. A fit counts as solved when
gt_match_score == 1. Results are merged across a run's final_eval/eval_* dirs
because cached fits (train-split overlap) are not re-written to the all-51 dir.

Run: python scripts/mips_11_6_report.py
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/mips-11-6/report.md'
ALL51 = ROOT / 'splits/mips_sr_targets_plus_refined.txt'
SOLVED_TASKS = {'rnn_abs_value_numerical', 'rnn_abs_value_of_diff_numerical',
                'rnn_add_mod_3_numerical', 'rnn_parity_last2_numerical',
                'rnn_unique2_numerical'}

METHODS = [
    # name, train split (None = no training), evolution run dir, final eval dir
    ('Base PySR', None, None, 'runs/709714/final_eval_baseline_10seed'),
    ('Train on all 51 (official 709714)', 'splits/mips_sr_targets_plus_refined.txt',
     'runs/709714', 'runs/709714/final_eval'),
    ('Train on solved 12', 'splits/mips_solved12.txt',
     'runs/mips-11-6-solved12', 'runs/mips-11-6-solved12/final_eval'),
    ('Train on solved 12 + 8/39 unsolved', 'splits/mips_solved12_unsolved8.txt',
     'runs/mips-11-6-solved12-unsolved8', 'runs/mips-11-6-solved12-unsolved8/final_eval'),
    ('Train on solved 12 + 20/39 unsolved', 'splits/mips_solved12_unsolved20.txt',
     'runs/mips-11-6-solved12-unsolved20', 'runs/mips-11-6-solved12-unsolved20/final_eval'),
]


def split(path):
    return (ROOT / path).read_text().split()


def load_fits(eval_root):
    """{(dataset, run_index): gt_match_score==1} merged over all eval_* dirs."""
    fits = {}
    for d in sorted((ROOT / eval_root).glob('slurm_pysr/eval_*')):
        if not (d / 'tasks.json').exists():
            continue
        tasks = json.loads((d / 'tasks.json').read_text())
        for i, t in enumerate(tasks):
            f = d / f'results/task_{i:06d}.json'
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            key = (r['dataset_name'], r['run_index'])
            assert key == (t['dataset_name'], t['run_index']), f
            fits.setdefault(key, r.get('gt_match_score') == 1)
    return fits


def score(fits, datasets, runs):
    present = [(d, r) for d in datasets for r in runs if (d, r) in fits]
    exact = sum(fits[k] for k in present)
    return exact, len(datasets) * len(runs), len(present)


def main():
    all51 = split(ALL51)
    solved12 = [d for d in all51 if d.split(':')[1] in SOLVED_TASKS]
    unsolved39 = [d for d in all51 if d.split(':')[1] not in SOLVED_TASKS]
    runs = range(10)
    rows, notes = [], []
    for name, train_split, run_dir, eval_root in METHODS:
        train = set(split(train_split)) if train_split else set()
        heldout = [d for d in unsolved39 if d not in train]
        if not (ROOT / eval_root).exists():
            rows.append([name, len(train) or '-', *['pending'] * 6])
            continue
        fits = load_fits(eval_root)
        cells = []
        incomplete = False
        for subset in (all51, solved12, unsolved39, heldout):
            if not subset:
                cells.append('-')
                continue
            e, n, p = score(fits, subset, runs)
            incomplete |= p < n
            cells.append(f'{100 * e / n:.1f}% ({e}/{n})')
        any_seed = {d for d in all51 if any(fits.get((d, r)) for r in runs)}
        groups = defaultdict(list)
        for d in all51:
            groups[d.split(':')[1]].append(d)
        tasks = {g for g, cs in groups.items() if all(c in any_seed for c in cs)}
        new_tasks = tasks - SOLVED_TASKS
        cells += [f'{len(any_seed)}/51', f'{len(tasks)}/17 ({len(new_tasks)}/12 unsolved)']
        if incomplete:
            notes.append(f'{name}: some fits missing in {eval_root}; percentages count missing as unsolved.')
        rows.append([name, len(train) or '-', *cells])

    headers = ['Method', '# train problems', 'All 51 fits', 'Solved-12 fits',
               'Unsolved-39 fits', 'Held-out unsolved fits', 'Problems solved (any seed)',
               'Tasks solved (any seed per component)']
    lines = ['# MIPS training-set-size ablation (11/6/26)', '',
             'All rows: evaluated on all 51 MIPS scalar problems (17 tasks), 10 seeds, eval seed 192, '
             '1e6 evals and 500 s per fit. Fit = exact ground-truth match (`gt_match_score == 1`). '
             '"Held-out unsolved" = unsolved problems not in that run\'s train split. '
             '"Solved 12" = the 12 problems of the 5 paper-solved tasks (abs_value, abs_value_of_diff, '
             'add_mod_3, parity_last2, unique2). Unsolved subsets are a nested random sample (seed 42), '
             'see `scripts/make_mips_subset_splits.py`. Evolution config is identical to 709714 apart '
             'from `--split` and `--test-split`.', '',
             '| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    lines += ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]
    if notes:
        lines += ['', *[f'- {n}' for n in notes]]
    lines += ['', '## Paths', '', '| Method | Train split | Evolution run | Final eval (all 51) |', '|---|---|---|---|']
    for name, train_split, run_dir, eval_root in METHODS:
        lines.append(f'| {name} | `{train_split or "-"}` | `{run_dir or "-"}` | `{eval_root}` |')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n')
    print(OUT.read_text())


if __name__ == '__main__':
    main()
