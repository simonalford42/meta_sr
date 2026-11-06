"""Read-only comparison of 941340 against 709715 SRBench ground truth."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    log = (ROOT / 'out/941340.out').read_text()
    errors = sum(int(n) for n in re.findall(r'WARNING: (\d+)/\d+ PySR tasks returned errors', log))
    tasks = sum(int(n) for n in re.findall(r'PySR SLURM eval: (\d+) tasks', log))
    print(f'Logged errors: {errors}/{tasks} ({errors/tasks:.2%}); includes repeated evaluations/cache hits')
    for batch in ['eval_0000', 'eval_0001']:
        directory = ROOT / 'runs/941340/slurm_pysr' / batch
        specs = json.loads((directory / 'tasks.json').read_text())
        results = json.loads((directory / 'combined.json').read_text())
        assert len(specs) == len(results)
        for spec, result in zip(specs, results):
            assert (spec['dataset_name'], spec['run_index']) == (result['dataset_name'], result['run_index'])
        for noise in [0, .001, .01, .1]:
            rows = [r for t, r in zip(specs, results) if t['target_noise'] == noise]
            print(batch, noise, 'GT', sum(r.get('gt_match_score') or 0 for r in rows) / len(rows),
                  'errors', sum(bool(r.get('error')) for r in rows), 'n', len(rows))
    train = set((ROOT / 'splits/barely_unsolvable.txt').read_text().split())
    for folder in ['srbench_gt_90s', 'srbench_gt_10seed_merged', 'srbench_gt_15m_single']:
        data = json.loads((ROOT / 'runs/709715' / folder / 'srbench_full_results.json').read_text())
        for label, subset in [('full', None), ('train', train)]:
            for noise in [0, .001, .01, .1]:
                rows = [r for r in data['results'].values()
                        if r['noise'] == noise and (subset is None or r['dataset'] in subset)]
                print(folder, label, noise, sum(bool(r['solved']) for r in rows) / len(rows), 'n', len(rows))


if __name__ == '__main__':
    main()
