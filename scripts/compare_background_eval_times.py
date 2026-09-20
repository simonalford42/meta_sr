#!/usr/bin/env python3
"""Read saved background reevaluation batches; never submit evaluations."""
import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/373692-vs-709715-background-timing'


def first_task(path):
    with path.open() as f:
        text = ''
        while block := f.read(65536):
            text += block
            try:
                return json.JSONDecoder().raw_decode(text.lstrip()[1:].lstrip())[0]
            except json.JSONDecodeError:
                pass
    raise ValueError(path)


def main():
    OUT.mkdir(exist_ok=True)
    records, batches = [], []
    for run in ['709715', '373692']:
        folder = ROOT / 'runs' / run
        log = (folder / 'run.log').read_text()
        old_positions = {'train_reeval': 0, 'validation': 0}
        old_logs = {split: re.findall(r'\[' + tag + r'\] gen (\d+) .*?(?:avg|reeval) GT match rate=([\d.]+)', log)
                    for split, tag in [('train_reeval', 'train reeval'), ('validation', 'val eval')]}
        for batch in sorted((folder / 'slurm_pysr').glob('eval_*')):
            if not (batch / 'tasks.json').exists():
                continue
            task = first_task(batch / 'tasks.json')
            index = task['run_index']
            if run == '709715':
                if index == 100000:
                    split = 'train_reeval'
                elif task['dataset_name'] == 'feynman_I_8_14':
                    split = 'validation'
                else:
                    continue
            else:
                if not 100000 <= index < 300000:
                    continue
                split = 'train_reeval' if index < 200000 else 'validation'
            # Identification also occupies the train seed band, but has multiple configs.
            with (batch / 'tasks.json').open() as f:
                tasks = json.load(f)
            if {t['config_id'] for t in tasks} != {0}:
                continue
            if run == '709715':
                gen, expected_score = old_logs[split][old_positions[split]]
                gen = int(gen)
                old_positions[split] += 1
            else:
                gen = (index - (100000 if split == 'train_reeval' else 200000)) // 10
                if str(gen) not in dict(old_logs[split]):
                    continue  # No completed score in the plotted/logged trajectory.
                expected_score = dict(old_logs[split])[str(gen)]
            assert task['pysr_kwargs']['max_evals'] == 1000000
            assert task['pysr_kwargs']['timeout_in_seconds'] == (500 if split == 'train_reeval' else 1500)
            assert task['pysr_wall_limit'] == (600 if split == 'train_reeval' else 1800)
            combined = batch / 'combined.json'
            if not combined.exists():
                batches.append(dict(run=run, split=split, generation=gen, batch=batch.name, tasks=len(tasks), results=0))
                continue
            results = json.loads(combined.read_text())
            assert len(results) == 200, (run, batch, len(results))
            score = sum(r.get('gt_match_score') or 0 for r in results) / len(results)
            assert abs(score-float(expected_score)) < 0.00006, (run, batch, gen, score, expected_score)
            keys = [(r['config_id'], r['dataset_name'], r['run_index']) for r in results]
            assert len(keys) == len(set(keys)), (run, batch, 'duplicate results')
            assert all((r['run_index'] - index) in range(10) for r in results)
            batches.append(dict(run=run, split=split, generation=gen, batch=batch.name, tasks=len(tasks), results=len(results)))
            for r in results:
                records.append(dict(run=run, split=split, generation=gen, batch=batch.name,
                    dataset=r['dataset_name'], run_index=r['run_index'], runtime_seconds=r.get('runtime_seconds'),
                    search_runtime_seconds=r.get('search_runtime_seconds'), error=bool(r.get('error')),
                    timed_out=bool(r.get('timed_out'))))
            print(run, split, gen, len(results), flush=True)
        if run == '709715':
            assert all(old_positions[k] == len(old_logs[k]) for k in old_positions)
    for name, rows in [('task_times.csv', records), ('batches.csv', batches)]:
        with (OUT / name).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    summaries = []
    for run in ['709715', '373692']:
        for split in ['train_reeval', 'validation']:
            group = [r for r in records if r['run'] == run and r['split'] == split]
            for scope in ['all', 'generations_0_20', 'generation_20', 'latest']:
                latest = max(r['generation'] for r in group)
                rows = [r for r in group if scope == 'all' or
                        (scope == 'generations_0_20' and r['generation'] <= 20) or
                        (scope == 'generation_20' and r['generation'] == 20) or
                        (scope == 'latest' and r['generation'] == latest)]
                values = [r['runtime_seconds'] for r in rows if r['runtime_seconds'] is not None]
                summaries.append(dict(run=run, split=split, scope=scope, generations=sorted({r['generation'] for r in rows}),
                    tasks=len(rows), timed_tasks=len(values), mean_seconds=statistics.mean(values) if values else None,
                    median_seconds=statistics.median(values) if values else None,
                    errors=sum(r['error'] for r in rows), timeouts=sum(r['timed_out'] for r in rows),
                    search_timed_tasks=sum(r['search_runtime_seconds'] is not None for r in rows)))
    (OUT / 'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
