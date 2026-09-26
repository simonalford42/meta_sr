#!/usr/bin/env python3
"""Plot fresh seed-level recoveries by evolution generation, including partial status."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]


def summarize(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    batch = json.loads((run / 'batch.json').read_text())
    results = Path(batch['batch_dir']) / 'results'
    rows = []
    for config_id, generation in enumerate(manifest['generations']):
        for split, worlds in [('Train', manifest['train_worlds']),
                              ('Held out', [w for w in manifest['worlds'] if w not in manifest['train_worlds']])]:
            complete = errors = solved = pending = 0
            for world in worlds:
                for run_index in range(len(manifest['seeds'])):
                    index = (config_id * len(manifest['worlds']) + manifest['worlds'].index(world)) * len(manifest['seeds']) + run_index
                    path = results / f'task_{index:06d}.json'
                    if not path.exists():
                        pending += 1
                        continue
                    result = json.loads(path.read_text())
                    assert (result['config_id'], result['dataset_name'], result['run_index']) == (config_id, world, run_index)
                    values = [float(p['test_nrmse']) for p in result.get('pareto_frontier') or []
                              if p.get('test_nrmse') is not None and math.isfinite(float(p['test_nrmse']))]
                    if result.get('error') or not values:
                        errors += 1
                        continue
                    complete += 1
                    solved += min(values) <= manifest['threshold']
            rows.append(dict(generation=generation['generation'], split=split, solved=solved,
                             complete=complete, errors=errors, pending=pending,
                             expected=len(worlds) * len(manifest['seeds'])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'figures/neuron_708907_generations.pdf')
    args = parser.parse_args()
    rows = summarize(args.run)
    with (args.run / 'generation_summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    for ax, split, color in zip(axes, ['Train', 'Held out'], ['#1F77B4', '#C33D3D']):
        selected = [r for r in rows if r['split'] == split]
        # Incomplete generations are gaps, never silently counted as failures.
        ax.plot([r['generation'] for r in selected],
                [r['solved'] if r['complete'] == r['expected'] else float('nan') for r in selected],
                'o-', color=color, markersize=4)
        ax.set(title=f'{split} ({selected[0]["expected"]} fits)', xlabel='Generation',
               ylabel='Number solved', ylim=(-0.5, selected[0]['expected'] + 0.5),
               xticks=[1, 3, 5, 7, 9, 11, 13, 15])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
    completed = sum(r['complete'] for r in rows)
    fig.suptitle(f'NeuronBench: test NRMSE ≤ 10⁻⁶ ({completed}/450 fits complete)', fontsize=11)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print('generation split solved complete errors pending')
    for r in rows:
        print(r['generation'], r['split'], r['solved'], r['complete'], r['errors'], r['pending'])
    print(f'Figure: {args.output}')


if __name__ == '__main__':
    main()
