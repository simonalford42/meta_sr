#!/usr/bin/env python3
"""Plot cached fresh-seed population diagnostics from one or more evolution runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='Run directories or population_reeval.json files')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent/'population_reevaluation')
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for run in args.runs:
        path = run/'population_reeval.json' if run.is_dir() else run
        data = json.loads(path.read_text())
        rows = sorted(data['generations'], key=lambda r:r['generation'])
        if not rows:
            raise ValueError(f'No completed diagnostic generations in {path}')
        label = f"{path.parent.name} ({data['context']['n_runs']} fresh seeds)"
        for ax, key in zip(axes, ['avg_score', 'expected_parent_score']):
            ax.plot([r['generation'] for r in rows], [r[key] for r in rows], marker='o',
                    markersize=3, label=label)
    for ax, title in zip(axes, ['Population mean', 'Expected parent fitness']):
        ax.set(title=title, xlabel='Generation', ylabel='Fresh-seed train fitness')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=min(3, len(args.runs)),
               frameon=False, fontsize=9)
    fig.suptitle('Population fitness on independent seeds')
    fig.tight_layout(rect=(0,.13,1,.95))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(args.output_dir/f'population_reevaluation.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
