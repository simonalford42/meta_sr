#!/usr/bin/env python3
"""Compare n=1 and population 1→3 train and fresh-seed reevaluation trajectories."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent / 'reevaluation_ablations20'


def main():
    with (OUT / 'scores.csv').open() as f:
        records = list(csv.DictReader(f))
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for run, label, color in [('373691', 'n=1, no reevaluation', 'C0'),
                              ('373694', 'Population 1 → 3', 'C3')]:
        for split, style, marker in [('Train', '-', 'o'), ('Train reevaluation', '--', 's')]:
            rows = sorted([r for r in records if r['run'] == run and r['split'] == split],
                          key=lambda r: int(r['generation']))
            gens = [int(r['generation']) for r in rows]
            assert len(gens) == len(set(gens)) and gens[0] == 0 and gens[-1] == 20
            ax.plot(gens, [float(r['score']) for r in rows], color=color,
                    linestyle=style, marker=marker, markersize=4, linewidth=2,
                    label=f'{label} — {split.lower()}')
    ax.set(xlabel='Generation', ylabel='GT match rate', xlim=(-.3, 20.3), ylim=(.4,.85),
           title='n=1 vs. population reevaluation 1 → 3')
    ax.set_xticks(range(0,21,2))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center',
               bbox_to_anchor=(.5,.055), ncol=2, frameon=False, fontsize=10)
    fig.text(.5,.018,'Solid: train selection score. Dashed: train reevaluation on 10 fresh seeds.\n'
             'Markers show observed evaluations; lines connect available generations.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0,.2,1,1))
    for ext in ['png','pdf']:
        fig.savefig(OUT / f'n1_vs_population_reeval.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
