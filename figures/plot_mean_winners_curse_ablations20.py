#!/usr/bin/env python3
"""Average observed winner's-curse points over generations 0–20, equally weighted."""
import csv
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'reevaluation_ablations20'
LABELS = {
    '373691': 'n=1', '373692': 'n=3', '373693': 'n=10',
    '373694': 'Population\n1 → 3', '373695': 'TTTS\ntop-k',
    '709715': '709715\n(first 20 gens)',
}


def main():
    with (OUT / 'winners_curse.csv').open() as f:
        points = list(csv.DictReader(f))
    rows = []
    for run, label in LABELS.items():
        subset = [r for r in points if r['run'] == run and 0 <= int(r['generation']) <= 20]
        generations = [int(r['generation']) for r in subset]
        assert len(generations) == len(set(generations)) and min(generations) == 0 and max(generations) == 20
        rows.append({'run': run, 'label': label.replace('\n', ' '),
                     'mean_winners_curse_pp': statistics.mean(float(r['winners_curse_pp']) for r in subset),
                     'observed_generations': len(subset),
                     'missing_generations': ','.join(map(str, sorted(set(range(21))-set(generations))))})
    with (OUT / 'mean_winners_curse.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    means = [r['mean_winners_curse_pp'] for r in rows]
    bars = ax.bar(range(6), means, color=[f'C{i}' for i in range(5)]+['#333333'], width=.65, zorder=3)
    ax.bar_label(bars, labels=[f'{v:.1f} pp' for v in means], padding=5, fontsize=12)
    ax.set_xticks(range(6), list(LABELS.values()))
    ax.set(ylabel='Mean winner’s curse (percentage points)',
           title='Average winner’s curse over generations 0–20', ylim=(0,max(means)*1.2))
    ax.grid(axis='y', alpha=.2, zorder=0)
    fig.text(.5,.04,'Equal weight per observed generation; missing evaluations are omitted.\n'
             'Observed generations per bar: '+', '.join(str(r['observed_generations']) for r in rows)+'. Includes generation 0.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0,.13,1,1))
    for ext in ['png','pdf']:
        fig.savefig(OUT / f'mean_winners_curse.{ext}', dpi=180)
    plt.close(fig)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
