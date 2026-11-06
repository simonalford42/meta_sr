#!/usr/bin/env python3
"""Plot matched train-selection minus fresh-seed train scores through generation 20."""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/reevaluation_ablations20'
RUNS = {
    '373691': 'n=1, no reevaluation',
    '373692': 'n=3, no reevaluation',
    '373693': 'n=10, no reevaluation',
    '373694': 'Population reevaluation: 1 → 3',
    '373695': 'TTTS, top-k (budget 20)',
    '709715': '709715 (first 20 generations)',
}
PATTERN = re.compile(
    r'\[train reeval\] gen (\d+) (.+?): reeval GT match rate=([\d.]+) '
    r'\(live=([\d.]+), winners_curse=([+\-\d.]+)\)')


def main():
    OUT.mkdir(exist_ok=True)
    records = []
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.axhline(0, color='#888888', lw=1, ls='--', zorder=0)
    for i, (run, label) in enumerate(RUNS.items()):
        series = []
        for gen, bundle, fresh, live, gap in PATTERN.findall((ROOT / 'runs' / run / 'run.log').read_text()):
            gen = int(gen)
            if gen > 20:
                continue
            fresh, live, gap = map(float, (fresh, live, gap))
            assert abs(live - fresh - gap) < 0.00011
            series.append(dict(run=run, label=label, generation=gen, train=live,
                               train_reevaluation=fresh, winners_curse_pp=100*gap, bundle=bundle))
        series.sort(key=lambda r:r['generation'])
        assert len({r['generation'] for r in series}) == len(series)
        assert series[0]['generation'] == 0 and series[-1]['generation'] == 20
        records.extend(series)
        ax.plot([r['generation'] for r in series], [r['winners_curse_pp'] for r in series],
                label=label, color='black' if run == '709715' else f'C{i}',
                marker=['o','s','^','D','v','P'][i], ms=4,
                lw=2.3 if run == '709715' else 1.7, alpha=.9)
    ax.set(xlabel='Generation', ylabel='Winner’s curse (percentage points)',
           xlim=(-.3,20.3), title='Winner’s curse over evolution\nTrain selection score − fresh-seed train reevaluation')
    ax.set_xticks(range(0,21,2))
    ax.grid(alpha=.18)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5,.045),
               ncol=3, frameon=False, fontsize=10)
    fig.text(.5,.018,'Markers show completed evaluations; lines connect available generations. No smoothing.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0,.17,1,1))
    for ext in ['png','pdf']:
        fig.savefig(OUT / f'winners_curse.{ext}', dpi=180)
    plt.close(fig)
    with (OUT / 'winners_curse.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n')
        w.writeheader(); w.writerows(records)
    for run in RUNS:
        r = [r for r in records if r['run']==run]
        print(run, len(r), 'points; endpoint',r[-1]['winners_curse_pp'])


if __name__ == '__main__':
    main()
