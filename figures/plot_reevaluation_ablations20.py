#!/usr/bin/env python3
"""Plot September 17 reevaluation ablations from local, completed run logs."""
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/reevaluation_ablations20'
RUNS = {
    '373691': '1 run, no reevaluation',
    '373692': '3 runs, no reevaluation',
    '373693': '10 runs, no reevaluation',
    '373694': 'Population reevaluation: 1 → 3',
    '373695': 'TTTS, budget 20 (top-k)',
}
SPLITS = ['Train', 'Train reevaluation', 'Validation']


def main():
    OUT.mkdir(exist_ok=True)
    rows, metadata, summaries = [], [], []
    for run, label in RUNS.items():
        folder = ROOT / 'runs' / run
        # Config precedes the multi-GB population history; avoid loading that history.
        with (folder / 'run_data.json').open() as f:
            prefix = f.read(24000).split('  "baseline":')[0]
        meta = json.loads(prefix.rstrip().rstrip(',') + '}')
        metadata.append({'run': run, 'label': label, **meta})
        log = (folder / 'run.log').read_text()
        assert 'Train reeval: enabled (10 runs/bundle' in log
        assert '20 datasets, 10 runs/bundle' in log
        for p in folder.glob('best_bundles/best_gen*.jl'):
            gen = int(re.fullmatch(r'best_gen(\d+)\.jl', p.name)[1])
            score = float(re.search(r'^# Bundle score: (.+)$', p.read_text(), re.M)[1])
            rows.append(dict(run=run, label=label, split='Train', generation=gen, score=score))
        initial = float(re.search(r'Best initial bundle: .*?\(score: ([\d.]+)\)', log)[1])
        rows.append(dict(run=run, label=label, split='Train', generation=0, score=initial))
        patterns = {
            'Train reevaluation': r'\[train reeval\] gen (\d+) .*?: reeval GT match rate=([\d.]+)',
            'Validation': r'\[val eval\] gen (\d+) .*?: avg GT match rate=([\d.]+)',
        }
        for split, pattern in patterns.items():
            for gen, score in re.findall(pattern, log):
                rows.append(dict(run=run, label=label, split=split, generation=int(gen), score=float(score)))
        summary = {'run': run, 'label': label}
        for split in SPLITS:
            series = sorted([r for r in rows if r['run'] == run and r['split'] == split], key=lambda r:r['generation'])
            assert series and len({r['generation'] for r in series}) == len(series)
            assert all(0 <= r['generation'] <= 20 and 0 <= r['score'] <= 1 for r in series)
            if split == 'Train':
                assert [r['generation'] for r in series] == list(range(21))
            summary[split] = series[-1]['score']
            summary[split + ' generation'] = series[-1]['generation']
            summary[split + ' points'] = len(series)
        summaries.append(summary)
    rows.sort(key=lambda r:(r['run'], SPLITS.index(r['split']), r['generation']))
    for name, data in [('scores.csv', rows), ('final_scores.csv', summaries)]:
        with (OUT / name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    def finish(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    def style(ax):
        ax.set(xlim=(-.3,20.3), ylim=(.25,1.0), xlabel='Generation', ylabel='GT match rate')
        ax.set_xticks(range(0,21,5))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
    fig, axes = plt.subplots(1,3,figsize=(14,4.6),sharey=True)
    for ax, split in zip(axes,SPLITS):
        for i,(run,label) in enumerate(RUNS.items()):
            series = [r for r in rows if r['run']==run and r['split']==split]
            ax.plot([r['generation'] for r in series], [r['score'] for r in series], marker='o', ms=3, color=f'C{i}',label=label)
        ax.set_title(split)
        style(ax)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Reevaluation ablations · 20 generations · best2')
    fig.tight_layout(rect=(0,.16,1,.95))
    finish(fig,'comparison')
    fig, axes = plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    for ax,(run,label) in zip(axes.flat,RUNS.items()):
        for i,split in enumerate(SPLITS):
            series = [r for r in rows if r['run']==run and r['split']==split]
            ax.plot([r['generation'] for r in series],[r['score'] for r in series],marker='o',ms=3,label=split,color=f'C{i}')
        style(ax)
        ax.set_title(f'{label}\nRun {run}')
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',bbox_to_anchor=(.5,.65),frameon=False)
    axes.flat[-1].text(.5,.25,'Markers = observed evaluations\nLines connect available generations\nFresh-seed reevaluation: 10 runs\nValidation: 10 runs',ha='center',transform=axes.flat[-1].transAxes)
    fig.tight_layout()
    finish(fig,'per_run')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
