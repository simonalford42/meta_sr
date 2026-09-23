"""Plot the Sep 21 90-second ablations. Run with --refresh to fetch source data.

Offline rerender: python figures/plot_90s_reevaluation_ablations.py
Scores are best-candidate train diagnostics (10 fresh seeds), not population means.
Missing diagnostics are left missing. Lines connect observed points only; no tails
are extrapolated. Evaluation counts are generation-end evolution eval_idx values,
including selection reevaluations but excluding baseline/diagnostic evaluations.
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/reevaluation_ablations_90s'
RUNS = [
    ('n1', 1, '671962', '7wdnkqss'), ('n1', 2, '750244', 'kwxib4uc'),
    ('n1-reeval', 1, '671963', 'hbonk3al'), ('n1-reeval', 2, '750245', 'iuqo0leg'),
    ('n1-TTTS', 1, '750241', 'ubg1ir57'), ('n1-TTTS', 2, '750250', '2sgnyvt8'),
    ('n3', 1, '671964', 'm642yus3'), ('n3', 2, '750247', '20af68tv'),
    ('n3-reeval', 1, '750239', 'igwan6cy'), ('n3-reeval', 2, '750248', 'sqk1poaq'),
    ('n3-TTTS', 1, '750242', 'egg7h6w2'), ('n3-TTTS', 2, '750251', 'apo0crji'),
]
FAILED = {'750247', '750248', '750250', '750251'}  # sacct confirmed 2026-09-23
PATTERN = re.compile(r'\[train reeval\] gen (\d+) (.+?): reeval GT match rate=([\d.]+) \(live=([\d.]+), winners_curse=([+\-\d.]+)\)')


def refresh():
    import wandb
    api = wandb.Api(timeout=90)
    records = []
    for method, seed, job, run_id in RUNS:
        run = api.run(f'simon-alford/meta-sr/{run_id}')
        history = [r for r in run.scan_history(keys=['generation', 'eval_idx', 'best_score'], page_size=1000)
                   if all(r.get(k) is not None for k in ['generation', 'eval_idx', 'best_score'])]
        counts = {int(r['generation']): int(r['eval_idx']) for r in history}
        log = (ROOT / 'runs' / job / 'run.log').read_text()
        points = []
        for g, bundle, reeval, live, curse in PATTERN.findall(log):
            g = int(g)
            assert g in counts, (job, g, 'missing generation count')
            points.append(dict(generation=g, eval_idx=counts[g], bundle=bundle,
                               train_score=float(live), train_reeval_score=float(reeval),
                               winners_curse=float(live)-float(reeval)))
        assert len({p['generation'] for p in points}) == len(points), job
        points.sort(key=lambda p: p['generation'])
        assert run.config['seed'] == seed
        assert run.config['generations'] == 15 and run.config['timeout'] == 90
        assert run.config['val_n_runs'] == 10
        records.append(dict(method=method, seed=seed, job_id=job, wandb_id=run_id,
                            status='FAILED' if job in FAILED else 'COMPLETED',
                            failure='OpenRouter HTTP 402: insufficient credits/key limit' if job in FAILED else None,
                            last_completed_generation=max(counts),
                            config={k:run.config.get(k) for k in ['seed','n_runs','reeval','n_reevals','reeval_budget','population_type','population','offspring','generations','timeout','budget_mode','val_n_runs','population_reeval_runs']},
                            generation_history=history, points=points))
        print(job, method, seed, max(counts), len(points), flush=True)
    (OUT / 'data.json').write_text(json.dumps(records, indent=2)+'\n')


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    records = json.loads((OUT / 'data.json').read_text())
    with (OUT / 'scores.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['method','seed','job_id','status','generation','eval_idx','bundle','train_score','train_reeval_score','winners_curse'])
        writer.writeheader()
        for r in records:
            for p in r['points']:
                writer.writerow({**{k:r[k] for k in ['method','seed','job_id','status']}, **p})
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    colors = ['#0072B2','#D55E00','#009E73']
    for row, group in enumerate(['n1','n3']):
        for col, xkey in enumerate(['generation','eval_idx']):
            draw(axes[row,col], records, group, xkey, 'train_reeval_score', colors)
            axes[row,col].set_title(f'{group}: train reevaluation vs {"generation" if col == 0 else "evaluations"}')
            axes[row,col].set_ylabel('Reevaluated train score')
            axes[row,col].set_ylim(0.30,0.90)
        draw(axes[2,row], records, group, 'generation','winners_curse', colors)
        axes[2,row].set_title(f'{group}: winner’s curse')
        axes[2,row].set_ylabel('Train score − reevaluated train score')
        axes[2,row].set_ylim(-0.06,0.36)
        axes[2,row].axhline(0,color='0.4',lw=.8)
    fig.suptitle('90-second PySR ablations · best-candidate train reevaluation',fontsize=17,y=.985)
    handles = [Line2D([],[],color='0.25',ls='-',label='Seed 1'),Line2D([],[],color='0.25',ls='--',label='Seed 2'),Line2D([],[],color='0.25',marker='x',ls='none',ms=8,label='Failed run: last observed diagnostic')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.96),ncol=3,frameon=False)
    fig.text(.06,.035,'Each point: best candidate reevaluated on training tasks with 10 fresh seeds. Lines join observed points; missing diagnostics are not filled.\nEvaluations = cumulative evolution seed-runs (including selection reevaluations; excluding diagnostics). One seed-run covers the train task set.\n8/12 runs completed 15 generations; 4 seed-2 runs failed (OpenRouter HTTP 402). n3-reeval uses population reevaluation, 3 → 10 seeds.',fontsize=9,va='bottom')
    fig.tight_layout(rect=(0,.105,1,.925),h_pad=2.2,w_pad=2.2)
    fig.savefig(OUT / 'train_reevaluation_six_panels.pdf')
    plt.close(fig)
    plot_combined(records)


def plot_combined(records):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#555555']
    methods = ['n1', 'n1-reeval', 'n1-TTTS', 'n3', 'n3-reeval', 'n3-TTTS']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, metric, title in zip(axes, ['train_reeval_score', 'winners_curse'],
                                 ['Reevaluated train score', 'Winner’s curse']):
        for method, color in zip(methods, colors):
            for r in records:
                if r['method'] != method:
                    continue
                p = r['points']
                ax.plot([v['eval_idx'] for v in p], [v[metric] for v in p],
                        color=color, ls='-' if r['seed'] == 1 else '--',
                        marker='o', ms=3, lw=1.5, alpha=1 if r['seed'] == 1 else .75,
                        label=method if r['seed'] == 1 else None)
                if r['status'] == 'FAILED':
                    ax.plot(p[-1]['eval_idx'], p[-1][metric], color=color, marker='x', ms=9, mew=2)
        ax.set_title(title)
        ax.set_xlabel('Cumulative evolution evaluations (seed-runs)')
        ax.set_ylabel('Reevaluated train score' if metric == 'train_reeval_score'
                      else 'Train score − reevaluated train score')
        ax.set_ylim((.30,.90) if metric == 'train_reeval_score' else (-.06,.36))
        ax.set_xlim(0,930)
        ax.grid(alpha=.2)
        ax.legend(ncol=3,fontsize=8,frameon=False,loc='upper left')
        if metric == 'winners_curse':
            ax.axhline(0,color='0.4',lw=.8)
    fig.suptitle('All six 90-second PySR ablations · evaluation-count comparison',fontsize=16,y=.98)
    handles = [Line2D([],[],color='0.25',ls='-',label='Seed 1'),
               Line2D([],[],color='0.25',ls='--',label='Seed 2'),
               Line2D([],[],color='0.25',marker='x',ls='none',ms=8,label='Failed run: last observed diagnostic')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False)
    fig.text(.06,.03,'10 fresh seeds per best-candidate train diagnostic; missing diagnostics are not filled. Four seed-2 runs failed (OpenRouter HTTP 402).\nEvaluation counts include selection reevaluations and exclude diagnostics. One seed-run covers the training task set.',fontsize=9)
    fig.tight_layout(rect=(0,.105,1,.85))
    fig.savefig(OUT / 'all_methods_eval_axis.pdf')
    plt.close(fig)


def draw(ax, records, group, xkey, ykey, colors):
    methods = [group,group+'-reeval',group+'-TTTS']
    for method, color in zip(methods, colors):
        for r in records:
            if r['method'] != method:
                continue
            points = r['points']
            ax.plot([p[xkey] for p in points],[p[ykey] for p in points],
                    color=color,ls='-' if r['seed']==1 else '--',marker='o',ms=3,lw=1.6,
                    label=method if r['seed']==1 else None,alpha=1 if r['seed']==1 else .75)
            if r['status']=='FAILED':
                ax.plot(points[-1][xkey],points[-1][ykey],color=color,marker='x',ms=9,mew=2)
    ax.set_xlabel('Generation' if xkey=='generation' else 'Cumulative evolution evaluations (seed-runs)')
    if xkey=='generation':
        ax.set_xlim(-.35,15.35)
        ax.set_xticks(range(0,16,3))
    ax.grid(alpha=.2)
    ax.legend(loc='upper left',ncol=3,fontsize=8,frameon=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.refresh:
        refresh()
    plot()
