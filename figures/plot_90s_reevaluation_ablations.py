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
    ('n1-TTTS', 1, '750241', 'ubg1ir57'), ('n1-TTTS', 2, '980600', 'rpgjrcdd'),
    ('n3', 1, '671964', 'm642yus3'), ('n3', 2, '980597', '0g6wvmco'),
    ('n3-reeval', 1, '750239', 'igwan6cy'), ('n3-reeval', 2, '980598', 'k3xgq41q'),
    ('n3-TTTS', 1, '750242', 'egg7h6w2'), ('n3-TTTS', 2, '980599', 'skg0nsbf'),
    ('n10', 1, '64604', 'sifnivw6'), ('n10', 2, '64605', 'wftnv7sd'),
    ('n1', 3, '64606', '4e15t4ig'),
]
SEED_STYLES = {1: '-', 2: '--', 3: ':'}
RESUMED = {'980597': ('750247', '20af68tv')}
PATTERN = re.compile(r'\[train reeval\] gen (\d+) (.+?): reeval GT match rate=([\d.]+) \(live=([\d.]+), winners_curse=([+\-\d.]+)\)')


def refresh():
    import wandb
    api = wandb.Api(timeout=90)
    def read_source(job, run_id):
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
        return run, history, points

    records = []
    for method, seed, job, run_id in RUNS:
        folder = ROOT / 'runs' / job
        if not (folder / 'final_eval_summary.json').exists():
            print(job, method, seed, 'not complete: excluded', flush=True)
            continue
        run, history, points = read_source(job, run_id)
        sources = [dict(job_id=job, wandb_id=run_id)]
        if job in RESUMED:
            parent_job, parent_id = RESUMED[job]
            _, prior_history, prior_points = read_source(parent_job, parent_id)
            boundary = min(int(h['generation']) for h in history)
            prior_count = max(h['eval_idx'] for h in prior_history if h['generation'] == boundary)
            resumed_count = max(h['eval_idx'] for h in history if h['generation'] == boundary)
            assert prior_count == resumed_count, (job, prior_count, resumed_count)
            # Prefer fresh resumed diagnostics if both sources have the boundary.
            merged = {p['generation']: p for p in prior_points if p['generation'] <= boundary}
            merged.update({p['generation']: p for p in points})
            points = list(merged.values())
            history = [h for h in prior_history if h['generation'] < boundary] + history
            sources.insert(0, dict(job_id=parent_job, wandb_id=parent_id))
        points.sort(key=lambda p: p['generation'])
        assert run.config['seed'] == seed
        assert run.config['generations'] == (3 if job in RESUMED else 15)
        assert run.config['timeout'] == 90 and run.config['val_n_runs'] == 10
        assert run.config['population_type'] == 'topk'
        assert run.config['population'] == run.config['offspring'] == 10
        assert run.config['n_runs'] == int(method.split('-')[0][1:])
        assert run.config['reeval'] == ('TTTS' if method.endswith('TTTS') else 'population' if method.endswith('reeval') else 'none')
        assert max(h['generation'] for h in history) == 15
        records.append(dict(method=method, seed=seed, job_id=job, wandb_id=run_id,
                            sources=sources, status='COMPLETED', failure=None,
                            last_completed_generation=15,
                            config={k:run.config.get(k) for k in ['seed','n_runs','reeval','n_reevals','reeval_budget','population_type','population','offspring','generations','timeout','budget_mode','val_n_runs','population_reeval_runs','continue_from']},
                            generation_history=history, points=points))
        print(job, method, seed, 'completed', len(points), flush=True)
    (OUT / 'data.json').write_text(json.dumps(records, indent=2)+'\n')


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    records = json.loads((OUT / 'data.json').read_text())
    with (OUT / 'scores.csv').open('w') as f:
        writer = csv.DictWriter(f, lineterminator='\n', fieldnames=['method','seed','job_id','status','generation','eval_idx','bundle','train_score','train_reeval_score','winners_curse'])
        writer.writeheader()
        for r in records:
            for p in r['points']:
                writer.writerow({**{k:r[k] for k in ['method','seed','job_id','status']}, **p})
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    colors = ['#0072B2','#D55E00','#009E73', '#000000']
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
    handles = [Line2D([], [], color='0.25', ls=SEED_STYLES[seed], label=f'Seed {seed}')
               for seed in sorted({r['seed'] for r in records})]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.96),ncol=3,frameon=False)
    fig.text(.06,.035,'Each point: best candidate reevaluated on training tasks with 10 fresh seeds. Lines join observed points; missing diagnostics are not filled.\nEvaluations = cumulative evolution seed-runs (including selection reevaluations; excluding diagnostics). One seed-run covers the train task set.\nCompleted seeds only; resumed n3 seed 2 includes its pre-resume history. n3-reeval uses population reevaluation, 3 → 10 seeds.',fontsize=9,va='bottom')
    fig.tight_layout(rect=(0,.105,1,.925),h_pad=2.2,w_pad=2.2)
    fig.savefig(OUT / 'train_reevaluation_six_panels.pdf')
    plt.close(fig)
    plot_combined(records)
    write_readme(records)


def plot_combined(records):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#555555', '#000000']
    methods = ['n1', 'n1-reeval', 'n1-TTTS', 'n3', 'n3-reeval', 'n3-TTTS', 'n10']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, xkey in enumerate(['eval_idx', 'generation']):
        for ax, metric, title in zip(axes[row], ['train_reeval_score', 'winners_curse'],
                                     ['Reevaluated train score', 'Winner’s curse']):
            for method, color in zip(methods, colors):
                for r in records:
                    if r['method'] != method:
                        continue
                    p = r['points']
                    ax.plot([v[xkey] for v in p], [v[metric] for v in p],
                            color=color, ls=SEED_STYLES[r['seed']],
                            marker='o', ms=3, lw=1.5, alpha=1 if r['seed'] == 1 else .75,
                            label=method if r['seed'] == 1 else None)
                    if r['status'] == 'FAILED':
                        ax.plot(p[-1][xkey], p[-1][metric], color=color, marker='x', ms=9, mew=2)
            ax.set_title(f'{title} vs {"evaluations" if row == 0 else "generation"}')
            ax.set_xlabel('Cumulative evolution evaluations (seed-runs)' if row == 0 else 'Generation')
            ax.set_ylabel('Reevaluated train score' if metric == 'train_reeval_score'
                          else 'Train score − reevaluated train score')
            ax.set_ylim((.30,.90) if metric == 'train_reeval_score' else (-.06,.36))
            ax.set_xlim((0, max(p['eval_idx'] for r in records for p in r['points']) * 1.04) if row == 0 else (-.35,15.35))
            if row == 1:
                ax.set_xticks(range(0,16,3))
            ax.grid(alpha=.2)
            ax.legend(ncol=2,fontsize=8,frameon=False,loc='upper left')
            if metric == 'winners_curse':
                ax.axhline(0,color='0.4',lw=.8)
    fig.suptitle('All seven 90-second PySR ablations · evaluation and generation comparisons',fontsize=16,y=.98)
    handles = [Line2D([], [], color='0.25', ls=SEED_STYLES[seed], label=f'Seed {seed}')
               for seed in sorted({r['seed'] for r in records})]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
    fig.text(.06,.03,'10 fresh seeds per best-candidate train diagnostic; missing diagnostics are not filled. Completed seeds only; resumed n3 seed 2 includes its earlier history.\nEvaluation counts include selection reevaluations and exclude diagnostics. One seed-run covers the training task set.',fontsize=9)
    fig.tight_layout(rect=(0,.085,1,.91), h_pad=2.5)
    fig.savefig(OUT / 'all_methods_eval_axis.pdf')
    plt.close(fig)


def write_readme(records):
    from datetime import datetime, timezone
    text = f"""# 90-second PySR reevaluation ablations

Updated {datetime.now(timezone.utc).isoformat()}. Includes {len(records)} completed method/seed combinations.

`train_reevaluation_six_panels.pdf` compares n1 and n3 with population reevaluation,
TTTS, and the n10 no-reevaluation reference in every panel. It shows reevaluated
train score versus generation and cumulative evolution evaluations, plus winner's
curse versus generation. `all_methods_eval_axis.pdf` compares all seven methods
on both axes. Solid = seed 1; dashed = seed 2; dotted = seed 3 when complete.
Each seed is a separate curve, not a seed average. Only completed runs are included.

Scores are best-candidate training diagnostics on 10 fresh seeds, parsed from
`[train reeval]` records in local run.log files (four-decimal logging precision).
These are not validation scores or population averages. Winner's curse is the
contemporaneous live train score minus the reevaluated score. Missing diagnostics
are not filled; lines connect available observations, with no extrapolation.

Unsampled W&B generation records supply evolution eval_idx at the submitted
generation, not the diagnostic completion step. Counts include initial population,
offspring and selection reevaluations; they exclude baseline, diagnostic and final
evaluations. A seed-run covers the entire training task set. The evaluation-axis
range includes the full n10 budget.

All runs use topk selection, population/offspring 10, best2 models, 90-second PySR
budgets and 15 generations. Population reevaluation tops up n1 to 3 seeds and n3
to 10; TTTS budgets are 10 and 30 per generation. n10 uses 10 initial seeds per
candidate without selection reevaluation.

The n3 seed-2 continuation merges original job 750247 with 980597. Their generation-12
evaluation counters agree at 390; no offset is applied. This is one seed, not two.
Failed attempts superseded by fresh retries are excluded. Completion requires a
local final_eval_summary.json and a generation-15 W&B record.

| Method | Seed | Source jobs | Last diagnostic generation |
|---|---:|---|---:|
"""
    for r in records:
        sources = ' → '.join(s['job_id'] for s in r['sources'])
        text += f"| {r['method']} | {r['seed']} | {sources} | {r['points'][-1]['generation']} |\n"
    included = {r['job_id'] for r in records}
    excluded = [f'{method} seed {seed} (job {job})' for method, seed, job, _ in RUNS if job not in included]
    text += '\nConfigured but not complete at refresh: ' + (', '.join(excluded) or 'none') + '.\n'
    text += """
Reproduce offline: `python figures/plot_90s_reevaluation_ablations.py`.
Add `--refresh` to reread local logs and W&B. `data.json` preserves the source
configuration, generation counters and diagnostics; `scores.csv` contains plotted
observations. Outputs are PDF only.
"""
    (OUT / 'README.md').write_text(text)


def draw(ax, records, group, xkey, ykey, colors):
    methods = [group,group+'-reeval',group+'-TTTS','n10']
    for method, color in zip(methods, colors):
        for r in records:
            if r['method'] != method:
                continue
            points = r['points']
            ax.plot([p[xkey] for p in points],[p[ykey] for p in points],
                    color=color,ls=SEED_STYLES[r['seed']],marker='o',ms=3,lw=1.6,
                    label=method if r['seed']==1 else None,alpha=1 if r['seed']==1 else .75)
            if r['status']=='FAILED':
                ax.plot(points[-1][xkey],points[-1][ykey],color=color,marker='x',ms=9,mew=2)
    ax.set_xlabel('Generation' if xkey=='generation' else 'Cumulative evolution evaluations (seed-runs)')
    if xkey=='generation':
        ax.set_xlim(-.35,15.35)
        ax.set_xticks(range(0,16,3))
    ax.grid(alpha=.2)
    ax.legend(loc='upper left',ncol=2,fontsize=8,frameon=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.refresh:
        refresh()
    plot()
